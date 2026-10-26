//! Нормалізація гучності за EBU R128 / ACX.
//!
//! Ланцюжок:
//! 1. Вимірювання інтегральної гучності (LUFS) та true peak (dBTP).
//! 2. Статичний gain до цільового LUFS.
//! 3. Look-ahead лімітер за sample peak із запасом під цільовий TP.
//! 4. Фінальна перевірка true peak: якщо після лімітера TP усе ще вище
//!    цілі, весь сигнал додатково затухає (гарантія ACX ≤ −3 dBTP).
//!
//! Типові цілі ACX: −19 LUFS та −3 dBTP.

use std::collections::VecDeque;

use anyhow::{bail, Result};
use ebur128::{EbuR128, Mode};

/// Типова ціль ACX за інтегральною гучністю.
pub const ACX_TARGET_LUFS: f64 = -19.0;
/// Типова ціль ACX за true peak.
pub const ACX_TRUE_PEAK_DB: f64 = -3.0;

/// Сигнал тихіший за це вважаємо тишею і не підсилюємо.
const SILENCE_LUFS: f64 = -70.0;
/// Тривалість look-ahead лімітера.
const LIMITER_LOOKAHEAD_MS: f64 = 5.0;
/// Час відновлення після спрацювання лімітера.
const LIMITER_RELEASE_MS: f64 = 100.0;
/// Розмір блоку обробки (семплів): лімітер не тримає весь сигнал у пам'яті.
const LIMITER_BLOCK: usize = 65_536;

/// Цільові параметри нормалізації.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessTarget {
    pub target_lufs: f64,
    pub true_peak_db: f64,
}

impl LoudnessTarget {
    pub const ACX: Self = Self {
        target_lufs: ACX_TARGET_LUFS,
        true_peak_db: ACX_TRUE_PEAK_DB,
    };

    pub fn new(target_lufs: f64, true_peak_db: f64) -> Self {
        Self {
            target_lufs,
            true_peak_db,
        }
    }

    /// Перевірка діапазонів (як у `ProjectSettings::validate`).
    pub fn validate(&self) -> Result<()> {
        if !self.target_lufs.is_finite() || !(-40.0..=-5.0).contains(&self.target_lufs) {
            bail!("loudness target_lufs must be in -40..=-5");
        }
        if !self.true_peak_db.is_finite() || !(-9.0..=0.0).contains(&self.true_peak_db) {
            bail!("loudness true_peak_db must be in -9..=0");
        }
        Ok(())
    }
}

impl Default for LoudnessTarget {
    fn default() -> Self {
        Self::ACX
    }
}

/// Результат вимірювання.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessMeasurement {
    /// Інтегральна гучність; `None` для тиші або занадто короткого сигналу.
    pub lufs: Option<f64>,
    /// True peak; `None`, якщо сигнал порожній.
    pub true_peak_db: Option<f64>,
}

/// Звіт про нормалізацію.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessReport {
    pub input_lufs: Option<f64>,
    pub input_true_peak_db: Option<f64>,
    pub output_lufs: Option<f64>,
    pub output_true_peak_db: Option<f64>,
    /// Статичний gain до цільового LUFS (дБ).
    pub gain_db: f64,
    /// Додаткове затухання після перевірки TP (дБ, ≤ 0).
    pub limiter_trim_db: f64,
    /// Скільки семплів було обмежено лімітером.
    pub limited_samples: usize,
    /// Чи застосовано нормалізацію (false для тиші/коротких сигналів).
    pub normalized: bool,
}

impl LoudnessReport {
    fn unchanged(measurement: LoudnessMeasurement) -> Self {
        Self {
            input_lufs: measurement.lufs,
            input_true_peak_db: measurement.true_peak_db,
            output_lufs: measurement.lufs,
            output_true_peak_db: measurement.true_peak_db,
            gain_db: 0.0,
            limiter_trim_db: 0.0,
            limited_samples: 0,
            normalized: false,
        }
    }
}

/// Виміряти інтегральну гучність і true peak.
pub fn measure(samples: &[f32], sample_rate: u32) -> Result<LoudnessMeasurement> {
    if sample_rate == 0 {
        bail!("sample rate must be positive");
    }
    if samples.is_empty() {
        return Ok(LoudnessMeasurement {
            lufs: None,
            true_peak_db: None,
        });
    }

    let mut meter = EbuR128::new(1, sample_rate, Mode::I | Mode::TRUE_PEAK)
        .map_err(|error| anyhow::anyhow!("create EBU R128 meter: {error}"))?;
    meter
        .add_frames_f32(samples)
        .map_err(|error| anyhow::anyhow!("analyze loudness: {error}"))?;

    let lufs = meter
        .loudness_global()
        .ok()
        .filter(|value| value.is_finite());
    let true_peak = meter
        .true_peak(0)
        .map_err(|error| anyhow::anyhow!("measure true peak: {error}"))?;
    let true_peak_db = linear_to_db(true_peak);
    Ok(LoudnessMeasurement { lufs, true_peak_db })
}

/// Нормалізувати сигнал до цілі; змінює `samples` на місці.
///
/// Тиша та сигнали коротші за вікно вимірювання (400 мс) не змінюються:
/// підсилювати їх означало б роздувати шум.
pub fn normalize(
    samples: &mut [f32],
    sample_rate: u32,
    target: LoudnessTarget,
) -> Result<LoudnessReport> {
    target.validate()?;
    if sample_rate == 0 {
        bail!("sample rate must be positive");
    }
    if samples.is_empty() {
        return Ok(LoudnessReport::unchanged(LoudnessMeasurement {
            lufs: None,
            true_peak_db: None,
        }));
    }

    let input = measure(samples, sample_rate)?;
    let Some(input_lufs) = input.lufs else {
        return Ok(LoudnessReport::unchanged(input));
    };
    if input_lufs <= SILENCE_LUFS {
        return Ok(LoudnessReport::unchanged(input));
    }

    let gain_db = target.target_lufs - input_lufs;
    let gain = db_to_linear(gain_db) as f32;
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    let ceiling = db_to_linear(target.true_peak_db) as f32;
    let lookahead = ms_to_samples(LIMITER_LOOKAHEAD_MS, sample_rate);
    let release = ms_to_samples(LIMITER_RELEASE_MS, sample_rate);
    let limited_samples = limit_in_place(samples, ceiling, lookahead, 1.0 / release.max(1) as f32);

    // Лімітер гарантує лише sample peak. Перевіряємо міжсемплові піки.
    let mut output = measure(samples, sample_rate)?;
    let mut limiter_trim_db = 0.0;
    if let Some(true_peak) = output.true_peak_db {
        if true_peak > target.true_peak_db {
            limiter_trim_db = target.true_peak_db - true_peak;
            let trim = db_to_linear(limiter_trim_db) as f32;
            for sample in samples.iter_mut() {
                *sample *= trim;
            }
            output = measure(samples, sample_rate)?;
        }
    }

    Ok(LoudnessReport {
        input_lufs: input.lufs,
        input_true_peak_db: input.true_peak_db,
        output_lufs: output.lufs,
        output_true_peak_db: output.true_peak_db,
        gain_db,
        limiter_trim_db,
        limited_samples,
        normalized: true,
    })
}

fn ms_to_samples(ms: f64, sample_rate: u32) -> usize {
    ((ms * sample_rate as f64) / 1000.0).round().max(1.0) as usize
}

fn db_to_linear(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn linear_to_db(linear: f64) -> Option<f64> {
    (linear > 0.0).then(|| 20.0 * linear.log10())
}

/// Look-ahead лімітер за sample peak.
///
/// Для кожного семпла береться мінімум потрібного gain у вікні наперед, тому
/// затухання починається за `lookahead` семплів до піку. Відновлення
/// обмежене `release_step` за семпл. Обидві умови разом гарантують, що
/// жоден семпл не перевищить `ceiling`.
fn limit_in_place(samples: &mut [f32], ceiling: f32, lookahead: usize, release_step: f32) -> usize {
    if samples.is_empty() || ceiling <= 0.0 {
        return 0;
    }
    let window = lookahead.max(1);
    let mut limited = 0usize;
    let mut previous_gain = 1.0f32;
    let mut start = 0usize;

    while start < samples.len() {
        let end = (start + LIMITER_BLOCK).min(samples.len());
        // Хвіст потрібен, щоб look-ahead бачив піки з наступного блоку.
        let look_end = (end + window - 1).min(samples.len());
        let mut required = Vec::with_capacity(look_end - start);
        for &sample in &samples[start..look_end] {
            let magnitude = sample.abs();
            required.push(if magnitude > ceiling {
                ceiling / magnitude
            } else {
                1.0
            });
        }

        let mut forward_min = vec![1.0f32; end - start];
        let mut candidates: VecDeque<usize> = VecDeque::new();
        for index in (0..required.len()).rev() {
            while let Some(&front) = candidates.front() {
                if front >= index + window {
                    candidates.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&back) = candidates.back() {
                if required[back] >= required[index] {
                    candidates.pop_back();
                } else {
                    break;
                }
            }
            candidates.push_back(index);
            if index < forward_min.len() {
                forward_min[index] = required[*candidates.front().expect("deque is non-empty")];
            }
        }

        for (offset, sample) in samples[start..end].iter_mut().enumerate() {
            let gain = forward_min[offset].min(previous_gain + release_step);
            if gain < 1.0 {
                limited += 1;
            }
            *sample *= gain;
            previous_gain = gain;
        }
        start = end;
    }

    limited
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;

    fn sine(freq: f64, amplitude: f32, seconds: f64) -> Vec<f32> {
        let len = (seconds * RATE as f64).round() as usize;
        (0..len)
            .map(|index| {
                let phase = 2.0 * std::f64::consts::PI * freq * index as f64 / RATE as f64;
                amplitude * phase.sin() as f32
            })
            .collect()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected} ± {tolerance}, got {actual}"
        );
    }

    #[test]
    fn measures_reference_tone() {
        let samples = sine(1_000.0, 0.1, 2.0);
        let measurement = measure(&samples, RATE).unwrap();
        // 1 кГц, −20 dBFS -> приблизно −22.98 LUFS (K-зважування).
        assert_close(measurement.lufs.unwrap(), -22.98, 0.1);
        assert_close(measurement.true_peak_db.unwrap(), -20.0, 0.1);
    }

    #[test]
    fn normalizes_reference_tone_to_acx() {
        let mut samples = sine(1_000.0, 0.1, 2.0);
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(report.normalized);
        assert_close(report.gain_db, 3.98, 0.1);
        assert_close(report.output_lufs.unwrap(), ACX_TARGET_LUFS, 0.1);
        assert!(
            report.output_true_peak_db.unwrap() <= ACX_TRUE_PEAK_DB + 1e-6,
            "true peak {} exceeds target",
            report.output_true_peak_db.unwrap()
        );
        assert_eq!(report.limiter_trim_db, 0.0);
    }

    #[test]
    fn limiter_keeps_transients_below_ceiling() {
        // Тихий фон + гучні короткі сплески: статичний gain підніме фон,
        // а лімітер має приборкати сплески.
        let mut samples = sine(440.0, 0.02, 4.0);
        for spike in [1_000usize, 50_000, 90_000] {
            for (offset, sample) in samples[spike..spike + 12].iter_mut().enumerate() {
                *sample = if offset % 2 == 0 { 0.99 } else { -0.99 };
            }
        }
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(report.normalized);
        assert!(report.limited_samples > 0);
        assert!(
            report.output_true_peak_db.unwrap() <= ACX_TRUE_PEAK_DB + 1e-6,
            "true peak {} exceeds target",
            report.output_true_peak_db.unwrap()
        );
        let peak = samples
            .iter()
            .fold(0.0f32, |max, sample| max.max(sample.abs()));
        assert!(peak <= db_to_linear(ACX_TRUE_PEAK_DB) as f32 + 1e-4);
    }

    #[test]
    fn silence_is_left_untouched() {
        let mut samples = vec![0.0f32; RATE as usize];
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(!report.normalized);
        assert_eq!(report.gain_db, 0.0);
        assert_eq!(report.input_lufs, None);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn short_signals_are_left_untouched() {
        let mut samples = sine(1_000.0, 0.1, 0.1);
        let original = samples.clone();
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(!report.normalized);
        assert_eq!(report.gain_db, 0.0);
        assert_eq!(samples, original);
    }

    #[test]
    fn empty_signal_is_left_untouched() {
        let mut samples: Vec<f32> = Vec::new();
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(!report.normalized);
        assert_eq!(report.input_lufs, None);
        assert_eq!(report.output_true_peak_db, None);
    }

    #[test]
    fn rejects_invalid_targets_and_zero_rate() {
        let mut samples = sine(1_000.0, 0.1, 1.0);
        assert!(normalize(&mut samples, RATE, LoudnessTarget::new(-50.0, -3.0)).is_err());
        assert!(normalize(&mut samples, RATE, LoudnessTarget::new(-19.0, 3.0)).is_err());
        assert!(normalize(&mut samples, 0, LoudnessTarget::ACX).is_err());
    }

    #[test]
    fn already_normalized_signal_stays_close() {
        let mut samples = sine(1_000.0, 0.1, 2.0);
        normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        let report = normalize(&mut samples, RATE, LoudnessTarget::ACX).unwrap();
        assert!(report.normalized);
        assert!(report.gain_db.abs() < 0.1);
        assert_close(report.output_lufs.unwrap(), ACX_TARGET_LUFS, 0.1);
    }
}
