//! Аудіо-діагностика готових глав: кліпінг, тиша, DC offset і рівень.
//!
//! Модуль не залежить від моделі чи сервера: на вході — mono float32 у
//! діапазоні [-1, 1]. Перевірки орієнтовані на аудіокниги: довгі паузи,
//! обрізані піки, постійна складова та занадто тихий/гучний матеріал.

use anyhow::{bail, Result};
use serde::Serialize;

/// Поріг, вище якого семпл вважається обрізаним.
pub const CLIPPING_THRESHOLD: f32 = 0.999;
/// Поріг тиші за замовчуванням.
pub const SILENCE_DBFS: f32 = -50.0;
/// Мінімальна тривалість паузи, яка потрапляє в `longest_silence`.
pub const MIN_SILENCE_MS: u32 = 300;
/// Пауза, довша за цю, вважається занадто довгою для аудіокниги.
pub const LONG_PAUSE_MS: u64 = 2_000;
/// Довга тиша на початку або в кінці глави.
pub const MAX_EDGE_SILENCE_MS: u64 = 2_000;
/// Постійна складова, вище якої варто попередити.
pub const DC_OFFSET_WARN: f32 = 0.01;
/// Занадто тихий середній рівень.
pub const LOW_RMS_DBFS: f32 = -40.0;
/// Пік майже на межі повної шкали.
pub const NEAR_FULL_SCALE_DBFS: f32 = -0.1;

/// Важливість діагностичної знахідки.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// Одна знахідка діагностики.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiagnosticIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
}

/// Інтервал тиші у таймлайні глави.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SilenceSpan {
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
}

/// Повний звіт про главу.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioDiagnostics {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_ms: u64,
    pub peak_dbfs: Option<f32>,
    pub rms_dbfs: Option<f32>,
    pub crest_factor_db: Option<f32>,
    pub dc_offset: f32,
    pub clipped_samples: usize,
    pub clipping_ratio: f32,
    pub silence_ratio: f32,
    pub leading_silence_ms: u64,
    pub trailing_silence_ms: u64,
    pub longest_silence: Option<SilenceSpan>,
    pub issues: Vec<DiagnosticIssue>,
}

/// Проаналізувати mono-сигнал.
pub fn analyze(samples: &[f32], sample_rate: u32) -> Result<AudioDiagnostics> {
    if sample_rate == 0 {
        bail!("sample rate must be positive");
    }
    let sample_count = samples.len();
    let duration_ms = sample_count as u64 * 1000 / u64::from(sample_rate);
    if samples.is_empty() {
        return Ok(AudioDiagnostics {
            sample_rate,
            sample_count: 0,
            duration_ms: 0,
            peak_dbfs: None,
            rms_dbfs: None,
            crest_factor_db: None,
            dc_offset: 0.0,
            clipped_samples: 0,
            clipping_ratio: 0.0,
            silence_ratio: 0.0,
            leading_silence_ms: 0,
            trailing_silence_ms: 0,
            longest_silence: None,
            issues: vec![issue(
                "empty_audio",
                Severity::Critical,
                "chapter contains no audio samples",
            )],
        });
    }

    let mut peak = 0.0f32;
    let mut sum = 0.0f64;
    let mut sum_squares = 0.0f64;
    let mut clipped_samples = 0usize;
    for &sample in samples {
        let magnitude = sample.abs();
        if magnitude > peak {
            peak = magnitude;
        }
        if magnitude >= CLIPPING_THRESHOLD {
            clipped_samples += 1;
        }
        sum += f64::from(sample);
        sum_squares += f64::from(sample) * f64::from(sample);
    }
    let rms = (sum_squares / sample_count as f64).sqrt();
    let peak_dbfs = amplitude_to_dbfs(peak);
    let rms_dbfs = amplitude_to_dbfs(rms as f32);
    let crest_factor_db = peak_dbfs.zip(rms_dbfs).map(|(peak, rms)| peak - rms);
    let dc_offset = (sum / sample_count as f64) as f32;
    let clipping_ratio = clipped_samples as f32 / sample_count as f32;

    let silence = silence_stats(samples, sample_rate);

    let mut issues = Vec::new();
    if peak <= f32::EPSILON {
        issues.push(issue(
            "silent_audio",
            Severity::Critical,
            "chapter is completely silent",
        ));
    } else {
        if clipped_samples > 0 {
            issues.push(issue(
                "clipping",
                Severity::Warning,
                format!(
                    "clipping detected in {clipped_samples} samples ({:.3}%)",
                    clipping_ratio * 100.0
                ),
            ));
        }
        if let Some(peak_dbfs) = peak_dbfs {
            if peak_dbfs > NEAR_FULL_SCALE_DBFS {
                issues.push(issue(
                    "near_full_scale",
                    Severity::Warning,
                    format!("peak is {peak_dbfs:.1} dBFS, close to full scale"),
                ));
            }
        }
        if let Some(rms_dbfs) = rms_dbfs {
            if rms_dbfs < LOW_RMS_DBFS {
                issues.push(issue(
                    "low_level",
                    Severity::Warning,
                    format!("average level is {rms_dbfs:.1} dBFS"),
                ));
            }
        }
    }
    if dc_offset.abs() > DC_OFFSET_WARN {
        issues.push(issue(
            "dc_offset",
            Severity::Warning,
            format!("DC offset is {:+.4}", dc_offset),
        ));
    }
    if silence.leading_ms > MAX_EDGE_SILENCE_MS {
        issues.push(issue(
            "leading_silence",
            Severity::Info,
            format!("{} ms of silence at the start", silence.leading_ms),
        ));
    }
    if silence.trailing_ms > MAX_EDGE_SILENCE_MS {
        issues.push(issue(
            "trailing_silence",
            Severity::Info,
            format!("{} ms of silence at the end", silence.trailing_ms),
        ));
    }
    if let Some(span) = silence.longest {
        if span.duration_ms > LONG_PAUSE_MS {
            issues.push(issue(
                "long_pause",
                Severity::Warning,
                format!(
                    "pause of {} ms at {}:{:02}.{:02}",
                    span.duration_ms,
                    span.start_ms / 60_000,
                    (span.start_ms / 1_000) % 60,
                    (span.start_ms % 1_000) / 10
                ),
            ));
        }
    }
    if duration_ms < 500 {
        issues.push(issue(
            "short_audio",
            Severity::Info,
            format!("chapter is only {duration_ms} ms long"),
        ));
    }

    Ok(AudioDiagnostics {
        sample_rate,
        sample_count,
        duration_ms,
        peak_dbfs,
        rms_dbfs,
        crest_factor_db,
        dc_offset,
        clipped_samples,
        clipping_ratio,
        silence_ratio: silence.ratio,
        leading_silence_ms: silence.leading_ms,
        trailing_silence_ms: silence.trailing_ms,
        longest_silence: silence.longest,
        issues,
    })
}

struct SilenceStats {
    ratio: f32,
    leading_ms: u64,
    trailing_ms: u64,
    longest: Option<SilenceSpan>,
}

fn silence_stats(samples: &[f32], sample_rate: u32) -> SilenceStats {
    let threshold = db_to_amplitude(SILENCE_DBFS);
    let min_samples = (u64::from(sample_rate) * u64::from(MIN_SILENCE_MS) / 1000).max(1) as usize;
    let mut silent_total = 0usize;
    let mut leading_ms = 0u64;
    let mut trailing_ms = 0u64;
    let mut longest: Option<SilenceSpan> = None;
    let mut run_start: Option<usize> = None;

    let mut record = |start: usize, end: usize| {
        let span = span(start, end, sample_rate);
        if start == 0 {
            leading_ms = span.duration_ms;
        }
        if end == samples.len() {
            trailing_ms = span.duration_ms;
        }
        if end - start >= min_samples
            && longest.is_none_or(|current| span.duration_ms > current.duration_ms)
        {
            longest = Some(span);
        }
    };

    for (index, &sample) in samples.iter().enumerate() {
        if sample.abs() < threshold {
            silent_total += 1;
            if run_start.is_none() {
                run_start = Some(index);
            }
        } else if let Some(start) = run_start.take() {
            record(start, index);
        }
    }
    if let Some(start) = run_start {
        record(start, samples.len());
    }

    SilenceStats {
        ratio: silent_total as f32 / samples.len() as f32,
        leading_ms,
        trailing_ms,
        longest,
    }
}

fn span(start: usize, end: usize, sample_rate: u32) -> SilenceSpan {
    let start_ms = start as u64 * 1000 / u64::from(sample_rate);
    let end_ms = end as u64 * 1000 / u64::from(sample_rate);
    SilenceSpan {
        start_ms,
        end_ms,
        duration_ms: (end - start) as u64 * 1000 / u64::from(sample_rate),
    }
}

fn amplitude_to_dbfs(amplitude: f32) -> Option<f32> {
    (amplitude > 0.0).then(|| 20.0 * amplitude.log10())
}

fn db_to_amplitude(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn issue(code: &str, severity: Severity, message: impl Into<String>) -> DiagnosticIssue {
    DiagnosticIssue {
        code: code.to_string(),
        severity,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 24_000;

    fn tone(seconds: f32, amplitude: f32) -> Vec<f32> {
        let count = (seconds * RATE as f32) as usize;
        (0..count)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * 440.0 * index as f32 / RATE as f32;
                amplitude * phase.sin()
            })
            .collect()
    }

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0.0; (seconds * RATE as f32) as usize]
    }

    fn codes(diagnostics: &AudioDiagnostics) -> Vec<&str> {
        diagnostics
            .issues
            .iter()
            .map(|issue| issue.code.as_str())
            .collect()
    }

    #[test]
    fn reports_complete_silence() {
        let diagnostics = analyze(&silence(1.0), RATE).unwrap();
        assert_eq!(diagnostics.peak_dbfs, None);
        assert_eq!(diagnostics.leading_silence_ms, 1_000);
        assert_eq!(diagnostics.trailing_silence_ms, 1_000);
        assert_eq!(diagnostics.silence_ratio, 1.0);
        assert!(codes(&diagnostics).contains(&"silent_audio"));
    }

    #[test]
    fn detects_clipping() {
        let mut samples = tone(1.0, 0.5);
        for sample in samples.iter_mut().take(10) {
            *sample = 1.0;
        }
        let diagnostics = analyze(&samples, RATE).unwrap();
        assert_eq!(diagnostics.clipped_samples, 10);
        assert!(codes(&diagnostics).contains(&"clipping"));
        assert!(codes(&diagnostics).contains(&"near_full_scale"));
    }

    #[test]
    fn finds_long_pause_between_tones() {
        let mut samples = tone(0.5, 0.4);
        samples.extend(silence(2.2));
        samples.extend(tone(0.5, 0.4));
        let diagnostics = analyze(&samples, RATE).unwrap();
        let pause = diagnostics.longest_silence.expect("pause is detected");
        assert!((2_190..=2_210).contains(&pause.duration_ms));
        assert_eq!(pause.start_ms, 500);
        assert!(codes(&diagnostics).contains(&"long_pause"));
    }

    #[test]
    fn reports_edges_and_dc_offset() {
        let mut samples = silence(0.5);
        samples.extend(std::iter::repeat_n(0.2f32, RATE as usize));
        samples.extend(silence(0.5));
        let diagnostics = analyze(&samples, RATE).unwrap();
        assert_eq!(diagnostics.leading_silence_ms, 500);
        assert_eq!(diagnostics.trailing_silence_ms, 500);
        assert!(diagnostics.dc_offset > 0.09);
        assert!(codes(&diagnostics).contains(&"dc_offset"));
    }

    #[test]
    fn flags_quiet_material() {
        let diagnostics = analyze(&tone(1.0, 0.001), RATE).unwrap();
        assert!(diagnostics.rms_dbfs.unwrap() < LOW_RMS_DBFS);
        assert!(codes(&diagnostics).contains(&"low_level"));
    }

    #[test]
    fn empty_and_invalid_input() {
        let diagnostics = analyze(&[], RATE).unwrap();
        assert_eq!(diagnostics.duration_ms, 0);
        assert_eq!(codes(&diagnostics), vec!["empty_audio"]);
        assert!(analyze(&[], 0).is_err());
    }
}
