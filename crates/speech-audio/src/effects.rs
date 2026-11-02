//! DSP-ефекти аудіо-процесора (ТЗ Етап 2: «алгоритми DSP (Pitch,
//! Reverb, Compression)»).
//!
//! Усі функції працюють із mono float32 у діапазоні [-1, 1], повертають
//! сигнал **тієї самої довжини** (тривалість не змінюється) і не залежать
//! від конкретної частоти дискретизації моделі — її треба передати явно.
//! Ефекти самодостатні: їх можна застосовувати ланцюжком у будь-якому
//! порядку, наприклад `compress → reverb → pitch_shift`.
//!
//! Кросфейд і LUFS-нормалізація живуть окремо: див. [`crate::crossfade`]
//! та [`crate::loudness`].

/// Зсув висоти тональності в півтонах.
///
/// Алгоритм двостадійний:
/// 1) лінійний ресемплінг із коефіцієнтом `2^(semitones/12)` (змінює
///    висоту і тривалість);
/// 2) WSOLA-розтягнення повертає вихідну тривалість без зміни висоти
///    (пошук найкращого вирівнювання фреймів ±25% та overlap-add з
///    вікном Ганна).
///
/// Довжина виходу завжди дорівнює довжині входу; при `semitones ≈ 0`,
/// поза межами ±24 півтони або короткому вході (< 40 мс) повертається
/// копія/ресемпл без time-stretching.
pub fn pitch_shift(samples: &[f32], sample_rate: u32, semitones: f32) -> Vec<f32> {
    if samples.is_empty() || !semitones.is_finite() || semitones.abs() < 0.01 {
        return samples.to_vec();
    }
    let ratio = 2f32.powf(semitones / 12.0);
    if !ratio.is_finite() || !(0.25..=4.0).contains(&ratio) {
        return samples.to_vec();
    }

    // Стадія 1: змінити висоту прискоренням/уповільненням читання.
    let fast = resample_playback(samples, ratio);
    // Стадія 2: повернути тривалість без зміни висоти.
    wsola_stretch(&fast, samples.len(), sample_rate)
}

/// Лінійний ресемплінг: читання вхідного сигналу кроком `ratio`
/// (більше за 1 — швидше, отже коротше).
fn resample_playback(samples: &[f32], ratio: f32) -> Vec<f32> {
    let n = samples.len();
    if n <= 1 {
        return samples.to_vec();
    }
    let out_len = ((n as f64) / f64::from(ratio)).round().max(1.0) as usize;
    (0..out_len)
        .map(|i| {
            let pos = f64::from(ratio) * i as f64;
            let idx = pos.floor() as usize;
            if idx + 1 < n {
                let t = (pos - idx as f64) as f32;
                samples[idx] * (1.0 - t) + samples[idx + 1] * t
            } else {
                samples[n - 1]
            }
        })
        .collect()
}

/// Розтягнення/стиснення WSOLA до заданої довжини без зміни висоти.
///
/// Синтезований крок дорівнює половині фрейма, аналітичний —
/// `крок / розтягнення`; для кожного фрейма вибирається зсув у межах
/// ±25% фрейма з максимальною нормалізованою кореляцією з уже
/// синтезованим перекриттям (тому голос не «рветься» на межах вікон).
fn wsola_stretch(input: &[f32], target_len: usize, sample_rate: u32) -> Vec<f32> {
    let frame = ((sample_rate as usize * 40 / 1000).max(64)) & !1; // ~40 мс, парний
    if target_len == 0 {
        return Vec::new();
    }
    // Вхід коротший за фрейм — time-stretch не застосовується.
    if input.len() <= frame {
        return resample_exact_length(input, target_len);
    }

    let hop = frame / 2;
    let search = (frame / 4).max(1);
    let stretch = target_len as f32 / input.len() as f32;
    let analysis_hop = hop as f32 / stretch;

    // Вікно Ганна з мінімальним рівнем: на краях синтезу перекриття
    // одне, тож фрейм має відновлюватися без артефактів затухання.
    let window: Vec<f32> = (0..frame)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / frame as f32;
            (0.5 - 0.5 * angle.cos()).max(1e-3)
        })
        .collect();

    let capacity = target_len + frame;
    let mut out = vec![0.0f32; capacity];
    let mut weight = vec![0.0f32; capacity];
    let mut analysis_pos = 0.0f64;
    let mut synth = 0usize;

    while synth < target_len {
        let base = analysis_pos.round() as i64;
        let max_base = (input.len() - frame) as i64;
        let mut chosen = base.clamp(0, max_base);

        // Пошук найкращого вирівнювання з уже синтезованим перекриттям.
        if synth > 0 {
            let mut best_score = f64::NEG_INFINITY;
            for delta in -(search as i64)..=(search as i64) {
                let cand = (base + delta).clamp(0, max_base);
                let score = normalized_correlation(
                    &input[cand as usize..cand as usize + hop],
                    &out[synth..synth + hop],
                );
                if score > best_score {
                    best_score = score;
                    chosen = cand;
                }
            }
        }

        let cand = chosen as usize;
        for k in 0..frame {
            out[synth + k] += input[cand + k] * window[k];
            weight[synth + k] += window[k];
        }
        synth += hop;
        analysis_pos += f64::from(analysis_hop);
    }

    (0..target_len)
        .map(|i| {
            if weight[i] > 1e-6 {
                out[i] / weight[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// Нормалізована кореляція двох однакових за довжиною відрізків (-1..1).
fn normalized_correlation(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

/// Лінійний ресемплінг до точної довжини (fallback для коротких входів).
fn resample_exact_length(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    if input.len() == 1 {
        return vec![input[0]; target_len];
    }
    let scale = (input.len() - 1) as f64 / (target_len - 1).max(1) as f64;
    (0..target_len)
        .map(|i| {
            if target_len == 1 {
                return input[0];
            }
            let pos = scale * i as f64;
            let idx = (pos.floor() as usize).min(input.len() - 2);
            let t = (pos - idx as f64) as f32;
            input[idx] * (1.0 - t) + input[idx + 1] * t
        })
        .collect()
}

/// Параметри реверберації (алгоритм Шройдера: 4 паралельні comb-фільтри
/// + 2 послідовні allpass, як у Freeverb; тюнінги масштабуються під
///   частоту дискретизації).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbParams {
    /// Розмір кімнати 0..1 (більше — довший хвіст).
    pub room_size: f32,
    /// Затухання високих частот 0..1 (більше — темніший реверб).
    pub damping: f32,
    /// Частка обробленого сигналу (wet) 0..1.
    pub wet: f32,
    /// Частка вихідного сигналу (dry) 0..1.
    pub dry: f32,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.3,
            wet: 0.25,
            dry: 1.0,
        }
    }
}

/// Comb-фільтр із демпфуванням (елемент ревербера).
struct Comb {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damping: f32,
    filter_state: f32,
}

impl Comb {
    fn new(delay: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay.max(1)],
            index: 0,
            feedback,
            damping,
            filter_state: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let out = self.buffer[self.index];
        self.filter_state = out * (1.0 - self.damping) + self.filter_state * self.damping;
        self.buffer[self.index] = input + self.filter_state * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();
        out
    }
}

/// Allpass-фільтр (фазова корекція ревербера).
struct Allpass {
    buffer: Vec<f32>,
    index: usize,
    gain: f32,
}

impl Allpass {
    fn new(delay: usize, gain: f32) -> Self {
        Self {
            buffer: vec![0.0; delay.max(1)],
            index: 0,
            gain,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buffer[self.index];
        let out = buffered - input;
        self.buffer[self.index] = input + buffered * self.gain;
        self.index = (self.index + 1) % self.buffer.len();
        out
    }
}

/// Комбіновані тюнінги Freeverb у семплах при 44.1 кГц.
const COMB_TUNING: [f32; 4] = [1116.0, 1188.0, 1277.0, 1356.0];
const ALLPASS_TUNING: [f32; 2] = [556.0, 441.0];

/// Реверберація. Вихід тієї самої довжини, що й вхід.
///
/// `wet = 0` — чистий dry-прохід (без обробки).
pub fn reverb(samples: &[f32], sample_rate: u32, params: &ReverbParams) -> Vec<f32> {
    let wet = params.wet.clamp(0.0, 1.0);
    let dry = params.dry.clamp(0.0, 1.0);
    if samples.is_empty() {
        return Vec::new();
    }
    if wet == 0.0 {
        return samples.iter().map(|&s| s * dry).collect();
    }

    let scale = f64::from(sample_rate) / 44_100.0;
    let tune = |base: f32| ((f64::from(base) * scale) as usize).max(1);

    // Як у Freeverb: 0.7..0.98 — стабільно на будь-яких тюнінгах.
    let feedback = 0.7 + 0.28 * params.room_size.clamp(0.0, 1.0);
    let damping = params.damping.clamp(0.0, 1.0);
    let mut combs: Vec<Comb> = COMB_TUNING
        .iter()
        .map(|&delay| Comb::new(tune(delay), feedback, damping))
        .collect();
    let mut allpasses: Vec<Allpass> = ALLPASS_TUNING
        .iter()
        .map(|&delay| Allpass::new(tune(delay), 0.5))
        .collect();

    samples
        .iter()
        .map(|&input| {
            let wet_signal: f32 = combs
                .iter_mut()
                .map(|comb| comb.process(input))
                .sum::<f32>()
                * 0.25;
            let wet_signal = allpasses
                .iter_mut()
                .fold(wet_signal, |signal, pass| pass.process(signal));
            input * dry + wet_signal * wet
        })
        .collect()
}

/// Параметри динамічного компресора (feed-forward, детектор —
/// наростаючий envelope follower за амплітудою, hard knee).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorParams {
    /// Поріг у dBFS, вище якого стиснення активується.
    pub threshold_db: f32,
    /// Стиснення: скільки dB на вході дає 1 dB на виході (≥ 1).
    pub ratio: f32,
    /// Час наростання в мілісекундах.
    pub attack_ms: f32,
    /// Час спадання в мілісекундах.
    pub release_ms: f32,
    /// Компенсація гучності після стиснення, dB.
    pub makeup_db: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 0.0,
        }
    }
}

/// Стиснення динаміки. Вихід тієї самої довжини, що й вхід.
///
/// Сигнал нижче порогу проходить без змін (біт-у-біт, `makeup_db = 0`).
pub fn compress(samples: &[f32], sample_rate: u32, params: &CompressorParams) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let ratio = params.ratio.max(1.0);
    let threshold = params.threshold_db;
    let makeup = 10f32.powf(params.makeup_db / 20.0);
    let attack = envelope_coefficient(params.attack_ms, sample_rate);
    let release = envelope_coefficient(params.release_ms, sample_rate);

    let mut envelope = 0.0f32;
    samples
        .iter()
        .map(|&sample| {
            let magnitude = sample.abs();
            envelope = if magnitude > envelope {
                attack * envelope + (1.0 - attack) * magnitude
            } else {
                release * envelope + (1.0 - release) * magnitude
            };

            let level_db = 20.0 * (envelope + 1e-9).log10();
            let over_db = level_db - threshold;
            let gain_db = if over_db > 0.0 && ratio > 1.0 {
                over_db * (1.0 / ratio - 1.0)
            } else {
                0.0
            };
            sample * 10f32.powf(gain_db / 20.0) * makeup
        })
        .collect()
}

/// Коефіцієнт експоненційного фільтра для заданого часу (мс).
fn envelope_coefficient(ms: f32, sample_rate: u32) -> f32 {
    let samples = (f64::from(ms) * f64::from(sample_rate) / 1000.0).max(1.0);
    (-1.0 / samples).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Оцінка частоти сигналу автокореляцією (з параболічним
    /// уточненням піку) по середній частині відрізка.
    fn estimate_frequency(samples: &[f32], sample_rate: u32) -> f32 {
        let segment = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let min_lag = (sample_rate as usize / 1000).max(2); // ≤ 1 кГц
        let max_lag = sample_rate as usize / 50; // ≥ 50 Гц

        let correlation_at = |lag: usize| -> f64 {
            let n = segment.len() - lag;
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                let x = f64::from(segment[i]);
                let y = f64::from(segment[i + lag]);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / ((na.sqrt() * nb.sqrt()).max(1e-12))
        };

        let scores: Vec<(usize, f64)> = (min_lag..=max_lag.min(segment.len() / 2))
            .map(|lag| (lag, correlation_at(lag)))
            .collect();
        let global_max = scores
            .iter()
            .map(|&(_, s)| s)
            .fold(f64::NEG_INFINITY, f64::max);
        // Найменший лаг із майже максимальним піком = базовий період
        // (кратні періоди теж корелюються, їх відсіюємо порогом).
        let threshold = global_max * 0.98;
        let (lag, peak) = scores
            .iter()
            .find(|&&(_, score)| score >= threshold)
            .copied()
            .expect("frequency peak not found");

        // Параболічна інтерполяція навколо піку для точності.
        let before = correlation_at(lag.saturating_sub(1));
        let after = correlation_at(lag + 1);
        let denom = before - 2.0 * peak + after;
        let shift = if denom.abs() > 1e-12 {
            (0.5 * (before - after) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        f64::from(sample_rate) as f32 / (lag as f32 + shift as f32)
    }

    fn sine(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let n = (f64::from(sample_rate) * f64::from(seconds)) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (std::f32::consts::TAU * frequency * t).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn pitch_shift_up_keeps_duration_and_raises_frequency() {
        let sample_rate = 24_000;
        let input = sine(440.0, sample_rate, 1.0);
        let output = pitch_shift(&input, sample_rate, 3.0);
        assert_eq!(output.len(), input.len(), "тривалість має зберегтися");
        assert!(output.iter().all(|s| s.is_finite()));

        let before = estimate_frequency(&input, sample_rate);
        let after = estimate_frequency(&output, sample_rate);
        let expected = before * 2f32.powf(3.0 / 12.0);
        let error = (after - expected).abs() / expected;
        assert!(
            error < 0.02,
            "expected ~{expected:.1} Hz, got {after:.1} Hz (error {error:.3})"
        );
    }

    #[test]
    fn pitch_shift_down_lowers_frequency() {
        let sample_rate = 24_000;
        let input = sine(440.0, sample_rate, 1.0);
        let output = pitch_shift(&input, sample_rate, -4.0);
        assert_eq!(output.len(), input.len());

        let before = estimate_frequency(&input, sample_rate);
        let after = estimate_frequency(&output, sample_rate);
        let expected = before * 2f32.powf(-4.0 / 12.0);
        let error = (after - expected).abs() / expected;
        assert!(
            error < 0.02,
            "expected ~{expected:.1} Hz, got {after:.1} Hz"
        );
    }

    #[test]
    fn pitch_shift_zero_and_short_input_are_passthrough() {
        let samples = vec![0.25; 64];
        assert_eq!(pitch_shift(&samples, 24_000, 0.0), samples);
        // Коротший за один фрейм (40 мс) — fallback-ресемплінг, довжина та сама.
        let short = sine(440.0, 24_000, 0.02);
        let output = pitch_shift(&short, 24_000, 2.0);
        assert_eq!(output.len(), short.len());
        assert!(output.iter().all(|s| s.is_finite()));
        assert_eq!(pitch_shift(&[], 24_000, 2.0), Vec::<f32>::new());
    }

    #[test]
    fn reverb_wet_zero_is_dry_passthrough() {
        let samples = sine(220.0, 24_000, 0.25);
        let params = ReverbParams {
            wet: 0.0,
            dry: 1.0,
            ..ReverbParams::default()
        };
        assert_eq!(reverb(&samples, 24_000, &params), samples);
        assert!(reverb(&[], 24_000, &ReverbParams::default()).is_empty());
    }

    #[test]
    fn reverb_impulse_produces_decaying_tail() {
        let mut input = vec![0.0f32; 12_000];
        input[0] = 1.0;
        let params = ReverbParams {
            room_size: 0.2,
            wet: 1.0,
            dry: 0.0,
            ..ReverbParams::default()
        };
        let output = reverb(&input, 24_000, &params);
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|s| s.is_finite()));

        let window_energy = |window: std::ops::Range<usize>| -> f64 {
            output[window]
                .iter()
                .map(|s| f64::from(*s) * f64::from(*s))
                .sum()
        };
        let early = window_energy(100..1100);
        let late = window_energy(6000..7000);
        assert!(early > 1e-6, "хвіст має існувати одразу після імпульсу");
        assert!(
            late < early * 0.05,
            "енергія хвоста має затихати: early={early:.6} late={late:.6}"
        );
    }

    #[test]
    fn reverb_processes_audio_in_place_length() {
        let samples = sine(330.0, 24_000, 0.5);
        let output = reverb(&samples, 24_000, &ReverbParams::default());
        assert_eq!(output.len(), samples.len());
        assert!(output.iter().all(|s| s.is_finite()));
        // З доданним wet сигнал відрізняється від сухого.
        assert_ne!(output, samples);
    }

    #[test]
    fn compressor_passes_quiet_signal_unchanged() {
        let samples = sine(440.0, 24_000, 0.1)
            .into_iter()
            .map(|s| s * 0.02)
            .collect::<Vec<f32>>();
        let output = compress(&samples, 24_000, &CompressorParams::default());
        assert_eq!(output, samples, "нижче порогу — біт-у-біт");
        assert!(compress(&[], 24_000, &CompressorParams::default()).is_empty());
    }

    #[test]
    fn compressor_reduces_loud_signal_to_expected_level() {
        let sample_rate = 24_000;
        let params = CompressorParams {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            makeup_db: 0.0,
        };
        let input = sine(440.0, sample_rate, 0.5);
        let output = compress(&input, sample_rate, &params);

        // Після налагодження (перші 100 мс) пік має сісти на
        // threshold + (in_peak - threshold)/ratio.
        let peak_db = |samples: &[f32]| -> f64 {
            let settled = &samples[samples.len() / 5..];
            let peak = settled.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            20.0 * (f64::from(peak) + 1e-12).log10()
        };
        let input_db = peak_db(&input);
        let output_db = peak_db(&output);
        let expected_db = -20.0 + (input_db + 20.0) / 4.0;
        assert!(
            (output_db - expected_db).abs() < 2.0,
            "expected ~{expected_db:.1} dBFS, got {output_db:.1} dBFS"
        );
        assert!(
            output_db < input_db - 6.0,
            "louder signal must be clearly reduced: in {input_db:.1}, out {output_db:.1}"
        );
        assert!(output.iter().all(|s| s.is_finite()));
    }
}
