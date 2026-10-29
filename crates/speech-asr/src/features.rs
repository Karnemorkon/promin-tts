//! W2V-BERT frontend: SeamlessM4T log-mel (80 bins) + stride 2.
//!
//! Використовуємо `asr-features` (MIT) замість власного порту fbank: саме
//! цим крейтом згенеровано еталонні ознаки для `speech-uk/w2v-bert-v3`.

use anyhow::{bail, Context, Result};
use asr_features::{
    extract_w2v_bert_features_from_samples, w2v_bert_frontend_config, W2vBertFrontendConfig,
};

pub use asr_features::FeatureMatrix;

/// Розмірність ознак після stride 2 (80 mel × 2).
pub const FEATURE_DIM: usize = 160;
/// Частота, яку очікує модель.
pub const SAMPLE_RATE: u32 = 16_000;

/// Конфігурація frontend для конкретної моделі.
pub fn config(model_source: &str) -> W2vBertFrontendConfig {
    w2v_bert_frontend_config(
        Some(model_source.to_string()),
        Some(SAMPLE_RATE),
        Some(80),
        Some(2),
        Some(FEATURE_DIM),
        Some(1.0),
    )
}

/// Обчислити ознаки для mono-сигналу.
pub fn extract_w2v_bert(
    samples: &[f32],
    sample_rate: u32,
    model_source: &str,
) -> Result<FeatureMatrix> {
    if samples.is_empty() {
        bail!("cannot extract ASR features from empty audio");
    }
    if sample_rate == 0 {
        bail!("sample rate must be positive");
    }
    let config = config(model_source);
    let features = extract_w2v_bert_features_from_samples(samples, sample_rate, &config)
        .context("extract W2V-BERT features")?;
    if features.rows == 0 || features.cols != FEATURE_DIM {
        bail!(
            "unexpected ASR feature shape: {}x{} (expected Nx{FEATURE_DIM})",
            features.rows,
            features.cols
        );
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_expected_shape_from_silence() {
        let samples = vec![0.0f32; SAMPLE_RATE as usize];
        let features = extract_w2v_bert(&samples, SAMPLE_RATE, "test").unwrap();
        assert!(features.rows > 0);
        assert_eq!(features.cols, FEATURE_DIM);
        assert_eq!(features.values.len(), features.rows * features.cols);
    }

    #[test]
    fn resamples_other_rates() {
        let samples = vec![0.0f32; 24_000];
        let features = extract_w2v_bert(&samples, 24_000, "test").unwrap();
        assert_eq!(features.cols, FEATURE_DIM);
        assert!(features.rows > 0);
    }

    #[test]
    fn rejects_empty_audio() {
        assert!(extract_w2v_bert(&[], SAMPLE_RATE, "test").is_err());
    }
}
