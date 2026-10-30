//! Ресемплінг референсного аудіо до 24 кГц (rubato, FFT).

use anyhow::{bail, Context, Result};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};

use crate::SAMPLE_RATE;

/// Ресемплить mono-сигнал до [`SAMPLE_RATE`] (24 кГц).
pub fn resample_to_24k(samples: Vec<f32>, sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate == 0 {
        bail!("sample rate must be positive");
    }
    if sample_rate == SAMPLE_RATE || samples.is_empty() {
        return Ok(samples);
    }

    let input_len = samples.len();
    let mut resampler = Fft::<f32>::new(
        sample_rate as usize,
        SAMPLE_RATE as usize,
        1024,
        1,
        FixedSync::Input,
    )
    .context("create resampler")?;
    let input_data = [samples];
    let input =
        SequentialSliceOfVecs::new(&input_data, 1, input_len).context("wrap resampler input")?;
    let output = resampler
        .process_all(&input, input_len, None)
        .context("resample reference audio")?;
    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn keeps_24k_untouched() {
        let samples = vec![0.1, -0.2, 0.3];
        assert_eq!(resample_to_24k(samples.clone(), 24_000).unwrap(), samples);
    }

    #[test]
    fn resamples_48k_to_24k() {
        let input: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let output = resample_to_24k(input, 48_000).unwrap();
        assert!(
            (output.len() as i64 - 24_000).abs() <= 2,
            "len {}",
            output.len()
        );
        assert!(output.iter().all(|value| value.is_finite()));
        let peak = output
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        assert!(peak > 0.4 && peak < 0.6, "peak {peak}");
    }

    #[test]
    fn resamples_16k_to_24k() {
        let input = vec![0.25f32; 16_000];
        let output = resample_to_24k(input, 16_000).unwrap();
        assert!(
            (output.len() as i64 - 24_000).abs() <= 2,
            "len {}",
            output.len()
        );
    }
}
