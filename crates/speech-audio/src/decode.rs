//! WAV (PCM/float) -> mono float32 у діапазоні [-1, 1].

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader};

/// Декодує WAV-байти у mono float32; повертає (samples, sample_rate).
///
/// Підтримуються PCM 8/16/24/32-bit та IEEE float 32-bit. Багатоканальні
/// записи усереднюються у моно (референс для voice encoder).
pub fn decode_wav_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = WavReader::new(Cursor::new(bytes)).context("parse WAV")?;
    let spec = reader.spec();
    if spec.channels == 0 {
        bail!("WAV has zero channels");
    }
    if spec.sample_rate == 0 {
        bail!("WAV has zero sample rate");
    }

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8) => reader
            .samples::<i8>()
            .map(|sample| sample.map(|value| f32::from(value) / 128.0))
            .collect::<Result<_, _>>()
            .context("read PCM8 samples")?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| f32::from(value) / 32768.0))
            .collect::<Result<_, _>>()
            .context("read PCM16 samples")?,
        (SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()
            .context("read PCM24 samples")?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / 2_147_483_648.0))
            .collect::<Result<_, _>>()
            .context("read PCM32 samples")?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .context("read float32 samples")?,
        (format, bits) => bail!("unsupported WAV format: {format:?} {bits}-bit"),
    };

    let channels = usize::from(spec.channels);
    let samples = if channels == 1 {
        interleaved
    } else {
        let mut mono = Vec::with_capacity(interleaved.len() / channels);
        for frame in interleaved.chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
        mono
    };
    Ok((samples, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav_bytes;

    #[test]
    fn decodes_pcm16_mono() {
        let bytes = wav_bytes(&[0, 16384, -16384], 24_000).unwrap();
        let (samples, rate) = decode_wav_f32(&bytes).unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(samples.len(), 3);
        assert!((samples[1] - 0.5).abs() < 1e-6);
        assert!((samples[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn decodes_float32_roundtrip() {
        let samples = [0.0, 0.25, -0.5, 1.0, -1.0];
        let bytes = crate::wav_bytes_f32(&samples, 24_000).unwrap();
        let (decoded, rate) = decode_wav_f32(&bytes).unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(decoded, samples);
    }

    #[test]
    fn averages_stereo_channels() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for sample in [32767, 0, -32768, 0] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let (samples, rate) = decode_wav_f32(&cursor.into_inner()).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 1e-4);
        assert!((samples[1] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode_wav_f32(b"not a wav").is_err());
    }
}
