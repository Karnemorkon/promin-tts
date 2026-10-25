//! Аудіо-хелпери: float32 -> PCM16, WAV (PCM16 mono).
//!
//! Порт `audio.cpp` з holos-tts-core: пікова нормалізація на весь масив,
//! clip у [-1, 1], округлення `lround(x * 32767)`, WAV 24 кГц mono.

use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};

/// Частота дискретизації моделі HolosTTS.
pub const SAMPLE_RATE: u32 = 24_000;

/// float32 (довільна амплітуда) -> int16 з піковою нормалізацією.
///
/// Нормалізація виконується на **весь** переданий масив (як у C++), тому
/// викликати її треба один раз після конкатенації всіх чанків.
pub fn float_to_pcm16(samples: &[f32]) -> Vec<i16> {
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if peak > 0.0 { 1.0 / peak } else { 1.0 };

    samples
        .iter()
        .map(|&v| {
            let x = (v * scale).clamp(-1.0, 1.0);
            // `f32::round` округлює половину від нуля, як std::lround.
            (x * 32767.0).round() as i16
        })
        .collect()
}

/// PCM16 mono -> байти WAV (44-байтний RIFF-заголовок + data).
pub fn wav_bytes(pcm: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).context("create WAV writer")?;
        for &sample in pcm {
            writer.write_sample(sample).context("write WAV sample")?;
        }
        writer.finalize().context("finalize WAV")?;
    }
    Ok(cursor.into_inner())
}

/// PCM16 mono -> WAV-файл.
pub fn write_wav(path: impl AsRef<Path>, pcm: &[i16], sample_rate: u32) -> Result<()> {
    let bytes = wav_bytes(pcm, sample_rate)?;
    std::fs::write(path.as_ref(), bytes)
        .with_context(|| format!("write WAV {}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_peak_to_full_scale() {
        let pcm = float_to_pcm16(&[0.0, 0.5, -0.25]);
        // peak = 0.5 -> scale 2.0 -> [0, 1.0, -0.5]
        assert_eq!(pcm, vec![0, 32767, -16384]);
    }

    #[test]
    fn clips_and_handles_silence() {
        assert_eq!(float_to_pcm16(&[]), Vec::<i16>::new());
        assert_eq!(float_to_pcm16(&[0.0, 0.0]), vec![0, 0]);
        // Після нормалізації пік завжди 1.0; перевіряємо, що знак збережено.
        let pcm = float_to_pcm16(&[-2.0, 1.0]);
        assert_eq!(pcm, vec![-32767, 16384]);
    }

    #[test]
    fn wav_header_is_pcm16_mono_24k() {
        let bytes = wav_bytes(&[0, 100, -100], SAMPLE_RATE).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            48_000
        );
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);
        assert_eq!(bytes.len(), 44 + 6);
        assert_eq!(i16::from_le_bytes(bytes[44..46].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(bytes[46..48].try_into().unwrap()), 100);
    }
}
