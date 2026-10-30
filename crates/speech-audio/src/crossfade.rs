//! Кросфейд між чанками синтезу та допоміжні функції для стрімінгу WAV.

use crate::SAMPLE_RATE;

/// Перетворити мілісекунди кросфейду в семпли.
pub fn crossfade_samples(crossfade_ms: u32, sample_rate: u32) -> usize {
    ((crossfade_ms as u64 * sample_rate as u64) / 1000) as usize
}

/// Склеїти чанки з рівнопотужним кросфейдом.
///
/// Останні `crossfade_samples` семплів попереднього результату змішуються з
/// першими семплами наступного чанка. Якщо чанк коротший за кросфейд,
/// перекриття обмежується його довжиною.
pub fn crossfade_concat(chunks: &[Vec<f32>], crossfade_samples: usize) -> Vec<f32> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };
    let mut out = first.clone();

    for next in &chunks[1..] {
        if next.is_empty() {
            continue;
        }
        let overlap = crossfade_samples.min(out.len()).min(next.len());
        if overlap == 0 {
            out.extend_from_slice(next);
            continue;
        }

        let split = out.len() - overlap;
        let tail = out[split..].to_vec();
        out.truncate(split);
        for i in 0..overlap {
            // t = 0 -> попередній чанк, t = 1 -> наступний.
            let t = if overlap > 1 {
                i as f32 / (overlap - 1) as f32
            } else {
                0.5
            };
            let angle = t * std::f32::consts::FRAC_PI_2;
            out.push(tail[i] * angle.cos() + next[i] * angle.sin());
        }
        out.extend_from_slice(&next[overlap..]);
    }
    out
}

/// PCM16 mono -> little-endian байти (для стрімінгу).
pub fn pcm16_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for sample in pcm {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// RIFF/WAVE заголовок для PCM16 mono.
///
/// `data_len = u32::MAX` використовується для потокового WAV, коли розмір
/// заздалегідь невідомий.
pub fn wav_header(sample_rate: u32, data_len: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    let riff_len = if data_len == u32::MAX {
        u32::MAX
    } else {
        36u32.saturating_add(data_len)
    };
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out
}

/// Потоковий WAV-заголовок (розміри невідомі).
pub fn streaming_wav_header(sample_rate: u32) -> Vec<u8> {
    wav_header(sample_rate, u32::MAX)
}

/// Заголовок для стандартного WAV 24 кГц.
pub fn default_streaming_wav_header() -> Vec<u8> {
    streaming_wav_header(SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenates_without_crossfade() {
        let chunks = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(crossfade_concat(&chunks, 0), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn crossfade_overlap_length() {
        let chunks = vec![vec![1.0; 4], vec![2.0; 4]];
        let out = crossfade_concat(&chunks, 2);
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[5] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn crossfade_limits_to_shorter_chunk() {
        let chunks = vec![vec![1.0], vec![2.0, 2.0, 2.0]];
        let out = crossfade_concat(&chunks, 10);
        assert_eq!(out.len(), 3);
        assert!(out[0] > 1.0 && out[0] < 2.5);
        assert!((out[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn single_chunk_is_unchanged() {
        let chunks = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(crossfade_concat(&chunks, 100), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_chunks() {
        assert!(crossfade_concat(&[], 10).is_empty());
        assert!(crossfade_concat(&[vec![], vec![]], 10).is_empty());
    }

    #[test]
    fn crossfade_ms_conversion() {
        assert_eq!(crossfade_samples(100, 24_000), 2400);
        assert_eq!(crossfade_samples(0, 24_000), 0);
    }

    #[test]
    fn wav_header_is_44_bytes() {
        let header = wav_header(24_000, 100);
        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[36..40], b"data");
        assert_eq!(u32::from_le_bytes(header[40..44].try_into().unwrap()), 100);
    }

    #[test]
    fn streaming_header_uses_max_sizes() {
        let header = streaming_wav_header(24_000);
        assert_eq!(
            u32::from_le_bytes(header[4..8].try_into().unwrap()),
            u32::MAX
        );
        assert_eq!(
            u32::from_le_bytes(header[40..44].try_into().unwrap()),
            u32::MAX
        );
    }
}
