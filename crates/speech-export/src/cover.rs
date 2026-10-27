//! Обкладинка: вбудований PNG і визначення розмірів зображення.

use std::sync::OnceLock;

/// Вбудована обкладинка Promin TTS (600×600 PNG).
const DEFAULT_COVER_PNG: &[u8] = include_bytes!("../assets/cover.png");

/// Обкладинка для вбудовування в аудіофайл.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverImage {
    pub mime: String,
    pub data: Vec<u8>,
}

impl CoverImage {
    /// Створити обкладинку з PNG або JPEG.
    pub fn new(data: Vec<u8>) -> Result<Self, &'static str> {
        let mime = sniff_image_mime(&data).ok_or("cover must be a PNG or JPEG image")?;
        Ok(Self {
            mime: mime.to_string(),
            data,
        })
    }

    pub fn is_png(&self) -> bool {
        self.mime == "image/png"
    }

    /// Ширина, висота та глибина кольору для FLAC picture block.
    ///
    /// Невідомі розміри повертаються як нулі — специфікація це дозволяє.
    pub fn dimensions(&self) -> (u32, u32, u32) {
        if self.is_png() {
            png_dimensions(&self.data).unwrap_or((0, 0, 0))
        } else {
            jpeg_dimensions(&self.data)
                .map(|(width, height)| (width, height, 24))
                .unwrap_or((0, 0, 0))
        }
    }
}

/// Визначити MIME за сигнатурою.
pub fn sniff_image_mime(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("image/png")
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else {
        None
    }
}

/// Вбудована обкладинка (копія; дані кешуються в статичній пам’яті).
pub fn default_cover() -> CoverImage {
    static CACHE: OnceLock<CoverImage> = OnceLock::new();
    CACHE
        .get_or_init(|| CoverImage {
            mime: "image/png".to_string(),
            data: DEFAULT_COVER_PNG.to_vec(),
        })
        .clone()
}

fn png_dimensions(data: &[u8]) -> Option<(u32, u32, u32)> {
    // PNG: 8-byte signature, 4-byte length, "IHDR", width, height, ...
    if data.len() < 26 || &data[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(data[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(data[20..24].try_into().ok()?);
    let color_type = data[25];
    let depth = match color_type {
        0 | 2 => 24,
        4 | 6 => 32,
        3 => 24,
        _ => 24,
    };
    Some((width, height, depth))
}

fn jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut position = 2usize;
    while position + 4 <= data.len() {
        if data[position] != 0xFF {
            position += 1;
            continue;
        }
        let marker = data[position + 1];
        position += 2;
        // Markers without a payload.
        if matches!(marker, 0xD8 | 0xD9) {
            continue;
        }
        if marker == 0xDA {
            return None; // start of scan before any SOF
        }
        let length = u16::from_be_bytes([data[position], data[position + 1]]) as usize;
        if length < 2 || position + length > data.len() {
            return None;
        }
        // SOF0..SOF15, excluding DHT (C4), JPG (C8) and DAC (CC).
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            if length < 7 {
                return None;
            }
            let height = u16::from_be_bytes([data[position + 3], data[position + 4]]) as u32;
            let width = u16::from_be_bytes([data[position + 5], data[position + 6]]) as u32;
            return Some((width, height));
        }
        position += length;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cover_is_valid_png() {
        let cover = default_cover();
        assert_eq!(cover.mime, "image/png");
        assert_eq!(cover.dimensions(), (600, 600, 32));
        assert!(cover.data.len() > 1_000);
    }

    #[test]
    fn rejects_unknown_image_data() {
        assert!(CoverImage::new(b"not an image".to_vec()).is_err());
    }
}
