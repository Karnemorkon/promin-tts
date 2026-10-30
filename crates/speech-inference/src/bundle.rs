//! Бандл HolosTTS `holos-onnx` (format_version 2).
//!
//! Новий формат моделі — не один ONNX-граф, а каталог:
//!
//! ```text
//! holos_onnx.json   # маніфест: вокабуляр, частота, етапи
//! voices.npz        # пресети голосів (names/values)
//! encoder.onnx      # токени -> acoustic/prosodic features + durations
//! prosody.onnx      # frame_to_token -> f0/energy
//! decoder.onnx      # exact-декодер на всю репліку
//! source.onnx       # (streaming) збудження для віконного декодера
//! decoder_window.onnx
//! ```
//!
//! Маніфест читається як є: жодних хардкодів частоти чи розмірності голосу,
//! щоб новий експорт моделі не вимагав правок коду.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Формат бандла, який ми вміємо читати.
pub const BUNDLE_FORMAT: &str = "holos-onnx";
/// Версія формату бандла.
pub const BUNDLE_FORMAT_VERSION: u32 = 2;
/// Ім'я маніфесту в каталозі бандла.
pub const BUNDLE_MANIFEST: &str = "holos_onnx.json";

/// Маніфест бандла (`holos_onnx.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct BundleManifest {
    pub format: String,
    pub format_version: u32,
    pub encoder: String,
    pub prosody: String,
    pub decoder: String,
    #[serde(default)]
    pub voices: Option<String>,
    #[serde(default)]
    pub vocab: Vec<String>,
    pub sample_rate: u32,
    pub samples_per_frame: usize,
    pub max_frames: usize,
    pub decoder_mode: String,
    pub alignment_format: String,
    #[serde(default)]
    pub streaming: Option<StreamingManifest>,
}

/// Параметри віконного (стрімінгового) декодера.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamingManifest {
    pub source: String,
    pub decoder: String,
    /// Скільки кадрів віддавати за один чанк.
    pub chunk_frames: usize,
    /// Скільки кадрів контексту з кожного боку вікна.
    pub context_frames: usize,
}

impl BundleManifest {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join(BUNDLE_MANIFEST);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read bundle manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse bundle manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != BUNDLE_FORMAT {
            bail!(
                "unsupported bundle format {:?} (expected {BUNDLE_FORMAT:?})",
                self.format
            );
        }
        if self.format_version != BUNDLE_FORMAT_VERSION {
            bail!(
                "unsupported bundle format version {} (expected {})",
                self.format_version,
                BUNDLE_FORMAT_VERSION
            );
        }
        if self.encoder.trim().is_empty() || self.prosody.trim().is_empty() {
            bail!("bundle manifest must name encoder and prosody stages");
        }
        if self.decoder.trim().is_empty() {
            bail!("bundle manifest must name a decoder stage");
        }
        if self.vocab.is_empty() {
            bail!("bundle manifest has an empty vocabulary");
        }
        if self.sample_rate == 0 {
            bail!("bundle sample_rate must be positive");
        }
        if self.samples_per_frame == 0 {
            bail!("bundle samples_per_frame must be positive");
        }
        if self.max_frames == 0 {
            bail!("bundle max_frames must be positive");
        }
        if self.decoder_mode != "exact" {
            bail!(
                "unsupported decoder mode {:?} (expected \"exact\")",
                self.decoder_mode
            );
        }
        if self.alignment_format != "gather" {
            bail!(
                "unsupported alignment format {:?} (expected \"gather\")",
                self.alignment_format
            );
        }
        if let Some(streaming) = &self.streaming {
            if streaming.chunk_frames == 0 {
                bail!("bundle streaming.chunk_frames must be positive");
            }
            if streaming.source.trim().is_empty() || streaming.decoder.trim().is_empty() {
                bail!("bundle streaming section must name source and decoder stages");
            }
        }
        for (field, file) in [
            ("encoder", self.encoder.as_str()),
            ("prosody", self.prosody.as_str()),
            ("decoder", self.decoder.as_str()),
        ] {
            validate_relative_path(field, file)?;
        }
        if let Some(voices) = &self.voices {
            validate_relative_path("voices", voices)?;
        }
        if let Some(streaming) = &self.streaming {
            validate_relative_path("streaming.source", &streaming.source)?;
            validate_relative_path("streaming.decoder", &streaming.decoder)?;
        }
        Ok(())
    }

    /// Шлях до етапу в каталозі бандла.
    pub fn stage_path(&self, dir: &Path, file: &str) -> PathBuf {
        dir.join(file)
    }

    /// Усі файли, від яких залежить аудіо (без самого маніфесту).
    ///
    /// Потрібен автозавантаженню: список етапів задає маніфест, а не код.
    pub fn files(&self) -> Vec<&str> {
        let mut files = vec![
            self.encoder.as_str(),
            self.prosody.as_str(),
            self.decoder.as_str(),
        ];
        if let Some(voices) = &self.voices {
            files.push(voices);
        }
        if let Some(streaming) = &self.streaming {
            files.push(streaming.source.as_str());
            files.push(streaming.decoder.as_str());
        }
        files
    }
}

/// Ім'я файлу в маніфесті мусить бути безпечним відносним шляхом.
///
/// Маніфест приходить із мережі, тому `../` або абсолютний шлях не повинні
/// дозволяти читати/перезаписувати файли поза каталогом бандла.
fn validate_relative_path(field: &str, file: &str) -> Result<()> {
    let path = Path::new(file);
    if file.trim().is_empty() {
        bail!("bundle manifest {field} must not be empty");
    }
    if path.is_absolute() {
        bail!("bundle manifest {field} must be a relative path, got {file:?}");
    }
    for component in path.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            bail!(
                "bundle manifest {field} must not contain '.', '..' or a root prefix, got {file:?}"
            );
        }
    }
    Ok(())
}

/// Завантажений бандл: маніфест, словник і пресети голосів.
#[derive(Debug, Clone)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: BundleManifest,
    pub voices: BTreeMap<String, Vec<f32>>,
    voice_dim: usize,
}

impl Bundle {
    /// Відкрити каталог бандла. `voice_dim` — очікувана розмірність ембеддингу.
    pub fn open(dir: impl AsRef<Path>, voice_dim: usize) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = BundleManifest::load(&dir)?;
        if voice_dim == 0 {
            bail!("voice dimension must be positive");
        }
        let voices = match &manifest.voices {
            Some(name) => load_voices(&dir.join(name), voice_dim)
                .with_context(|| format!("load voices from {}", dir.join(name).display()))?,
            None => BTreeMap::new(),
        };
        Ok(Self {
            dir,
            manifest,
            voices,
            voice_dim,
        })
    }

    pub fn voice_dim(&self) -> usize {
        self.voice_dim
    }

    pub fn sample_rate(&self) -> u32 {
        self.manifest.sample_rate
    }

    pub fn samples_per_frame(&self) -> usize {
        self.manifest.samples_per_frame
    }

    pub fn max_frames(&self) -> usize {
        self.manifest.max_frames
    }

    pub fn vocab(&self) -> &[String] {
        &self.manifest.vocab
    }

    pub fn voice_names(&self) -> Vec<&str> {
        self.voices.keys().map(String::as_str).collect()
    }

    /// Шлях до файлу етапу.
    pub fn stage(&self, file: &str) -> PathBuf {
        self.dir.join(file)
    }
}

/// Прочитати `voices.npz` (zip із `names.npy` + `values.npy`).
pub fn load_voices(path: &Path, voice_dim: usize) -> Result<BTreeMap<String, Vec<f32>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("read zip {}", path.display()))?;
    let names = read_npy_entry(&mut archive, "names.npy")
        .with_context(|| format!("read names.npy from {}", path.display()))?;
    let values = read_npy_entry(&mut archive, "values.npy")
        .with_context(|| format!("read values.npy from {}", path.display()))?;

    let names = names
        .unicode_strings()
        .context("voices names must be a numpy unicode array")?;
    let values = values
        .f32_matrix()
        .context("voices values must be a float32 matrix")?;

    if values.len() != names.len() {
        bail!(
            "voices names and embeddings have different lengths: {} vs {}",
            names.len(),
            values.len()
        );
    }
    let mut voices = BTreeMap::new();
    for (name, embedding) in names.into_iter().zip(values) {
        if embedding.len() != voice_dim {
            bail!(
                "voice {name:?} has {} floats, expected {voice_dim}",
                embedding.len()
            );
        }
        if !embedding.iter().all(|value| value.is_finite()) {
            bail!("voice {name:?} contains NaN or Inf");
        }
        if voices.insert(name.clone(), embedding).is_some() {
            bail!("duplicate voice name {name:?}");
        }
    }
    Ok(voices)
}

fn read_npy_entry(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<NpyArray> {
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("archive has no {name}"))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {name}"))?;
    NpyArray::parse(&bytes)
}

/// Мінімальний читач NPY: `<f4` та `<U*` (numpy unicode).
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub descr: String,
    pub shape: Vec<usize>,
    data: Vec<u8>,
}

impl NpyArray {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            bail!("not a numpy .npy file");
        }
        let major = bytes[6];
        let (header_start, header_len) = match major {
            1 => {
                if bytes.len() < 10 {
                    bail!("truncated numpy header");
                }
                (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
            }
            2 | 3 => {
                if bytes.len() < 12 {
                    bail!("truncated numpy header");
                }
                (
                    12usize,
                    u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                )
            }
            other => bail!("unsupported numpy version {other}"),
        };
        let header_end = header_start
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .context("numpy header length is out of bounds")?;
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .context("numpy header is not utf-8")?;

        let descr = header_value(header, "descr").context("numpy header has no descr")?;
        let descr = descr.trim_matches(|ch| ch == '\'' || ch == '"').to_string();
        let fortran = header_value(header, "fortran_order")
            .context("numpy header has no fortran_order")?
            .trim()
            .eq_ignore_ascii_case("true");
        if fortran {
            bail!("fortran-order numpy arrays are not supported");
        }
        let shape =
            parse_shape(header_value(header, "shape").context("numpy header has no shape")?)?;

        Ok(Self {
            descr,
            shape,
            data: bytes[header_end..].to_vec(),
        })
    }

    fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Декодувати `<U*` у рядки.
    pub fn unicode_strings(&self) -> Result<Vec<String>> {
        let width = self
            .descr
            .strip_prefix("<U")
            .or_else(|| self.descr.strip_prefix("=U"))
            .and_then(|value| value.parse::<usize>().ok())
            .context("expected a numpy unicode array")?;
        let count = self.element_count();
        let expected = count
            .checked_mul(width)
            .and_then(|units| units.checked_mul(4))
            .context("numpy unicode array is too large")?;
        if self.data.len() != expected {
            bail!(
                "numpy unicode array has {} data bytes, expected {expected}",
                self.data.len()
            );
        }
        let mut out = Vec::with_capacity(count);
        for chunk in self.data.chunks_exact(width.max(1) * 4) {
            let mut text = String::with_capacity(width);
            for unit in chunk.as_chunks::<4>().0 {
                let code = u32::from_le_bytes([unit[0], unit[1], unit[2], unit[3]]);
                if code == 0 {
                    continue;
                }
                let ch = char::from_u32(code)
                    .with_context(|| format!("invalid unicode code point {code:#x}"))?;
                text.push(ch);
            }
            out.push(text);
        }
        Ok(out)
    }

    /// Декодувати `<f4` у матрицю рядків.
    pub fn f32_matrix(&self) -> Result<Vec<Vec<f32>>> {
        if self.descr != "<f4" && self.descr != "=f4" && self.descr != "|f4" {
            bail!("expected a float32 numpy array, got {:?}", self.descr);
        }
        if self.shape.len() != 2 {
            bail!("expected a 2-D float32 array, got shape {:?}", self.shape);
        }
        let expected = self
            .element_count()
            .checked_mul(4)
            .context("numpy float array is too large")?;
        if self.data.len() != expected {
            bail!(
                "numpy float array has {} data bytes, expected {expected}",
                self.data.len()
            );
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut values = Vec::with_capacity(cols);
            for col in 0..cols {
                let offset = (row * cols + col) * 4;
                let bytes = &self.data[offset..offset + 4];
                values.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            out.push(values);
        }
        Ok(out)
    }
}

fn header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let key_pattern = format!("'{key}'");
    let key_start = header.find(&key_pattern)?;
    let after_key = &header[key_start + key_pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    // `shape` містить кому всередині дужок, тому для нього шукаємо `)`.
    let end = if rest.starts_with('(') {
        rest.find(')').map(|index| index + 1).unwrap_or(rest.len())
    } else {
        rest.find(',').unwrap_or(rest.len())
    };
    Some(rest[..end].trim())
}

fn parse_shape(value: &str) -> Result<Vec<usize>> {
    let value = value.trim();
    let inner = value
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .with_context(|| format!("invalid numpy shape {value:?}"))?;
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        dims.push(
            part.parse::<usize>()
                .with_context(|| format!("invalid numpy dimension {part:?}"))?,
        );
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_unicode_names() {
        let header = "{'descr': '<U4', 'fortran_order': False, 'shape': (2,), }";
        let mut data = Vec::new();
        for text in ["ab", "cd"] {
            for ch in text.chars() {
                data.extend_from_slice(&(ch as u32).to_le_bytes());
            }
            for _ in text.len()..4 {
                data.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        let array = NpyArray::parse(&npy(header, &data)).unwrap();
        assert_eq!(array.unicode_strings().unwrap(), vec!["ab", "cd"]);
    }

    #[test]
    fn parses_float_matrix() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let array = NpyArray::parse(&npy(header, &data)).unwrap();
        assert_eq!(
            array.f32_matrix().unwrap(),
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
    }

    #[test]
    fn rejects_fortran_order_and_bad_magic() {
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }";
        assert!(NpyArray::parse(&npy(header, &[0, 0, 0, 0])).is_err());
        assert!(NpyArray::parse(b"not-numpy").is_err());
    }

    #[test]
    fn validates_manifest() {
        let manifest = BundleManifest {
            format: BUNDLE_FORMAT.into(),
            format_version: BUNDLE_FORMAT_VERSION,
            encoder: "encoder.onnx".into(),
            prosody: "prosody.onnx".into(),
            decoder: "decoder.onnx".into(),
            voices: Some("voices.npz".into()),
            vocab: vec!["$".into(), "a".into()],
            sample_rate: 24_000,
            samples_per_frame: 480,
            max_frames: 1_000,
            decoder_mode: "exact".into(),
            alignment_format: "gather".into(),
            streaming: None,
        };
        manifest.validate().unwrap();
        assert_eq!(
            manifest.files(),
            vec!["encoder.onnx", "prosody.onnx", "decoder.onnx", "voices.npz"]
        );

        let mut broken = manifest.clone();
        broken.format_version = 3;
        assert!(broken.validate().is_err());

        let mut broken = manifest;
        broken.samples_per_frame = 0;
        assert!(broken.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_manifest_paths() {
        let manifest = BundleManifest {
            format: BUNDLE_FORMAT.into(),
            format_version: BUNDLE_FORMAT_VERSION,
            encoder: "encoder.onnx".into(),
            prosody: "prosody.onnx".into(),
            decoder: "decoder.onnx".into(),
            voices: Some("voices.npz".into()),
            vocab: vec!["a".into()],
            sample_rate: 24_000,
            samples_per_frame: 480,
            max_frames: 1_000,
            decoder_mode: "exact".into(),
            alignment_format: "gather".into(),
            streaming: None,
        };
        for unsafe_path in ["../encoder.onnx", "/etc/passwd", "a/../../b.onnx", ""] {
            let mut broken = manifest.clone();
            broken.encoder = unsafe_path.into();
            assert!(
                broken.validate().is_err(),
                "accepted unsafe path {unsafe_path:?}"
            );
        }
        // Вкладені каталоги всередині бандла дозволені.
        let mut nested = manifest;
        nested.encoder = "stages/encoder.onnx".into();
        nested.validate().unwrap();
    }
}
