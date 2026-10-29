//! Ідентичність моделі синтезу.
//!
//! Кеш тейків і знімки job-ів мусять знати, якою саме моделлю згенеровано
//! аудіо. Інакше після оновлення моделі старі тейки вважатимуться валідними
//! і змішаються з новими в одній главі.
//!
//! Ідентичність — це не «версія застосунку», а відбиток файлів: SHA-256 графа
//! та словника фонем. Вона стабільна між перезапусками і змінюється навіть
//! тоді, коли файл моделі замінили без зміни імені (`holos_cpu_int8.onnx`).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Версія формату ідентичності (не версія моделі).
pub const MODEL_IDENTITY_SCHEMA: u32 = 1;

/// Ліцензія вагів HolosTTS.
pub const DEFAULT_MODEL_LICENSE: &str = "MIT";

/// Родина моделей за замовчуванням.
pub const DEFAULT_MODEL_ID: &str = "holos-tts";

/// Незмінна ідентичність завантаженої моделі.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Версія структури цього запису.
    #[serde(default = "default_schema")]
    pub schema: u32,
    /// Родина моделей, напр. `holos-tts`.
    pub id: String,
    /// Варіант у межах родини, напр. `holos_cpu_int8` або `onnx`.
    pub variant: String,
    /// SHA-256 файлу ONNX-графа.
    pub sha256: String,
    /// SHA-256 словника фонем (`vocab.json`).
    pub vocab_sha256: String,
    /// Частота дискретизації аудіо на виході моделі.
    pub sample_rate: u32,
    /// Розмірність вектора голосу.
    pub voice_dim: usize,
    /// Ліцензія вагів.
    pub license: String,
}

fn default_schema() -> u32 {
    MODEL_IDENTITY_SCHEMA
}

impl ModelIdentity {
    /// Обчислити ідентичність із файлів моделі та словника.
    pub fn compute(
        model_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
        sample_rate: u32,
        voice_dim: usize,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let vocab_path = vocab_path.as_ref();
        let variant = model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .unwrap_or("model")
            .to_string();
        Ok(Self {
            schema: MODEL_IDENTITY_SCHEMA,
            id: DEFAULT_MODEL_ID.to_string(),
            variant,
            sha256: sha256_file(model_path)?,
            vocab_sha256: sha256_file(vocab_path)?,
            sample_rate,
            voice_dim,
            license: DEFAULT_MODEL_LICENSE.to_string(),
        })
    }

    /// Перевизначити родину моделей (напр. для майбутнього бандла `holos-onnx`).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Перевизначити варіант моделі.
    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = variant.into();
        self
    }

    /// Коротке позначення для логів: `holos-tts/holos_cpu_int8@a1b2c3d4e5f6`.
    pub fn short(&self) -> String {
        let hash = self.sha256.get(..12).unwrap_or(&self.sha256);
        format!("{}/{}@{hash}", self.id, self.variant)
    }

    /// Перевірити, що запис придатний для порівняння.
    pub fn validate(&self) -> Result<()> {
        if self.schema != MODEL_IDENTITY_SCHEMA {
            anyhow::bail!(
                "unsupported model identity schema: {} (expected {})",
                self.schema,
                MODEL_IDENTITY_SCHEMA
            );
        }
        if self.id.trim().is_empty() {
            anyhow::bail!("model id must not be empty");
        }
        if self.sha256.len() != 64 || !self.sha256.chars().all(|ch| ch.is_ascii_hexdigit()) {
            anyhow::bail!("model sha256 is invalid");
        }
        if self.vocab_sha256.len() != 64
            || !self.vocab_sha256.chars().all(|ch| ch.is_ascii_hexdigit())
        {
            anyhow::bail!("model vocab sha256 is invalid");
        }
        if self.sample_rate == 0 {
            anyhow::bail!("model sample rate must be positive");
        }
        if self.voice_dim == 0 {
            anyhow::bail!("model voice dimension must be positive");
        }
        Ok(())
    }
}

/// SHA-256 файлу, порахований потоково (модель не читається в пам'ять цілком).
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read {} for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 набору файлів: імена та вміст у заданому порядку.
///
/// Потрібен для багатоетапного бандла, де аудіо залежить від кількох графів.
/// Ім'я файлу входить у геш, щоб перейменування або перестановка етапів
/// змінювали ідентичність.
pub fn sha256_files(paths: &[PathBuf]) -> Result<String> {
    let mut hasher = Sha256::new();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("file name is not valid utf-8: {}", path.display()))?;
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        let file =
            File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = reader
                .read(&mut buffer)
                .with_context(|| format!("read {} for hashing", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        hasher.update([0u8]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 довільного байтового буфера.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("promin-model-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    fn cleanup(path: &std::path::Path) {
        if let Some(dir) = path.parent() {
            std::fs::remove_dir_all(dir).ok();
        }
    }

    #[test]
    fn hashes_files_and_builds_identity() {
        let model = temp_file("model.onnx", b"model-bytes");
        let vocab = temp_file("vocab.json", b"{\"a\":0}");
        let identity = ModelIdentity::compute(&model, &vocab, 24_000, 256).unwrap();
        identity.validate().unwrap();
        assert_eq!(identity.variant, "model");
        assert_eq!(identity.sample_rate, 24_000);
        assert_eq!(identity.voice_dim, 256);
        assert_eq!(
            identity.sha256,
            "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5"
        );
        cleanup(&model);
        cleanup(&vocab);
    }

    #[test]
    fn identity_changes_with_model_bytes() {
        let model_a = temp_file("a.onnx", b"aaa");
        let model_b = temp_file("b.onnx", b"bbb");
        let vocab = temp_file("vocab.json", b"{}");
        let a = ModelIdentity::compute(&model_a, &vocab, 24_000, 256).unwrap();
        let b = ModelIdentity::compute(&model_b, &vocab, 24_000, 256).unwrap();
        assert_ne!(a.sha256, b.sha256);
        assert_ne!(a.short(), b.short());
        cleanup(&model_a);
        cleanup(&model_b);
        cleanup(&vocab);
    }

    #[test]
    fn roundtrips_through_json() {
        let model = temp_file("model.onnx", b"x");
        let vocab = temp_file("vocab.json", b"{}");
        let identity = ModelIdentity::compute(&model, &vocab, 24_000, 256)
            .unwrap()
            .with_id("holos-tts")
            .with_variant("v2");
        let json = serde_json::to_string(&identity).unwrap();
        let parsed: ModelIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, identity);
        cleanup(&model);
        cleanup(&vocab);
    }

    #[test]
    fn hashes_multiple_files_deterministically() {
        let a = temp_file("a.onnx", b"alpha");
        let b = temp_file("b.onnx", b"beta");
        let first = sha256_files(&[a.clone(), b.clone()]).unwrap();
        let second = sha256_files(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        // Порядок і вміст впливають на результат.
        assert_ne!(first, sha256_files(&[b.clone(), a.clone()]).unwrap());
        std::fs::write(&b, b"gamma").unwrap();
        assert_ne!(first, sha256_files(&[a.clone(), b.clone()]).unwrap());
        assert_eq!(sha256_bytes(b"alpha"), sha256_file(&a).unwrap());
        cleanup(&a);
        cleanup(&b);
    }

    #[test]
    fn rejects_broken_identity() {
        let model = temp_file("model.onnx", b"x");
        let vocab = temp_file("vocab.json", b"{}");
        let mut identity = ModelIdentity::compute(&model, &vocab, 24_000, 256).unwrap();
        identity.sha256 = "nope".into();
        assert!(identity.validate().is_err());
        cleanup(&model);
        cleanup(&vocab);
    }
}
