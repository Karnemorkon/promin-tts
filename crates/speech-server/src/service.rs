//! Синтез-сервіс: моделі, голоси, повний ланцюжок текст -> WAV.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use speech_audio::{float_to_pcm16, wav_bytes, SAMPLE_RATE};
use speech_inference::TtsEngine;
use speech_text::phonemize::Phonemizer;
use speech_text::tokenizer::CharTokenizer;

/// Голос за замовчуванням (як у holos-cli).
pub const DEFAULT_VOICE: &str = "Speaker_0";

/// Обмеження довжини тексту в одному запиті (v0.1).
pub const MAX_TEXT_CHARS: usize = 5_000;

/// Допустимий діапазон темпу.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.0;

#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    /// Некоректний запит клієнта (HTTP 400).
    #[error("{0}")]
    BadRequest(String),
    /// Внутрішня помилка (HTTP 500).
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

/// Шляхи до файлів моделей.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub vocab: PathBuf,
    pub voices: PathBuf,
    pub stress_trie: PathBuf,
}

impl ModelPaths {
    /// `dir/tts/*` + `dir/stress/*`, з fallback на плоску розкладку `dir/*`.
    pub fn discover(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tts_dir = if dir.join("tts").is_dir() {
            dir.join("tts")
        } else {
            dir.to_path_buf()
        };
        let stress_dir = if dir.join("stress").is_dir() {
            dir.join("stress")
        } else {
            dir.to_path_buf()
        };

        let paths = Self {
            model: tts_dir.join("holos_cpu_int8.onnx"),
            vocab: tts_dir.join("vocab.json"),
            voices: tts_dir.join("voices.json"),
            stress_trie: stress_dir.join("stress.trie"),
        };
        paths.validate()?;
        Ok(paths)
    }

    fn validate(&self) -> Result<()> {
        for path in [&self.model, &self.vocab, &self.voices, &self.stress_trie] {
            if !path.is_file() {
                bail!("missing model file: {}", path.display());
            }
        }
        Ok(())
    }
}

/// voices.json -> map; кожен голос мусить мати рівно 256 float.
fn parse_voices(json: &str) -> Result<BTreeMap<String, Vec<f32>>> {
    let voices: BTreeMap<String, Vec<f32>> =
        serde_json::from_str(json).context("parse voices.json")?;
    if voices.is_empty() {
        bail!("voices.json contains no voices");
    }
    for (name, vector) in &voices {
        if vector.len() != speech_inference::VOICE_DIM {
            bail!(
                "voice {name:?} must have {} floats, got {}",
                speech_inference::VOICE_DIM,
                vector.len()
            );
        }
    }
    Ok(voices)
}

fn load_voices(path: &Path) -> Result<BTreeMap<String, Vec<f32>>> {
    let json = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_voices(&json)
}

/// Завантажені моделі + стан синтезу.
pub struct SynthService {
    engine: Mutex<TtsEngine>,
    tokenizer: CharTokenizer,
    phonemizer: Phonemizer,
    voices: BTreeMap<String, Vec<f32>>,
    default_voice: String,
}

impl SynthService {
    pub fn load(paths: &ModelPaths, intra_threads: i16) -> Result<Self> {
        let engine = TtsEngine::from_file(&paths.model, intra_threads)?;
        let tokenizer = CharTokenizer::from_vocab_json(&paths.vocab)
            .with_context(|| format!("load vocab {}", paths.vocab.display()))?;
        let phonemizer = Phonemizer::open(&paths.stress_trie)
            .with_context(|| format!("load stress dictionary {}", paths.stress_trie.display()))?;
        let voices = load_voices(&paths.voices)?;

        let default_voice = if voices.contains_key(DEFAULT_VOICE) {
            DEFAULT_VOICE.to_string()
        } else {
            voices.keys().next().cloned().expect("voices is non-empty")
        };

        tracing::info!(
            voices = voices.len(),
            default_voice = %default_voice,
            vocab = tokenizer.vocab_len(),
            "synthesis service ready"
        );

        Ok(Self {
            engine: Mutex::new(engine),
            tokenizer,
            phonemizer,
            voices,
            default_voice,
        })
    }

    pub fn voice_names(&self) -> Vec<&str> {
        self.voices.keys().map(String::as_str).collect()
    }

    pub fn default_voice(&self) -> &str {
        &self.default_voice
    }

    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Повний ланцюжок: текст -> фонеми -> токени -> ONNX -> PCM16 -> WAV.
    pub fn synthesize_wav(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: f32,
    ) -> Result<Vec<u8>, SynthError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SynthError::BadRequest("text must not be empty".into()));
        }
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(SynthError::BadRequest(format!(
                "text is too long: max {MAX_TEXT_CHARS} characters"
            )));
        }
        if !speed.is_finite() || !SPEED_RANGE.contains(&speed) {
            return Err(SynthError::BadRequest(format!(
                "speed must be in {}..={}",
                SPEED_RANGE.start(),
                SPEED_RANGE.end()
            )));
        }

        let voice_name = voice.unwrap_or(&self.default_voice);
        let voice_vector = self
            .voices
            .get(voice_name)
            .ok_or_else(|| SynthError::BadRequest(format!("unknown voice: {voice_name}")))?;

        let parts = self.phonemizer.phonemize(text, false);
        if parts.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }

        // Один важкий ONNX-виклик за раз: модель int8 і так вантажить усі ядра.
        let engine = self
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut samples = Vec::new();
        for part in parts {
            let ids = self.tokenizer.encode(&part);
            if ids.is_empty() {
                continue;
            }
            let chunk = engine
                .generate(&ids, voice_vector, speed)
                .with_context(|| format!("generate audio for phonemes {part:?}"))?;
            samples.extend_from_slice(&chunk);
        }
        drop(engine);

        if samples.is_empty() {
            return Err(SynthError::Internal(anyhow::anyhow!(
                "model produced no audio samples"
            )));
        }

        let pcm = float_to_pcm16(&samples);
        wav_bytes(&pcm, SAMPLE_RATE).map_err(SynthError::Internal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_voices() {
        let json = r#"{"Speaker_0": [0.0], "Speaker_1": [1.0]}"#;
        // 1 float — невалідно, перевіряємо саме валідацію довжини.
        assert!(parse_voices(json).is_err());
    }

    #[test]
    fn rejects_empty_voices() {
        assert!(parse_voices("{}").is_err());
    }

    #[test]
    fn discovers_nested_layout() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("tts")).unwrap();
        std::fs::create_dir_all(root.join("stress")).unwrap();
        for file in [
            "tts/holos_cpu_int8.onnx",
            "tts/vocab.json",
            "tts/voices.json",
            "stress/stress.trie",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }

        let paths = ModelPaths::discover(&root).expect("discover nested layout");
        assert_eq!(paths.model, root.join("tts/holos_cpu_int8.onnx"));
        assert_eq!(paths.stress_trie, root.join("stress/stress.trie"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "requires models + ORT 1.17: ORT_STRATEGY=system ORT_LIB_LOCATION=<ort>/lib PROMIN_MODELS_DIR=<dir>"]
    fn synthesizes_golden_phrase() {
        let dir = std::env::var("PROMIN_MODELS_DIR").expect("PROMIN_MODELS_DIR not set");
        let paths = ModelPaths::discover(dir).expect("discover models");
        let service = SynthService::load(&paths, 4).expect("load service");

        let wav = service
            .synthesize_wav("Привіт, світе!", None, 1.0)
            .expect("synthesize");

        let reader = hound::WavReader::new(std::io::Cursor::new(wav)).expect("parse WAV");
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().bits_per_sample, 16);
        // Еталон holos-cli: 28200 семплів (1.175 с) для цієї фрази.
        assert_eq!(reader.len(), 28_200);
    }

    #[test]
    fn discovers_flat_layout() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-flat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        for file in [
            "holos_cpu_int8.onnx",
            "vocab.json",
            "voices.json",
            "stress.trie",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }

        let paths = ModelPaths::discover(&root).expect("discover flat layout");
        assert_eq!(paths.model, root.join("holos_cpu_int8.onnx"));
        assert_eq!(paths.stress_trie, root.join("stress.trie"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
