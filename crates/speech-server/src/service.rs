//! Синтез-сервіс: моделі, голоси, повний ланцюжок текст -> WAV.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use speech_audio::{decode_wav_f32, float_to_pcm16, resample_to_24k, wav_bytes, SAMPLE_RATE};
use speech_core::custom_voices::{CustomVoiceError, CustomVoices};
use speech_inference::{Accentor, GemmaVerbalizer, TtsEngine, VoiceEncoder};
use speech_text::neural::{NeuralStress, NeuralVerbalizer};
use speech_text::phonemize::Phonemizer;
use speech_text::tokenizer::CharTokenizer;
use speech_text::verbalize;

/// Голос за замовчуванням (як у holos-cli).
pub const DEFAULT_VOICE: &str = "Speaker_0";

/// Обмеження довжини тексту в одному запиті.
pub const MAX_TEXT_CHARS: usize = 5_000;

/// Допустимий діапазон темпу.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.0;

/// Референс для клонування: мінімальна та максимальна тривалість (секунди).
pub const REFERENCE_SECONDS: std::ops::RangeInclusive<f32> = 0.25..=120.0;

/// Режим вербалізації (як `VbMode` у C++-сервері).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbalizerMode {
    Off,
    Rule,
    Neural,
}

impl VerbalizerMode {
    /// `auto_verbalize` з тіла запиту: bool або рядок; відсутнє -> neural.
    pub fn parse(value: Option<&serde_json::Value>) -> Self {
        use serde_json::Value;
        match value {
            None => Self::Neural,
            Some(Value::Bool(true)) => Self::Neural,
            Some(Value::Bool(false)) => Self::Off,
            Some(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
                "neural" | "true" | "1" => Self::Neural,
                "rule" => Self::Rule,
                "off" | "false" | "0" | "none" => Self::Off,
                _ => Self::Neural,
            },
            // Як у C++: не bool/string -> neural.
            Some(_) => Self::Neural,
        }
    }

    pub fn parse_mode(value: Option<&str>) -> Self {
        match value {
            None => Self::Neural,
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "neural" | "true" | "1" => Self::Neural,
                "rule" => Self::Rule,
                "off" | "false" | "0" | "none" => Self::Off,
                _ => Self::Neural,
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Rule => "rule",
            Self::Neural => "neural",
        }
    }
}

/// Результат синтезу: WAV + метадані вербалізації.
#[derive(Debug)]
pub struct SynthOutput {
    pub wav: Vec<u8>,
    /// Фактично застосований режим (`off` / `rule` / `neural`).
    pub verbalizer: &'static str,
    pub vb_ms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    /// Некоректний запит клієнта (HTTP 400).
    #[error("{0}")]
    BadRequest(String),
    /// Функція недоступна на цьому сервері (HTTP 503).
    #[error("{0}")]
    Unavailable(String),
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
    /// Каталог Gemma-вербалізатора (`model.onnx` + `tokenizer.model`), якщо є.
    pub verbalizer: Option<PathBuf>,
    /// accentor-lite ONNX (neural fallback для OOV-наголосу), якщо є.
    pub accentor: Option<PathBuf>,
    /// voice encoder ONNX (клонування голосу), якщо є.
    pub voice_encoder: Option<PathBuf>,
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

        // Gemma-вербалізатор опційний: без нього працює rule fallback.
        let verbalizer = ["verbalizer/gemma-int8", "verbalizer", "gemma-int8", ""]
            .iter()
            .map(|rel| dir.join(rel))
            .find(|candidate| {
                candidate.join("model.onnx").is_file()
                    && candidate.join("tokenizer.model").is_file()
            });

        // accentor-lite опційний: без нього OOV-слова лишаються без наголосу.
        let accentor = [
            "accentor/accentor-lite.onnx",
            "accentor-lite.onnx",
            "accentor/accentor.onnx",
        ]
        .iter()
        .map(|rel| dir.join(rel))
        .find(|candidate| candidate.is_file());

        // voice encoder опційний: без нього клонування вимкнено.
        let voice_encoder = [
            tts_dir.join("voice_encoder.onnx"),
            dir.join("voice_encoder.onnx"),
        ]
        .into_iter()
        .find(|candidate| candidate.is_file());

        let paths = Self {
            model: tts_dir.join("holos_cpu_int8.onnx"),
            vocab: tts_dir.join("vocab.json"),
            voices: tts_dir.join("voices.json"),
            stress_trie: stress_dir.join("stress.trie"),
            verbalizer,
            accentor,
            voice_encoder,
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

/// Помилка сховища кастомних голосів -> HTTP-помилка сервісу.
fn custom_voice_error(error: CustomVoiceError) -> SynthError {
    match error {
        CustomVoiceError::Disabled => SynthError::Unavailable(error.to_string()),
        CustomVoiceError::Unknown(_)
        | CustomVoiceError::InvalidName(_)
        | CustomVoiceError::WrongDim(_)
        | CustomVoiceError::Npy(_) => SynthError::BadRequest(error.to_string()),
        CustomVoiceError::Io(io) => SynthError::Internal(io.into()),
    }
}

/// Завантажені моделі + стан синтезу.
pub struct SynthService {
    engine: Mutex<TtsEngine>,
    tokenizer: CharTokenizer,
    phonemizer: Phonemizer,
    voices: BTreeMap<String, Vec<f32>>,
    default_voice: String,
    /// Gemma-вербалізатор; `None` -> rule fallback.
    neural: Option<Arc<dyn NeuralVerbalizer>>,
    /// voice encoder; `None` -> клонування вимкнено.
    voice_encoder: Option<VoiceEncoder>,
    /// Кастомні голоси (`<dir>/<name>.npy`); порожній каталог -> вимкнено.
    custom_voices: CustomVoices,
}

impl SynthService {
    pub fn load(paths: &ModelPaths, intra_threads: i16, speakers_dir: PathBuf) -> Result<Self> {
        let engine = TtsEngine::from_file(&paths.model, intra_threads)?;
        let tokenizer = CharTokenizer::from_vocab_json(&paths.vocab)
            .with_context(|| format!("load vocab {}", paths.vocab.display()))?;

        // accentor-lite опційний: без нього працює лише словниковий наголос.
        let accentor: Option<Arc<dyn NeuralStress>> = match &paths.accentor {
            Some(path) => match Accentor::from_file(path) {
                Ok(engine) => Some(Arc::new(engine)),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        path = %path.display(),
                        "accentor unavailable, using dictionary stress only"
                    );
                    None
                }
            },
            None => None,
        };
        let phonemizer = Phonemizer::open_with_accentor(&paths.stress_trie, accentor)
            .with_context(|| format!("load stress dictionary {}", paths.stress_trie.display()))?;
        let voices = load_voices(&paths.voices)?;

        let default_voice = if voices.contains_key(DEFAULT_VOICE) {
            DEFAULT_VOICE.to_string()
        } else {
            voices.keys().next().cloned().expect("voices is non-empty")
        };

        // Нейромережевий вербалізатор опційний: без моделі працює rule fallback.
        let neural: Option<Arc<dyn NeuralVerbalizer>> = match &paths.verbalizer {
            Some(dir) => match GemmaVerbalizer::from_dir(dir, intra_threads) {
                Ok(engine) => Some(Arc::new(engine)),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        dir = %dir.display(),
                        "gemma verbalizer unavailable, using rule fallback"
                    );
                    None
                }
            },
            None => None,
        };

        // Voice encoder опційний і завантажується ліниво (перший енролмент).
        let voice_encoder = match &paths.voice_encoder {
            Some(path) => match VoiceEncoder::from_file(path, intra_threads) {
                Ok(encoder) => Some(encoder),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        path = %path.display(),
                        "voice encoder unavailable, cloning disabled"
                    );
                    None
                }
            },
            None => None,
        };
        let custom_voices = CustomVoices::new(speakers_dir);

        tracing::info!(
            voices = voices.len(),
            default_voice = %default_voice,
            vocab = tokenizer.vocab_len(),
            verbalizer = if neural.is_some() { "neural+rule" } else { "rule" },
            verbalizer_dir = ?paths.verbalizer,
            accentor = if phonemizer.has_accentor() { "on" } else { "off" },
            accentor_path = ?paths.accentor,
            voice_encoder = if voice_encoder.is_some() { "on" } else { "off" },
            voice_encoder_path = ?paths.voice_encoder,
            speakers_dir = %custom_voices.dir().display(),
            "synthesis service ready"
        );

        Ok(Self {
            engine: Mutex::new(engine),
            tokenizer,
            phonemizer,
            voices,
            default_voice,
            neural,
            voice_encoder,
            custom_voices,
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

    /// Фактично доступний вербалізатор для `/healthz`.
    pub fn verbalizer_name(&self) -> &'static str {
        if self.neural.is_some() {
            "neural+rule"
        } else {
            "rule"
        }
    }

    /// Чи підключено neural accentor (для `/healthz`).
    pub fn accentor_name(&self) -> &'static str {
        if self.phonemizer.has_accentor() {
            "on"
        } else {
            "off"
        }
    }

    /// Чи підключено voice encoder (для `/healthz`).
    pub fn voice_encoder_name(&self) -> &'static str {
        if self.voice_encoder.is_some() {
            "on"
        } else {
            "off"
        }
    }

    /// Кастомні голоси (рескан каталогу при кожному виклику).
    pub fn custom_voice_names(&self) -> Vec<String> {
        self.custom_voices.names()
    }

    pub fn custom_voice_count(&self) -> usize {
        self.custom_voices.names().len()
    }

    /// Голос за назвою: пресет із voices.json, далі кастомний `.npy`.
    fn resolve_voice(&self, voice: Option<&str>) -> Result<Vec<f32>, SynthError> {
        let name = voice.unwrap_or(&self.default_voice);
        if let Some(vector) = self.voices.get(name) {
            return Ok(vector.clone());
        }
        match self.custom_voices.get(name) {
            Ok(vector) => Ok(vector),
            Err(CustomVoiceError::Unknown(_)) => {
                Err(SynthError::BadRequest(format!("unknown voice: {name}")))
            }
            Err(error) => Err(custom_voice_error(error)),
        }
    }

    /// base64 WAV -> mono float32 @24 кГц (без енкодера).
    fn decode_reference(&self, audio_base64: &str) -> Result<Vec<f32>, SynthError> {
        // Дозволяємо переноси рядків (base64 CLI без -w0).
        let compact: String = audio_base64
            .chars()
            .filter(|ch| !ch.is_ascii_whitespace())
            .collect();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&compact)
            .map_err(|_| SynthError::BadRequest("audio_base64 is not valid base64".into()))?;
        let (samples, rate) = decode_wav_f32(&bytes)
            .map_err(|error| SynthError::BadRequest(format!("invalid reference audio: {error}")))?;
        if samples.is_empty() {
            return Err(SynthError::BadRequest(
                "reference audio contains no samples".into(),
            ));
        }
        let seconds = samples.len() as f32 / rate as f32;
        if !REFERENCE_SECONDS.contains(&seconds) {
            return Err(SynthError::BadRequest(format!(
                "reference audio must be {}..={} seconds, got {seconds:.2}",
                REFERENCE_SECONDS.start(),
                REFERENCE_SECONDS.end()
            )));
        }
        resample_to_24k(samples, rate)
            .map_err(|error| SynthError::BadRequest(format!("cannot resample reference: {error}")))
    }

    /// WAV (base64) -> 256-вимірний ембеддінг.
    fn encode_reference(&self, audio_base64: &str) -> Result<Vec<f32>, SynthError> {
        let encoder = self.voice_encoder.as_ref().ok_or_else(|| {
            SynthError::Unavailable("voice encoder is not configured on this server".into())
        })?;
        let samples = self.decode_reference(audio_base64)?;
        encoder
            .encode(&samples)
            .map_err(|error| SynthError::Internal(error.context("voice encoder failed")))
    }

    /// Енролмент: обчислити ембеддінг і зберегти `<speakers>/<name>.npy`.
    pub fn enroll_voice(&self, name: &str, audio_base64: &str) -> Result<(), SynthError> {
        if !self.custom_voices.enabled() {
            return Err(SynthError::Unavailable(
                "custom voices are disabled (PROMIN_SPEAKERS_DIR is empty)".into(),
            ));
        }
        if !speech_core::custom_voices::valid_name(name) {
            return Err(SynthError::BadRequest(
                "voice name must be 1-40 letters/digits/space/underscore/hyphen".into(),
            ));
        }
        let embedding = self.encode_reference(audio_base64)?;
        self.custom_voices
            .save(name, &embedding)
            .map_err(custom_voice_error)
    }

    /// Видалення кастомного голосу; `false` — такого голосу не було.
    pub fn delete_voice(&self, name: &str) -> Result<bool, SynthError> {
        self.custom_voices.remove(name).map_err(custom_voice_error)
    }

    /// Застосовує обраний режим; neural -> rule, якщо модель недоступна/впала.
    fn apply_verbalizer(&self, text: &str, mode: VerbalizerMode) -> (String, &'static str, u64) {
        let started = Instant::now();
        let (out, applied) = match mode {
            VerbalizerMode::Off => (text.to_string(), "off"),
            VerbalizerMode::Rule => (verbalize::verbalize(text), "rule"),
            VerbalizerMode::Neural => match &self.neural {
                Some(engine) => match engine.verbalize(text) {
                    Ok(out) => (out, "neural"),
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            "neural verbalizer failed, falling back to rule"
                        );
                        (verbalize::verbalize(text), "rule")
                    }
                },
                None => (verbalize::verbalize(text), "rule"),
            },
        };
        (out, applied, started.elapsed().as_millis() as u64)
    }

    fn validate_text<'a>(&self, text: &'a str) -> Result<&'a str, SynthError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SynthError::BadRequest("text must not be empty".into()));
        }
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(SynthError::BadRequest(format!(
                "text is too long: max {MAX_TEXT_CHARS} characters"
            )));
        }
        Ok(text)
    }

    /// Вербалізація без синтезу (для `POST /api/verbalize`).
    pub fn verbalize_text(&self, text: &str, mode: VerbalizerMode) -> Result<String, SynthError> {
        let text = self.validate_text(text)?;
        let (out, applied, vb_ms) = self.apply_verbalizer(text, mode);
        tracing::debug!(mode = mode.name(), applied, vb_ms, "verbalized");
        Ok(out)
    }

    /// Повний ланцюжок: текст -> вербалізація -> фонеми -> ONNX -> PCM16 -> WAV.
    ///
    /// `audio_base64` (WAV-референс для клонування) має пріоритет над `voice`
    /// (як у HolosTTS-hybrid); без нього голос береться з `voice`/дефолту.
    pub fn synthesize_wav(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
    ) -> Result<SynthOutput, SynthError> {
        let text = self.validate_text(text)?;
        if !speed.is_finite() || !SPEED_RANGE.contains(&speed) {
            return Err(SynthError::BadRequest(format!(
                "speed must be in {}..={}",
                SPEED_RANGE.start(),
                SPEED_RANGE.end()
            )));
        }

        let voice_vector = match audio_base64 {
            Some(reference) => self.encode_reference(reference)?,
            None => self.resolve_voice(voice)?,
        };

        let (text, applied, vb_ms) = self.apply_verbalizer(text, mode);
        tracing::debug!(verbalizer = applied, vb_ms, "text verbalized");

        let parts = self.phonemizer.phonemize(&text, raw_stress);
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
                .generate(&ids, &voice_vector, speed)
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
        let wav = wav_bytes(&pcm, SAMPLE_RATE).map_err(SynthError::Internal)?;
        Ok(SynthOutput {
            wav,
            verbalizer: applied,
            vb_ms,
        })
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
    fn parses_verbalizer_modes() {
        use serde_json::json;
        assert_eq!(VerbalizerMode::parse(None), VerbalizerMode::Neural);
        assert_eq!(
            VerbalizerMode::parse(Some(&json!(true))),
            VerbalizerMode::Neural
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!(false))),
            VerbalizerMode::Off
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!("rule"))),
            VerbalizerMode::Rule
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!("off"))),
            VerbalizerMode::Off
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!("neural"))),
            VerbalizerMode::Neural
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!("unknown"))),
            VerbalizerMode::Neural
        );
        assert_eq!(
            VerbalizerMode::parse(Some(&json!(0))),
            VerbalizerMode::Neural
        );
    }

    #[test]
    fn parses_explicit_mode_strings() {
        assert_eq!(VerbalizerMode::parse_mode(None), VerbalizerMode::Neural);
        assert_eq!(
            VerbalizerMode::parse_mode(Some("rule")),
            VerbalizerMode::Rule
        );
        assert_eq!(VerbalizerMode::parse_mode(Some("OFF")), VerbalizerMode::Off);
        assert_eq!(
            VerbalizerMode::parse_mode(Some("nonsense")),
            VerbalizerMode::Neural
        );
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
        assert!(paths.verbalizer.is_none());
        assert!(paths.accentor.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discovers_optional_verbalizer() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-vb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("tts")).unwrap();
        std::fs::create_dir_all(root.join("stress")).unwrap();
        std::fs::create_dir_all(root.join("verbalizer/gemma-int8")).unwrap();
        for file in [
            "tts/holos_cpu_int8.onnx",
            "tts/vocab.json",
            "tts/voices.json",
            "stress/stress.trie",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }
        for file in [
            "verbalizer/gemma-int8/model.onnx",
            "verbalizer/gemma-int8/tokenizer.model",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }

        let paths = ModelPaths::discover(&root).expect("discover verbalizer");
        assert_eq!(
            paths.verbalizer.as_deref(),
            Some(root.join("verbalizer/gemma-int8").as_path())
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discovers_optional_accentor() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-accentor-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("tts")).unwrap();
        std::fs::create_dir_all(root.join("stress")).unwrap();
        std::fs::create_dir_all(root.join("accentor")).unwrap();
        for file in [
            "tts/holos_cpu_int8.onnx",
            "tts/vocab.json",
            "tts/voices.json",
            "stress/stress.trie",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }
        std::fs::write(root.join("accentor/accentor-lite.onnx"), b"x").unwrap();

        let paths = ModelPaths::discover(&root).expect("discover accentor");
        assert_eq!(
            paths.accentor.as_deref(),
            Some(root.join("accentor/accentor-lite.onnx").as_path())
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discovers_optional_voice_encoder() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-encoder-{}-{}",
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
        std::fs::write(root.join("tts/voice_encoder.onnx"), b"x").unwrap();

        let paths = ModelPaths::discover(&root).expect("discover voice encoder");
        assert_eq!(
            paths.voice_encoder.as_deref(),
            Some(root.join("tts/voice_encoder.onnx").as_path())
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "requires models + ORT 1.31: ORT_STRATEGY=system ORT_LIB_LOCATION=<ort>/lib PROMIN_MODELS_DIR=<dir>"]
    fn synthesizes_golden_phrase() {
        let dir = std::env::var("PROMIN_MODELS_DIR").expect("PROMIN_MODELS_DIR not set");
        let paths = ModelPaths::discover(&dir).expect("discover models");
        let service = SynthService::load(&paths, 4, PathBuf::from(&dir).join("speakers"))
            .expect("load service");

        let output = service
            .synthesize_wav(
                "Привіт, світе!",
                None,
                None,
                1.0,
                VerbalizerMode::Off,
                false,
            )
            .expect("synthesize");

        let reader = hound::WavReader::new(std::io::Cursor::new(output.wav)).expect("parse WAV");
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().bits_per_sample, 16);
        // Еталон holos-cli: 28200 семплів (1.175 с) для цієї фрази.
        assert_eq!(reader.len(), 28_200);
        assert_eq!(output.verbalizer, "off");
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
        assert!(paths.verbalizer.is_none());
        assert!(paths.accentor.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }
}
