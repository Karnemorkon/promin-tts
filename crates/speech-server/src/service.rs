//! Синтез-сервіс: моделі, голоси, повний ланцюжок текст -> WAV.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use speech_audio::{
    crossfade_concat, crossfade_samples, decode_wav_f32, float_to_pcm16, float_to_pcm16_clipped,
    pcm16_bytes, resample_to_24k, wav_bytes, SAMPLE_RATE,
};
use speech_core::custom_voices::{CustomVoiceError, CustomVoices};
use speech_core::projects::{ChunkTiming, LoudnessReport as ProjectLoudnessReport};
use speech_inference::{Accentor, GemmaVerbalizer, TtsEngine, VoiceEncoder};
use speech_text::neural::{NeuralStress, NeuralVerbalizer};
use speech_text::phonemize::Phonemizer;
use speech_text::pronunciation::Pronouncer;
use speech_text::segment::{chunk_text, sentence_chunks};
use speech_text::tokenizer::CharTokenizer;
use speech_text::verbalize;

use crate::pronunciation::PronunciationStore;

/// Голос за замовчуванням (як у holos-cli).
pub const DEFAULT_VOICE: &str = "Speaker_0";

/// Обмеження довжини тексту в одному запиті.
pub const MAX_TEXT_CHARS: usize = 5_000;

/// Обмеження для асинхронних завдань / потокового синтезу довгого тексту.
pub const MAX_JOB_TEXT_CHARS: usize = 200_000;

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
    /// Кількість чанків, з яких складено аудіо.
    pub chunks: usize,
    /// Тривалість аудіо в мілісекундах.
    pub audio_ms: u64,
}

/// PCM-результат синтезу (без WAV-контейнера).
#[derive(Debug)]
pub struct SynthPcmOutput {
    pub pcm: Vec<i16>,
    pub verbalizer: &'static str,
    pub vb_ms: u64,
    pub chunks: usize,
}

impl SynthPcmOutput {
    pub fn audio_ms(&self) -> u64 {
        self.pcm.len() as u64 * 1000 / SAMPLE_RATE as u64
    }
}

/// Результат синтезу глави проєкту: PCM, таймінги чанків і звіт гучності.
#[derive(Debug)]
pub struct SynthChapterOutput {
    pub pcm: Vec<i16>,
    pub verbalizer: &'static str,
    pub vb_ms: u64,
    pub chunks: usize,
    /// Межі чанків після кросфейду (для SRT/VTT).
    pub chunk_timings: Vec<ChunkTiming>,
    /// `None`, якщо нормалізацію вимкнено або сигнал надто короткий.
    pub loudness: Option<ProjectLoudnessReport>,
    pub duration_ms: u64,
}

/// Метадані, готові до відправки клієнту перед першим аудіо-чанком.
#[derive(Debug, Clone)]
pub struct StreamMeta {
    pub verbalizer: &'static str,
    pub vb_ms: u64,
    pub chunks: usize,
}

/// Підсумок потокового синтезу.
#[derive(Debug, Clone)]
pub struct StreamSummary {
    pub verbalizer: &'static str,
    pub vb_ms: u64,
    pub chunks: usize,
    pub audio_ms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    /// Некоректний запит клієнта (HTTP 400).
    #[error("{0}")]
    BadRequest(String),
    /// Функція недоступна на цьому сервері (HTTP 503).
    #[error("{0}")]
    Unavailable(String),
    /// Синтез скасовано клієнтом (HTTP 409).
    #[error("cancelled")]
    Cancelled,
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
        // Кожен обов'язковий файл шукаємо окремо: це підтримує змішану
        // розкладку (flat-моделі + опційний `tts/voice_encoder.onnx`).
        let tts_file = |name: &str| {
            let nested = dir.join("tts").join(name);
            if nested.is_file() {
                nested
            } else {
                dir.join(name)
            }
        };
        let stress_trie = ["stress/stress.trie", "stress.trie"]
            .iter()
            .map(|rel| dir.join(rel))
            .find(|candidate| candidate.is_file())
            .unwrap_or_else(|| dir.join("stress/stress.trie"));

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
            dir.join("tts/voice_encoder.onnx"),
            dir.join("voice_encoder.onnx"),
        ]
        .into_iter()
        .find(|candidate| candidate.is_file());

        let paths = Self {
            model: tts_file("holos_cpu_int8.onnx"),
            vocab: tts_file("vocab.json"),
            voices: tts_file("voices.json"),
            stress_trie,
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
    /// Глобальний словник вимови; `None` -> без замін.
    pronunciation: Option<Arc<PronunciationStore>>,
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
            pronunciation: None,
        })
    }

    /// Підключити глобальний словник вимови (до загортання в `Arc`).
    pub fn set_pronunciation_store(&mut self, store: Arc<PronunciationStore>) {
        self.pronunciation = Some(store);
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

    /// Застосовує словник вимови, потім обраний режим вербалізації;
    /// neural -> rule, якщо модель недоступна/впала.
    fn apply_verbalizer(
        &self,
        text: &str,
        mode: VerbalizerMode,
        pronunciation: Option<&Pronouncer>,
    ) -> (String, &'static str, u64) {
        let started = Instant::now();
        let pronounced = match pronunciation {
            Some(pronouncer) => pronouncer.apply(text),
            None => match &self.pronunciation {
                Some(store) => store.pronouncer().apply(text),
                None => std::borrow::Cow::Borrowed(text),
            },
        };
        let text = pronounced.as_ref();
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

    fn validate_text_with_limit<'a>(
        &self,
        text: &'a str,
        limit: usize,
    ) -> Result<&'a str, SynthError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SynthError::BadRequest("text must not be empty".into()));
        }
        if text.chars().count() > limit {
            return Err(SynthError::BadRequest(format!(
                "text is too long: max {limit} characters"
            )));
        }
        Ok(text)
    }

    fn validate_text<'a>(&self, text: &'a str) -> Result<&'a str, SynthError> {
        self.validate_text_with_limit(text, MAX_TEXT_CHARS)
    }

    fn validate_long_text<'a>(&self, text: &'a str) -> Result<&'a str, SynthError> {
        self.validate_text_with_limit(text, MAX_JOB_TEXT_CHARS)
    }

    /// Вербалізація без синтезу (для `POST /api/verbalize`).
    pub fn verbalize_text(&self, text: &str, mode: VerbalizerMode) -> Result<String, SynthError> {
        let text = self.validate_text(text)?;
        let (out, applied, vb_ms) = self.apply_verbalizer(text, mode, None);
        tracing::debug!(mode = mode.name(), applied, vb_ms, "verbalized");
        Ok(out)
    }

    fn validate_speed(&self, speed: f32) -> Result<(), SynthError> {
        if !speed.is_finite() || !SPEED_RANGE.contains(&speed) {
            return Err(SynthError::BadRequest(format!(
                "speed must be in {}..={}",
                SPEED_RANGE.start(),
                SPEED_RANGE.end()
            )));
        }
        Ok(())
    }

    /// Фонеми -> ONNX -> float32-семпли (один або кілька викликів моделі).
    fn synthesize_verbalized_f32(
        &self,
        text: &str,
        voice_vector: &[f32],
        speed: f32,
        raw_stress: bool,
    ) -> Result<Vec<f32>, SynthError> {
        let parts = self.phonemizer.phonemize(text, raw_stress);
        if parts.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }

        let mut samples = Vec::new();
        for part in parts {
            let ids = self.tokenizer.encode(&part);
            if ids.is_empty() {
                continue;
            }
            // Один важкий ONNX-виклик за раз: модель int8 і так вантажить усі ядра.
            let engine = self
                .engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let chunk = engine
                .generate(&ids, voice_vector, speed)
                .with_context(|| format!("generate audio for phonemes {part:?}"))?;
            drop(engine);
            samples.extend_from_slice(&chunk);
        }

        if samples.is_empty() {
            return Err(SynthError::Internal(anyhow::anyhow!(
                "model produced no audio samples"
            )));
        }
        Ok(samples)
    }

    /// Короткий синтез (без нарізки тексту) у PCM16.
    pub fn synthesize_pcm(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
    ) -> Result<SynthPcmOutput, SynthError> {
        self.synthesize_pcm_with_pronunciation(
            text,
            voice,
            audio_base64,
            speed,
            mode,
            raw_stress,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_pcm_with_pronunciation(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        pronunciation: Option<&Pronouncer>,
    ) -> Result<SynthPcmOutput, SynthError> {
        let text = self.validate_text(text)?;
        self.validate_speed(speed)?;

        let voice_vector = match audio_base64 {
            Some(reference) => self.encode_reference(reference)?,
            None => self.resolve_voice(voice)?,
        };

        let (verbalized, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, pronunciation);
        tracing::debug!(verbalizer, vb_ms, "text verbalized");
        let samples =
            self.synthesize_verbalized_f32(&verbalized, &voice_vector, speed, raw_stress)?;
        Ok(SynthPcmOutput {
            pcm: float_to_pcm16(&samples),
            verbalizer,
            vb_ms,
            chunks: 1,
        })
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
        let pcm = self.synthesize_pcm(text, voice, audio_base64, speed, mode, raw_stress)?;
        let wav = wav_bytes(&pcm.pcm, SAMPLE_RATE).map_err(SynthError::Internal)?;
        Ok(SynthOutput {
            wav,
            verbalizer: pcm.verbalizer,
            vb_ms: pcm.vb_ms,
            chunks: pcm.chunks,
            audio_ms: pcm.audio_ms(),
        })
    }

    /// Довгий текст: нарізка на чанки, кросфейд, одна пікова нормалізація.
    ///
    /// `progress(done, total)` викликається перед першим чанком і після кожного;
    /// `false` означає скасування.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_long_pcm(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        max_chars: usize,
        crossfade_ms: u32,
        progress: Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<SynthPcmOutput, SynthError> {
        self.synthesize_long_pcm_with_pronunciation(
            text,
            voice,
            audio_base64,
            speed,
            mode,
            raw_stress,
            max_chars,
            crossfade_ms,
            None,
            progress,
        )
    }

    /// Довгий синтез із додатковим (проєктним) словником вимови.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_long_pcm_with_pronunciation(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        max_chars: usize,
        crossfade_ms: u32,
        pronunciation: Option<&Pronouncer>,
        mut progress: Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<SynthPcmOutput, SynthError> {
        let text = self.validate_long_text(text)?;
        self.validate_speed(speed)?;
        if max_chars == 0 {
            return self.synthesize_pcm_with_pronunciation(
                text,
                voice,
                audio_base64,
                speed,
                mode,
                raw_stress,
                pronunciation,
            );
        }

        let voice_vector = match audio_base64 {
            Some(reference) => self.encode_reference(reference)?,
            None => self.resolve_voice(voice)?,
        };
        let (verbalized, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, pronunciation);
        tracing::debug!(verbalizer, vb_ms, "long text verbalized");

        let chunks = chunk_text(&verbalized, max_chars);
        if chunks.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }
        let total = chunks.len();

        if let Some(cb) = progress.as_deref_mut() {
            if !cb(0, total) {
                return Err(SynthError::Cancelled);
            }
        }

        let mut audio_chunks = Vec::with_capacity(total);
        for (index, chunk) in chunks.iter().enumerate() {
            let samples =
                self.synthesize_verbalized_f32(chunk, &voice_vector, speed, raw_stress)?;
            audio_chunks.push(samples);
            if let Some(cb) = progress.as_deref_mut() {
                if !cb(index + 1, total) {
                    return Err(SynthError::Cancelled);
                }
            }
        }

        let overlap = crossfade_samples(crossfade_ms.min(200), SAMPLE_RATE);
        let samples = crossfade_concat(&audio_chunks, overlap);
        Ok(SynthPcmOutput {
            pcm: float_to_pcm16(&samples),
            verbalizer,
            vb_ms,
            chunks: total,
        })
    }

    /// Синтез глави проєкту: таймінги чанків, LUFS-нормалізація та PCM.
    ///
    /// На відміну від [`Self::synthesize_long_pcm_with_pronunciation`], тут
    /// зберігаються межі кожного чанка після кросфейду, а нормалізація
    /// виконується до конверсії в PCM16 (без повторної пікової нормалізації).
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_chapter_pcm(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        max_chars: usize,
        crossfade_ms: u32,
        pronunciation: Option<&Pronouncer>,
        loudness_target: Option<speech_audio::loudness::LoudnessTarget>,
        mut progress: Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<SynthChapterOutput, SynthError> {
        let text = self.validate_long_text(text)?;
        self.validate_speed(speed)?;

        let voice_vector = match audio_base64 {
            Some(reference) => self.encode_reference(reference)?,
            None => self.resolve_voice(voice)?,
        };
        let (verbalized, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, pronunciation);
        tracing::debug!(verbalizer, vb_ms, "chapter text verbalized");

        let chunks = chunk_text(&verbalized, max_chars);
        if chunks.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }
        let total = chunks.len();
        let timing_texts = timing_texts(text, &chunks, max_chars);

        if let Some(cb) = progress.as_deref_mut() {
            if !cb(0, total) {
                return Err(SynthError::Cancelled);
            }
        }

        let mut audio_chunks = Vec::with_capacity(total);
        for (index, chunk) in chunks.iter().enumerate() {
            let samples =
                self.synthesize_verbalized_f32(chunk, &voice_vector, speed, raw_stress)?;
            audio_chunks.push(samples);
            if let Some(cb) = progress.as_deref_mut() {
                if !cb(index + 1, total) {
                    return Err(SynthError::Cancelled);
                }
            }
        }

        let overlap = crossfade_samples(crossfade_ms.min(200), SAMPLE_RATE);
        let mut chunk_timings = Vec::with_capacity(total);
        let mut cursor = 0usize;
        for (index, audio) in audio_chunks.iter().enumerate() {
            let start = if index == 0 {
                0
            } else {
                cursor.saturating_sub(overlap)
            };
            let end = start + audio.len();
            chunk_timings.push(ChunkTiming {
                start_ms: samples_to_ms(start),
                end_ms: samples_to_ms(end),
                text: timing_texts[index].clone(),
            });
            cursor = end;
        }

        let mut samples = crossfade_concat(&audio_chunks, overlap);
        let duration_ms = samples_to_ms(samples.len());
        let mut loudness = None;
        if let Some(target) = loudness_target {
            let report = speech_audio::loudness::normalize(&mut samples, SAMPLE_RATE, target)
                .map_err(SynthError::Internal)?;
            if report.normalized {
                loudness = Some(ProjectLoudnessReport {
                    input_lufs: report.input_lufs.unwrap_or(-70.0),
                    output_lufs: report.output_lufs.unwrap_or(-70.0),
                    input_true_peak_db: report.input_true_peak_db.unwrap_or(-120.0),
                    output_true_peak_db: report.output_true_peak_db.unwrap_or(-120.0),
                    gain_db: report.gain_db,
                    target_lufs: target.target_lufs,
                    target_true_peak_db: target.true_peak_db,
                    limited: report.limited_samples > 0,
                });
            }
        }
        let pcm = if loudness.is_some() {
            float_to_pcm16_clipped(&samples)
        } else {
            float_to_pcm16(&samples)
        };

        Ok(SynthChapterOutput {
            pcm,
            verbalizer,
            vb_ms,
            chunks: total,
            chunk_timings,
            loudness,
            duration_ms,
        })
    }

    /// Потоковий синтез: `on_ready` -> метадані, `on_chunk` -> PCM16-байти.
    ///
    /// `false` з будь-якого колбека означає, що клієнт відключився.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_stream_pcm<F, G>(
        &self,
        text: &str,
        voice: Option<&str>,
        audio_base64: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        max_chars: usize,
        mut on_ready: F,
        mut on_chunk: G,
    ) -> Result<StreamSummary, SynthError>
    where
        F: FnMut(&StreamMeta) -> bool,
        G: FnMut(&[u8]) -> bool,
    {
        let text = self.validate_long_text(text)?;
        self.validate_speed(speed)?;

        let voice_vector = match audio_base64 {
            Some(reference) => self.encode_reference(reference)?,
            None => self.resolve_voice(voice)?,
        };
        let (verbalized, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, None);
        let chunks = sentence_chunks(&verbalized, max_chars);
        if chunks.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }

        let meta = StreamMeta {
            verbalizer,
            vb_ms,
            chunks: chunks.len(),
        };
        if !on_ready(&meta) {
            return Err(SynthError::Cancelled);
        }

        let mut audio_ms = 0u64;
        for chunk in &chunks {
            let samples =
                self.synthesize_verbalized_f32(chunk, &voice_vector, speed, raw_stress)?;
            let pcm = float_to_pcm16(&samples);
            audio_ms += pcm.len() as u64 * 1000 / SAMPLE_RATE as u64;
            if !on_chunk(&pcm16_bytes(&pcm)) {
                break;
            }
        }

        Ok(StreamSummary {
            verbalizer,
            vb_ms,
            chunks: chunks.len(),
            audio_ms,
        })
    }
}

fn samples_to_ms(samples: usize) -> u64 {
    samples as u64 * 1000 / SAMPLE_RATE as u64
}

/// Текст для субтитрів: оригінальні чанки, якщо вербалізація зберегла
/// кількість чанків; інакше — вербалізовані (номери вже розгорнуто).
fn timing_texts(original: &str, spoken: &[String], max_chars: usize) -> Vec<String> {
    let original_chunks = chunk_text(original, max_chars);
    if original_chunks.len() == spoken.len() {
        original_chunks
    } else {
        spoken.to_vec()
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
    #[test]
    fn discovers_mixed_flat_and_nested_layout() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-mixed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("tts")).unwrap();
        for file in [
            "holos_cpu_int8.onnx",
            "vocab.json",
            "voices.json",
            "stress.trie",
        ] {
            std::fs::write(root.join(file), b"x").unwrap();
        }
        // Автозавантаження опційного encoder-а створює `tts/` у flat-каталозі.
        std::fs::write(root.join("tts/voice_encoder.onnx"), b"x").unwrap();

        let paths = ModelPaths::discover(&root).expect("discover mixed layout");
        assert_eq!(paths.model, root.join("holos_cpu_int8.onnx"));
        assert_eq!(paths.vocab, root.join("vocab.json"));
        assert_eq!(paths.voices, root.join("voices.json"));
        assert_eq!(paths.stress_trie, root.join("stress.trie"));
        assert_eq!(
            paths.voice_encoder.as_deref(),
            Some(root.join("tts/voice_encoder.onnx").as_path())
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
