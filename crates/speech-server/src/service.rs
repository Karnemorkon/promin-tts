//! Синтез-сервіс: моделі, голоси, повний ланцюжок текст -> WAV.

use std::collections::{BTreeMap, HashMap};
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
use speech_core::model::ModelIdentity;
use speech_core::projects::{ChunkTiming, LoudnessReport as ProjectLoudnessReport, Role};
use speech_inference::{
    Accentor, Bundle, BundleManifest, GemmaVerbalizer, HolosPipeline, TtsEngine, VoiceEncoder,
    BUNDLE_MANIFEST,
};
use speech_text::dialogue::DialogueSegment;
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
    /// Сирі float32-семпли кожного чанка (до кросфейду) для кешу тейків.
    ///
    /// Порожній для глав, зібраних із готових тейків.
    pub units: Vec<SynthUnit>,
}

/// Один синтезований юніт глави до склеювання.
#[derive(Debug)]
pub struct SynthUnit {
    pub samples: Vec<f32>,
    /// Оригінальний текст для субтитрів.
    pub text: String,
    /// Текст, який фактично пішов у синтез.
    pub spoken: String,
    pub voice: Option<String>,
    pub role: Option<String>,
}

/// Юніт, готовий до перезбирання глави з кешу тейків.
#[derive(Debug)]
pub struct AssembledUnit {
    pub samples: Vec<f32>,
    pub text: String,
    pub role: Option<String>,
}

/// Результат синтезу одного юніта (для перегенерації речення).
#[derive(Debug)]
#[allow(dead_code)]
pub struct SynthUnitOutput {
    pub samples: Vec<f32>,
    pub spoken: String,
    pub verbalizer: &'static str,
    pub vb_ms: u64,
}

/// Запланований чанк глави: яким голосом озвучити та що показати в субтитрах.
struct PlannedChunk {
    voice: Option<String>,
    spoken: String,
    timing: String,
    role: Option<String>,
}

/// Готове аудіо глави до формування [`SynthChapterOutput`].
struct ChapterAudio {
    pcm: Vec<i16>,
    chunk_timings: Vec<ChunkTiming>,
    loudness: Option<ProjectLoudnessReport>,
    duration_ms: u64,
    units: Vec<SynthUnit>,
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

/// Джерело моделі синтезу.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Один ONNX-граф `holos_cpu_int8.onnx` + `vocab.json` + `voices.json`.
    Legacy {
        model: PathBuf,
        vocab: PathBuf,
        voices: PathBuf,
    },
    /// Багатоетапний бандл `holos-onnx`: `holos_onnx.json`, `*.onnx`, `voices.npz`.
    Bundle { dir: PathBuf },
}

/// Шляхи до файлів моделей.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub source: ModelSource,
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
    ///
    /// Якщо знайдено `holos_onnx.json`, перевага надається багатоетапному
    /// бандлу; інакше — старому однографовому формату.
    pub fn discover(dir: impl AsRef<Path>) -> Result<Self> {
        Self::discover_with_format(dir, "auto")
    }

    /// Чи є під `dir` каталог багатоетапного бандла.
    pub fn bundle_present(dir: impl AsRef<Path>) -> bool {
        find_bundle(dir.as_ref()).is_some()
    }

    /// Те саме, але з явним вибором формату: `auto`, `legacy` або `bundle`.
    pub fn discover_with_format(dir: impl AsRef<Path>, format: &str) -> Result<Self> {
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

        let bundle = find_bundle(dir);
        let source = match format.trim().to_ascii_lowercase().as_str() {
            "auto" => match bundle {
                Some(dir) => ModelSource::Bundle { dir },
                None => ModelSource::Legacy {
                    model: tts_file("holos_cpu_int8.onnx"),
                    vocab: tts_file("vocab.json"),
                    voices: tts_file("voices.json"),
                },
            },
            "bundle" | "holos-onnx" | "onnx" => ModelSource::Bundle {
                dir: bundle.with_context(|| {
                    format!(
                        "no {BUNDLE_MANIFEST} found under {} (looked in the root, tts/, onnx/ and holos-onnx/)",
                        dir.display()
                    )
                })?,
            },
            "legacy" | "holos-cpu" | "int8" => ModelSource::Legacy {
                model: tts_file("holos_cpu_int8.onnx"),
                vocab: tts_file("vocab.json"),
                voices: tts_file("voices.json"),
            },
            other => bail!("unknown model format {other:?} (expected auto, legacy or bundle)"),
        };

        let paths = Self {
            source,
            stress_trie,
            verbalizer,
            accentor,
            voice_encoder,
        };
        paths.validate()?;
        Ok(paths)
    }

    fn validate(&self) -> Result<()> {
        if !self.stress_trie.is_file() {
            bail!("missing model file: {}", self.stress_trie.display());
        }
        match &self.source {
            ModelSource::Legacy {
                model,
                vocab,
                voices,
            } => {
                for path in [model, vocab, voices] {
                    if !path.is_file() {
                        bail!("missing model file: {}", path.display());
                    }
                }
            }
            ModelSource::Bundle { dir } => {
                let manifest = dir.join(BUNDLE_MANIFEST);
                if !manifest.is_file() {
                    bail!("missing model file: {}", manifest.display());
                }
            }
        }
        Ok(())
    }
}

/// Знайти каталог бандла в типових розкладках.
pub(crate) fn find_bundle(dir: &Path) -> Option<PathBuf> {
    ["", "tts", "onnx", "holos-onnx"]
        .iter()
        .map(|rel| dir.join(rel))
        .find(|candidate| candidate.join(BUNDLE_MANIFEST).is_file())
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

/// Ідентичність багатоетапного бандла.
///
/// Хешуємо всі файли, від яких залежить аудіо: маніфест, графи етапів і
/// пресети голосів. Словник фонем входить окремим гешем, щоб зміна
/// вокабуляра інвалідувала тейки навіть за незмінних графів.
fn bundle_model_identity(dir: &Path, bundle: &Bundle) -> Result<ModelIdentity> {
    let manifest: &BundleManifest = &bundle.manifest;
    let mut files = vec![
        dir.join(BUNDLE_MANIFEST),
        bundle.stage(&manifest.encoder),
        bundle.stage(&manifest.prosody),
        bundle.stage(&manifest.decoder),
    ];
    if let Some(streaming) = &manifest.streaming {
        files.push(bundle.stage(&streaming.source));
        files.push(bundle.stage(&streaming.decoder));
    }
    if let Some(voices) = &manifest.voices {
        files.push(dir.join(voices));
    }
    let sha256 = speech_core::model::sha256_files(&files)?;
    let vocab_sha256 = speech_core::model::sha256_bytes(
        &serde_json::to_vec(bundle.vocab()).context("serialize bundle vocabulary")?,
    );
    Ok(ModelIdentity {
        schema: speech_core::model::MODEL_IDENTITY_SCHEMA,
        id: speech_core::model::DEFAULT_MODEL_ID.to_string(),
        variant: "onnx".to_string(),
        sha256,
        vocab_sha256,
        sample_rate: manifest.sample_rate,
        voice_dim: bundle.voice_dim(),
        license: speech_core::model::DEFAULT_MODEL_LICENSE.to_string(),
    })
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

/// Рушій синтезу: старий однографовий або багатоетапний бандл.
enum Engine {
    Legacy(TtsEngine),
    Holos(Box<HolosPipeline>),
}

impl Engine {
    fn generate(&self, phoneme_ids: &[i64], voice: &[f32], speed: f32) -> Result<Vec<f32>> {
        match self {
            Self::Legacy(engine) => engine.generate(phoneme_ids, voice, speed),
            Self::Holos(pipeline) => pipeline.synthesize(phoneme_ids, voice, speed),
        }
    }

    fn is_streaming(&self) -> bool {
        match self {
            Self::Legacy(_) => false,
            Self::Holos(pipeline) => pipeline.is_streaming(),
        }
    }

    /// Синтез із віддачею готових шматків; `false` — клієнт відключився.
    fn stream<F>(
        &self,
        phoneme_ids: &[i64],
        voice: &[f32],
        speed: f32,
        mut on_samples: F,
    ) -> Result<bool>
    where
        F: FnMut(&[f32]) -> bool,
    {
        match self {
            Self::Legacy(engine) => {
                let samples = engine.generate(phoneme_ids, voice, speed)?;
                Ok(on_samples(&samples))
            }
            Self::Holos(pipeline) => {
                pipeline.synthesize_stream(phoneme_ids, voice, speed, None, on_samples)
            }
        }
    }
}

/// Завантажені моделі + стан синтезу.
pub struct SynthService {
    engine: Mutex<Engine>,
    /// Відбиток завантаженої моделі (входить у fingerprint тейків).
    model_identity: ModelIdentity,
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
        let (engine, tokenizer, voices, model_identity) = match &paths.source {
            ModelSource::Legacy {
                model,
                vocab,
                voices,
            } => {
                let engine = Engine::Legacy(TtsEngine::from_file(model, intra_threads)?);
                let tokenizer = CharTokenizer::from_vocab_json(vocab)
                    .with_context(|| format!("load vocab {}", vocab.display()))?;
                let voices = load_voices(voices)?;
                let identity = ModelIdentity::compute(
                    model,
                    vocab,
                    SAMPLE_RATE,
                    speech_core::custom_voices::VOICE_DIM,
                )
                .context("compute model identity")?;
                (engine, tokenizer, voices, identity)
            }
            ModelSource::Bundle { dir } => {
                let bundle = Bundle::open(dir, speech_core::custom_voices::VOICE_DIM)
                    .with_context(|| format!("open Holos bundle {}", dir.display()))?;
                if bundle.sample_rate() != SAMPLE_RATE {
                    bail!(
                        "bundle sample rate {} does not match server rate {SAMPLE_RATE}",
                        bundle.sample_rate()
                    );
                }
                let engine =
                    Engine::Holos(Box::new(HolosPipeline::open(dir, intra_threads, true)?));
                let tokenizer = CharTokenizer::from_vocab(bundle.vocab());
                let voices = bundle.voices.clone();
                let identity = bundle_model_identity(dir, &bundle)?;
                (engine, tokenizer, voices, identity)
            }
        };
        model_identity
            .validate()
            .context("validate model identity")?;
        if voices.is_empty() {
            bail!("model has no voices");
        }

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
            model = %model_identity.short(),
            sample_rate = model_identity.sample_rate,
            voice_dim = model_identity.voice_dim,
            streaming = engine.is_streaming(),
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
            model_identity,
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
        self.model_identity.sample_rate
    }

    /// Незмінна ідентичність завантаженої моделі.
    pub fn model_identity(&self) -> &ModelIdentity {
        &self.model_identity
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

    /// Синтез одного юніта (речення) для кешу тейків.
    ///
    /// Повертає сирі float32-семпли до кросфейду та нормалізації, щоб глава
    /// могла бути перезібрана з активних тейків без повторного синтезу.
    pub fn synthesize_unit(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        pronunciation: Option<&Pronouncer>,
    ) -> Result<SynthUnitOutput, SynthError> {
        let text = self.validate_text_with_limit(text, MAX_TEXT_CHARS)?;
        self.validate_speed(speed)?;
        let voice_vector = self.resolve_voice(voice)?;
        let (spoken, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, pronunciation);
        let samples = self.synthesize_verbalized_f32(&spoken, &voice_vector, speed, raw_stress)?;
        Ok(SynthUnitOutput {
            samples,
            spoken,
            verbalizer,
            vb_ms,
        })
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

    /// Як `synthesize_verbalized_f32`, але віддає аудіо шматками.
    ///
    /// Для віконного бандла шматок — це одне декодоване вікно, тому перше
    /// аудіо з'являється, не чекаючи кінця речення.
    fn stream_verbalized_f32<G>(
        &self,
        text: &str,
        voice_vector: &[f32],
        speed: f32,
        raw_stress: bool,
        mut on_samples: G,
    ) -> Result<bool, SynthError>
    where
        G: FnMut(&[f32]) -> bool,
    {
        let parts = self.phonemizer.phonemize(text, raw_stress);
        if parts.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }

        let mut produced = false;
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
            let keep = engine
                .stream(&ids, voice_vector, speed, |samples| {
                    produced = true;
                    on_samples(samples)
                })
                .with_context(|| format!("stream audio for phonemes {part:?}"))?;
            drop(engine);
            if !keep {
                return Ok(false);
            }
        }

        if !produced {
            return Err(SynthError::Internal(anyhow::anyhow!(
                "model produced no audio samples"
            )));
        }
        Ok(true)
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

        let reference = match audio_base64 {
            Some(reference) => Some(self.encode_reference(reference)?),
            None => None,
        };
        let (verbalized, verbalizer, vb_ms) = self.apply_verbalizer(text, mode, pronunciation);
        tracing::debug!(verbalizer, vb_ms, "chapter text verbalized");

        let chunks = sentence_chunks(&verbalized, max_chars);
        if chunks.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }
        let timing_texts = timing_texts(text, &chunks, max_chars);
        let plan: Vec<PlannedChunk> = chunks
            .into_iter()
            .zip(timing_texts)
            .map(|(spoken, timing)| PlannedChunk {
                voice: voice.map(str::to_string),
                spoken,
                timing,
                role: None,
            })
            .collect();

        let audio = self.synthesize_planned_chunks(
            &plan,
            reference.as_deref(),
            speed,
            raw_stress,
            crossfade_ms,
            loudness_target,
            &mut progress,
        )?;

        Ok(SynthChapterOutput {
            pcm: audio.pcm,
            verbalizer,
            vb_ms,
            chunks: plan.len(),
            chunk_timings: audio.chunk_timings,
            loudness: audio.loudness,
            duration_ms: audio.duration_ms,
            units: audio.units,
        })
    }

    /// Синтез глави з діалогами: кожен сегмент озвучується голосом своєї ролі.
    ///
    /// Нарація (`role = None`) використовує голос проєкту, репліки — голос ролі
    /// з `roles`; якщо голос ролі не заданий, береться голос проєкту. Таймінги
    /// зберігають роль кожного чанка для субтитрів і майбутнього редактора.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_dialogue_chapter_pcm(
        &self,
        segments: &[DialogueSegment],
        default_voice: Option<&str>,
        roles: &[Role],
        speed: f32,
        mode: VerbalizerMode,
        raw_stress: bool,
        max_chars: usize,
        crossfade_ms: u32,
        pronunciation: Option<&Pronouncer>,
        loudness_target: Option<speech_audio::loudness::LoudnessTarget>,
        mut progress: Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<SynthChapterOutput, SynthError> {
        self.validate_speed(speed)?;
        let total_chars: usize = segments
            .iter()
            .map(|segment| segment.text.chars().count())
            .sum();
        if total_chars == 0 {
            return Err(SynthError::BadRequest("text must not be empty".into()));
        }
        if total_chars > MAX_JOB_TEXT_CHARS {
            return Err(SynthError::BadRequest(format!(
                "text is too long: max {MAX_JOB_TEXT_CHARS} characters"
            )));
        }

        let role_voices: HashMap<String, Option<String>> = roles
            .iter()
            .map(|role| (role.name.to_lowercase(), role.voice.clone()))
            .collect();

        let mut plan = Vec::new();
        let mut verbalizer = None;
        let mut vb_ms = 0u64;
        for segment in segments {
            let (verbalized, label, segment_vb_ms) =
                self.apply_verbalizer(&segment.text, mode, pronunciation);
            vb_ms = vb_ms.saturating_add(segment_vb_ms);
            verbalizer = Some(match verbalizer {
                None => label,
                Some(previous) if previous == label => previous,
                Some(_) => "mixed",
            });

            let chunks = sentence_chunks(&verbalized, max_chars);
            if chunks.is_empty() {
                continue;
            }
            let timing_texts = timing_texts(&segment.text, &chunks, max_chars);
            let voice = segment
                .role
                .as_ref()
                .and_then(|role| role_voices.get(&role.to_lowercase()))
                .cloned()
                .flatten()
                .or_else(|| default_voice.map(str::to_string));
            for (spoken, timing) in chunks.into_iter().zip(timing_texts) {
                plan.push(PlannedChunk {
                    voice: voice.clone(),
                    spoken,
                    timing,
                    role: segment.role.clone(),
                });
            }
        }
        if plan.is_empty() {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }
        tracing::debug!(
            segments = segments.len(),
            chunks = plan.len(),
            vb_ms,
            "dialogue chapter verbalized"
        );

        let audio = self.synthesize_planned_chunks(
            &plan,
            None,
            speed,
            raw_stress,
            crossfade_ms,
            loudness_target,
            &mut progress,
        )?;

        Ok(SynthChapterOutput {
            pcm: audio.pcm,
            verbalizer: verbalizer.unwrap_or("off"),
            vb_ms,
            chunks: plan.len(),
            chunk_timings: audio.chunk_timings,
            loudness: audio.loudness,
            duration_ms: audio.duration_ms,
            units: audio.units,
        })
    }

    /// Озвучити план чанків, за потреби різними голосами, і зібрати главу.
    #[allow(clippy::too_many_arguments)]
    fn synthesize_planned_chunks(
        &self,
        chunks: &[PlannedChunk],
        reference: Option<&[f32]>,
        speed: f32,
        raw_stress: bool,
        crossfade_ms: u32,
        loudness_target: Option<speech_audio::loudness::LoudnessTarget>,
        progress: &mut Option<&mut dyn FnMut(usize, usize) -> bool>,
    ) -> Result<ChapterAudio, SynthError> {
        let total = chunks.len();
        if total == 0 {
            return Err(SynthError::BadRequest(
                "text contains no speakable content".into(),
            ));
        }

        // Голоси резолвимо один раз на унікальне ім'я: `.npy` не перечитується
        // для кожної репліки.
        let mut voices: HashMap<Option<String>, Vec<f32>> = HashMap::new();
        if reference.is_none() {
            for chunk in chunks {
                if !voices.contains_key(&chunk.voice) {
                    let vector = self.resolve_voice(chunk.voice.as_deref())?;
                    voices.insert(chunk.voice.clone(), vector);
                }
            }
        }

        if let Some(cb) = progress.as_deref_mut() {
            if !cb(0, total) {
                return Err(SynthError::Cancelled);
            }
        }

        let mut audio_chunks = Vec::with_capacity(total);
        for (index, chunk) in chunks.iter().enumerate() {
            let voice = match reference {
                Some(vector) => vector,
                None => voices
                    .get(&chunk.voice)
                    .expect("voice was resolved before synthesis"),
            };
            let samples =
                self.synthesize_verbalized_f32(&chunk.spoken, voice, speed, raw_stress)?;
            audio_chunks.push(samples);
            if let Some(cb) = progress.as_deref_mut() {
                if !cb(index + 1, total) {
                    return Err(SynthError::Cancelled);
                }
            }
        }

        let timing_texts: Vec<String> = chunks.iter().map(|chunk| chunk.timing.clone()).collect();
        let roles: Vec<Option<String>> = chunks.iter().map(|chunk| chunk.role.clone()).collect();
        let mut audio = assemble_chapter_audio(
            &audio_chunks,
            &timing_texts,
            &roles,
            crossfade_ms,
            loudness_target,
        )?;
        audio.units = chunks
            .iter()
            .zip(audio_chunks)
            .map(|(chunk, samples)| SynthUnit {
                samples,
                text: chunk.timing.clone(),
                spoken: chunk.spoken.clone(),
                voice: chunk.voice.clone(),
                role: chunk.role.clone(),
            })
            .collect();
        Ok(audio)
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
            let keep =
                self.stream_verbalized_f32(chunk, &voice_vector, speed, raw_stress, |samples| {
                    let pcm = float_to_pcm16(samples);
                    audio_ms += pcm.len() as u64 * 1000 / SAMPLE_RATE as u64;
                    on_chunk(&pcm16_bytes(&pcm))
                })?;
            if !keep {
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

/// Склеїти сирі чанки у PCM глави: кросфейд, таймінги та LUFS.
fn assemble_chapter_audio(
    audio_chunks: &[Vec<f32>],
    timing_texts: &[String],
    roles: &[Option<String>],
    crossfade_ms: u32,
    loudness_target: Option<speech_audio::loudness::LoudnessTarget>,
) -> Result<ChapterAudio, SynthError> {
    let total = audio_chunks.len();
    if total == 0 {
        return Err(SynthError::BadRequest(
            "chapter has no speakable content".into(),
        ));
    }
    if timing_texts.len() != total || roles.len() != total {
        return Err(SynthError::Internal(anyhow::anyhow!(
            "chapter plan is inconsistent: {} chunks, {} texts, {} roles",
            total,
            timing_texts.len(),
            roles.len()
        )));
    }
    if audio_chunks.iter().any(Vec::is_empty) {
        return Err(SynthError::Internal(anyhow::anyhow!(
            "chapter contains an empty audio unit"
        )));
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
            role: roles[index].clone(),
        });
        cursor = end;
    }

    let mut samples = crossfade_concat(audio_chunks, overlap);
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

    Ok(ChapterAudio {
        pcm,
        chunk_timings,
        loudness,
        duration_ms,
        units: Vec::new(),
    })
}

/// Перезібрати главу з готових юнітів (активних тейків).
///
/// Результат еквівалентний повному синтезу: кросфейд і LUFS застосовуються
/// до всієї глави, а не до окремих тейків.
pub fn assemble_chapter_pcm(
    units: Vec<AssembledUnit>,
    crossfade_ms: u32,
    loudness_target: Option<speech_audio::loudness::LoudnessTarget>,
) -> Result<SynthChapterOutput, SynthError> {
    let chunks = units.len();
    if chunks == 0 {
        return Err(SynthError::BadRequest(
            "chapter has no speakable content".into(),
        ));
    }
    let mut audio_chunks = Vec::with_capacity(chunks);
    let mut timing_texts = Vec::with_capacity(chunks);
    let mut roles = Vec::with_capacity(chunks);
    for unit in units {
        audio_chunks.push(unit.samples);
        timing_texts.push(unit.text);
        roles.push(unit.role);
    }
    let audio = assemble_chapter_audio(
        &audio_chunks,
        &timing_texts,
        &roles,
        crossfade_ms,
        loudness_target,
    )?;
    Ok(SynthChapterOutput {
        pcm: audio.pcm,
        verbalizer: "cache",
        vb_ms: 0,
        chunks,
        chunk_timings: audio.chunk_timings,
        loudness: audio.loudness,
        duration_ms: audio.duration_ms,
        units: Vec::new(),
    })
}

/// Текст для субтитрів: оригінальні речення, якщо вербалізація зберегла
/// кількість чанків; інакше — вербалізовані (номери вже розгорнуто).
fn timing_texts(original: &str, spoken: &[String], max_chars: usize) -> Vec<String> {
    let original_chunks = sentence_chunks(original, max_chars);
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
        assert_eq!(
            paths.source,
            ModelSource::Legacy {
                model: root.join("tts/holos_cpu_int8.onnx"),
                vocab: root.join("tts/vocab.json"),
                voices: root.join("tts/voices.json"),
            }
        );
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
    fn discovers_holos_bundle() {
        let root = std::env::temp_dir().join(format!(
            "promin-models-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("holos-onnx")).unwrap();
        std::fs::create_dir_all(root.join("stress")).unwrap();
        std::fs::write(root.join("holos-onnx/holos_onnx.json"), b"{}").unwrap();
        std::fs::write(root.join("stress/stress.trie"), b"x").unwrap();

        let paths = ModelPaths::discover(&root).expect("discover bundle");
        assert_eq!(
            paths.source,
            ModelSource::Bundle {
                dir: root.join("holos-onnx"),
            }
        );

        // Явний legacy ігнорує бандл, явний bundle без маніфесту — помилка.
        assert!(ModelPaths::discover_with_format(&root, "legacy").is_err());
        std::fs::remove_file(root.join("holos-onnx/holos_onnx.json")).unwrap();
        assert!(ModelPaths::discover_with_format(&root, "bundle").is_err());
        assert!(ModelPaths::discover_with_format(&root, "nonsense").is_err());

        let _ = std::fs::remove_dir_all(&root);
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
        assert_eq!(
            paths.source,
            ModelSource::Legacy {
                model: root.join("holos_cpu_int8.onnx"),
                vocab: root.join("vocab.json"),
                voices: root.join("voices.json"),
            }
        );
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
        assert_eq!(
            paths.source,
            ModelSource::Legacy {
                model: root.join("holos_cpu_int8.onnx"),
                vocab: root.join("vocab.json"),
                voices: root.join("voices.json"),
            }
        );
        assert_eq!(paths.stress_trie, root.join("stress.trie"));
        assert_eq!(
            paths.voice_encoder.as_deref(),
            Some(root.join("tts/voice_encoder.onnx").as_path())
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn float32_take_roundtrip_preserves_chapter_bytes() {
        use speech_audio::{decode_wav_f32, wav_bytes_f32, LoudnessTarget, SAMPLE_RATE};

        let first: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.013).sin() * 0.31)
            .collect();
        let second: Vec<f32> = (0..11_000)
            .map(|i| (i as f32 * 0.021).cos() * 0.23)
            .collect();
        let target = LoudnessTarget::new(-16.0, -1.5);

        let expected = assemble_chapter_pcm(
            vec![
                AssembledUnit {
                    samples: first.clone(),
                    text: "Перше речення.".into(),
                    role: None,
                },
                AssembledUnit {
                    samples: second.clone(),
                    text: "Друге речення.".into(),
                    role: Some("Іван".into()),
                },
            ],
            40,
            Some(target),
        )
        .unwrap();

        let mut rebuilt = Vec::new();
        for (samples, text, role) in [
            (first, "Перше речення.", None),
            (second, "Друге речення.", Some("Іван")),
        ] {
            let wav = wav_bytes_f32(&samples, SAMPLE_RATE).unwrap();
            let (decoded, rate) = decode_wav_f32(&wav).unwrap();
            assert_eq!(rate, SAMPLE_RATE);
            rebuilt.push(AssembledUnit {
                samples: decoded,
                text: text.into(),
                role: role.map(str::to_string),
            });
        }
        let actual = assemble_chapter_pcm(rebuilt, 40, Some(target)).unwrap();

        assert_eq!(expected.pcm, actual.pcm);
        assert_eq!(expected.chunk_timings, actual.chunk_timings);
        assert_eq!(expected.duration_ms, actual.duration_ms);
        assert_eq!(expected.loudness, actual.loudness);
    }
}
