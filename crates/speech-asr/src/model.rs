//! ONNX W2V-BERT CTC model: ознаки -> logits -> greedy CTC -> SentencePiece.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use asr_features::W2vBertFrontendConfig;
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    Value,
};
use sentencepiece::SentencePieceProcessor;

use crate::decode::greedy_ctc;
use crate::features::{config as frontend_config, extract_w2v_bert, FEATURE_DIM};

/// Репозиторій моделі за замовчуванням.
pub const DEFAULT_MODEL_ID: &str = "speech-uk/w2v-bert-v3-onnx";
/// CTC blank id у `speech-uk/w2v-bert-v3`.
pub const BLANK_ID: u32 = 0;
/// CTC unknown id (SentencePiece `<unk>`).
pub const UNK_ID: u32 = 1;

/// Файли, без яких модель не завантажиться (окрім зовнішніх ваг).
pub const BASE_FILES: &[&str] = &["model.onnx", "tokenizer.model"];
/// Зовнішні ваги для графа `speech-uk/w2v-bert-v3`.
pub const EXTERNAL_DATA_FILE: &str = "model.onnx_data";

/// Результат одного розпізнавання.
#[derive(Debug, Clone)]
pub struct AsrTranscript {
    pub text: String,
    pub tokens: Vec<u32>,
    /// Кількість вихідних кадрів CTC (40 мс на кадр).
    pub frames: usize,
    pub audio_ms: u64,
    pub inference_ms: u64,
}

/// Завантажена ASR-модель.
pub struct AsrModel {
    /// Тримаємо environment живим, доки живе сесія.
    _env: Arc<Environment>,
    session: Session,
    tokenizer: SentencePieceProcessor,
    output_name: String,
    frontend: W2vBertFrontendConfig,
    model_id: String,
    threads: i16,
}

impl AsrModel {
    /// Завантажити `model.onnx` + `tokenizer.model` з каталогу.
    pub fn load(model_dir: impl AsRef<Path>, model_id: &str, threads: i16) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.model");
        for path in [&model_path, &tokenizer_path] {
            if !path.is_file() {
                bail!("ASR file is missing: {}", path.display());
            }
        }

        let threads = threads.max(1);
        let env = Arc::new(
            Environment::builder()
                .with_name("promin-asr")
                .build()
                .context("create ONNX Runtime environment for ASR")?,
        );
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(threads)?
            .with_model_from_file(&model_path)
            .with_context(|| format!("load ASR model {}", model_path.display()))?;

        if session.inputs.is_empty() {
            bail!("ASR model has no inputs");
        }
        let output_name = session
            .outputs
            .iter()
            .find(|output| output.name == "logits")
            .or_else(|| session.outputs.first())
            .map(|output| output.name.clone())
            .context("ASR model has no outputs")?;

        let tokenizer = SentencePieceProcessor::open(&tokenizer_path)
            .with_context(|| format!("load SentencePiece {}", tokenizer_path.display()))?;

        Ok(Self {
            _env: env,
            session,
            tokenizer,
            output_name,
            frontend: frontend_config(model_id),
            model_id: model_id.to_string(),
            threads,
        })
    }

    /// Розпізнати mono float32 у діапазоні [-1, 1].
    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<AsrTranscript> {
        if samples.is_empty() {
            bail!("cannot transcribe empty audio");
        }
        if sample_rate == 0 {
            bail!("sample rate must be positive");
        }
        let started = Instant::now();
        let features = extract_w2v_bert(samples, sample_rate, &self.model_id)?;
        let rows = features.rows;
        let cols = features.cols;
        if cols != FEATURE_DIM {
            bail!("unexpected ASR feature width: {cols}");
        }
        let input = CowArray::from(
            Array::from_shape_vec(IxDyn(&[1, rows, cols]), features.values)
                .context("build ASR input tensor")?,
        );
        let value = Value::from_array(self.session.allocator(), &input)
            .context("create ASR input value")?;
        let output_index = self
            .session
            .outputs
            .iter()
            .position(|output| output.name == self.output_name)
            .unwrap_or(0);
        let outputs = self
            .session
            .run(vec![value])
            .context("ONNX session run (ASR)")?;
        let output = outputs
            .get(output_index)
            .context("ASR model returned no outputs")?;
        let tensor = output
            .try_extract::<f32>()
            .context("extract ASR logits as f32")?;
        let view = tensor.view();
        let shape = view.shape();
        if shape.len() != 3 || shape[0] != 1 {
            bail!("unexpected ASR logits shape: {shape:?}");
        }
        let frames = shape[1];
        let vocab = shape[2];
        let logits = view
            .as_slice()
            .context("ASR logits tensor is not contiguous")?;
        let tokens = greedy_ctc(logits, frames, vocab, BLANK_ID)?;
        let text = self
            .tokenizer
            .decode_piece_ids(&tokens)
            .context("decode ASR SentencePiece tokens")?
            .trim()
            .to_string();
        Ok(AsrTranscript {
            text,
            tokens,
            frames,
            audio_ms: samples.len() as u64 * 1000 / u64::from(sample_rate),
            inference_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn threads(&self) -> i16 {
        self.threads
    }

    pub fn frontend(&self) -> &W2vBertFrontendConfig {
        &self.frontend
    }
}

/// Файли, яких бракує для запуску моделі.
///
/// `model.onnx` для цієї моделі — лише граф (кілька МБ), а ваги лежать у
/// `model.onnx_data`. Якщо граф великий, зовнішні ваги не обов'язкові.
pub fn missing_model_files(model_dir: impl AsRef<Path>) -> Vec<&'static str> {
    let model_dir = model_dir.as_ref();
    let mut missing = Vec::new();
    for file in BASE_FILES {
        if !non_empty(&model_dir.join(file)) {
            missing.push(*file);
        }
    }
    let graph = model_dir.join("model.onnx");
    let graph_is_small = graph
        .metadata()
        .map(|meta| meta.len() < 64 * 1024 * 1024)
        .unwrap_or(true);
    if graph_is_small && !non_empty(&model_dir.join(EXTERNAL_DATA_FILE)) {
        missing.push(EXTERNAL_DATA_FILE);
    }
    missing
}

/// Чи всі файли моделі на місці.
pub fn model_files_present(model_dir: impl AsRef<Path>) -> bool {
    missing_model_files(model_dir).is_empty()
}

/// Шлях до каталогу моделі за замовчуванням.
pub fn default_model_dir(models_dir: impl AsRef<Path>) -> PathBuf {
    models_dir.as_ref().join("asr").join("w2v-bert-v3")
}

fn non_empty(path: &Path) -> bool {
    path.metadata()
        .map(|meta| meta.is_file() && meta.len() > 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_files_are_reported() {
        let dir = std::env::temp_dir().join(format!("promin-asr-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let missing = missing_model_files(&dir);
        assert!(missing.contains(&"model.onnx"));
        assert!(missing.contains(&"tokenizer.model"));
        assert!(missing.contains(&EXTERNAL_DATA_FILE));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "requires PROMIN_ASR_TEST_MODEL_DIR and PROMIN_ASR_TEST_WAV"]
    fn transcribes_reference_wav() {
        let model_dir = std::env::var("PROMIN_ASR_TEST_MODEL_DIR")
            .expect("PROMIN_ASR_TEST_MODEL_DIR is not set");
        let wav_path =
            std::env::var("PROMIN_ASR_TEST_WAV").expect("PROMIN_ASR_TEST_WAV is not set");
        let model = AsrModel::load(model_dir, DEFAULT_MODEL_ID, 4).unwrap();
        let mut reader = hound::WavReader::open(wav_path).unwrap();
        let spec = reader.spec();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect();
        let transcript = model.transcribe(&samples, spec.sample_rate).unwrap();
        assert!(!transcript.text.is_empty());
        println!("{}", transcript.text);
    }
}
