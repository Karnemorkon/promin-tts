//! ONNX voice encoder: wave float32 [1, N] @24 кГц -> ембеддінг [1, 256].
//!
//! Порт `voice_encoder.cpp`: сесія створюється ліниво при першому `encode`
//! (енкодер потрібен лише для клонування, тож не платимо RSS і час старту).
//! Пам'ять понад усе: mem pattern вимкнено, виклики рідкісні.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    GraphOptimizationLevel, Value,
};

use crate::VOICE_DIM;

/// Завантажена ORT-сесія енкодера (створюється один раз).
struct EncoderSession {
    /// Тримає Environment живим, поки живе сесія.
    _env: Arc<Environment>,
    session: Session,
}

impl EncoderSession {
    fn load(path: &Path, intra_threads: i16) -> Result<Self> {
        let env = Arc::new(
            Environment::builder()
                .with_name("promin-tts-voice-encoder")
                .build()
                .context("create ONNX Runtime environment (voice encoder)")?,
        );
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(intra_threads.max(1))?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // Тимчасові буфери звільняються після Run (менший пік RSS).
            .with_memory_pattern(false)?
            .with_model_from_file(path)
            .with_context(|| format!("load ONNX model {}", path.display()))?;

        let input_name = session
            .inputs
            .first()
            .map(|input| input.name.clone())
            .context("voice encoder has no inputs")?;
        let output_name = session
            .outputs
            .first()
            .map(|output| output.name.clone())
            .context("voice encoder has no outputs")?;

        tracing::info!(
            model = %path.display(),
            input = %input_name,
            output = %output_name,
            threads = intra_threads.max(1),
            "voice encoder loaded"
        );
        Ok(Self { _env: env, session })
    }
}

/// Ліниво завантажуваний voice encoder.
pub struct VoiceEncoder {
    path: PathBuf,
    intra_threads: i16,
    session: Mutex<Option<Arc<EncoderSession>>>,
}

impl VoiceEncoder {
    /// Перевіряє доступність файлу; сама сесія створюється при першому encode.
    pub fn from_file(path: impl AsRef<Path>, intra_threads: i16) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::File::open(&path)
            .with_context(|| format!("open voice encoder model {}", path.display()))?;
        Ok(Self {
            path,
            intra_threads,
            session: Mutex::new(None),
        })
    }

    /// Чи налаштований енкодер (файл доступний при створенні).
    pub fn ready(&self) -> bool {
        self.path.is_file()
    }

    fn session(&self) -> Result<Arc<EncoderSession>> {
        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_none() {
            *guard = Some(Arc::new(EncoderSession::load(
                &self.path,
                self.intra_threads,
            )?));
        }
        Ok(Arc::clone(guard.as_ref().expect("session is initialized")))
    }

    /// Обчислює 256-вимірний ембеддінг; `wave` — mono float32 @24 кГц.
    pub fn encode(&self, wave: &[f32]) -> Result<Vec<f32>> {
        if wave.is_empty() {
            bail!("empty audio for voice encoder");
        }
        let session = self.session()?;

        let input = CowArray::from(Array::from_shape_vec(
            IxDyn(&[1, wave.len()]),
            wave.to_vec(),
        )?);
        let value = Value::from_array(session.session.allocator(), &input)?;
        let outputs = session
            .session
            .run(vec![value])
            .context("ONNX session run (voice encoder)")?;

        let tensor = outputs[0]
            .try_extract::<f32>()
            .context("extract voice encoder output")?;
        let view = tensor.view();
        let data = view
            .as_slice()
            .context("voice encoder output is not contiguous")?;
        if data.len() != VOICE_DIM {
            bail!(
                "voice encoder expected {VOICE_DIM}-dim output, got {}",
                data.len()
            );
        }
        Ok(data.to_vec())
    }
}

impl std::fmt::Debug for VoiceEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceEncoder")
            .field("path", &self.path)
            .field("intra_threads", &self.intra_threads)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_model() {
        assert!(VoiceEncoder::from_file("/definitely/missing/encoder.onnx", 1).is_err());
    }

    #[test]
    fn encoder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VoiceEncoder>();
    }
}
