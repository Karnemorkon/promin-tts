use std::{path::Path, sync::Arc};

use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
};

/// ONNX-based TTS generator engine (holos_cpu_int8.onnx).
pub struct TtsEngine {
    /// Kept alive so that the session stays valid.
    _env: Arc<Environment>,
    session: Session,
}

impl TtsEngine {
    pub fn from_file(model_path: impl AsRef<Path>, intra_threads: i16) -> ort::OrtResult<Self> {
        let env = Arc::new(Environment::builder().with_name("promin-tts").build()?);
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(intra_threads)?
            .with_model_from_file(model_path)?;
        Ok(Self { _env: env, session })
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.session
            .inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect()
    }

    pub fn output_names(&self) -> Vec<&str> {
        self.session
            .outputs
            .iter()
            .map(|o| o.name.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires models + ORT 1.17: ORT_STRATEGY=system ORT_LIB_LOCATION=<ort>/lib PROMIN_MODELS_DIR=<dir>"]
    fn load_holos_model() {
        let dir = std::env::var("PROMIN_MODELS_DIR").expect("PROMIN_MODELS_DIR not set");
        let model = Path::new(&dir).join("tts/holos_cpu_int8.onnx");
        let engine = TtsEngine::from_file(&model, 4).expect("load TTS model");
        assert!(!engine.input_names().is_empty());
        assert!(!engine.output_names().is_empty());
    }
}
