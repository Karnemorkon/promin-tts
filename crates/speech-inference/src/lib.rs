pub mod accentor;
pub mod engine;
pub mod gemma;
pub mod voice_encoder;

pub use accentor::Accentor;
pub use engine::{TtsEngine, VOICE_DIM};
pub use gemma::GemmaVerbalizer;
pub use voice_encoder::VoiceEncoder;

pub fn crate_name() -> &'static str {
    "speech-inference"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-inference");
    }
}
