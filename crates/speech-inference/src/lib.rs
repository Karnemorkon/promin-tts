pub mod accentor;
pub mod engine;
pub mod gemma;

pub use accentor::Accentor;
pub use engine::{TtsEngine, VOICE_DIM};
pub use gemma::GemmaVerbalizer;

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
