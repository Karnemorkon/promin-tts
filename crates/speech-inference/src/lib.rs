pub mod accentor;
pub mod bundle;
pub mod engine;
pub mod gemma;
pub mod pipeline;
pub mod voice_encoder;

pub use accentor::Accentor;
pub use bundle::{Bundle, BundleManifest, BUNDLE_MANIFEST};
pub use engine::{TtsEngine, VOICE_DIM};
pub use gemma::GemmaVerbalizer;
pub use pipeline::HolosPipeline;
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
