pub mod engine;

pub use engine::{TtsEngine, VOICE_DIM};

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
