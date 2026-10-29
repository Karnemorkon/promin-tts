//! Бізнес-логіка Promin TTS: кастомні голоси, проєкти, черга генерації.

pub mod custom_voices;
pub mod jobs;
pub mod model;
pub mod projects;
pub mod takes;

pub fn crate_name() -> &'static str {
    "speech-core"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-core");
    }
}
