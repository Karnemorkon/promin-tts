//! Back-ASR: українське розпізнавання мовлення та порівняння зі скриптом.
//!
//! Модуль навмисно не залежить від сервера: [`AsrModel`] приймає mono float32
//! у діапазоні [-1, 1], а [`diff::compare`] працює з будь-якими рядками.
//! Сервер використовує його для перевірки готових глав без Python у рантаймі.

pub mod decode;
pub mod diff;
pub mod features;
pub mod model;

pub use decode::greedy_ctc;
pub use diff::{compare, normalize_text, DiffError, DiffReport, ErrorKind};
pub use features::{extract_w2v_bert, FeatureMatrix, FEATURE_DIM};
pub use model::{AsrModel, AsrTranscript, BLANK_ID, DEFAULT_MODEL_ID, UNK_ID};

/// Назва крейта для smoke-тестів.
pub fn crate_name() -> &'static str {
    "speech-asr"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-asr");
    }
}
