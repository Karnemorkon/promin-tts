pub mod ipa;
pub mod phonemize;
mod repl;
pub mod stress;
pub mod text_prep;
pub mod tokenizer;
pub mod utf8;

pub fn crate_name() -> &'static str {
    "speech-text"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-text");
    }
}
