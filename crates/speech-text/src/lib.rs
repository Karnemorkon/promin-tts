pub mod dialogue;
pub mod gemma_prep;
#[cfg(feature = "import")]
pub mod import;
pub mod ipa;
pub mod neural;
pub mod phonemize;
pub mod pronunciation;
mod repl;
pub mod segment;
pub mod stress;
pub mod text_prep;
pub mod tokenizer;
pub mod utf8;
pub mod verbalize;

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
