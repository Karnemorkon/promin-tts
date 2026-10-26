//! Текст -> фонеми: split_to_parts -> normalize_part -> stress -> ipa::transcribe.
//! Порт phonemize.cpp: словниковий наголос + neural accentor для OOV.

use std::path::Path;
use std::sync::Arc;

use crate::ipa;
use crate::neural::NeuralStress;
use crate::stress::StressDictionary;
use crate::text_prep;

/// Слово: українські кириличні літери + апострофи (ʼ, ’, ').
fn is_word_cp(c: char) -> bool {
    ('\u{0400}'..='\u{04FF}').contains(&c) || c == '\u{02BC}' || c == '\u{2019}' || c == '\''
}

const ACUTE: char = '\u{0301}';

/// Словниковий наголос; якщо словник не дав U+0301 — accentor (як у C++).
fn stress_word(dict: &StressDictionary, accentor: Option<&dyn NeuralStress>, word: &str) -> String {
    let stressed = dict.stress(word);
    if stressed.contains(ACUTE) {
        return stressed;
    }
    let Some(accentor) = accentor else {
        return stressed;
    };
    match accentor.stress(word) {
        Ok(neural) if neural.contains(ACUTE) => neural,
        Ok(_) => stressed,
        Err(error) => {
            tracing::warn!(error = %error, word, "accentor failed, keeping dictionary form");
            stressed
        }
    }
}

/// Наголос по кожному слову тексту (словник + опційний accentor).
pub fn stress_text(
    dict: &StressDictionary,
    accentor: Option<&dyn NeuralStress>,
    text: &str,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    for ch in text.chars() {
        if is_word_cp(ch) {
            word.push(ch);
        } else {
            if !word.is_empty() {
                out.push_str(&stress_word(dict, accentor, &word));
                word.clear();
            }
            out.push(ch);
        }
    }
    if !word.is_empty() {
        out.push_str(&stress_word(dict, accentor, &word));
    }
    out
}

/// Фонемізатор: словниковий наголос (+ accentor) + IPA-транскрипція.
pub struct Phonemizer {
    dict: StressDictionary,
    accentor: Option<Arc<dyn NeuralStress>>,
}

impl Phonemizer {
    pub fn open(stress_trie: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::open_with_accentor(stress_trie, None)
    }

    /// `accentor` — опційний neural fallback для слів поза словником.
    pub fn open_with_accentor(
        stress_trie: impl AsRef<Path>,
        accentor: Option<Arc<dyn NeuralStress>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            dict: StressDictionary::open(stress_trie)?,
            accentor,
        })
    }

    pub fn has_accentor(&self) -> bool {
        self.accentor.is_some()
    }

    /// Фонемний рядок на кожну частину тексту.
    /// `raw_stress = true` — пропустити автонаголос (наголоси вже в тексті).
    pub fn phonemize(&self, text: &str, raw_stress: bool) -> Vec<String> {
        let mut result = Vec::new();
        for part in text_prep::split_to_parts(text, true) {
            let norm = text_prep::normalize_part(&part);
            if norm.is_empty() {
                continue;
            }
            let stressed = if raw_stress {
                norm
            } else {
                stress_text(&self.dict, self.accentor.as_deref(), &norm)
            };
            let ph = ipa::transcribe(&stressed).unwrap_or_default();
            if !ph.is_empty() {
                result.push(ph);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn trie_path() -> Option<PathBuf> {
        let dir = std::env::var("PROMIN_MODELS_DIR").ok()?;
        let nested = PathBuf::from(&dir).join("stress/stress.trie");
        if nested.exists() {
            return Some(nested);
        }
        let flat = PathBuf::from(&dir).join("stress.trie");
        flat.exists().then_some(flat)
    }

    #[test]
    fn phonemizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Phonemizer>();
    }

    #[test]
    fn word_chars() {
        assert!(is_word_cp('п'));
        assert!(is_word_cp('\''));
        assert!(is_word_cp('\u{02BC}'));
        assert!(!is_word_cp(' '));
        assert!(!is_word_cp('.'));
        assert!(!is_word_cp('1'));
    }

    #[test]
    #[ignore = "requires stress.trie (PROMIN_MODELS_DIR)"]
    fn phonemize_simple() {
        let path = trie_path().expect("PROMIN_MODELS_DIR with stress.trie");
        let p = Phonemizer::open(&path).expect("load dictionary");
        let parts = p.phonemize("Привіт, світе!", false);
        let all = parts.join(" ");
        println!("PHONEMES: {all}");
        assert!(!parts.is_empty());
        assert!(all.chars().all(|c| !('\u{0400}'..='\u{04FF}').contains(&c)));
        assert!(all.contains('\u{02C8}'), "at least one stressed vowel");
    }

    // Golden-дані згенеровано C++ еталоном (holos-tts-core: Phonemizer).
    #[test]
    #[ignore = "requires stress.trie (PROMIN_MODELS_DIR)"]
    fn phonemize_golden() {
        let path = trie_path().expect("PROMIN_MODELS_DIR with stress.trie");
        let p = Phonemizer::open(&path).expect("load dictionary");
        let cases = [
            ("привіт", "prɪˈʋʲit."),
            ("місто", "ˈmʲistɔ."),
            ("щастя", "ˈʃt͡ʃasʲtʲɐ."),
            ("українською", "ʊkrɐˈjinʲsʲkɔjʊ."),
            ("Привіт, світе!", "prɪˈʋʲit, ˈsʲʋʲite!"),
            ("Слава Україні!", "ˈsɫaʋɐ ʊkrɐˈjinʲi!"),
            ("Ґанок, їжак і солов'ї.", "ˈɡanɔk, jiˈʒak i sɔɫɔˈʋji."),
        ];
        for (text, expected) in cases {
            assert_eq!(p.phonemize(text, false).join(" "), expected, "text: {text}");
        }
    }
}
