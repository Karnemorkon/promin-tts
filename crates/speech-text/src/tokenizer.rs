//! Посимвольний токенізатор (vocab.json: масив односимвольних рядків).

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    index: HashMap<String, i64>,
}

impl CharTokenizer {
    /// Побудова зі словника; за парністю з Python dict — last-wins.
    pub fn from_vocab(vocab: &[String]) -> Self {
        let mut index = HashMap::with_capacity(vocab.len());
        for (i, ch) in vocab.iter().enumerate() {
            index.insert(ch.clone(), i as i64);
        }
        Self { index }
    }

    pub fn from_vocab_json(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path.as_ref())?;
        let vocab: Vec<String> = serde_json::from_str(&data)?;
        Ok(Self::from_vocab(&vocab))
    }

    /// Токенизація рядка по codepoint; невідомі символи пропускаються.
    /// BOS/EOS НЕ додаються (додає генератор).
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            if let Some(&id) = self.index.get(&s) {
                ids.push(id);
            }
        }
        ids
    }

    pub fn vocab_len(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_chars() {
        let tok = CharTokenizer::from_vocab(&["а".into(), "б".into(), "в".into()]);
        assert_eq!(tok.encode("абв"), vec![0, 1, 2]);
        assert_eq!(tok.encode("а?б"), vec![0, 1]);
    }
}
