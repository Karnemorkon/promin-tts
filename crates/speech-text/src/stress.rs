//! Словниковий наголос через marisa-trie (stress.trie, 2.9 млн форм).

use crate::utf8;
use rsmarisa::{Agent, Trie};
use std::path::Path;

const TRIE_SEP: u8 = 0xFF;
const REC_SEP: u8 = 0xFF;
const POS_SEP: u8 = 0xFE;
const ACUTE: &str = "\u{0301}";

/// Байти payload -> позиції наголосу (індекси codepoint, "після літери").
/// >1 POS-варіантів -> порожній (неоднозначність, варіант A).
fn parse_payload(payload: &[u8]) -> Vec<usize> {
    let mut records: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    while start < payload.len() {
        match payload[start..].iter().position(|&b| b == REC_SEP) {
            Some(rel) => {
                let sep = start + rel;
                if sep > start {
                    records.push(&payload[start..sep]);
                }
                start = sep + 1;
            }
            None => {
                if start < payload.len() {
                    records.push(&payload[start..]);
                }
                break;
            }
        }
    }
    records.retain(|r| !r.is_empty());
    if records.len() > 1 {
        return Vec::new();
    }
    let Some(rec) = records.first() else {
        return Vec::new();
    };
    let accent_bytes = match rec.iter().position(|&b| b == POS_SEP) {
        Some(p) => &rec[..p],
        None => rec,
    };
    accent_bytes.iter().map(|&b| b as usize).collect()
}

pub struct StressDictionary {
    trie: Trie,
}

impl StressDictionary {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("bad trie path"))?;
        let mut trie = Trie::new();
        trie.load(path_str)
            .map_err(|e| anyhow::anyhow!("Не вдалося завантажити {}: {e}", path.display()))?;
        Ok(Self { trie })
    }

    pub fn len(&self) -> usize {
        self.trie.num_keys()
    }

    pub fn is_empty(&self) -> bool {
        self.trie.num_keys() == 0
    }

    /// Позиції наголосу для слова (без наголосу -> порожній вектор).
    pub fn accent_positions(&self, word: &str) -> Vec<usize> {
        if word.is_empty() {
            return Vec::new();
        }
        let lower = utf8::unicode_lower(word);
        let title = utf8::unicode_title(word);
        for w in [word, lower.as_str(), title.as_str()] {
            let mut q = Vec::with_capacity(w.len() + 1);
            q.extend_from_slice(w.as_bytes());
            q.push(TRIE_SEP);
            let mut agent = Agent::new();
            agent.set_query_bytes(&q);
            while self.trie.predictive_search(&mut agent) {
                let key = agent.key();
                let k = key.as_bytes();
                if k.len() <= q.len() {
                    continue;
                }
                let payload = &k[q.len()..];
                let pos = parse_payload(payload);
                if !pos.is_empty() || payload.is_empty() {
                    return pos;
                }
            }
        }
        Vec::new()
    }

    /// Слово з U+0301 після наголошених літер (або без змін).
    pub fn stress(&self, word: &str) -> String {
        let mut pos = self.accent_positions(word);
        if pos.is_empty() {
            return word.to_string();
        }
        pos.sort_unstable_by(|a, b| b.cmp(a));
        let mut s = word.to_string();
        for p in pos {
            s = utf8::insert_at_cp(&s, p, ACUTE);
        }
        s
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
    #[ignore = "requires stress.trie (PROMIN_MODELS_DIR)"]
    fn golden_stress() {
        let Some(path) = trie_path() else {
            return;
        };
        let dict = StressDictionary::open(&path).expect("load trie");
        assert!(dict.len() > 2_000_000);
        assert_eq!(dict.stress("привіт"), "приві\u{0301}т");
        assert_eq!(dict.stress("місто"), "мі\u{0301}сто");
        assert_eq!(dict.stress("щастя"), "ща\u{0301}стя");
        assert_eq!(dict.stress("українською"), "украї\u{0301}нською");
        assert_eq!(dict.stress("Привіт"), "Приві\u{0301}т");
        assert_eq!(dict.stress("Hello"), "Hello");
    }
}
