//! accentor-lite: нейромережевий наголос для слів поза словником.
//!
//! Порт `accentor.cpp`: вхід `sequences` int64 `[1, 29]` (id символів,
//! lowercase), вихід `logits` float `[1, 29]`; argmax -> позиція U+0301.
//! Модель маленька (1.2 MB), тому 1 потік, як у C++.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    GraphOptimizationLevel, Value,
};
use speech_text::neural::NeuralStress;
use speech_text::utf8;

/// Максимальна довжина слова в токенах (як `kMaxLen` у C++).
const MAX_LEN: usize = 29;
const ACUTE: &str = "\u{0301}";

/// Lowercase як у C++ `utf8_lower`: ASCII + кирилиця, решта без змін.
fn accentor_lower(cp: u32) -> u32 {
    match cp {
        0x41..=0x5A => cp + 0x20,     // A-Z
        0x0410..=0x042F => cp + 0x20, // А-Я
        0x0400..=0x040F => cp + 0x50, // Ѐ-Џ (Є, І, Ї)
        0x0490 => 0x0491,             // Ґ -> ґ
        _ => cp,
    }
}

/// Словник accentor-lite: PAD=0, <UNK>=1, апострофи=2, а-я/є/і/ї/ґ=3..35.
fn char_id(cp: u32) -> i64 {
    match cp {
        0x0027 | 0x02BC | 0x2019 => 2,
        0x0430 => 3,
        0x0431 => 4,
        0x0432 => 5,
        0x0433 => 6,
        0x0434 => 7,
        0x0435 => 8,
        0x0436 => 9,
        0x0437 => 10,
        0x0438 => 11,
        0x0439 => 12,
        0x043A => 13,
        0x043B => 14,
        0x043C => 15,
        0x043D => 16,
        0x043E => 17,
        0x043F => 18,
        0x0440 => 19,
        0x0441 => 20,
        0x0442 => 21,
        0x0443 => 22,
        0x0444 => 23,
        0x0445 => 24,
        0x0446 => 25,
        0x0447 => 26,
        0x0448 => 27,
        0x0449 => 28,
        0x044C => 29,
        0x044E => 30,
        0x044F => 31,
        0x0454 => 32,
        0x0456 => 33,
        0x0457 => 34,
        0x0491 => 35,
        _ => 1,
    }
}

/// Токенізує слово в `[1, 29]`: lowercase codepoints, PAD до кінця.
fn encode_word(word: &str) -> [i64; MAX_LEN] {
    let mut ids = [0i64; MAX_LEN];
    for (slot, ch) in ids.iter_mut().zip(word.chars()) {
        *slot = char_id(accentor_lower(ch as u32));
    }
    ids
}

/// ONNX accentor-lite.
pub struct Accentor {
    /// Тримає Environment живим, поки живе сесія.
    _env: Arc<Environment>,
    session: Session,
    output_idx: usize,
}

impl Accentor {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let env = Arc::new(
            Environment::builder()
                .with_name("promin-tts-accentor")
                .build()
                .context("create ONNX Runtime environment (accentor)")?,
        );
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(path)
            .with_context(|| format!("load ONNX model {}", path.display()))?;

        if session.inputs.len() != 1 || session.inputs[0].name != "sequences" {
            bail!(
                "accentor model must have a single `sequences` input, got {:?}",
                session
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>()
            );
        }
        let output_idx = session
            .outputs
            .iter()
            .position(|output| output.name == "logits")
            .context("accentor model has no logits output")?;

        tracing::info!(
            model = %path.display(),
            threads = 1,
            "accentor loaded"
        );

        Ok(Self {
            _env: env,
            session,
            output_idx,
        })
    }

    /// Повертає слово з U+0301 (або без змін, якщо модель не впевнена).
    pub fn stress_word(&self, word: &str) -> Result<String> {
        if word.is_empty() {
            return Ok(String::new());
        }

        let ids = encode_word(word);
        let input = CowArray::from(Array::from_shape_vec(IxDyn(&[1, MAX_LEN]), ids.to_vec())?);
        let value = Value::from_array(self.session.allocator(), &input)?;
        let outputs = self
            .session
            .run(vec![value])
            .context("ONNX session run (accentor)")?;

        let tensor = outputs[self.output_idx]
            .try_extract::<f32>()
            .context("extract accentor logits")?;
        let view = tensor.view();
        let shape = view.shape();
        let classes = match shape {
            [1, classes] | [classes] => *classes,
            other => bail!("unexpected accentor logits shape: {other:?}"),
        };
        if classes < MAX_LEN {
            bail!("accentor logits have {classes} classes, expected at least {MAX_LEN}");
        }
        let slice = view
            .as_slice()
            .context("accentor logits are not contiguous")?;

        // Перший максимум (строге >), як у C++.
        let mut best = 0usize;
        let mut best_value = slice[0];
        for (index, &value) in slice.iter().enumerate().take(MAX_LEN).skip(1) {
            if value > best_value {
                best_value = value;
                best = index;
            }
        }

        // Python: word[:index+1] + U+0301 + word[index+1:] (shift=1).
        let cp_len = word.chars().count();
        let pos = (best + 1).min(cp_len);
        Ok(utf8::insert_at_cp(word, pos, ACUTE))
    }
}

impl NeuralStress for Accentor {
    fn stress(&self, word: &str) -> Result<String> {
        self.stress_word(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_characters_like_cpp() {
        assert_eq!(char_id(accentor_lower('а' as u32)), 3);
        assert_eq!(char_id(accentor_lower('я' as u32)), 31);
        assert_eq!(char_id(accentor_lower('ґ' as u32)), 35);
        assert_eq!(char_id(accentor_lower('Ї' as u32)), 34);
        assert_eq!(char_id(accentor_lower('\'' as u32)), 2);
        assert_eq!(char_id(0x02BC), 2);
        assert_eq!(char_id(0x2019), 2);
        // Латиниця та невідомі символи -> <UNK>=1.
        assert_eq!(char_id(accentor_lower('A' as u32)), 1);
        assert_eq!(char_id(accentor_lower('é' as u32)), 1);
    }

    #[test]
    fn lowercases_like_cpp() {
        assert_eq!(accentor_lower('A' as u32), 'a' as u32);
        assert_eq!(accentor_lower('А' as u32), 'а' as u32);
        assert_eq!(accentor_lower('Є' as u32), 'є' as u32);
        assert_eq!(accentor_lower('І' as u32), 'і' as u32);
        assert_eq!(accentor_lower('Ґ' as u32), 'ґ' as u32);
        // C++ не чіпає Latin-1 uppercase (на відміну від utf8::cp_lower).
        assert_eq!(accentor_lower(0xC9), 0xC9);
    }

    #[test]
    fn encodes_and_pads_words() {
        let ids = encode_word("Ґанок");
        assert_eq!(&ids[..5], &[35, 3, 16, 17, 13]);
        assert!(ids[5..].iter().all(|&id| id == 0));

        let long = "а".repeat(MAX_LEN + 5);
        let ids = encode_word(&long);
        assert_eq!(ids.len(), MAX_LEN);
        assert!(ids.iter().all(|&id| id == 3));
    }

    #[test]
    fn accentor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Accentor>();
    }
}
