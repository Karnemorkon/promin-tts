//! Порівняння розпізнаного тексту зі скриптом: нормалізація, WER/CER,
//! построкова вирівнювальна діагностика (заміни, пропуски, вставки).

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// Тип помилки на рівні слова.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorKind {
    /// Слово скрипта замінено іншим.
    Substitution,
    /// Слово скрипта не розпізнано.
    Deletion,
    /// У розпізнаному тексті є зайве слово.
    Insertion,
}

/// Одна помилка вирівнювання.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffError {
    pub kind: ErrorKind,
    /// Індекс слова у нормалізованому скрипті (0-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_index: Option<usize>,
    /// Індекс слова у нормалізованому розпізнаному тексті (0-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis: Option<String>,
}

/// Результат порівняння одного фрагмента.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffReport {
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub correct: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    /// Word Error Rate: `(S + D + I) / reference_words`.
    pub wer: f32,
    /// Character Error Rate: Levenshtein по символах без пробілів.
    pub cer: f32,
    pub reference_chars: usize,
    pub hypothesis_chars: usize,
    pub char_errors: usize,
    pub errors: Vec<DiffError>,
}

impl DiffReport {
    /// Загальна кількість помилок на рівні слів.
    pub fn word_errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// Чи є хоч одна помилка.
    pub fn has_errors(&self) -> bool {
        self.word_errors() > 0
    }
}

/// Нормалізувати текст для порівняння.
///
/// NFC, нижній регістр, апострофи видаляються, розділові знаки та дефіси
/// стають пробілами, пробіли згортаються. Літери й цифри зберігаються.
pub fn normalize_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending_space = false;
    for ch in input.nfc() {
        for lower in ch.to_lowercase() {
            if lower.is_alphanumeric() {
                if pending_space && !out.is_empty() {
                    out.push(' ');
                }
                out.push(lower);
                pending_space = false;
            } else if matches!(lower, '\'' | '’' | 'ʼ' | '`' | '´') {
                // Апостроф не є межею слова: п'ять == пять.
            } else {
                pending_space = true;
            }
        }
    }
    out
}

/// Розбити нормалізований текст на слова.
pub fn tokenize(input: &str) -> Vec<String> {
    normalize_text(input)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// Порівняти скрипт із розпізнаним текстом.
pub fn compare(reference: &str, hypothesis: &str) -> DiffReport {
    let reference_normalized = normalize_text(reference);
    let hypothesis_normalized = normalize_text(hypothesis);
    let reference_words: Vec<&str> = reference_normalized.split_whitespace().collect();
    let hypothesis_words: Vec<&str> = hypothesis_normalized.split_whitespace().collect();
    let alignment = align_words(&reference_words, &hypothesis_words);

    let reference_chars: Vec<char> = reference_normalized
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    let hypothesis_chars: Vec<char> = hypothesis_normalized
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    let char_errors = levenshtein(&reference_chars, &hypothesis_chars);
    let word_errors = alignment.substitutions + alignment.deletions + alignment.insertions;

    DiffReport {
        reference_words: reference_words.len(),
        hypothesis_words: hypothesis_words.len(),
        correct: alignment.correct,
        substitutions: alignment.substitutions,
        deletions: alignment.deletions,
        insertions: alignment.insertions,
        wer: error_rate(word_errors, reference_words.len()),
        cer: error_rate(char_errors, reference_chars.len()),
        reference_chars: reference_chars.len(),
        hypothesis_chars: hypothesis_chars.len(),
        char_errors,
        errors: alignment.errors,
    }
}

struct Alignment {
    correct: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    errors: Vec<DiffError>,
}

fn align_words(reference: &[&str], hypothesis: &[&str]) -> Alignment {
    let n = reference.len();
    let m = hypothesis.len();
    let width = m + 1;
    let mut dp = vec![0usize; (n + 1) * width];
    for (i, row) in dp.chunks_mut(width).enumerate() {
        row[0] = i;
    }
    for (j, value) in dp.iter_mut().take(width).enumerate() {
        *value = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            let substitution = dp[(i - 1) * width + (j - 1)] + cost;
            let deletion = dp[(i - 1) * width + j] + 1;
            let insertion = dp[i * width + (j - 1)] + 1;
            dp[i * width + j] = substitution.min(deletion).min(insertion);
        }
    }

    let mut alignment = Alignment {
        correct: 0,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
        errors: Vec::new(),
    };
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            if dp[i * width + j] == dp[(i - 1) * width + (j - 1)] + cost {
                if cost == 0 {
                    alignment.correct += 1;
                } else {
                    alignment.substitutions += 1;
                    alignment.errors.push(DiffError {
                        kind: ErrorKind::Substitution,
                        reference_index: Some(i - 1),
                        hypothesis_index: Some(j - 1),
                        reference: Some(reference[i - 1].to_string()),
                        hypothesis: Some(hypothesis[j - 1].to_string()),
                    });
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i * width + j] == dp[(i - 1) * width + j] + 1 {
            alignment.deletions += 1;
            alignment.errors.push(DiffError {
                kind: ErrorKind::Deletion,
                reference_index: Some(i - 1),
                hypothesis_index: None,
                reference: Some(reference[i - 1].to_string()),
                hypothesis: None,
            });
            i -= 1;
            continue;
        }
        if j > 0 && dp[i * width + j] == dp[i * width + (j - 1)] + 1 {
            alignment.insertions += 1;
            alignment.errors.push(DiffError {
                kind: ErrorKind::Insertion,
                reference_index: None,
                hypothesis_index: Some(j - 1),
                reference: None,
                hypothesis: Some(hypothesis[j - 1].to_string()),
            });
            j -= 1;
            continue;
        }
        // Недосяжно для коректної DP-таблиці; захист від нескінченного циклу.
        break;
    }
    alignment.errors.reverse();
    alignment
}

fn error_rate(errors: usize, reference: usize) -> f32 {
    if reference == 0 {
        if errors == 0 {
            0.0
        } else {
            1.0
        }
    } else {
        errors as f32 / reference as f32
    }
}

fn levenshtein<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> usize {
    if reference.is_empty() {
        return hypothesis.len();
    }
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (i, reference_item) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, hypothesis_item) in hypothesis.iter().enumerate() {
            let cost = usize::from(reference_item != hypothesis_item);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_case_punctuation_and_spaces() {
        assert_eq!(normalize_text("  Привіт,   СВІТЕ! "), "привіт світе");
        assert_eq!(normalize_text("п'ять"), "пять");
        assert_eq!(normalize_text("будь-який"), "будь який");
        assert_eq!(normalize_text("«Ґанок» — 2026."), "ґанок 2026");
    }

    #[test]
    fn identical_text_has_zero_errors() {
        let report = compare("Привіт, світе!", "привіт світе");
        assert_eq!(report.reference_words, 2);
        assert_eq!(report.correct, 2);
        assert_eq!(report.wer, 0.0);
        assert_eq!(report.cer, 0.0);
        assert!(!report.has_errors());
    }

    #[test]
    fn substitution_is_counted() {
        let report = compare("кіт сидить", "кот сидить");
        assert_eq!(report.substitutions, 1);
        assert_eq!(report.deletions, 0);
        assert_eq!(report.insertions, 0);
        assert_eq!(report.correct, 1);
        assert!((report.wer - 0.5).abs() < 1e-6);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].kind, ErrorKind::Substitution);
        assert_eq!(report.errors[0].reference.as_deref(), Some("кіт"));
        assert_eq!(report.errors[0].hypothesis.as_deref(), Some("кот"));
    }

    #[test]
    fn deletion_is_counted() {
        let report = compare("один два три", "один три");
        assert_eq!(report.deletions, 1);
        assert_eq!(report.correct, 2);
        assert!((report.wer - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(report.errors[0].kind, ErrorKind::Deletion);
        assert_eq!(report.errors[0].reference_index, Some(1));
        assert_eq!(report.errors[0].hypothesis_index, None);
    }

    #[test]
    fn insertion_is_counted() {
        let report = compare("один три", "один два три");
        assert_eq!(report.insertions, 1);
        assert_eq!(report.correct, 2);
        assert!((report.wer - 0.5).abs() < 1e-6);
        assert_eq!(report.errors[0].kind, ErrorKind::Insertion);
        assert_eq!(report.errors[0].reference_index, None);
        assert_eq!(report.errors[0].hypothesis_index, Some(1));
    }

    #[test]
    fn empty_reference_and_hypothesis() {
        let report = compare("", "");
        assert_eq!(report.wer, 0.0);
        assert_eq!(report.cer, 0.0);
        assert!(report.errors.is_empty());

        let report = compare("", "зайве");
        assert_eq!(report.wer, 1.0);
        assert_eq!(report.cer, 1.0);
        assert_eq!(report.insertions, 1);
    }

    #[test]
    fn cer_uses_characters_without_spaces() {
        let report = compare("кіт", "кот");
        assert_eq!(report.reference_chars, 3);
        assert_eq!(report.char_errors, 1);
        assert!((report.cer - 1.0 / 3.0).abs() < 1e-6);
    }
}
