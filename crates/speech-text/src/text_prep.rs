//! Нормалізація тексту перед фонемізацією (порт text_prep.cpp / Python reference).

use crate::repl::to_fancy_template;
use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("invalid regex")
}

fn re_sub(pattern: &str, replacement: &str, text: &str) -> String {
    let template = to_fancy_template(replacement);
    re(pattern)
        .replace_all(text, template.as_str())
        .into_owned()
}

fn nfkc(s: &str) -> String {
    s.nfkc().collect()
}

fn last_cp(s: &str) -> Option<char> {
    s.chars().next_back()
}

/// Об'єднання переносів рядків: `слово\n` -> `слово. `, решта `\n` -> пробіл.
pub fn collapse_newlines(text: &str) -> String {
    let mut s = re_sub(r"(\w+[^.,!:\-])\n", "$1. ", text);
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == '\n' {
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    std::mem::swap(&mut s, &mut out);
    s
}

/// Розбиття на частини по `.!? :` + пробіл; `group` згруповує короткі (<=100 cp).
pub fn split_to_parts(text: &str, group: bool) -> Vec<String> {
    let t = collapse_newlines(text);
    let cps: Vec<char> = t.chars().collect();
    let mut parts = Vec::new();
    let mut tail = String::new();
    let mut tail_cp = 0usize;
    let last = cps.len().saturating_sub(1);
    for (pos, &ch) in cps.iter().enumerate() {
        tail.push(ch);
        tail_cp += 1;
        let is_break = matches!(ch, '.' | '!' | '?' | ':');
        if is_break && pos < last && cps[pos + 1] == ' ' {
            if group && tail_cp <= 100 {
                continue;
            }
            parts.push(std::mem::take(&mut tail));
            tail_cp = 0;
        }
    }
    parts.push(tail);
    parts
}

fn strip_wrappers(text: &str) -> String {
    let s = re_sub(r"^\s+", "", text);
    let s = re_sub(r"\s+$", "", &s);
    re_sub("\"", "", &s)
}

/// Розмітка наголосу `+` -> U+0301, потім NFKC.
pub fn apply_stress_markup(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '+' {
            s.push('\u{0301}');
        } else {
            s.push(ch);
        }
    }
    nfkc(&s)
}

fn standardize_dashes(text: &str) -> String {
    re_sub(
        "[\u{1806}\u{2010}\u{2011}\u{2012}\u{2013}\u{2014}\u{2015}\u{207B}\u{208B}\u{2212}\u{2E3A}\u{2E3B}]",
        "-",
        text,
    )
}

fn ensure_terminal_punct(text: String) -> String {
    let mut s = text;
    if let Some(last) = last_cp(&s) {
        if !matches!(last, '.' | '!' | '?' | ':' | '-') {
            s.push('.');
        }
    }
    s
}

/// Повна нормалізація однієї частини тексту.
pub fn normalize_part(part: &str) -> String {
    let s = strip_wrappers(part);
    if s.is_empty() {
        return String::new();
    }
    let s = apply_stress_markup(&s);
    let s = standardize_dashes(&s);
    let s = ensure_terminal_punct(s);
    re_sub(" - ", ": ", &s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_basic() {
        assert_eq!(collapse_newlines("привіт\nсвіте"), "привіт. світе");
        assert_eq!(collapse_newlines("a\nb"), "a b");
    }

    #[test]
    fn split_grouped() {
        let parts = split_to_parts("Один. Два. Три", true);
        assert_eq!(parts.len(), 1);
        let parts = split_to_parts("Один. Два.", false);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn terminal_punct() {
        assert_eq!(normalize_part("привіт"), "привіт.");
        assert_eq!(normalize_part("\"привіт\"!"), "привіт!");
    }

    #[test]
    fn stress_markup_plus() {
        assert_eq!(apply_stress_markup("приві+т"), "приві\u{0301}т");
    }
}
