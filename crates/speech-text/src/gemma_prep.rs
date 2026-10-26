//! Підготовка тексту для Gemma-вербалізатора (порт `gemma_prep` з C++).
//!
//! - `match_bare_digits` / `spell_digits` — поцифрове проговорювання довгих
//!   чисел і номерів без участі моделі;
//! - `add_spacing` — групування довгих цифрових послідовностей по 3, щоб
//!   SentencePiece не різав їх непередбачувано (з захистом карток/IBAN).

use fancy_regex::Regex;

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("invalid gemma_prep regex")
}

/// Чи містить текст ASCII-цифри.
pub fn has_ascii_digits(text: &str) -> bool {
    text.bytes().any(|b| b.is_ascii_digit())
}

fn is_ascii_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0x0B)
}

/// Trim ASCII whitespace + NBSP, як у C++ `trim_left`.
fn trim_left(s: &[u8]) -> usize {
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if is_ascii_ws(c) {
            i += 1;
            continue;
        }
        if c == 0xC2 && i + 1 < s.len() && s[i + 1] == 0xA0 {
            i += 2;
            continue;
        }
        break;
    }
    i
}

fn trim_right(s: &[u8], begin: usize) -> usize {
    let mut end = s.len();
    while end > begin {
        let c = s[end - 1];
        if is_ascii_ws(c) {
            end -= 1;
            continue;
        }
        if end >= begin + 2 && c == 0xA0 && s[end - 2] == 0xC2 {
            end -= 2;
            continue;
        }
        break;
    }
    end
}

/// Розбиття цифр на групи по 3 з кінця: `123456789` -> `123 456 789`.
pub fn group_digits(digits: &str) -> String {
    let bytes = digits.as_bytes();
    let mut parts: Vec<&str> = Vec::new();
    let mut end = bytes.len();
    while end > 0 {
        let begin = end.saturating_sub(3);
        parts.push(&digits[begin..end]);
        end = begin;
    }
    parts.reverse();
    parts.join(" ")
}

/// Якщо весь текст — цифри (7+ цифр або провідний нуль) — повертає їх.
pub fn match_bare_digits(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let b = trim_left(bytes);
    let e = trim_right(bytes, b);
    if b >= e {
        return None;
    }
    let mut digits = String::with_capacity(e - b);
    for &c in &bytes[b..e] {
        if !c.is_ascii_digit() {
            return None;
        }
        digits.push(char::from(c));
    }
    if digits.len() >= 7 || (digits.len() >= 2 && digits.starts_with('0')) {
        Some(digits)
    } else {
        None
    }
}

const DIGIT_NAMES: [&str; 10] = [
    "нуль",
    "один",
    "два",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
];

/// `"007"` -> `"нуль нуль сім"`.
pub fn spell_digits(digits: &str) -> String {
    let mut out = String::new();
    for c in digits.chars() {
        let Some(index) = c.to_digit(10) else {
            continue;
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(DIGIT_NAMES[index as usize]);
    }
    out
}

const OTHER_RE: &str = r"(?i)картк|карта|iban|айбан|телефон|подзвон|дзвон|код\b|на рахунок|cvv|імей|imei|паспорт|ідентифікатор";
const RUN_RE: &str = r"(?<![\w:/.\-№])(\d{10,13})(?![\w:/.\-])";
const GROUPED_RE: &str = r"(?<![\w:/.\-№])(\d{1,4}(?:[ \x{a0}]\d{1,4})+)(?![\w:/.\-])";

/// Групує довгі цифрові послідовності (порт `add_spacing`).
pub fn add_spacing(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if re(OTHER_RE).is_match(text).unwrap_or(false) {
        return text.to_string();
    }

    let s = re(RUN_RE)
        .replace_all(text, |caps: &fancy_regex::Captures<'_, str>| {
            let digits = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if digits.is_empty() || digits.starts_with('0') || digits.starts_with("380") {
                return digits.to_string();
            }
            group_digits(digits)
        })
        .into_owned();

    re(GROUPED_RE)
        .replace_all(&s, |caps: &fancy_regex::Captures<'_, str>| {
            let raw = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if raw.is_empty() {
                return String::new();
            }

            let bytes = raw.as_bytes();
            let mut parts: Vec<&str> = Vec::new();
            let mut start = 0usize;
            let mut i = 0usize;
            while i < bytes.len() {
                if bytes[i] == b' ' {
                    parts.push(&raw[start..i]);
                    i += 1;
                    start = i;
                } else if bytes[i] == 0xC2 && i + 1 < bytes.len() && bytes[i + 1] == 0xA0 {
                    parts.push(&raw[start..i]);
                    i += 2;
                    start = i;
                } else {
                    i += 1;
                }
            }
            parts.push(&raw[start..]);

            let digits: String = parts.concat();
            if digits.is_empty() {
                return raw.to_string();
            }
            let all4 = parts.len() >= 3 && parts.iter().all(|p| p.len() == 4);
            if all4 {
                return raw.to_string();
            }
            if digits.starts_with('0') || digits.starts_with("380") {
                return raw.to_string();
            }
            let mut grouped_ok = parts[0].len() <= 3;
            for part in &parts[1..] {
                if part.len() != 3 {
                    grouped_ok = false;
                }
            }
            if grouped_ok {
                return raw.to_string();
            }
            if digits.len() < 5 || digits.len() > 15 {
                return raw.to_string();
            }
            group_digits(&digits)
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_digit_routing() {
        assert_eq!(match_bare_digits("12345678").as_deref(), Some("12345678"));
        assert_eq!(match_bare_digits("000000").as_deref(), Some("000000"));
        assert_eq!(match_bare_digits("007").as_deref(), Some("007"));
        assert_eq!(match_bare_digits("0000").as_deref(), Some("0000"));
        assert_eq!(
            match_bare_digits("123456789012").as_deref(),
            Some("123456789012")
        );
        assert_eq!(
            match_bare_digits("  918273645  ").as_deref(),
            Some("918273645")
        );
        assert_eq!(match_bare_digits("42"), None);
        assert_eq!(match_bare_digits("2026"), None);
        assert_eq!(match_bare_digits("10 000 001"), None);
        assert_eq!(match_bare_digits("У звіті 12345678."), None);
        assert_eq!(match_bare_digits(""), None);
    }

    #[test]
    fn spells_digits() {
        assert_eq!(
            spell_digits("12345678"),
            "один два три чотири п'ять шість сім вісім"
        );
        assert_eq!(spell_digits("007"), "нуль нуль сім");
        assert_eq!(
            spell_digits("10000001"),
            "один нуль нуль нуль нуль нуль нуль один"
        );
    }

    #[test]
    fn groups_digits() {
        assert_eq!(group_digits("123456789"), "123 456 789");
        assert_eq!(group_digits("1000"), "1 000");
        assert_eq!(group_digits("12"), "12");
    }

    #[test]
    fn spacing_matches_cpp_cases() {
        let cases = [
            (
                "Збитки оцінили в 7000000000 доларів.",
                "Збитки оцінили в 7 000 000 000 доларів.",
            ),
            (
                "Борг сягнув 2625414042726 гривень.",
                "Борг сягнув 2 625 414 042 726 гривень.",
            ),
            ("Оборот 9464922334 долари.", "Оборот 9 464 922 334 долари."),
            (
                "Картка 1234567890123456 заблокована.",
                "Картка 1234567890123456 заблокована.",
            ),
            (
                "Мій телефон 0671234567, дзвоніть.",
                "Мій телефон 0671234567, дзвоніть.",
            ),
            (
                "Наберіть 380671234567 звідусіль.",
                "Наберіть 380671234567 звідусіль.",
            ),
            (
                "Ідентифікатор 1234567890 у базі.",
                "Ідентифікатор 1234567890 у базі.",
            ),
            (
                "Рейс о 12:30 коштує 1234567890 гривень.",
                "Рейс о 12:30 коштує 1 234 567 890 гривень.",
            ),
            (
                "Артикул AB1234567890 на складі.",
                "Артикул AB1234567890 на складі.",
            ),
            (
                "Версія 1.2345678901 не існує.",
                "Версія 1.2345678901 не існує.",
            ),
            ("У нас 123456789 позицій.", "У нас 123456789 позицій."),
            (
                "IBAN UA123456789012345678901234567890.",
                "IBAN UA123456789012345678901234567890.",
            ),
            (
                "Рахунок на 1111 386,40 грн треба сплатити до 25.07.2026.",
                "Рахунок на 1 111 386,40 грн треба сплатити до 25.07.2026.",
            ),
            ("Борг 12 3456 789 гривень.", "Борг 123 456 789 гривень."),
            ("Оборот 1 111 386 гривень.", "Оборот 1 111 386 гривень."),
            ("Сума 10 000 001 грн.", "Сума 10 000 001 грн."),
            (
                "Оплата 4149 6293 1234 5678 пройшла.",
                "Оплата 4149 6293 1234 5678 пройшла.",
            ),
            (
                "Дзвони 067 123 45 67 зранку.",
                "Дзвони 067 123 45 67 зранку.",
            ),
            ("Рахунок 25.07.2026 і 12:30.", "Рахунок 25.07.2026 і 12:30."),
            ("", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(add_spacing(input), expected, "input: {input:?}");
        }
    }
}
