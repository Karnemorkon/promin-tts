//! Rule-based вербалізація цифр, дат і часу (порт `verbalizer.cpp`).
//!
//! Покриває кардинальні числа до 999 999 999 999 (12 цифр), дати
//! `ДД.ММ.РРРР` / `ДД/ММ/РРРР`, час `ГГ:ХХ` / `ГГ-ХХ`. Не потребує
//! нейромоделей і використовується як fallback для Gemma.

/// Чи містить текст ASCII-цифри (гейт для будь-якої вербалізації).
pub fn needs_verbalize(text: &str) -> bool {
    text.bytes().any(|b| b.is_ascii_digit())
}

const UNITS: [&str; 20] = [
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
    "десять",
    "одинадцять",
    "дванадцять",
    "тринадцять",
    "чотирнадцять",
    "п'ятнадцять",
    "шістнадцять",
    "сімнадцять",
    "вісімнадцять",
    "дев'ятнадцять",
];
const UNITS_FEM: [&str; 10] = [
    "нуль",
    "одна",
    "дві",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
];
const TENS: [&str; 10] = [
    "",
    "",
    "двадцять",
    "тридцять",
    "сорок",
    "п'ятдесят",
    "шістьдесят",
    "сімдесят",
    "вісімдесят",
    "дев'яносто",
];
const HUNDREDS: [&str; 10] = [
    "",
    "сто",
    "двісті",
    "триста",
    "чотириста",
    "п'ятсот",
    "шістьсот",
    "сімсот",
    "вісімсот",
    "дев'ятсот",
];

fn say_upto999(n: i32, fem: bool) -> String {
    if n <= 0 {
        return String::new();
    }
    let mut out = String::new();
    let h = (n / 100) as usize;
    let r = n % 100;
    if h != 0 {
        out.push_str(HUNDREDS[h]);
    }
    if r >= 20 {
        let t = (r / 10) as usize;
        let u = (r % 10) as usize;
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(TENS[t]);
        if u != 0 {
            out.push(' ');
            out.push_str(if fem { UNITS_FEM[u] } else { UNITS[u] });
        }
    } else if r >= 10 {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(UNITS[r as usize]);
    } else if r > 0 {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(if fem {
            UNITS_FEM[r as usize]
        } else {
            UNITS[r as usize]
        });
    }
    out
}

fn plural_form<'a>(n: i32, one: &'a str, few: &'a str, many: &'a str) -> &'a str {
    let last2 = n % 100;
    let last1 = n % 10;
    if last1 == 1 && last2 != 11 {
        one
    } else if (2..=4).contains(&last1) && !(10..=20).contains(&last2) {
        few
    } else {
        many
    }
}

/// Ціле число словами (як `say_int` у C++).
pub fn say_int(n: i64) -> String {
    if n == 0 {
        return "нуль".to_string();
    }
    if n < 0 {
        return format!("мінус {}", say_int(-n));
    }

    let mut rest = n;
    let billions = (rest / 1_000_000_000) as i32;
    rest %= 1_000_000_000;
    let millions = (rest / 1_000_000) as i32;
    rest %= 1_000_000;
    let thousands = (rest / 1_000) as i32;
    let units = (rest % 1_000) as i32;

    let mut out = String::new();
    let mut add = |s: String| {
        if s.is_empty() {
            return;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&s);
    };

    if billions != 0 {
        add(say_upto999(billions, false));
        add(plural_form(billions, "мільярд", "мільярди", "мільярдів").to_string());
    }
    if millions != 0 {
        add(say_upto999(millions, false));
        add(plural_form(millions, "мільйон", "мільйони", "мільйонів").to_string());
    }
    if thousands != 0 {
        add(say_upto999(thousands, true));
        add(plural_form(thousands, "тисяча", "тисячі", "тисяч").to_string());
    }
    if units != 0 {
        add(say_upto999(units, false));
    }
    out
}

fn atoi_n(bytes: &[u8], pos: usize, len: usize) -> i32 {
    let mut v = 0i32;
    for k in 0..len {
        let Some(&c) = bytes.get(pos + k) else {
            return -1;
        };
        if !c.is_ascii_digit() {
            return -1;
        }
        v = v * 10 + i32::from(c - b'0');
    }
    v
}

/// Вербалізує числа/дати/час. Якщо цифр немає — повертає текст без змін.
pub fn verbalize(text: &str) -> String {
    if !needs_verbalize(text) {
        return text.to_string();
    }

    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out: Vec<u8> = Vec::with_capacity(n * 2);
    let mut i = 0usize;

    while i < n {
        // Час: ГГ:ХХ (2+2), Г:ХХ (1+2), ГГ-ХХ, Г-ХХ.
        {
            let mut hlen = 0usize;
            while i + hlen < n && bytes[i + hlen].is_ascii_digit() && hlen < 2 {
                hlen += 1;
            }
            if hlen >= 1 && i + hlen < n && matches!(bytes[i + hlen], b':' | b'-') {
                let mpos = i + hlen + 1;
                let mut mlen = 0usize;
                while mpos + mlen < n && bytes[mpos + mlen].is_ascii_digit() && mlen < 2 {
                    mlen += 1;
                }
                if mlen == 2 {
                    let hh = atoi_n(bytes, i, hlen);
                    let mm = atoi_n(bytes, mpos, mlen);
                    if (0..=23).contains(&hh) && (0..=59).contains(&mm) {
                        if mm == 0 {
                            let h12 = if hh == 0 {
                                12
                            } else if hh > 12 {
                                hh - 12
                            } else {
                                hh
                            };
                            out.extend_from_slice(say_upto999(h12, true).as_bytes());
                            out.extend_from_slice(" годині".as_bytes());
                        } else {
                            out.extend_from_slice(say_int(i64::from(hh)).as_bytes());
                            out.extend_from_slice(" годині ".as_bytes());
                            out.extend_from_slice(say_upto999(mm, true).as_bytes());
                            out.extend_from_slice(" хвилин".as_bytes());
                        }
                        i = mpos + mlen;
                        continue;
                    }
                }
            }
        }

        // Дата ДД.ММ.РРРР або ДД/ММ/РРРР.
        if i + 10 <= n
            && bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && matches!(bytes[i + 2], b'.' | b'/')
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4].is_ascii_digit()
            && matches!(bytes[i + 5], b'.' | b'/')
            && bytes[i + 6].is_ascii_digit()
            && bytes[i + 7].is_ascii_digit()
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
        {
            let dd = atoi_n(bytes, i, 2);
            let mm = atoi_n(bytes, i + 3, 2);
            let yyyy = atoi_n(bytes, i + 6, 4);
            if (1..=31).contains(&dd) && (1..=12).contains(&mm) && yyyy >= 1000 {
                out.extend_from_slice(say_upto999(dd, true).as_bytes());
                out.extend_from_slice(" числа ".as_bytes());
                out.extend_from_slice(say_upto999(mm, true).as_bytes());
                out.extend_from_slice(" місяця ".as_bytes());
                out.extend_from_slice(say_int(i64::from(yyyy)).as_bytes());
                out.extend_from_slice(" року".as_bytes());
                i += 10;
                continue;
            }
        }

        // Кардинальне число (обрізка до 12 цифр, як у C++).
        if bytes[i].is_ascii_digit() {
            let mut j = i;
            while j < n && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let num = &bytes[i..j];
            let truncated = &num[..num.len().min(12)];
            match std::str::from_utf8(truncated)
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
            {
                Some(value) => out.extend_from_slice(say_int(value).as_bytes()),
                None => out.extend_from_slice(num),
            }
            i = j;
            continue;
        }

        out.push(bytes[i]);
        i += 1;
    }

    String::from_utf8(out).expect("verbalizer preserves valid UTF-8")
}

fn is_ascii_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0x0B)
}

/// Trim ASCII whitespace + NBSP (як Python `str.strip` / PCRE `\s`).
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

/// Trim ASCII whitespace + NBSP (як Python `str.strip` / PCRE `\s`).
pub fn strip_ws(s: &str) -> String {
    let bytes = s.as_bytes();
    let b = trim_left(bytes);
    let e = trim_right(bytes, b);
    s[b..e].to_string()
}

fn lower_utf8(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if c < 0x80 {
            out.push(c.to_ascii_lowercase());
            i += 1;
            continue;
        }
        if (c == 0xD0 || c == 0xD1) && i + 1 < s.len() {
            let mut cp = (u32::from(c & 0x1F) << 6) | u32::from(s[i + 1] & 0x3F);
            if (0x0410..=0x042F).contains(&cp) {
                cp += 0x20;
            } else if (0x0400..=0x040F).contains(&cp) {
                cp += 0x50;
            } else if cp == 0x0490 {
                cp = 0x0491;
            }
            out.push(0xC0 | ((cp >> 6) as u8));
            out.push(0x80 | ((cp & 0x3F) as u8));
            i += 2;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn cp_count(s: &[u8]) -> usize {
    s.iter().filter(|&&c| (c & 0xC0) != 0x80).count()
}

fn is_abbrev_before(s: &[u8], dot: usize) -> bool {
    let mut j = dot;
    while j > 0 {
        let c = s[j - 1];
        if c.is_ascii_alphabetic() || c >= 0x80 {
            j -= 1;
        } else {
            break;
        }
    }
    if j == dot {
        return false;
    }
    let w = lower_utf8(&s[j..dot]);
    if cp_count(&w) == 1 {
        return true;
    }
    const ABBREV: [&str; 34] = [
        "рт",
        "ст",
        "ім",
        "вул",
        "просп",
        "буд",
        "кв",
        "обл",
        "тис",
        "млн",
        "млрд",
        "грн",
        "коп",
        "см",
        "мм",
        "кг",
        "км",
        "стор",
        "рис",
        "табл",
        "гл",
        "англ",
        "укр",
        "доц",
        "проф",
        "акад",
        "див",
        "ін",
        "др",
        "канд",
        "ім",
        "тоб",
        "напр",
        "прибл",
    ];
    ABBREV.iter().any(|a| w == a.as_bytes())
}

/// Розбиття на речення — дзеркало `split_sentences` з C++ (з абревіатурами).
pub fn split_sentences(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut punct_end = 0usize;
        match bytes[i] {
            b'.' => {
                if is_abbrev_before(bytes, i) {
                    i += 1;
                    continue;
                }
                punct_end = i + 1;
            }
            b'!' | b'?' => punct_end = i + 1,
            0xE2 if i + 2 < n && bytes[i + 1] == 0x80 && bytes[i + 2] == 0xA6 => {
                punct_end = i + 3;
            }
            _ => {}
        }

        if punct_end > 0 && punct_end < n && is_ascii_ws(bytes[punct_end]) {
            let part = strip_ws(&text[start..punct_end]);
            if !part.is_empty() {
                out.push(part);
            }
            while punct_end < n && is_ascii_ws(bytes[punct_end]) {
                punct_end += 1;
            }
            start = punct_end;
            i = punct_end;
        } else {
            i += 1;
        }
    }

    if start < n {
        let part = strip_ws(&text[start..]);
        if !part.is_empty() {
            out.push(part);
        }
    }
    if out.is_empty() && !text.is_empty() {
        out.push(text.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_detects_only_ascii_digits() {
        assert!(needs_verbalize("2026"));
        assert!(needs_verbalize("ціна 5"));
        assert!(!needs_verbalize("без цифр"));
        assert!(!needs_verbalize("½ ²"));
    }

    #[test]
    fn says_cardinals() {
        assert_eq!(say_int(0), "нуль");
        assert_eq!(say_int(1), "один");
        assert_eq!(say_int(2), "два");
        assert_eq!(say_int(11), "одинадцять");
        assert_eq!(say_int(21), "двадцять один");
        assert_eq!(say_int(100), "сто");
        assert_eq!(say_int(101), "сто один");
        assert_eq!(say_int(1000), "одна тисяча");
        assert_eq!(say_int(2000), "дві тисячі");
        assert_eq!(say_int(5000), "п'ять тисяч");
        assert_eq!(say_int(1_000_000), "один мільйон");
        assert_eq!(say_int(2_000_000), "два мільйони");
        assert_eq!(say_int(5_000_000), "п'ять мільйонів");
        assert_eq!(say_int(1_000_000_000), "один мільярд");
        assert_eq!(say_int(-5), "мінус п'ять");
    }

    #[test]
    fn verbalizes_numbers_dates_and_time() {
        assert_eq!(verbalize("2026"), "дві тисячі двадцять шість");
        assert_eq!(verbalize("14:30"), "чотирнадцять годині тридцять хвилин");
        assert_eq!(verbalize("14:00"), "дві годині");
        assert_eq!(verbalize("00:00"), "дванадцять годині");
        assert_eq!(
            verbalize("25.07.2026"),
            "двадцять п'ять числа сім місяця дві тисячі двадцять шість року"
        );
        assert_eq!(verbalize("50%"), "п'ятдесят%");
        assert_eq!(verbalize("без цифр"), "без цифр");
    }

    #[test]
    fn truncates_to_twelve_digits() {
        assert_eq!(verbalize("1000000000000"), "сто мільярдів");
    }

    #[test]
    fn splits_sentences_and_keeps_abbreviations() {
        assert_eq!(
            split_sentences("Перше. Друге! Третє?"),
            vec!["Перше.", "Друге!", "Третє?"]
        );
        assert_eq!(
            split_sentences("Тиск 750 мм рт. ст. і все."),
            vec!["Тиск 750 мм рт. ст. і все."]
        );
        assert_eq!(
            split_sentences("Це … далі. Кінець."),
            vec!["Це …", "далі.", "Кінець."]
        );
        assert!(split_sentences("").is_empty());
    }
}
