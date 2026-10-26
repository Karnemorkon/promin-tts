//! Розбиття тексту на речення та безпечні для синтезу чанки.
//!
//! Правила враховують українську пунктуацію, скорочення (`т.д.`, `грн.`,
//! `ім.`), ініціали (`Т. Шевченко`), десяткові дроби (`3.14`) та лапки після
//! розділового знака (`«...».`). Довгі речення додатково ріжуться за комами,
//! тире або межами слів, щоб жоден чанк не перевищував ліміт.

/// Розділові знаки, що завершують речення.
fn is_terminal(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '…')
}

/// Закривальні лапки/дужки, які можуть стояти після розділового знака.
fn is_closing(ch: char) -> bool {
    matches!(
        ch,
        '»' | '”' | '’' | '\'' | '"' | '“' | '‘' | ')' | ']' | '}'
    )
}

/// Скорочення, після яких крапка не завершує речення.
const ABBREVIATIONS: &[&str] = &[
    "ім",
    "т.д",
    "т.п",
    "т.зв",
    "і.т.д",
    "н.е",
    "н. е",
    "напр",
    "див",
    "пор",
    "рис",
    "табл",
    "буд",
    "вул",
    "просп",
    "кв",
    "тел",
    "моб",
    "англ",
    "укр",
    "рос",
    "нім",
    "фр",
    "лат",
    "грец",
    "поч",
    "кін",
    "прим",
    "ред",
    "пер",
    "вид",
    "розд",
    "гл",
    "стол",
    "грн",
    "коп",
    "тис",
    "млн",
    "млрд",
    "р",
    "рр",
    "ст",
    "с",
    "м",
    "см",
    "мм",
    "км",
    "кг",
    "г",
    "мг",
    "л",
    "мл",
    "хв",
    "сек",
    "год",
    "о",
    "ч",
    "п",
    "пп",
    "доц",
    "проф",
    "акад",
    "канд",
    "наук",
    "техн",
    "екон",
    "політ",
];

/// Чи токен перед крапкою є скороченням або ініціалом.
fn is_abbreviation_or_initial(token: &str) -> bool {
    let token = token.trim_end_matches('.');
    if token.is_empty() {
        return false;
    }
    if ABBREVIATIONS.contains(&token.to_lowercase().as_str()) {
        return true;
    }
    // Ініціал: одна велика літера (Т., І., М.).
    let mut chars = token.chars();
    match (chars.next(), chars.next()) {
        (Some(first), None) => first.is_uppercase(),
        _ => false,
    }
}

/// Слово/токен безпосередньо перед позицією `end` (не включаючи її).
fn token_before(chars: &[char], end: usize) -> String {
    let mut start = end;
    while start > 0 {
        let ch = chars[start - 1];
        if ch.is_alphanumeric() || ch == '.' || ch == '\'' || ch == '’' || ch == '-' {
            start -= 1;
        } else {
            break;
        }
    }
    chars[start..end].iter().collect()
}

fn push_trimmed(out: &mut Vec<String>, text: &str) {
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
}

/// Розбити текст на речення, зберігаючи пунктуацію та лапки.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;

    while i < chars.len() {
        if !is_terminal(chars[i]) {
            i += 1;
            continue;
        }

        // Знаходимо кінець серії розділових знаків і закривальних лапок.
        let mut end = i + 1;
        let mut terminal_count = 1;
        while end < chars.len() {
            if is_terminal(chars[end]) {
                terminal_count += 1;
                end += 1;
            } else if is_closing(chars[end]) {
                end += 1;
            } else {
                break;
            }
        }

        let at_boundary = end == chars.len() || chars[end].is_whitespace();
        if at_boundary {
            let token = token_before(&chars, i);
            // `...`, `?!` тощо — завжди межа; інакше перевіряємо скорочення.
            let boundary = terminal_count > 1 || !is_abbreviation_or_initial(&token);
            if boundary {
                let sentence: String = chars[start..end].iter().collect();
                push_trimmed(&mut out, &sentence);
                start = end;
                while start < chars.len() && chars[start].is_whitespace() {
                    start += 1;
                }
                i = start;
                continue;
            }
        }

        i = end;
    }

    if start < chars.len() {
        let sentence: String = chars[start..].iter().collect();
        push_trimmed(&mut out, &sentence);
    }
    out
}

/// Розбити занадто довге речення за комами/тире/пробілами.
fn split_long_sentence(sentence: &str, max_chars: usize) -> Vec<String> {
    let chars: Vec<char> = sentence.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        while start < chars.len() && chars[start].is_whitespace() {
            start += 1;
        }
        if start >= chars.len() {
            break;
        }

        let remaining = chars.len() - start;
        if remaining <= max_chars {
            let piece: String = chars[start..].iter().collect();
            push_trimmed(&mut out, &piece);
            break;
        }

        let limit = start + max_chars;
        // Шукаємо останню межу (кому, крапку з комою, двокрапку, тире або
        // пробіл) у межах ліміту, віддаючи перевагу розділовим знакам.
        let mut best_delim = None;
        let mut best_space = None;
        for (idx, ch) in chars.iter().enumerate().take(limit).skip(start) {
            if matches!(*ch, ',' | ';' | ':' | '—' | '–') {
                best_delim = Some(idx + 1);
            } else if ch.is_whitespace() {
                best_space = Some(idx);
            }
        }

        let cut = best_delim
            .or(best_space)
            .filter(|&idx| idx > start)
            .unwrap_or(limit);
        let piece: String = chars[start..cut].iter().collect();
        push_trimmed(&mut out, &piece);
        start = cut;
    }

    out
}

/// Згрупувати речення у чанки не довші за `max_chars`.
///
/// Порожній `max_chars` або `0` означає «не різати» (повертається весь текст
/// одним чанком). Довгі речення ріжуться за комами/тире/пробілами.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    if max_chars == 0 {
        return vec![text.to_string()];
    }

    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        if sentence.chars().count() > max_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            chunks.extend(split_long_sentence(&sentence, max_chars));
            continue;
        }

        let extra = if current.is_empty() { 0 } else { 1 };
        if current.chars().count() + extra + sentence.chars().count() > max_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&sentence);
    }

    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_basic_sentences() {
        assert_eq!(
            split_sentences("Привіт, світе! Як справи? Добре."),
            vec!["Привіт, світе!", "Як справи?", "Добре."]
        );
    }

    #[test]
    fn keeps_abbreviations() {
        assert_eq!(
            split_sentences("Ціна 250 грн. Зустріч о 14:30. Т.д. і т.п."),
            vec!["Ціна 250 грн. Зустріч о 14:30.", "Т.д. і т.п."]
        );
    }

    #[test]
    fn keeps_inline_abbreviation() {
        assert_eq!(
            split_sentences("Це т.д. приклад. Далі."),
            vec!["Це т.д. приклад.", "Далі."]
        );
    }

    #[test]
    fn keeps_initials() {
        assert_eq!(
            split_sentences("Т. Шевченко народився в 1814 р. Це факт."),
            vec!["Т. Шевченко народився в 1814 р. Це факт."]
        );
    }

    #[test]
    fn keeps_decimals() {
        assert_eq!(
            split_sentences("Число 3.14 — це пі. Далі йде текст."),
            vec!["Число 3.14 — це пі.", "Далі йде текст."]
        );
    }

    #[test]
    fn handles_ellipsis_and_quotes() {
        assert_eq!(
            split_sentences("Він сказав: «Так...». Потім пішов."),
            vec!["Він сказав: «Так...».", "Потім пішов."]
        );
    }

    #[test]
    fn handles_exclamation_question() {
        assert_eq!(
            split_sentences("Невже?! Це правда."),
            vec!["Невже?!", "Це правда."]
        );
    }

    #[test]
    fn keeps_abbreviation_before_closing_paren() {
        assert_eq!(
            split_sentences("Це (т.д.) приклад. Далі."),
            vec!["Це (т.д.) приклад.", "Далі."]
        );
    }

    #[test]
    fn chunks_short_text_into_one() {
        assert_eq!(chunk_text("Привіт. Світ.", 100), vec!["Привіт. Світ."]);
    }

    #[test]
    fn chunks_respect_limit() {
        let text = "Перше речення. Друге речення. Третє речення. Четверте речення.";
        let chunks = chunk_text(text, 20);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 20, "chunk too long: {chunk:?}");
        }
        assert_eq!(chunks.join(" "), text);
    }

    #[test]
    fn splits_long_sentence() {
        let text =
            "Це дуже довге речення, яке містить багато слів, і його треба розбити на частини.";
        let chunks = chunk_text(text, 25);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 25, "chunk too long: {chunk:?}");
        }
    }

    #[test]
    fn empty_text_gives_no_chunks() {
        assert!(chunk_text("   ", 100).is_empty());
        assert!(chunk_text("", 100).is_empty());
    }

    #[test]
    fn zero_limit_keeps_text() {
        assert_eq!(chunk_text("А. Б.", 0), vec!["А. Б."]);
    }
}
