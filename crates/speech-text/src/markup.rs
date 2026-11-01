//! Інлайн-розмітка керування синтезом у тексті.
//!
//! Підтримувані директиви (всередині квадратних дужок):
//! - `[pause 500ms]` — пауза мовчання тривалістю N мс.
//! - `[rate:1.2]` — зміна швидкості мовлення для подальшого тексту.
//!
//! Приклад:
/// Один елемент розмітки або speakable-фрагмент тексту.
#[derive(Debug, Clone, PartialEq)]
pub enum MarkupUnit {
    /// Чистий текст для синтезу.
    Text(String),
    /// Пауза мовчання (мс).
    Pause(u64),
    /// Швидкість мовлення для наступних фрагментів.
    Rate(f32),
}

impl MarkupUnit {
    /// Тривалість паузи у зразках (фреймах) при SAMPLE_RATE = 24 кГц.
    pub fn pause_samples(&self, sample_rate: u32) -> Option<usize> {
        match self {
            Self::Pause(ms) => Some((*ms as f64 / 1000.0 * sample_rate as f64).round() as usize),
            _ => None,
        }
    }
}

/// Помилка парсингу розмітки.
#[derive(Debug, thiserror::Error)]
pub enum MarkupError {
    #[error("невідома директива: {0}")]
    Unknown(String),
    #[error("некоректне значення паузи: {0}")]
    PauseValue(String),
    #[error("некоректне значення швидкості: {0}")]
    RateValue(String),
    #[error("нерозпарсена розмітка: {0}")]
    Unmatched(String),
}

/// Розібрати текст на послідовність одиниць розмітки.
///
/// Текст поза директивами потрапляє в `MarkupUnit::Text`. Direktorivi видаляються
/// із вихідного тексту; між текстовими фрагментами доджується один пробіл, щоб
/// не 스лить слова.
pub fn parse_markup(text: &str) -> Result<Vec<MarkupUnit>, MarkupError> {
    let mut units: Vec<MarkupUnit> = Vec::new();
    let mut buf: String = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '[' {
            // Спробуємо знайти закриваючу дужку.
            if let Some(close) = chars[i + 1..].iter().position(|c| *c == ']') {
                let close = i + 1 + close;
                // Зліва accumulated текст стає текстовим блоком.
                let t = std::mem::take(&mut buf);
                if !t.trim().is_empty() {
                    units.push(MarkupUnit::Text(t.trim().to_string()));
                }
                let directive: String = chars[i + 1..close].iter().collect();
                let trimmed = directive.trim();

                if let Some(rest) = trimmed.strip_prefix("pause") {
                    let ms_str = rest.trim();
                    let ms: u64 = if let Some(s) = ms_str.strip_suffix("ms") {
                        s.trim()
                            .parse::<u64>()
                            .map_err(|_| MarkupError::PauseValue(ms_str.to_string()))?
                    } else if let Some(s) = ms_str.strip_suffix('s') {
                        s.trim()
                            .parse::<u64>()
                            .map_err(|_| MarkupError::PauseValue(ms_str.to_string()))?
                            * 1000
                    } else {
                        return Err(MarkupError::PauseValue(ms_str.to_string()));
                    };
                    units.push(MarkupUnit::Pause(ms));
                } else if let Some(rest) = trimmed.strip_prefix("rate:") {
                    let rate: f32 = rest
                        .trim()
                        .parse::<f32>()
                        .map_err(|_| MarkupError::RateValue(rest.trim().to_string()))?;
                    if rate <= 0.0 {
                        return Err(MarkupError::RateValue(rest.trim().to_string()));
                    }
                    units.push(MarkupUnit::Rate(rate));
                } else {
                    return Err(MarkupError::Unknown(directive));
                }

                i = close + 1;
                continue;
            } else {
                // Немає закриваючої дужки — [ — це буквальний символ.
                buf.push('[');
                i += 1;
                continue;
            }
        }

        buf.push(chars[i]);
        i += 1;
    }

    // Залишок тексту після останньої директиви.
    let t = std::mem::take(&mut buf);
    if !t.trim().is_empty() {
        units.push(MarkupUnit::Text(t.trim().to_string()));
    }

    if units.is_empty() {
        units.push(MarkupUnit::Text(text.to_string()));
    }

    Ok(units)
}

/// Чистий speakable-текст (без пауз і змін швидкості), з'єднаний пробілами.
pub fn speakable_text(units: &[MarkupUnit]) -> String {
    let parts: Vec<String> = units
        .iter()
        .filter_map(|u| match u {
            MarkupUnit::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    parts.join(" ")
}

/// Загальна тривалість усіх пауз у мілісекундах.
pub fn total_pause_ms(units: &[MarkupUnit]) -> u64 {
    units
        .iter()
        .filter_map(|u| match u {
            MarkupUnit::Pause(ms) => Some(*ms),
            _ => None,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_stays_text() {
        let units = parse_markup("Привіт, світе.").unwrap();
        assert_eq!(
            format!("{:?}", units),
            format!("{:?}", vec![MarkupUnit::Text("Привіт, світе.".to_string())])
        );
    }

    #[test]
    fn pause_directive() {
        let units = parse_markup("Текст [pause 500ms] далі.").unwrap();
        assert_eq!(
            format!("{:?}", units),
            format!(
                "{:?}",
                vec![
                    MarkupUnit::Text("Текст".to_string()),
                    MarkupUnit::Pause(500),
                    MarkupUnit::Text("далі.".to_string()),
                ]
            )
        );
    }

    #[test]
    fn rate_directive() {
        let units = parse_markup("А [rate:1.2] Б.").unwrap();
        assert_eq!(
            format!("{:?}", units),
            format!(
                "{:?}",
                vec![
                    MarkupUnit::Text("А".to_string()),
                    MarkupUnit::Rate(1.2),
                    MarkupUnit::Text("Б.".to_string()),
                ]
            )
        );
    }

    #[test]
    fn multiple_directives() {
        let units = parse_markup("[pause 200ms] X [rate:0.8] Y [pause 1s] Z").unwrap();
        assert_eq!(
            format!("{:?}", units),
            format!(
                "{:?}",
                vec![
                    MarkupUnit::Pause(200),
                    MarkupUnit::Text("X".to_string()),
                    MarkupUnit::Rate(0.8),
                    MarkupUnit::Text("Y".to_string()),
                    MarkupUnit::Pause(1000),
                    MarkupUnit::Text("Z".to_string()),
                ]
            )
        );
    }

    #[test]
    fn pause_without_ms_suffix_is_invalid() {
        assert!(parse_markup("[pause 500]").is_err());
    }

    #[test]
    fn zero_or_negative_rate_is_invalid() {
        assert!(parse_markup("[rate:0]").is_err());
        assert!(parse_markup("[rate:-1]").is_err());
    }

    #[test]
    fn unknown_directive_is_invalid() {
        assert!(parse_markup("[banana]").is_err());
    }

    #[test]
    fn speakable_text_skips_pauses() {
        let units = parse_markup("А [pause 500ms] Б [rate:1.2] В.").unwrap();
        assert_eq!(speakable_text(&units), "А Б В.");
    }

    #[test]
    fn total_pause_ms_sums() {
        let units = parse_markup("[pause 200ms] X [pause 300ms]").unwrap();
        assert_eq!(total_pause_ms(&units), 500);
    }

    #[test]
    fn empty_text_gives_one_text_unit() {
        let units = parse_markup("").unwrap();
        assert_eq!(units, vec![MarkupUnit::Text("".to_string())]);
    }

    #[test]
    fn unclosed_bracket_is_text() {
        let units = parse_markup("привіт [здійснюємо").unwrap();
        assert_eq!(
            units,
            vec![MarkupUnit::Text("привіт [здійснюємо".to_string())]
        );
    }
}
