//! Розмітка діалогів у тексті глави.
//!
//! Ролі задаються в проєкті, а в тексті репліки позначаються одним із двох
//! способів на початку рядка:
//!
//! ```text
//! Іван: Привіт, Маріє!
//! [Марія] Привіт, Іване.
//! ```
//!
//! Усе, що не починається з імені зареєстрованої ролі, вважається нарацією
//! (голос проєкту). Порожні рядки розділяють абзаци й не дають зливати сусідні
//! репліки однієї ролі в один сегмент.

/// Один сегмент глави: нарація (`role = None`) або репліка ролі.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogueSegment {
    pub role: Option<String>,
    pub text: String,
}

/// Розбити текст глави на сегменти за зареєстрованими ролями.
///
/// `roles` — канонічні назви ролей (без дужок і двокрапок). Порівняння
/// регістронезалежне для Unicode. Невідомі префікси залишаються частиною
/// нарації, тому звичайний текст із двокрапкою не ламається.
pub fn split_dialogue(text: &str, roles: &[&str]) -> Vec<DialogueSegment> {
    let roles: Vec<(String, &str)> = roles
        .iter()
        .filter(|role| !role.trim().is_empty())
        .map(|role| (role.to_lowercase(), *role))
        .collect();

    let mut segments: Vec<DialogueSegment> = Vec::new();
    let mut current: Option<DialogueSegment> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            if let Some(segment) = current.take() {
                segments.push(segment);
            }
            continue;
        }

        let (role, content) = match marked_line(line, &roles) {
            Some((role, content)) => (Some(role.to_string()), content),
            None => (None, line),
        };
        if content.trim().is_empty() {
            // Рядок-етикетка на кшталт `Іван:` без тексту.
            continue;
        }

        match &mut current {
            Some(segment) if segment.role == role => {
                if !segment.text.is_empty() {
                    segment.text.push(' ');
                }
                segment.text.push_str(content);
            }
            _ => {
                if let Some(segment) = current.take() {
                    segments.push(segment);
                }
                current = Some(DialogueSegment {
                    role,
                    text: content.to_string(),
                });
            }
        }
    }

    if let Some(segment) = current {
        segments.push(segment);
    }
    segments
}

/// Розпізнати `[Роль] текст`, `[Роль]: текст` або `Роль: текст`.
fn marked_line<'a>(line: &'a str, roles: &[(String, &'a str)]) -> Option<(&'a str, &'a str)> {
    if let Some(rest) = line.strip_prefix('[') {
        if let Some(close) = rest.find(']') {
            let name = rest[..close].trim();
            if let Some(canonical) = find_role(name, roles) {
                let mut content = rest[close + 1..].trim_start();
                if let Some(stripped) = content.strip_prefix(':') {
                    content = stripped.trim_start();
                }
                return Some((canonical, content));
            }
        }
    }

    let colon = line.find(':')?;
    let name = line[..colon].trim();
    if name.is_empty() {
        return None;
    }
    let canonical = find_role(name, roles)?;
    Some((canonical, line[colon + 1..].trim_start()))
}

fn find_role<'a>(name: &str, roles: &[(String, &'a str)]) -> Option<&'a str> {
    let lowered = name.to_lowercase();
    roles
        .iter()
        .find(|(role, _)| role == &lowered)
        .map(|(_, canonical)| *canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(role: Option<&str>, text: &str) -> DialogueSegment {
        DialogueSegment {
            role: role.map(str::to_string),
            text: text.to_string(),
        }
    }

    #[test]
    fn plain_text_is_narration() {
        assert_eq!(
            split_dialogue("Перше речення. Друге речення.", &["Іван"]),
            vec![segment(None, "Перше речення. Друге речення.")]
        );
    }

    #[test]
    fn splits_colon_dialogue() {
        assert_eq!(
            split_dialogue(
                "Був теплий вечір.\nІван: Привіт, Маріє!\nМарія: Привіт!",
                &["Іван", "Марія"]
            ),
            vec![
                segment(None, "Був теплий вечір."),
                segment(Some("Іван"), "Привіт, Маріє!"),
                segment(Some("Марія"), "Привіт!"),
            ]
        );
    }

    #[test]
    fn supports_bracket_markup() {
        assert_eq!(
            split_dialogue("[Іван] Привіт!\n[Марія]: Добрий день.", &["Іван", "Марія"]),
            vec![
                segment(Some("Іван"), "Привіт!"),
                segment(Some("Марія"), "Добрий день."),
            ]
        );
    }

    #[test]
    fn matches_roles_case_insensitively_and_keeps_canonical_name() {
        assert_eq!(
            split_dialogue("іван: Текст.", &["Іван"]),
            vec![segment(Some("Іван"), "Текст.")]
        );
        assert_eq!(
            split_dialogue("[МАРІЯ] Текст.", &["Марія"]),
            vec![segment(Some("Марія"), "Текст.")]
        );
    }

    #[test]
    fn unknown_prefix_stays_narration() {
        assert_eq!(
            split_dialogue("Примітка: це не роль.", &["Іван"]),
            vec![segment(None, "Примітка: це не роль.")]
        );
        assert_eq!(
            split_dialogue("Іванко: Це інше ім'я.", &["Іван"]),
            vec![segment(None, "Іванко: Це інше ім'я.")]
        );
    }

    #[test]
    fn multiword_role_names_work() {
        assert_eq!(
            split_dialogue("Іван Петрович: Добридень.", &["Іван Петрович"]),
            vec![segment(Some("Іван Петрович"), "Добридень.")]
        );
    }

    #[test]
    fn consecutive_lines_of_same_role_merge() {
        assert_eq!(
            split_dialogue("Іван: Перша.\nІван: Друга.", &["Іван"]),
            vec![segment(Some("Іван"), "Перша. Друга.")]
        );
    }

    #[test]
    fn blank_line_breaks_same_role_segments() {
        assert_eq!(
            split_dialogue("Іван: Перша.\n\nІван: Друга.", &["Іван"]),
            vec![
                segment(Some("Іван"), "Перша."),
                segment(Some("Іван"), "Друга."),
            ]
        );
    }

    #[test]
    fn empty_reply_is_skipped() {
        assert_eq!(
            split_dialogue("Іван:\nМарія: Текст.", &["Іван", "Марія"]),
            vec![segment(Some("Марія"), "Текст.")]
        );
    }

    #[test]
    fn colon_inside_dialogue_text_is_preserved() {
        assert_eq!(
            split_dialogue("Іван: Час: 14:30.", &["Іван"]),
            vec![segment(Some("Іван"), "Час: 14:30.")]
        );
    }

    #[test]
    fn blank_text_produces_no_segments() {
        assert!(split_dialogue("   \n\n  ", &["Іван"]).is_empty());
    }
}
