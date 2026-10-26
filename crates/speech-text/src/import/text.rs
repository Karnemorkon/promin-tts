//! Імпорт простого тексту та Markdown.

use super::util;
use super::BookFormat;
use super::{ImportError, ImportedBook, ImportedChapter};

const PLAIN_PART_CHARS: usize = 50_000;

pub(crate) fn import_plain(bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let text = util::decode_text(bytes);
    let chapters = split_plain_text(&text);
    Ok(ImportedBook {
        title: String::new(),
        author: String::new(),
        language: "uk".to_string(),
        format: BookFormat::Txt,
        chapters,
    })
}

pub(crate) fn import_markdown(bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let text = util::decode_text(bytes);
    let chapters = split_markdown(&text);
    Ok(ImportedBook {
        title: String::new(),
        author: String::new(),
        language: "uk".to_string(),
        format: BookFormat::Markdown,
        chapters,
    })
}

/// Розбити простий текст на глави за заголовками або за розміром.
pub fn split_plain_text(text: &str) -> Vec<ImportedChapter> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = text.lines().collect();
    let candidates: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            let next_is_blank = lines
                .get(index + 1)
                .is_none_or(|next| next.trim().is_empty());
            util::looks_like_heading(line, next_is_blank)
        })
        .map(|(index, _)| index)
        .collect();

    if candidates.len() >= 2 {
        let mut chapters = Vec::new();
        let first = candidates[0];
        let preface = lines[..first].join("\n").trim().to_string();
        if !preface.is_empty() {
            chapters.push(ImportedChapter {
                title: "Передмова".to_string(),
                text: preface,
            });
        }
        for (position, &start) in candidates.iter().enumerate() {
            let end = candidates.get(position + 1).copied().unwrap_or(lines.len());
            let title = util::normalize_inline(lines[start]);
            let body = lines[start..end].join("\n").trim().to_string();
            if body.is_empty() {
                continue;
            }
            chapters.push(ImportedChapter { title, text: body });
        }
        if !chapters.is_empty() {
            return chapters;
        }
    }

    let parts = split_by_size(&text, PLAIN_PART_CHARS);
    parts
        .into_iter()
        .enumerate()
        .filter(|(_, part)| !part.trim().is_empty())
        .map(|(index, part)| ImportedChapter {
            title: format!("Частина {}", index + 1),
            text: part,
        })
        .collect()
}

/// Розбити Markdown на глави за ATX-заголовками.
pub fn split_markdown(text: &str) -> Vec<ImportedChapter> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = text.lines().collect();
    let headings: Vec<(usize, usize, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            parse_atx_heading(line).map(|(level, title)| (index, level, title))
        })
        .collect();

    if headings.is_empty() {
        let stripped = util::strip_markdown(&text);
        if stripped.trim().is_empty() {
            return Vec::new();
        }
        return vec![ImportedChapter {
            title: "Розділ 1".to_string(),
            text: stripped,
        }];
    }

    let split_level = choose_heading_level(&headings);
    let split_indices: Vec<usize> = headings
        .iter()
        .filter(|(_, level, _)| *level == split_level)
        .map(|(index, _, _)| *index)
        .collect();

    if split_indices.len() < 2 {
        let title = headings
            .iter()
            .find(|(_, level, _)| *level == split_level)
            .map(|(_, _, title)| title.clone())
            .unwrap_or_else(|| "Розділ 1".to_string());
        let stripped = util::strip_markdown(&text);
        return vec![ImportedChapter {
            title,
            text: stripped,
        }];
    }

    let mut chapters = Vec::new();
    let first = split_indices[0];
    let preface = util::strip_markdown(&lines[..first].join("\n"));
    if !preface.trim().is_empty() {
        chapters.push(ImportedChapter {
            title: "Передмова".to_string(),
            text: preface,
        });
    }
    for (position, &start) in split_indices.iter().enumerate() {
        let end = split_indices
            .get(position + 1)
            .copied()
            .unwrap_or(lines.len());
        let (_, _, title) = headings
            .iter()
            .find(|(index, _, _)| *index == start)
            .expect("heading index exists");
        let body = util::strip_markdown(&lines[start..end].join("\n"));
        if body.trim().is_empty() {
            continue;
        }
        chapters.push(ImportedChapter {
            title: title.clone(),
            text: body,
        });
    }
    chapters
}

fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim().to_string();
    if title.is_empty() {
        return None;
    }
    Some((hashes, title))
}

fn choose_heading_level(headings: &[(usize, usize, String)]) -> usize {
    let mut counts = [0usize; 3];
    for (_, level, _) in headings {
        if (1..=3).contains(level) {
            counts[level - 1] += 1;
        }
    }
    if counts[0] >= 2 {
        1
    } else if counts[1] >= 2 {
        2
    } else if counts[2] >= 2 {
        3
    } else {
        headings
            .iter()
            .map(|(_, level, _)| *level)
            .min()
            .unwrap_or(1)
    }
}

fn split_by_size(text: &str, max_chars: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        let paragraph_chars = paragraph.chars().count();
        if paragraph_chars > max_chars {
            if !current.trim().is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            parts.extend(crate::segment::chunk_text(paragraph, max_chars));
            continue;
        }
        let current_chars = current.chars().count();
        if current_chars > 0 && current_chars + 2 + paragraph_chars > max_chars {
            parts.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    if parts.is_empty() && !text.trim().is_empty() {
        parts.push(text.trim().to_string());
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_splits_on_chapter_headings() {
        let text = "Передмова.\n\nРОЗДІЛ 1\n\nПерший текст.\n\nРОЗДІЛ 2\n\nДругий текст.\n";
        let chapters = split_plain_text(text);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].title, "Передмова");
        assert_eq!(chapters[1].title, "РОЗДІЛ 1");
        assert!(chapters[1].text.contains("Перший текст."));
        assert_eq!(chapters[2].title, "РОЗДІЛ 2");
    }

    #[test]
    fn plain_text_without_headings_splits_by_size() {
        let text = "абв ".repeat(40_000);
        let chapters = split_plain_text(&text);
        assert!(chapters.len() >= 2);
        assert!(chapters[0].title.starts_with("Частина"));
    }

    #[test]
    fn markdown_splits_and_strips_markup() {
        let text = "# Книга\n\nВступ.\n\n## Розділ 1\n\n**Жирний** текст і [посилання](https://example.com).\n\n## Розділ 2\n\n- пункт\n- другий\n";
        let chapters = split_markdown(text);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].title, "Передмова");
        assert_eq!(chapters[1].title, "Розділ 1");
        assert!(chapters[1].text.contains("Жирний текст і посилання."));
        assert!(!chapters[1].text.contains("**"));
        assert_eq!(chapters[2].title, "Розділ 2");
        assert!(chapters[2].text.contains("пункт"));
    }
}
