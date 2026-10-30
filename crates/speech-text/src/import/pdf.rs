//! PDF: лише текстовий шар. Скановані та зашифровані PDF відхиляються.
//!
//! Розбиття на глави: закладки (TOC), далі евристика заголовків, а якщо
//! заголовків немає — групування сторінок до цільового розміру.

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use pdf_extract::{decode_text_string, output_doc_page, Document, PlainTextOutput};

use super::util;
use super::{BookFormat, ImportError, ImportedBook, ImportedChapter, MAX_TOTAL_CHARS};

const MAX_PDF_PAGES: usize = 2_000;
const MIN_TEXT_CHARS: usize = 50;
const MIN_CHAPTER_CHARS: usize = 20;
const TARGET_PAGE_CHARS: usize = 8_000;
const MAX_HEADING_CHARS: usize = 100;

struct PdfPage {
    number: u32,
    lines: Vec<String>,
}

/// Імпортувати PDF. `filename` використовується як запасна назва книги.
pub(crate) fn import(bytes: &[u8], filename: &str) -> Result<ImportedBook, ImportError> {
    // pdf-extract/lopdf можуть панікувати на пошкоджених документах. Не
    // дозволяємо одному файлу зірвати весь HTTP-обробник.
    match catch_unwind(AssertUnwindSafe(|| import_inner(bytes, filename))) {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!("PDF parser panicked");
            Err(ImportError::invalid("invalid PDF"))
        }
    }
}

fn import_inner(bytes: &[u8], filename: &str) -> Result<ImportedBook, ImportError> {
    let document = Document::load_mem(bytes)
        .map_err(|error| ImportError::invalid(format!("invalid PDF: {error}")))?;

    if document.was_encrypted() || document.is_encrypted() {
        return Err(ImportError::invalid(
            "encrypted PDF is not supported; remove the password first",
        ));
    }

    let page_map = document.get_pages();
    if page_map.is_empty() {
        return Err(ImportError::invalid("PDF contains no pages"));
    }
    if page_map.len() > MAX_PDF_PAGES {
        return Err(ImportError::limit(format!(
            "PDF has too many pages: {} (max {MAX_PDF_PAGES})",
            page_map.len()
        )));
    }

    let mut pages = Vec::with_capacity(page_map.len());
    let mut extracted_chars = 0usize;
    for number in page_map.keys().copied() {
        let mut raw = String::new();
        {
            let mut output = PlainTextOutput::new(&mut raw);
            output_doc_page(&document, &mut output, number).map_err(|error| {
                ImportError::invalid(format!(
                    "cannot extract text from PDF page {number}: {error}"
                ))
            })?;
        }
        extracted_chars = extracted_chars.saturating_add(raw.chars().count());
        if extracted_chars > MAX_TOTAL_CHARS.saturating_mul(2) {
            return Err(ImportError::limit(format!(
                "PDF text is too long: more than {} chars",
                MAX_TOTAL_CHARS
            )));
        }
        pages.push(PdfPage {
            number,
            lines: clean_lines(&raw),
        });
    }

    remove_repeated_headers(&mut pages);
    let text_chars: usize = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| line.chars().count())
                .sum::<usize>()
        })
        .sum();
    if text_chars < MIN_TEXT_CHARS
        || (pages.len() >= 5 && text_chars / pages.len() < MIN_TEXT_CHARS / 5)
    {
        return Err(ImportError::invalid(
            "PDF has no text layer; scanned PDFs are not supported (OCR is required)",
        ));
    }

    let (metadata_title, metadata_author) = metadata(&document);
    let title = metadata_title
        .or_else(|| filename_title(filename))
        .unwrap_or_else(|| "Без назви".to_string());

    let mut chapters = chapters_from_toc(&document, &pages);
    if chapters.len() < 2 {
        chapters = chapters_from_headings(&pages);
    }
    if chapters.len() < 2 {
        chapters = chapters_from_pages(&pages, &title);
    }

    Ok(ImportedBook {
        title,
        author: metadata_author.unwrap_or_default(),
        language: "uk".to_string(),
        format: BookFormat::Pdf,
        chapters,
    })
}

/// Очистити рядки сторінки: прибрати м'які переноси, керуючі символи й
/// зайві пробіли. Порожні рядки зберігаються як межі абзаців.
fn clean_lines(raw: &str) -> Vec<String> {
    let mut lines = Vec::new();
    for line in raw.replace('\u{00ad}', "").lines() {
        let line: String = line
            .chars()
            .filter(|ch| !ch.is_control() || *ch == '\t')
            .collect();
        let line = util::normalize_inline(&line);
        if line.is_empty() {
            if lines.last().is_some_and(|last: &String| !last.is_empty()) {
                lines.push(String::new());
            }
        } else {
            lines.push(line);
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

/// Колонтитули та номери сторінок зазвичай повторюються на кожній сторінці.
fn remove_repeated_headers(pages: &mut [PdfPage]) {
    if pages.len() < 3 {
        for page in pages.iter_mut() {
            page.lines.retain(|line| !is_page_number(line));
        }
        return;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for page in pages.iter() {
        let mut seen = HashSet::new();
        let non_empty: Vec<&String> = page.lines.iter().filter(|line| !line.is_empty()).collect();
        for line in non_empty
            .iter()
            .take(2)
            .chain(non_empty.iter().rev().take(2))
        {
            let key = line.to_lowercase();
            if (3..=MAX_HEADING_CHARS).contains(&key.chars().count()) && seen.insert(key.clone()) {
                *counts.entry(key).or_default() += 1;
            }
        }
    }
    let threshold = (pages.len() / 2).max(3);
    let repeated: HashSet<String> = counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold)
        .map(|(line, _)| line)
        .collect();

    for page in pages.iter_mut() {
        page.lines
            .retain(|line| !is_page_number(line) && !repeated.contains(&line.to_lowercase()));
    }
}

fn is_page_number(line: &str) -> bool {
    let trimmed = line
        .trim()
        .trim_matches(|ch| matches!(ch, '-' | '—' | '–'))
        .trim();
    !trimmed.is_empty() && trimmed.len() <= 8 && trimmed.chars().all(|ch| ch.is_ascii_digit())
}

fn metadata(document: &Document) -> (Option<String>, Option<String>) {
    let info = document.trailer.get(b"Info").ok().and_then(|object| {
        if let Ok(id) = object.as_reference() {
            document.get_dictionary(id).ok()
        } else {
            object.as_dict().ok()
        }
    });
    let Some(info) = info else {
        return (None, None);
    };
    let value = |key: &[u8]| {
        info.get(key)
            .ok()
            .and_then(|object| decode_text_string(object).ok())
            .map(|text| util::normalize_inline(&text))
            .filter(|text| !text.is_empty())
    };
    (value(b"Title"), value(b"Author"))
}

fn filename_title(filename: &str) -> Option<String> {
    let stem = Path::new(filename).file_stem()?.to_str()?;
    let title = util::normalize_inline(&stem.replace(['_', '-'], " "));
    (!title.is_empty()).then_some(title)
}

/// Спроба 1: закладки PDF. Використовуємо перший пункт на кожній сторінці.
fn chapters_from_toc(document: &Document, pages: &[PdfPage]) -> Vec<ImportedChapter> {
    let Ok(toc) = document.get_toc() else {
        return Vec::new();
    };
    let mut entries: Vec<(usize, String)> = Vec::new();
    for item in toc.toc {
        if item.page == 0 || item.page > pages.len() {
            continue;
        }
        let title = util::normalize_inline(&item.title);
        if title.is_empty() {
            continue;
        }
        if entries.iter().any(|(page, _)| *page == item.page) {
            continue;
        }
        entries.push((item.page, title));
    }
    entries.sort_by_key(|(page, _)| *page);
    if entries.len() < 2 {
        return Vec::new();
    }

    let mut chapters = Vec::new();
    let first_page = entries[0].0;
    if first_page > 1 {
        let text = pages_text(&pages[..first_page - 1], "");
        if text.chars().count() >= MIN_CHAPTER_CHARS {
            chapters.push(ImportedChapter {
                title: "Початок".to_string(),
                text,
            });
        }
    }

    for (index, (page, title)) in entries.iter().enumerate() {
        let start = page - 1;
        let end = entries
            .get(index + 1)
            .map(|(next, _)| next - 1)
            .unwrap_or(pages.len());
        let text = pages_text(&pages[start..end], title);
        if text.chars().count() < MIN_CHAPTER_CHARS {
            continue;
        }
        chapters.push(ImportedChapter {
            title: title.clone(),
            text,
        });
    }

    if chapters.len() >= 2 {
        chapters
    } else {
        Vec::new()
    }
}

/// Спроба 2: рядки, схожі на заголовки («Розділ 1», «Глава», «I. Назва»).
fn chapters_from_headings(pages: &[PdfPage]) -> Vec<ImportedChapter> {
    let mut chapters = Vec::new();
    let mut current_title = String::new();
    let mut current_text = String::new();
    let mut heading_count = 0usize;

    for page in pages {
        for line in &page.lines {
            if is_heading(line) {
                flush_chapter(&mut chapters, &current_title, &mut current_text);
                current_title = util::normalize_inline(line);
                heading_count += 1;
            } else {
                append_line(&mut current_text, line);
            }
        }
        if !current_text.is_empty() {
            current_text.push('\n');
        }
    }
    flush_chapter(&mut chapters, &current_title, &mut current_text);

    if heading_count >= 2 && chapters.len() >= 2 {
        chapters
    } else {
        Vec::new()
    }
}

/// Спроба 3: групування сторінок. Використовується, коли заголовків немає.
fn chapters_from_pages(pages: &[PdfPage], book_title: &str) -> Vec<ImportedChapter> {
    let mut groups: Vec<(u32, u32, String)> = Vec::new();
    let mut start_page: Option<u32> = None;
    let mut last_page = 0u32;
    let mut current = String::new();

    for page in pages {
        let text = page_text(page);
        if text.is_empty() {
            continue;
        }
        start_page.get_or_insert(page.number);
        last_page = page.number;
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&text);
        if current.chars().count() >= TARGET_PAGE_CHARS {
            groups.push((
                start_page.take().unwrap_or(page.number),
                last_page,
                std::mem::take(&mut current),
            ));
        }
    }
    if !current.is_empty() {
        groups.push((start_page.unwrap_or(last_page.max(1)), last_page, current));
    }

    if groups.len() == 1 {
        return vec![ImportedChapter {
            title: book_title.to_string(),
            text: groups.pop().map(|(_, _, text)| text).unwrap_or_default(),
        }];
    }

    groups
        .into_iter()
        .filter(|(_, _, text)| text.chars().count() >= MIN_CHAPTER_CHARS)
        .map(|(start, end, text)| ImportedChapter {
            title: format!("Сторінки {start}–{end}"),
            text,
        })
        .collect()
}

fn flush_chapter(chapters: &mut Vec<ImportedChapter>, title: &str, text: &mut String) {
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_CHAPTER_CHARS {
        text.clear();
        return;
    }
    let title = if title.trim().is_empty() {
        if chapters.is_empty() {
            "Початок".to_string()
        } else {
            format!("Розділ {}", chapters.len() + 1)
        }
    } else {
        title.trim().to_string()
    };
    chapters.push(ImportedChapter {
        title,
        text: trimmed.to_string(),
    });
    text.clear();
}

fn pages_text(pages: &[PdfPage], title: &str) -> String {
    let mut text = String::new();
    for page in pages {
        let page = page_text(page);
        if page.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&page);
    }
    // Закладка часто вказує на сторінку, де заголовок повторюється першим
    // рядком. Не дублюємо його в тексті глави.
    let normalized_title = util::normalize_inline(title);
    if !normalized_title.is_empty() {
        let mut lines = text.lines();
        if let Some(first) = lines.next() {
            if util::normalize_inline(first) == normalized_title {
                text = lines.collect::<Vec<_>>().join("\n").trim().to_string();
            }
        }
    }
    text
}

fn page_text(page: &PdfPage) -> String {
    let mut text = String::new();
    for line in &page.lines {
        append_line(&mut text, line);
    }
    text.trim().to_string()
}

fn append_line(text: &mut String, line: &str) {
    let line = line.trim();
    if line.is_empty() {
        if !text.is_empty() && !text.ends_with("\n\n") {
            text.push_str("\n\n");
        }
        return;
    }
    if !text.is_empty() && !text.ends_with('\n') {
        if text.ends_with('-') {
            text.pop();
        } else {
            text.push(' ');
        }
    }
    text.push_str(line);
}

fn is_heading(line: &str) -> bool {
    let line = line.trim();
    let chars = line.chars().count();
    if !(2..=MAX_HEADING_CHARS).contains(&chars) {
        return false;
    }
    if line.ends_with(['.', '!', '?', ';', ',']) {
        return false;
    }
    if line.split_whitespace().count() > 14 {
        return false;
    }

    let lower = line.to_lowercase();
    const KEYWORDS: &[&str] = &[
        "розділ",
        "глава",
        "частина",
        "книга",
        "пісня",
        "пролог",
        "епілог",
        "передмова",
        "післямова",
        "chapter",
        "part",
        "book",
        "prologue",
        "epilogue",
    ];
    for keyword in KEYWORDS {
        if let Some(rest) = lower.strip_prefix(keyword) {
            if rest
                .chars()
                .next()
                .is_none_or(|ch| ch.is_whitespace() || matches!(ch, '.' | ':' | '№' | '#'))
            {
                return true;
            }
        }
    }

    if is_roman_heading(line) || is_numbered_heading(line) {
        return true;
    }

    let letters = line.chars().filter(|ch| ch.is_alphabetic()).count();
    let uppercase = line.chars().filter(|ch| ch.is_uppercase()).count();
    letters >= 4
        && uppercase.saturating_mul(100) >= letters.saturating_mul(80)
        && line.split_whitespace().count() <= 12
}

fn is_roman_heading(line: &str) -> bool {
    let token: String = line
        .chars()
        .take_while(|ch| !ch.is_whitespace() && !matches!(ch, '.' | ')'))
        .collect();
    if token.is_empty()
        || token.len() > 8
        || !token
            .chars()
            .all(|ch| matches!(ch, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
    {
        return false;
    }
    let rest = line[token.len()..].trim_start_matches(['.', ')', ' ']);
    rest.is_empty() || rest.chars().next().is_some_and(char::is_uppercase)
}

fn is_numbered_heading(line: &str) -> bool {
    let digits: String = line.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() || digits.len() > 4 {
        return false;
    }
    let rest = line[digits.len()..].trim_start();
    if rest.is_empty() {
        return false;
    }
    let rest = rest
        .strip_prefix('.')
        .or_else(|| rest.strip_prefix(')'))
        .unwrap_or(rest);
    let rest = rest.trim_start();
    rest.chars().next().is_some_and(char::is_uppercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_headings() {
        assert!(is_heading("Розділ 1"));
        assert!(is_heading("Глава 12. Повернення"));
        assert!(is_heading("I. Початок"));
        assert!(is_heading("ПРОЛОГ"));
        assert!(is_heading("1. Вступ"));
        assert!(!is_heading("Звичайний рядок тексту."));
        assert!(!is_heading("2024 рік"));
        assert!(!is_heading(""));
    }

    #[test]
    fn strips_page_numbers_and_repeated_headers() {
        let mut pages = vec![
            PdfPage {
                number: 1,
                lines: vec!["Книга".into(), "Текст 1".into(), "1".into()],
            },
            PdfPage {
                number: 2,
                lines: vec!["Книга".into(), "Текст 2".into(), "2".into()],
            },
            PdfPage {
                number: 3,
                lines: vec!["Книга".into(), "Текст 3".into(), "3".into()],
            },
        ];
        remove_repeated_headers(&mut pages);
        assert_eq!(pages[0].lines, vec!["Текст 1"]);
        assert_eq!(pages[2].lines, vec!["Текст 3"]);
    }

    #[test]
    fn groups_pages_when_no_headings() {
        let pages = vec![
            PdfPage {
                number: 1,
                lines: vec!["а".repeat(4_500)],
            },
            PdfPage {
                number: 2,
                lines: vec!["б".repeat(4_500)],
            },
            PdfPage {
                number: 3,
                lines: vec!["в".repeat(4_500)],
            },
        ];
        let chapters = chapters_from_pages(&pages, "Книга");
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "Сторінки 1–2");
        assert_eq!(chapters[1].title, "Сторінки 3–3");
    }
}
