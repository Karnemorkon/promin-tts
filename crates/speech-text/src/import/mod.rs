//! Імпорт книг: EPUB, FB2, DOCX, TXT, Markdown.
//!
//! Модуль увімкнений feature-флагом `import`, щоб `speech-inference` не тягнув
//! `zip`/`quick-xml`/`html-escape`.

mod docx;
mod epub;
mod fb2;
mod text;
mod util;

pub use text::{split_markdown, split_plain_text};

use std::path::Path;

use thiserror::Error;

/// Максимальний розмір завантаженого файлу.
pub const MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Максимальний сумарний розмір розпакованих даних.
pub const MAX_UNCOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
/// Максимальна кількість записів у ZIP.
pub const MAX_ZIP_ENTRIES: usize = 5_000;
/// Максимальна кількість глав.
pub const MAX_CHAPTERS: usize = 2_000;
/// Максимальна довжина тексту глави (символів).
pub const MAX_CHAPTER_CHARS: usize = 200_000;
/// Максимальна сумарна довжина тексту книги (символів).
pub const MAX_TOTAL_CHARS: usize = 2_000_000;
/// Підозрілий коефіцієнт стиснення ZIP-запису.
pub const MAX_COMPRESSION_RATIO: u64 = 1_000;
/// Максимальна довжина назви глави.
pub const MAX_TITLE_CHARS: usize = 500;

/// Помилка імпорту.
#[derive(Debug, Error)]
pub enum ImportError {
    #[error("unsupported book format")]
    UnsupportedFormat,
    #[error("invalid book: {0}")]
    Invalid(String),
    #[error("book exceeds limit: {0}")]
    LimitExceeded(String),
    #[error("zip error: {0}")]
    Zip(String),
    #[error("xml error: {0}")]
    Xml(String),
}

impl ImportError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub(crate) fn limit(message: impl Into<String>) -> Self {
        Self::LimitExceeded(message.into())
    }
}

/// Формат книги.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookFormat {
    Epub,
    Fb2,
    Docx,
    Txt,
    Markdown,
}

impl BookFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Epub => "epub",
            Self::Fb2 => "fb2",
            Self::Docx => "docx",
            Self::Txt => "txt",
            Self::Markdown => "markdown",
        }
    }
}

/// Одна глава імпортованої книги.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedChapter {
    pub title: String,
    pub text: String,
}

/// Результат імпорту.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedBook {
    pub title: String,
    pub author: String,
    pub language: String,
    pub format: BookFormat,
    pub chapters: Vec<ImportedChapter>,
}

/// Визначити формат за розширенням, а якщо воно невідоме — за вмістом.
pub fn detect_format(filename: &str, bytes: &[u8]) -> Option<BookFormat> {
    if let Some(ext) = Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
    {
        match ext.as_str() {
            "epub" => return Some(BookFormat::Epub),
            "fb2" => return Some(BookFormat::Fb2),
            "docx" => return Some(BookFormat::Docx),
            "md" | "markdown" => return Some(BookFormat::Markdown),
            "txt" => return Some(BookFormat::Txt),
            _ => {}
        }
    }
    sniff_format(bytes)
}

fn sniff_format(bytes: &[u8]) -> Option<BookFormat> {
    if bytes.starts_with(b"PK\x03\x04") {
        if let Ok(mut zip) = util::open_zip(bytes) {
            if zip.by_name("META-INF/container.xml").is_ok() {
                return Some(BookFormat::Epub);
            }
            if zip.by_name("word/document.xml").is_ok() {
                return Some(BookFormat::Docx);
            }
        }
        return None;
    }
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]);
    if head.contains("<FictionBook") {
        return Some(BookFormat::Fb2);
    }
    if head.contains("<?xml") {
        return None;
    }
    Some(BookFormat::Txt)
}

/// Імпортувати книгу, визначивши формат.
pub fn import_book(filename: &str, bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    if bytes.len() > MAX_UPLOAD_BYTES {
        return Err(ImportError::limit(format!(
            "file is too large: {} bytes (max {MAX_UPLOAD_BYTES})",
            bytes.len()
        )));
    }
    let format = detect_format(filename, bytes).ok_or(ImportError::UnsupportedFormat)?;
    import_bytes(format, bytes)
}

/// Імпортувати книгу у явно заданому форматі.
pub fn import_bytes(format: BookFormat, bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let book = match format {
        BookFormat::Epub => epub::import(bytes)?,
        BookFormat::Fb2 => fb2::import(bytes)?,
        BookFormat::Docx => docx::import(bytes)?,
        BookFormat::Txt => text::import_plain(bytes)?,
        BookFormat::Markdown => text::import_markdown(bytes)?,
    };
    finalize_book(book)
}

/// Обрізати/розбити глави, застосувати ліміти та заповнити порожні назви.
fn finalize_book(mut book: ImportedBook) -> Result<ImportedBook, ImportError> {
    let mut chapters = Vec::new();
    for chapter in book.chapters {
        let text = chapter.text.trim();
        if text.is_empty() {
            continue;
        }
        let base_title = util::normalize_inline(&chapter.title);
        let parts = crate::segment::chunk_text(text, MAX_CHAPTER_CHARS);
        let part_count = parts.len().max(1);
        for (part_index, part) in parts.into_iter().enumerate() {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let mut title = if base_title.is_empty() {
                format!("Розділ {}", chapters.len() + 1)
            } else {
                base_title.clone()
            };
            if part_count > 1 {
                title = format!("{title} ({}/{})", part_index + 1, part_count);
            }
            chapters.push(ImportedChapter {
                title: util::truncate_chars(&title, MAX_TITLE_CHARS),
                text: part.to_string(),
            });
        }
    }

    if chapters.is_empty() {
        return Err(ImportError::invalid("book contains no text"));
    }
    if chapters.len() > MAX_CHAPTERS {
        return Err(ImportError::limit(format!(
            "too many chapters: {} (max {MAX_CHAPTERS})",
            chapters.len()
        )));
    }
    let total_chars: usize = chapters
        .iter()
        .map(|chapter| chapter.text.chars().count())
        .sum();
    if total_chars > MAX_TOTAL_CHARS {
        return Err(ImportError::limit(format!(
            "book text is too long: {total_chars} chars (max {MAX_TOTAL_CHARS})"
        )));
    }

    book.title = util::truncate_chars(&util::normalize_inline(&book.title), MAX_TITLE_CHARS);
    if book.title.is_empty() {
        book.title = "Без назви".to_string();
    }
    book.author = util::normalize_inline(&book.author);
    if book.language.trim().is_empty() {
        book.language = "uk".to_string();
    }
    book.chapters = chapters;
    Ok(book)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_by_extension_and_magic() {
        assert_eq!(detect_format("book.epub", b""), Some(BookFormat::Epub));
        assert_eq!(detect_format("book.FB2", b""), Some(BookFormat::Fb2));
        assert_eq!(detect_format("book.docx", b""), Some(BookFormat::Docx));
        assert_eq!(detect_format("book.md", b""), Some(BookFormat::Markdown));
        assert_eq!(detect_format("book.txt", b""), Some(BookFormat::Txt));
        assert_eq!(
            detect_format("book", b"<FictionBook></FictionBook>"),
            Some(BookFormat::Fb2)
        );
    }

    #[test]
    fn rejects_oversized_upload() {
        let bytes = vec![0u8; MAX_UPLOAD_BYTES + 1];
        assert!(matches!(
            import_book("book.txt", &bytes),
            Err(ImportError::LimitExceeded(_))
        ));
    }
}
