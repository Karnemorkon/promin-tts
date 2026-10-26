//! Спільні хелпери імпорту: безпечний ZIP, кодування, Markdown, заголовки.

use std::io::{Cursor, Read};
use std::sync::OnceLock;

use fancy_regex::Regex as FancyRegex;
use quick_xml::events::{BytesStart, BytesText};
use quick_xml::XmlVersion;
use regex::Regex;
use zip::ZipArchive;

use super::{ImportError, MAX_COMPRESSION_RATIO, MAX_UNCOMPRESSED_BYTES, MAX_ZIP_ENTRIES};

pub(crate) type Zip = ZipArchive<Cursor<Vec<u8>>>;

/// Відкрити ZIP і перевірити його на zip-bomb / traversal / шифрування.
pub(crate) fn open_zip(bytes: &[u8]) -> Result<Zip, ImportError> {
    let mut zip = ZipArchive::new(Cursor::new(bytes.to_vec()))
        .map_err(|error| ImportError::Zip(error.to_string()))?;
    validate_zip(&mut zip)?;
    Ok(zip)
}

fn validate_zip(zip: &mut Zip) -> Result<(), ImportError> {
    if zip.len() > MAX_ZIP_ENTRIES {
        return Err(ImportError::limit(format!(
            "zip has too many entries: {} (max {MAX_ZIP_ENTRIES})",
            zip.len()
        )));
    }
    let mut total = 0u64;
    for index in 0..zip.len() {
        let entry = zip
            .by_index(index)
            .map_err(|error| ImportError::Zip(error.to_string()))?;
        let name = entry
            .name()
            .map_err(|error| ImportError::Zip(error.to_string()))?
            .into_owned();
        if entry.encrypted() {
            return Err(ImportError::invalid(format!("encrypted zip entry: {name}")));
        }
        if entry.is_symlink() {
            return Err(ImportError::invalid(format!("symlink in zip: {name}")));
        }
        if entry.enclosed_name().is_none()
            || name.split(['/', '\\']).any(|component| component == "..")
        {
            return Err(ImportError::invalid(format!("unsafe zip path: {name}")));
        }
        let size = entry.size();
        if size > MAX_UNCOMPRESSED_BYTES {
            return Err(ImportError::limit(format!(
                "zip entry is too large: {name} ({size} bytes)"
            )));
        }
        total = total.saturating_add(size);
        if total > MAX_UNCOMPRESSED_BYTES {
            return Err(ImportError::limit(format!(
                "zip expands beyond {MAX_UNCOMPRESSED_BYTES} bytes"
            )));
        }
        let compressed = entry.compressed_size();
        if compressed == 0 && size > 0 {
            return Err(ImportError::invalid(format!(
                "invalid compressed size for {name}"
            )));
        }
        if compressed > 0 && size / compressed > MAX_COMPRESSION_RATIO {
            return Err(ImportError::limit(format!(
                "suspicious compression ratio in {name}: {size}/{compressed}"
            )));
        }
    }
    Ok(())
}

/// Прочитати запис ZIP із перевіркою розміру.
pub(crate) fn read_zip_entry(zip: &mut Zip, name: &str) -> Result<Vec<u8>, ImportError> {
    let mut entry = zip
        .by_name(name)
        .map_err(|_| ImportError::invalid(format!("zip entry not found: {name}")))?;
    if entry.size() > MAX_UNCOMPRESSED_BYTES {
        return Err(ImportError::limit(format!(
            "zip entry is too large: {name}"
        )));
    }
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry
        .by_ref()
        .take(MAX_UNCOMPRESSED_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| ImportError::Zip(error.to_string()))?;
    if bytes.len() as u64 > MAX_UNCOMPRESSED_BYTES {
        return Err(ImportError::limit(format!(
            "zip entry exceeded limit while reading: {name}"
        )));
    }
    Ok(bytes)
}

/// Чи є запис у ZIP.
pub(crate) fn zip_has(zip: &mut Zip, name: &str) -> bool {
    zip.by_name(name).is_ok()
}

/// Декодувати текст: BOM, XML-декларація, UTF-8, fallback Windows-1251.
pub(crate) fn decode_text(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return decode_utf16(rest, true);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(rest, false);
    }
    if let Some(encoding) = sniff_encoding(bytes) {
        match encoding.to_ascii_lowercase().as_str() {
            "windows-1251" | "cp1251" | "win-1251" => {
                return encoding_rs::WINDOWS_1251.decode(bytes).0.into_owned();
            }
            "koi8-r" => return encoding_rs::KOI8_R.decode(bytes).0.into_owned(),
            "iso-8859-1" | "latin1" | "windows-1252" => {
                return encoding_rs::WINDOWS_1252.decode(bytes).0.into_owned();
            }
            _ => {}
        }
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => encoding_rs::WINDOWS_1251.decode(bytes).0.into_owned(),
    }
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            if little_endian {
                u16::from_le_bytes([pair[0], pair[1]])
            } else {
                u16::from_be_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16_lossy(&units)
}

fn sniff_encoding(bytes: &[u8]) -> Option<String> {
    let head = &bytes[..bytes.len().min(512)];
    let head = String::from_utf8_lossy(head);
    let index = head.find("encoding")?;
    let rest = &head[index + "encoding".len()..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

/// Значення атрибута XML (локальне ім'я, без namespace).
pub(crate) fn attr(event: &BytesStart<'_>, name: &str) -> Option<String> {
    event
        .attributes()
        .flatten()
        .find(|attribute| attribute.key.local_name().as_ref() == name)
        .and_then(|attribute| attribute.normalized_value(XmlVersion::Implicit1_0).ok())
        .map(|value| value.into_owned())
}

/// Текст XML-події з декодуванням HTML-сутностей.
pub(crate) fn text_content(text: &BytesText<'_>) -> String {
    html_escape::decode_html_entities(&text.xml10_content()).into_owned()
}

/// Нормалізувати пробіли в один рядок.
pub(crate) fn normalize_inline(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Обрізати рядок до `max` символів (не байтів).
pub(crate) fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

fn link_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"!?\[([^\]]*)\]\([^)]*\)").expect("valid link regex"))
}

fn tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>\n]{1,200}>").expect("valid tag regex"))
}

fn emphasis_re() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r"(\*\*|__|~~|`)(.+?)\1|(\*|_)([^*_\n]+?)\3").expect("valid emphasis regex")
    })
}

fn list_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(?:[-*+]|\d+[.)])\s+").expect("valid list regex"))
}

fn heading_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*#{1,6}\s+").expect("valid heading regex"))
}

/// Прибрати Markdown-розмітку, залишивши текст для TTS.
pub(crate) fn strip_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let mut line = line.to_string();
        while let Some(rest) = line.trim_start().strip_prefix('>') {
            line = rest.trim_start().to_string();
        }
        if is_horizontal_rule(&line) {
            continue;
        }
        line = heading_marker_re().replace(&line, "").into_owned();
        line = list_marker_re().replace(&line, "").into_owned();
        line = link_re().replace_all(&line, "$1").into_owned();
        line = tag_re().replace_all(&line, "").into_owned();
        line = emphasis_re().replace_all(&line, "$2$4").into_owned();
        out.push_str(line.trim_end());
        out.push('\n');
    }
    html_escape::decode_html_entities(&out).into_owned()
}

fn is_horizontal_rule(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3
        && (trimmed.chars().all(|ch| ch == '-')
            || trimmed.chars().all(|ch| ch == '*')
            || trimmed.chars().all(|ch| ch == '_'))
}

/// Чи схожий рядок на заголовок глави в простому тексті.
pub(crate) fn looks_like_heading(line: &str, next_is_blank: bool) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 120 {
        return false;
    }
    let upper = trimmed.to_uppercase();
    const KEYWORDS: &[&str] = &[
        "CHAPTER",
        "РОЗДІЛ",
        "ГЛАВА",
        "ЧАСТИНА",
        "PART",
        "BOOK",
        "КНИГА",
        "ПРОЛОГ",
        "ЕПІЛОГ",
        "PROLOGUE",
        "EPILOGUE",
    ];
    if KEYWORDS.iter().any(|keyword| upper.starts_with(keyword)) {
        return true;
    }
    let letters = trimmed.chars().filter(|ch| ch.is_alphabetic()).count();
    letters >= 3
        && trimmed == upper
        && trimmed.chars().count() <= 60
        && next_is_blank
        && !trimmed.ends_with(['.', ',', ';', ':', '!', '?'])
}
