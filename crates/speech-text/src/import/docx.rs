//! DOCX: word/document.xml + styles.xml + docProps/core.xml.

use std::collections::HashMap;

use quick_xml::events::Event;
use quick_xml::Reader;

use super::util;
use super::{BookFormat, ImportError, ImportedBook, ImportedChapter};

#[derive(Default)]
struct Meta {
    title: String,
    author: String,
    language: String,
}

struct Paragraph {
    text: String,
    heading_level: Option<usize>,
}

pub(crate) fn import(bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let mut zip = util::open_zip(bytes)?;
    let document = util::read_zip_entry(&mut zip, "word/document.xml")?;
    let document = util::decode_text(&document);
    let styles = if util::zip_has(&mut zip, "word/styles.xml") {
        let bytes = util::read_zip_entry(&mut zip, "word/styles.xml")?;
        parse_styles(&util::decode_text(&bytes))
    } else {
        HashMap::new()
    };
    let meta = if util::zip_has(&mut zip, "docProps/core.xml") {
        let bytes = util::read_zip_entry(&mut zip, "docProps/core.xml")?;
        parse_core(&util::decode_text(&bytes))?
    } else {
        Meta::default()
    };
    let paragraphs = parse_document(&document, &styles)?;
    let chapters = build_chapters(&paragraphs);
    if chapters.is_empty() {
        return Err(ImportError::invalid("DOCX contains no readable text"));
    }
    let title = if meta.title.trim().is_empty() {
        chapters
            .first()
            .map(|chapter| chapter.title.clone())
            .unwrap_or_default()
    } else {
        meta.title
    };
    Ok(ImportedBook {
        title,
        author: meta.author,
        language: if meta.language.is_empty() {
            "uk".to_string()
        } else {
            meta.language
        },
        format: BookFormat::Docx,
        chapters,
    })
}

fn parse_styles(xml: &str) -> HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut styles = HashMap::new();
    let mut current_id: Option<String> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                match event.local_name().as_ref() {
                    "style" => {
                        current_id = util::attr(&event, "styleId");
                    }
                    "name" => {
                        if let Some(id) = current_id.clone() {
                            if let Some(name) = util::attr(&event, "val") {
                                styles.insert(id, name);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(event)) => {
                if event.local_name().as_ref() == "style" {
                    current_id = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    styles
}

fn parse_core(xml: &str) -> Result<Meta, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut meta = Meta::default();
    let mut field: Option<&'static str> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                match event.local_name().as_ref() {
                    "title" => field = Some("title"),
                    "creator" => field = Some("creator"),
                    "language" => field = Some("language"),
                    _ => {}
                }
            }
            Ok(Event::Text(text)) => {
                if let Some(field) = field {
                    let value = util::normalize_inline(&util::text_content(&text));
                    if !value.is_empty() {
                        match field {
                            "title" if meta.title.is_empty() => meta.title = value,
                            "creator" if meta.author.is_empty() => meta.author = value,
                            "language" if meta.language.is_empty() => meta.language = value,
                            _ => {}
                        }
                    }
                }
            }
            Ok(Event::End(event)) => {
                if matches!(
                    event.local_name().as_ref(),
                    "title" | "creator" | "language"
                ) {
                    field = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }
    Ok(meta)
}

fn parse_document(
    xml: &str,
    styles: &HashMap<String, String>,
) -> Result<Vec<Paragraph>, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();
    let mut paragraphs = Vec::new();
    let mut current: Option<Paragraph> = None;
    let mut in_text = false;
    let mut style: Option<String> = None;
    let mut outline: Option<usize> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                "p" => {
                    current = Some(Paragraph {
                        text: String::new(),
                        heading_level: None,
                    });
                    style = None;
                    outline = None;
                }
                "pStyle" => style = util::attr(&event, "val"),
                "outlineLvl" => {
                    outline = util::attr(&event, "val").and_then(|value| value.parse().ok())
                }
                "t" => in_text = true,
                "tab" => {
                    if let Some(paragraph) = current.as_mut() {
                        paragraph.text.push('\t');
                    }
                }
                "br" => {
                    if let Some(paragraph) = current.as_mut() {
                        paragraph.text.push('\n');
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(event)) => match event.local_name().as_ref() {
                "pStyle" => style = util::attr(&event, "val"),
                "outlineLvl" => {
                    outline = util::attr(&event, "val").and_then(|value| value.parse().ok())
                }
                "tab" => {
                    if let Some(paragraph) = current.as_mut() {
                        paragraph.text.push('\t');
                    }
                }
                "br" => {
                    if let Some(paragraph) = current.as_mut() {
                        paragraph.text.push('\n');
                    }
                }
                _ => {}
            },
            Ok(Event::Text(text)) => {
                if in_text {
                    if let Some(paragraph) = current.as_mut() {
                        paragraph.text.push_str(&util::text_content(&text));
                    }
                }
            }
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                "t" => in_text = false,
                "p" => {
                    if let Some(mut paragraph) = current.take() {
                        paragraph.text = normalize_paragraph(&paragraph.text);
                        if !paragraph.text.is_empty() {
                            paragraph.heading_level =
                                heading_level(outline, style.as_deref(), styles);
                            paragraphs.push(paragraph);
                        }
                    }
                    style = None;
                    outline = None;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }
    Ok(paragraphs)
}

fn heading_level(
    outline: Option<usize>,
    style: Option<&str>,
    styles: &HashMap<String, String>,
) -> Option<usize> {
    if let Some(level) = outline {
        if level < 9 {
            return Some(level + 1);
        }
    }
    let style = style?;
    let name = styles.get(style).map(String::as_str).unwrap_or(style);
    let name = name.trim().to_ascii_lowercase();
    if let Some(rest) = name.strip_prefix("heading") {
        let rest = rest.trim();
        if let Ok(level) = rest.parse::<usize>() {
            if (1..=9).contains(&level) {
                return Some(level);
            }
        }
    }
    let digits: String = name.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if name.contains("heading") || style.to_ascii_lowercase().contains("heading") {
        if let Ok(level) = digits.parse::<usize>() {
            if (1..=9).contains(&level) {
                return Some(level);
            }
        }
    }
    None
}

fn build_chapters(paragraphs: &[Paragraph]) -> Vec<ImportedChapter> {
    if paragraphs.is_empty() {
        return Vec::new();
    }
    let split_level = choose_split_level(paragraphs);
    let Some(split_level) = split_level else {
        let title = paragraphs
            .iter()
            .find(|paragraph| paragraph.heading_level.is_some())
            .map(|paragraph| paragraph.text.clone())
            .unwrap_or_default();
        let text = paragraphs
            .iter()
            .map(|paragraph| paragraph.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        return vec![ImportedChapter { title, text }];
    };

    let mut chapters = Vec::new();
    let mut current_title = String::new();
    let mut current_text = String::new();
    let mut started = false;

    for paragraph in paragraphs {
        if paragraph.heading_level == Some(split_level) {
            if started && !current_text.trim().is_empty() {
                chapters.push(ImportedChapter {
                    title: current_title.clone(),
                    text: current_text.trim().to_string(),
                });
            } else if !started && !current_text.trim().is_empty() {
                chapters.push(ImportedChapter {
                    title: "Передмова".to_string(),
                    text: current_text.trim().to_string(),
                });
            }
            current_title = paragraph.text.clone();
            current_text = format!("{}\n\n", paragraph.text);
            started = true;
        } else {
            current_text.push_str(&paragraph.text);
            current_text.push_str("\n\n");
        }
    }
    if started && !current_text.trim().is_empty() {
        chapters.push(ImportedChapter {
            title: current_title,
            text: current_text.trim().to_string(),
        });
    } else if !current_text.trim().is_empty() {
        chapters.push(ImportedChapter {
            title: "Передмова".to_string(),
            text: current_text.trim().to_string(),
        });
    }
    chapters
}

fn choose_split_level(paragraphs: &[Paragraph]) -> Option<usize> {
    let mut counts = [0usize; 3];
    for paragraph in paragraphs {
        if let Some(level) = paragraph.heading_level {
            if (1..=3).contains(&level) {
                counts[level - 1] += 1;
            }
        }
    }
    if counts[0] >= 2 {
        Some(1)
    } else if counts[1] >= 2 {
        Some(2)
    } else if counts[2] >= 2 {
        Some(3)
    } else {
        None
    }
}

fn normalize_paragraph(raw: &str) -> String {
    let mut lines = Vec::new();
    for line in raw.lines() {
        let line = util::normalize_inline(line);
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_heading_levels_from_style_and_outline() {
        let mut styles = HashMap::new();
        styles.insert("Heading1".to_string(), "heading 1".to_string());
        assert_eq!(heading_level(None, Some("Heading1"), &styles), Some(1));
        assert_eq!(heading_level(Some(1), None, &styles), Some(2));
        assert_eq!(heading_level(None, Some("Normal"), &styles), None);
    }

    #[test]
    fn builds_chapters_from_headings() {
        let paragraphs = vec![
            Paragraph {
                text: "Вступ".to_string(),
                heading_level: None,
            },
            Paragraph {
                text: "Розділ 1".to_string(),
                heading_level: Some(1),
            },
            Paragraph {
                text: "Перший текст.".to_string(),
                heading_level: None,
            },
            Paragraph {
                text: "Розділ 2".to_string(),
                heading_level: Some(1),
            },
            Paragraph {
                text: "Другий текст.".to_string(),
                heading_level: None,
            },
        ];
        let chapters = build_chapters(&paragraphs);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].title, "Передмова");
        assert_eq!(chapters[1].title, "Розділ 1");
        assert!(chapters[1].text.contains("Перший текст."));
        assert_eq!(chapters[2].title, "Розділ 2");
    }
}
