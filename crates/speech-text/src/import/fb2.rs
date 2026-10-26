//! FB2 (FictionBook 2): XML, часто Windows-1251.

use quick_xml::events::Event;
use quick_xml::Reader;

use super::util;
use super::{BookFormat, ImportError, ImportedBook, ImportedChapter};

#[derive(Default)]
struct Meta {
    title: String,
    authors: Vec<String>,
    language: String,
}

struct RawChapter {
    title: String,
    text: String,
}

pub(crate) fn import(bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let xml = util::decode_text(bytes);
    let meta = parse_meta(&xml)?;
    let chapters = parse_body(&xml)?;
    if chapters.is_empty() {
        return Err(ImportError::invalid("FB2 contains no readable text"));
    }
    Ok(ImportedBook {
        title: meta.title,
        author: meta.authors.join(", "),
        language: if meta.language.is_empty() {
            "uk".to_string()
        } else {
            meta.language
        },
        format: BookFormat::Fb2,
        chapters: chapters
            .into_iter()
            .enumerate()
            .map(|(index, chapter)| ImportedChapter {
                title: if chapter.title.is_empty() {
                    format!("Розділ {}", index + 1)
                } else {
                    chapter.title
                },
                text: chapter.text,
            })
            .filter(|chapter| !chapter.text.trim().is_empty())
            .collect(),
    })
}

fn parse_meta(xml: &str) -> Result<Meta, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut meta = Meta::default();
    let mut in_title_info = false;
    let mut field: Option<&'static str> = None;
    let mut author: Option<Author> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                match event.local_name().as_ref() {
                    "title-info" => in_title_info = true,
                    "book-title" if in_title_info => field = Some("title"),
                    "lang" if in_title_info => field = Some("lang"),
                    "author" if in_title_info => author = Some(Author::default()),
                    "first-name" if author.is_some() => field = Some("first"),
                    "middle-name" if author.is_some() => field = Some("middle"),
                    "last-name" if author.is_some() => field = Some("last"),
                    _ => {}
                }
            }
            Ok(Event::Text(text)) => {
                if let Some(field) = field {
                    let value = util::normalize_inline(&util::text_content(&text));
                    if !value.is_empty() {
                        match field {
                            "title" if meta.title.is_empty() => meta.title = value,
                            "lang" if meta.language.is_empty() => meta.language = value,
                            "first" => {
                                if let Some(author) = author.as_mut() {
                                    author.first = value;
                                }
                            }
                            "middle" => {
                                if let Some(author) = author.as_mut() {
                                    author.middle = value;
                                }
                            }
                            "last" => {
                                if let Some(author) = author.as_mut() {
                                    author.last = value;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                "title-info" => in_title_info = false,
                "book-title" | "lang" | "first-name" | "middle-name" | "last-name" => {
                    field = None;
                }
                "author" => {
                    if let Some(author) = author.take() {
                        let name = author.full_name();
                        if !name.is_empty() {
                            meta.authors.push(name);
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }
    Ok(meta)
}

#[derive(Default)]
struct Author {
    first: String,
    middle: String,
    last: String,
}

impl Author {
    fn full_name(&self) -> String {
        [self.first.trim(), self.middle.trim(), self.last.trim()]
            .iter()
            .filter(|part| !part.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn parse_body(xml: &str) -> Result<Vec<RawChapter>, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();
    let mut body_depth = 0usize;
    let mut skip_body = false;
    let mut skip_binary = false;
    let mut section_depth = 0usize;
    let mut title_depth = 0usize;
    let mut title_buf = String::new();
    let mut current: Option<RawChapter> = None;
    let mut body_text = String::new();
    let mut chapters = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => {
                let name = event.local_name().as_ref().to_string();
                match name.as_str() {
                    "body" => {
                        if body_depth == 0 {
                            skip_body = util::attr(&event, "name")
                                .map(|name| name.eq_ignore_ascii_case("notes"))
                                .unwrap_or(false);
                        }
                        body_depth += 1;
                    }
                    "binary" => skip_binary = true,
                    "section" if !skip_body && body_depth > 0 && !skip_binary => {
                        if section_depth == 0 {
                            current = Some(RawChapter {
                                title: String::new(),
                                text: String::new(),
                            });
                        }
                        section_depth += 1;
                    }
                    "title" if !skip_body && section_depth > 0 && !skip_binary => {
                        title_depth += 1;
                        if title_depth == 1 {
                            title_buf.clear();
                        }
                    }
                    "p" | "subtitle" | "v" | "text-author" | "date" | "epigraph" | "cite"
                    | "poem" | "stanza"
                        if !skip_body && section_depth > 0 && !skip_binary =>
                    {
                        if let Some(current) = current.as_mut() {
                            current.text.push_str("\n\n");
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(event)) => {
                if event.local_name().as_ref() == "empty-line" && section_depth > 0 {
                    if let Some(current) = current.as_mut() {
                        current.text.push_str("\n\n");
                    }
                }
            }
            Ok(Event::Text(text)) => {
                if skip_body || skip_binary {
                    continue;
                }
                let value = util::text_content(&text);
                if title_depth > 0 {
                    title_buf.push_str(&value);
                } else if section_depth > 0 {
                    if let Some(current) = current.as_mut() {
                        current.text.push_str(&value);
                    }
                } else if body_depth > 0 {
                    body_text.push_str(&value);
                }
            }
            Ok(Event::CData(text)) => {
                if skip_body || skip_binary {
                    continue;
                }
                let value = text.xml10_content().into_owned();
                if title_depth > 0 {
                    title_buf.push_str(&value);
                } else if section_depth > 0 {
                    if let Some(current) = current.as_mut() {
                        current.text.push_str(&value);
                    }
                } else if body_depth > 0 {
                    body_text.push_str(&value);
                }
            }
            Ok(Event::End(event)) => {
                let name = event.local_name().as_ref().to_string();
                match name.as_str() {
                    "body" => {
                        body_depth = body_depth.saturating_sub(1);
                        if body_depth == 0 {
                            skip_body = false;
                        }
                    }
                    "binary" => skip_binary = false,
                    "section" if !skip_body && section_depth > 0 && !skip_binary => {
                        section_depth = section_depth.saturating_sub(1);
                        if section_depth == 0 {
                            if let Some(mut chapter) = current.take() {
                                chapter.text = normalize_text(&chapter.text);
                                chapter.title = util::normalize_inline(&chapter.title);
                                if !chapter.text.is_empty() {
                                    chapters.push(chapter);
                                }
                            }
                        }
                    }
                    "title" if title_depth > 0 => {
                        title_depth -= 1;
                        if title_depth == 0 {
                            let title = util::normalize_inline(&title_buf);
                            if let Some(current) = current.as_mut() {
                                if section_depth == 1 && current.title.is_empty() {
                                    current.title = title;
                                } else if !title.is_empty() {
                                    current.text.push_str(&title);
                                    current.text.push_str("\n\n");
                                }
                            }
                        }
                    }
                    "p" | "subtitle" | "v" | "text-author" | "date" | "epigraph" | "cite"
                    | "poem" | "stanza"
                        if !skip_body && section_depth > 0 && !skip_binary =>
                    {
                        if let Some(current) = current.as_mut() {
                            current.text.push_str("\n\n");
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }

    if chapters.is_empty() {
        let text = normalize_text(&body_text);
        if !text.is_empty() {
            chapters.push(RawChapter {
                title: String::new(),
                text,
            });
        }
    }
    Ok(chapters)
}

fn normalize_text(raw: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in raw.lines() {
        let line = util::normalize_inline(line);
        if line.is_empty() {
            if lines.last().is_some_and(|line| !line.is_empty()) {
                lines.push(String::new());
            }
        } else {
            lines.push(line);
        }
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FB2: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<FictionBook>
  <description>
    <title-info>
      <book-title>Тестова книга</book-title>
      <author><first-name>Тарас</first-name><last-name>Шевченко</last-name></author>
      <lang>uk</lang>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Розділ 1</p></title>
      <p>Перший текст.</p>
      <section><title><p>Підрозділ</p></title><p>Вкладений текст.</p></section>
    </section>
    <section>
      <title><p>Розділ 2</p></title>
      <p>Другий текст.</p>
    </section>
  </body>
</FictionBook>"#;

    #[test]
    fn parses_metadata_and_sections() {
        let meta = parse_meta(FB2).unwrap();
        assert_eq!(meta.title, "Тестова книга");
        assert_eq!(meta.authors, vec!["Тарас Шевченко"]);
        assert_eq!(meta.language, "uk");

        let chapters = parse_body(FB2).unwrap();
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "Розділ 1");
        assert!(chapters[0].text.contains("Перший текст."));
        assert!(chapters[0].text.contains("Підрозділ"));
        assert!(chapters[0].text.contains("Вкладений текст."));
        assert_eq!(chapters[1].title, "Розділ 2");
    }

    #[test]
    fn decodes_windows_1251() {
        let xml = FB2.replace("encoding=\"utf-8\"", "encoding=\"windows-1251\"");
        let (encoded, _, _) = encoding_rs::WINDOWS_1251.encode(&xml);
        let decoded = util::decode_text(&encoded);
        assert!(decoded.contains("Тестова книга"));
    }
}
