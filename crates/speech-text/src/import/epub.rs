//! EPUB 2/3: container.xml -> OPF -> spine -> XHTML, з NCX/nav-заголовками.

use std::collections::{HashMap, HashSet};

use quick_xml::events::Event;
use quick_xml::Reader;

use super::util;
use super::{BookFormat, ImportError, ImportedBook, ImportedChapter};

const MIN_CHAPTER_CHARS: usize = 20;

struct ManifestItem {
    href: String,
    media_type: String,
    properties: String,
}

#[derive(Default)]
struct Package {
    title: String,
    authors: Vec<String>,
    language: String,
    manifest: HashMap<String, ManifestItem>,
    spine: Vec<String>,
}

struct RawChapter {
    title: String,
    anchor: Option<String>,
    from_heading: bool,
    text: String,
}

pub(crate) fn import(bytes: &[u8]) -> Result<ImportedBook, ImportError> {
    let mut zip = util::open_zip(bytes)?;
    let container = util::read_zip_entry(&mut zip, "META-INF/container.xml")?;
    let container = util::decode_text(&container);
    let opf_path = parse_container(&container)?;
    let opf = util::read_zip_entry(&mut zip, &opf_path)
        .map_err(|_| ImportError::invalid(format!("missing OPF: {opf_path}")))?;
    let opf = util::decode_text(&opf);
    let package = parse_opf(&opf)?;

    let mut nav_entries: Vec<(String, String)> = Vec::new();
    if let Some(item) = package.manifest.values().find(|item| {
        item.properties
            .split_whitespace()
            .any(|value| value == "nav")
    }) {
        let nav_path = resolve_href(&opf_path, &item.href);
        if let Ok(bytes) = util::read_zip_entry(&mut zip, &nav_path) {
            nav_entries.extend(parse_nav(&util::decode_text(&bytes), &nav_path));
        }
    }
    if let Some(item) = package
        .manifest
        .values()
        .find(|item| item.media_type == "application/x-dtbncx+xml")
    {
        let ncx_path = resolve_href(&opf_path, &item.href);
        if let Ok(bytes) = util::read_zip_entry(&mut zip, &ncx_path) {
            nav_entries.extend(parse_ncx(&util::decode_text(&bytes), &ncx_path));
        }
    }

    let mut nav_titles: HashMap<String, String> = HashMap::new();
    let mut nav_order: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (key, title) in nav_entries {
        nav_titles
            .entry(key.clone())
            .or_insert_with(|| title.clone());
        let file = key.split('#').next().unwrap_or(&key).to_string();
        nav_order.entry(file).or_default().push((key, title));
    }

    let mut chapters = Vec::new();
    let mut used_nav: HashSet<String> = HashSet::new();
    for idref in &package.spine {
        let Some(item) = package.manifest.get(idref) else {
            continue;
        };
        if !matches!(
            item.media_type.as_str(),
            "application/xhtml+xml" | "text/html"
        ) {
            continue;
        }
        let href = resolve_href(&opf_path, &item.href);
        let bytes = match util::read_zip_entry(&mut zip, &href) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let html = util::decode_text(&bytes);
        let raw_chapters = html_to_chapters(&html);
        let mut cursor = 0usize;
        for mut chapter in raw_chapters {
            if chapter.title.is_empty() {
                let anchor_key = chapter
                    .anchor
                    .as_ref()
                    .map(|anchor| format!("{href}#{anchor}"));
                if let Some(title) = anchor_key
                    .as_ref()
                    .and_then(|key| nav_titles.get(key))
                    .cloned()
                {
                    chapter.title = clean_title(&title);
                    if let Some(key) = anchor_key {
                        used_nav.insert(key);
                    }
                } else if let Some(entries) = nav_order.get(&href) {
                    while cursor < entries.len() && used_nav.contains(&entries[cursor].0) {
                        cursor += 1;
                    }
                    if let Some((key, title)) = entries.get(cursor) {
                        chapter.title = clean_title(title);
                        used_nav.insert(key.clone());
                    }
                }
            }
            if chapter.title.is_empty() {
                if let Some(title) = nav_titles.get(&href) {
                    chapter.title = clean_title(title);
                }
            }
            if chapter.title.is_empty() {
                chapter.title = format!("Розділ {}", chapters.len() + 1);
            }
            chapters.push(ImportedChapter {
                title: chapter.title,
                text: chapter.text,
            });
        }
    }

    if chapters.is_empty() {
        return Err(ImportError::invalid("EPUB contains no readable chapters"));
    }
    let mut title = package.title;
    if title.trim().is_empty() {
        title = chapters[0].title.clone();
    }
    Ok(ImportedBook {
        title,
        author: package.authors.join(", "),
        language: package.language,
        format: BookFormat::Epub,
        chapters,
    })
}

fn parse_container(xml: &str) -> Result<String, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                if event.local_name().as_ref() == "rootfile" {
                    if let Some(path) = util::attr(&event, "full-path") {
                        if !path.trim().is_empty() {
                            return Ok(path);
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }
    Err(ImportError::invalid("container.xml has no rootfile"))
}

fn parse_opf(xml: &str) -> Result<Package, ImportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut package = Package::default();
    let mut buf = Vec::new();
    let mut capture: Option<&'static str> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                match event.local_name().as_ref() {
                    "title" => capture = Some("title"),
                    "creator" => capture = Some("author"),
                    "language" => capture = Some("language"),
                    "item" => {
                        let id = util::attr(&event, "id").unwrap_or_default();
                        let item = ManifestItem {
                            href: util::attr(&event, "href").unwrap_or_default(),
                            media_type: util::attr(&event, "media-type").unwrap_or_default(),
                            properties: util::attr(&event, "properties").unwrap_or_default(),
                        };
                        if !id.is_empty() {
                            package.manifest.insert(id, item);
                        }
                    }
                    "itemref" => {
                        let linear = util::attr(&event, "linear").unwrap_or_default();
                        if !linear.eq_ignore_ascii_case("no") {
                            if let Some(idref) = util::attr(&event, "idref") {
                                package.spine.push(idref);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(text)) => {
                if let Some(field) = capture {
                    let value = util::text_content(&text);
                    let value = util::normalize_inline(&value);
                    if !value.is_empty() {
                        match field {
                            "title" if package.title.is_empty() => package.title = value,
                            "author" if !package.authors.iter().any(|a| a == &value) => {
                                package.authors.push(value)
                            }
                            "language" if package.language.is_empty() => package.language = value,
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
                    capture = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(ImportError::Xml(error.to_string())),
            _ => {}
        }
        buf.clear();
    }
    if package.language.is_empty() {
        package.language = "uk".to_string();
    }
    Ok(package)
}

fn parse_ncx(xml: &str, ncx_path: &str) -> Vec<(String, String)> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut capture = false;
    let mut entries = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                "navPoint" => labels.push(None),
                "text" => capture = true,
                "content" => push_ncx_entry(&event, ncx_path, &labels, &mut entries),
                _ => {}
            },
            Ok(Event::Empty(event)) => {
                if event.local_name().as_ref() == "content" {
                    push_ncx_entry(&event, ncx_path, &labels, &mut entries);
                }
            }
            Ok(Event::Text(text)) => {
                if capture {
                    let value = util::text_content(&text);
                    if let Some(label) = labels.last_mut() {
                        match label {
                            Some(label) => label.push_str(&value),
                            None => *label = Some(value),
                        }
                    }
                }
            }
            Ok(Event::End(event)) => match event.local_name().as_ref() {
                "navPoint" => {
                    labels.pop();
                }
                "text" => capture = false,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    entries
}

fn push_ncx_entry(
    event: &quick_xml::events::BytesStart<'_>,
    ncx_path: &str,
    labels: &[Option<String>],
    entries: &mut Vec<(String, String)>,
) {
    let Some(src) = util::attr(event, "src") else {
        return;
    };
    let Some(Some(label)) = labels.last() else {
        return;
    };
    let (file, anchor) = resolve_ref(ncx_path, &src);
    let key = match anchor {
        Some(anchor) => format!("{file}#{anchor}"),
        None => file,
    };
    entries.push((key, util::normalize_inline(label)));
}

fn parse_nav(xml: &str, nav_path: &str) -> Vec<(String, String)> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut entries = Vec::new();
    let mut nav_depth = 0usize;
    let mut in_toc = false;
    let mut capture: Option<(String, String)> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => {
                let name = event.local_name().as_ref().to_string();
                if name == "nav" {
                    nav_depth += 1;
                    if nav_depth == 1 {
                        in_toc = util::attr(&event, "type")
                            .map(|value| value.split_whitespace().any(|part| part == "toc"))
                            .unwrap_or(false);
                    }
                } else if name == "a" && in_toc && capture.is_none() {
                    if let Some(href) = util::attr(&event, "href") {
                        let (file, anchor) = resolve_ref(nav_path, &href);
                        let key = match anchor {
                            Some(anchor) => format!("{file}#{anchor}"),
                            None => file,
                        };
                        capture = Some((key, String::new()));
                    }
                }
            }
            Ok(Event::Text(text)) => {
                if let Some((_, title)) = capture.as_mut() {
                    title.push_str(&util::text_content(&text));
                }
            }
            Ok(Event::End(event)) => {
                let name = event.local_name().as_ref().to_string();
                if name == "a" {
                    if let Some((key, title)) = capture.take() {
                        let title = util::normalize_inline(&title);
                        if !title.is_empty() {
                            entries.push((key, title));
                        }
                    }
                } else if name == "nav" {
                    nav_depth = nav_depth.saturating_sub(1);
                    if nav_depth == 0 {
                        in_toc = false;
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    entries
}

fn html_to_chapters(html: &str) -> Vec<RawChapter> {
    let split_level = choose_split_level(html);
    let mut reader = Reader::from_str(html);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut skip_depth: Option<usize> = None;
    let mut segments: Vec<RawChapter> = vec![RawChapter {
        title: String::new(),
        anchor: None,
        from_heading: false,
        text: String::new(),
    }];
    let mut heading: Option<(String, String)> = None;
    let mut heading_anchor: Option<String> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => {
                let name = event.local_name().as_ref().to_string();
                stack.push(name.clone());
                if skip_depth.is_none()
                    && matches!(
                        name.as_str(),
                        "script" | "style" | "head" | "svg" | "math" | "nav"
                    )
                {
                    skip_depth = Some(stack.len());
                }
                if skip_depth.is_some() {
                    continue;
                }
                if name == "br" {
                    segments.last_mut().unwrap().text.push('\n');
                }
                if is_block(&name) {
                    segments.last_mut().unwrap().text.push_str("\n\n");
                }
                if split_level == Some(name.as_str()) {
                    let current = segments.last().unwrap();
                    if !current.text.trim().is_empty() || current.from_heading {
                        segments.push(RawChapter {
                            title: String::new(),
                            anchor: None,
                            from_heading: false,
                            text: String::new(),
                        });
                    }
                    heading = Some((name.clone(), String::new()));
                    heading_anchor = util::attr(&event, "id");
                } else if matches!(name.as_str(), "h1" | "h2" | "h3") {
                    heading = Some((name.clone(), String::new()));
                } else if heading.is_some() && name == "a" && heading_anchor.is_none() {
                    heading_anchor = util::attr(&event, "id");
                }
            }
            Ok(Event::Empty(event)) => {
                if skip_depth.is_none() && event.local_name().as_ref() == "br" {
                    segments.last_mut().unwrap().text.push('\n');
                }
            }
            Ok(Event::Text(text)) => {
                if skip_depth.is_some() {
                    continue;
                }
                let value = util::text_content(&text);
                if let Some((_, heading_text)) = heading.as_mut() {
                    heading_text.push_str(&value);
                } else {
                    segments.last_mut().unwrap().text.push_str(&value);
                }
            }
            Ok(Event::CData(text)) => {
                if skip_depth.is_some() {
                    continue;
                }
                let value = text.xml10_content().into_owned();
                if let Some((_, heading_text)) = heading.as_mut() {
                    heading_text.push_str(&value);
                } else {
                    segments.last_mut().unwrap().text.push_str(&value);
                }
            }
            Ok(Event::End(event)) => {
                let name = event.local_name().as_ref().to_string();
                if let Some((tag, heading_text)) = heading.take() {
                    if tag == name {
                        let title = clean_title(&heading_text);
                        let segment = segments.last_mut().unwrap();
                        if !title.is_empty() && segment.title.is_empty() {
                            segment.title = title;
                        }
                        if !heading_text.trim().is_empty() {
                            segment.text.push_str(&heading_text);
                            segment.text.push_str("\n\n");
                        }
                        if split_level == Some(name.as_str()) {
                            segment.anchor = heading_anchor.take();
                            segment.from_heading = true;
                        }
                    } else {
                        heading = Some((tag, heading_text));
                    }
                }
                if skip_depth == Some(stack.len()) {
                    skip_depth = None;
                }
                stack.pop();
                if skip_depth.is_none() && is_block(&name) {
                    segments.last_mut().unwrap().text.push_str("\n\n");
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    let mut chapters = Vec::new();
    for segment in segments {
        let text = normalize_text(&segment.text);
        if text.chars().count() < MIN_CHAPTER_CHARS {
            continue;
        }
        chapters.push(RawChapter {
            title: segment.title,
            anchor: segment.anchor,
            from_heading: segment.from_heading,
            text,
        });
    }
    chapters
}

fn choose_split_level(html: &str) -> Option<&'static str> {
    let mut counts = [0usize; 3];
    let mut reader = Reader::from_str(html);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                "h1" => counts[0] += 1,
                "h2" => counts[1] += 1,
                "h3" => counts[2] += 1,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    if counts[0] >= 2 {
        Some("h1")
    } else if counts[1] >= 2 {
        Some("h2")
    } else if counts[2] >= 2 {
        Some("h3")
    } else {
        None
    }
}

fn is_block(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "section"
            | "article"
            | "blockquote"
            | "pre"
            | "ul"
            | "ol"
            | "li"
            | "tr"
            | "table"
            | "figure"
            | "figcaption"
            | "hr"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
    )
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

fn clean_title(raw: &str) -> String {
    let title = util::normalize_inline(raw);
    for needle in ["CHAPTER ", "Chapter ", "РОЗДІЛ ", "Розділ "] {
        if let Some(index) = title.rfind(needle) {
            if index > 0 {
                let prefix = title[..index].trim_end();
                if prefix.ends_with(['.', '!', '?', '”', '"']) {
                    return title[index..].trim().to_string();
                }
            }
        }
    }
    title
}

fn resolve_href(base_file: &str, href: &str) -> String {
    resolve_ref(base_file, href).0
}

fn resolve_ref(base_file: &str, href: &str) -> (String, Option<String>) {
    let (href, anchor) = match href.split_once('#') {
        Some((path, anchor)) => (path, Some(anchor.to_string())),
        None => (href, None),
    };
    let href = percent_decode(href);
    let base_dir = base_file.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let mut parts: Vec<String> = base_dir
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_string)
        .collect();
    for segment in href.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_string()),
        }
    }
    (parts.join("/"), anchor)
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = &input[index + 1..index + 3];
            if let Ok(value) = u8::from_str_radix(hex, 16) {
                out.push(value);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_paths_and_anchors() {
        assert_eq!(
            resolve_ref("OEBPS/content.opf", "Text/ch1.xhtml#part"),
            ("OEBPS/Text/ch1.xhtml".to_string(), Some("part".to_string()))
        );
        assert_eq!(
            resolve_ref("OEBPS/Text/ch1.xhtml", "../Images/../ch2.xhtml"),
            ("OEBPS/ch2.xhtml".to_string(), None)
        );
        assert_eq!(
            resolve_href("content.opf", "Text/ch%201.xhtml"),
            "Text/ch 1.xhtml"
        );
    }

    #[test]
    fn splits_long_documents_by_headings() {
        let html = r#"<html><body>
            <h1>Книга</h1>
            <p>Вступ.</p>
            <h2 id="c1">Розділ 1</h2>
            <p>Перший текст.</p>
            <h2 id="c2">Розділ 2</h2>
            <p>Другий текст.</p>
        </body></html>"#;
        let chapters = html_to_chapters(html);
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "Розділ 1");
        assert!(chapters[0].text.contains("Перший текст."));
        assert_eq!(chapters[1].title, "Розділ 2");
    }

    #[test]
    fn strips_caption_prefix_from_title() {
        assert_eq!(
            clean_title("I hope Mr. Bingley will like it. CHAPTER II."),
            "CHAPTER II."
        );
        assert_eq!(
            clean_title("CHAPTER I. Down the Rabbit-Hole"),
            "CHAPTER I. Down the Rabbit-Hole"
        );
    }
}
