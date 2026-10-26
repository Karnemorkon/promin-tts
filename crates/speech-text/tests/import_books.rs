#![cfg(feature = "import")]

//! Інтеграційні тести імпорту книг на синтетичних фікстурах.

use std::io::{Cursor, Write};

use speech_text::import::{import_book, import_bytes, BookFormat, ImportError};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, data) in entries {
        writer.start_file(*name, options).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

const CONTAINER: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#;

fn epub3_fixture() -> Vec<u8> {
    let opf = r#"<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Тестова EPUB3</dc:title>
    <dc:creator>Автор Тестовий</dc:creator>
    <dc:language>uk</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"#;
    let nav = r#"<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body><nav epub:type="toc"><ol>
  <li><a href="ch1.xhtml#c1">Розділ 1</a></li>
  <li><a href="ch2.xhtml#c2">Розділ 2</a></li>
</ol></nav></body></html>"#;
    let ch1 = r#"<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 id="c1">Розділ 1</h2><p>Перший абзац.</p><p>Другий абзац.</p>
</body></html>"#;
    let ch2 = r#"<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 id="c2">Розділ 2</h2><p>Третій абзац.</p>
</body></html>"#;
    build_zip(&[
        ("META-INF/container.xml", CONTAINER.as_bytes()),
        ("OEBPS/content.opf", opf.as_bytes()),
        ("OEBPS/nav.xhtml", nav.as_bytes()),
        ("OEBPS/ch1.xhtml", ch1.as_bytes()),
        ("OEBPS/ch2.xhtml", ch2.as_bytes()),
    ])
}

fn epub2_fixture() -> Vec<u8> {
    let opf = r#"<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Тестова EPUB2</dc:title>
    <dc:creator>Другий Автор</dc:creator>
    <dc:language>uk</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="c1"/></spine>
</package>"#;
    let ncx = r#"<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="test"/></head>
  <docTitle><text>Тестова EPUB2</text></docTitle>
  <navMap>
    <navPoint id="n1" playOrder="1">
      <navLabel><text>I hope Mr. Bingley will like it. CHAPTER II.</text></navLabel>
      <content src="ch1.xhtml#c1"/>
    </navPoint>
    <navPoint id="n2" playOrder="2">
      <navLabel><text>CHAPTER III</text></navLabel>
      <content src="ch1.xhtml#c2"/>
    </navPoint>
  </navMap>
</ncx>"#;
    let ch1 = r#"<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 id="c1"><img src="heading.png" alt=""/></h2><p>Текст перший, довший за двадцять символів.</p>
<h2 id="c2">CHAPTER III</h2><p>Текст другий, теж достатньо довгий.</p>
</body></html>"#;
    build_zip(&[
        ("META-INF/container.xml", CONTAINER.as_bytes()),
        ("OEBPS/content.opf", opf.as_bytes()),
        ("OEBPS/toc.ncx", ncx.as_bytes()),
        ("OEBPS/ch1.xhtml", ch1.as_bytes()),
    ])
}

#[test]
fn imports_epub3_with_nav() {
    let bytes = epub3_fixture();
    assert_eq!(
        speech_text::import::detect_format("book", &bytes),
        Some(BookFormat::Epub)
    );
    let book = import_book("book.epub", &bytes).unwrap();
    assert_eq!(book.title, "Тестова EPUB3");
    assert_eq!(book.author, "Автор Тестовий");
    assert_eq!(book.language, "uk");
    assert_eq!(book.format, BookFormat::Epub);
    assert_eq!(book.chapters.len(), 2);
    assert_eq!(book.chapters[0].title, "Розділ 1");
    assert!(book.chapters[0].text.contains("Перший абзац."));
    assert!(book.chapters[0].text.contains("Другий абзац."));
    assert_eq!(book.chapters[1].title, "Розділ 2");
    assert!(book.chapters[1].text.contains("Третій абзац."));
}

#[test]
fn imports_epub2_ncx_and_cleans_image_headings() {
    let bytes = epub2_fixture();
    let book = import_bytes(BookFormat::Epub, &bytes).unwrap();
    assert_eq!(book.title, "Тестова EPUB2");
    assert_eq!(book.author, "Другий Автор");
    assert_eq!(book.chapters.len(), 2);
    assert_eq!(book.chapters[0].title, "CHAPTER II.");
    assert!(book.chapters[0].text.contains("Текст перший"));
    assert_eq!(book.chapters[1].title, "CHAPTER III");
    assert!(book.chapters[1].text.contains("Текст другий"));
}

const FB2: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Кобзар тест</book-title>
      <author><first-name>Тарас</first-name><last-name>Шевченко</last-name></author>
      <lang>uk</lang>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Розділ 1</p></title>
      <p>Думи мої, думи мої.</p>
      <section><title><p>Підрозділ</p></title><p>Вкладений текст.</p></section>
    </section>
    <section>
      <title><p>Розділ 2</p></title>
      <p>Літа орел, літа сизий.</p>
    </section>
  </body>
</FictionBook>"#;

#[test]
fn imports_fb2_utf8() {
    let book = import_book("kobzar.fb2", FB2.as_bytes()).unwrap();
    assert_eq!(book.title, "Кобзар тест");
    assert_eq!(book.author, "Тарас Шевченко");
    assert_eq!(book.language, "uk");
    assert_eq!(book.chapters.len(), 2);
    assert_eq!(book.chapters[0].title, "Розділ 1");
    assert!(book.chapters[0].text.contains("Думи мої"));
    assert!(book.chapters[0].text.contains("Вкладений текст."));
    assert_eq!(book.chapters[1].title, "Розділ 2");
}

#[test]
fn imports_fb2_windows_1251() {
    let xml = FB2.replace("encoding=\"utf-8\"", "encoding=\"windows-1251\"");
    let (encoded, _, had_errors) = encoding_rs::WINDOWS_1251.encode(&xml);
    assert!(!had_errors);
    let book = import_book("kobzar.fb2", &encoded).unwrap();
    assert_eq!(book.title, "Кобзар тест");
    assert_eq!(book.author, "Тарас Шевченко");
    assert!(book.chapters[0].text.contains("Думи мої, думи мої."));
}

fn docx_fixture() -> Vec<u8> {
    let document = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Вступ до книги.</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Розділ 1</w:t></w:r></w:p>
    <w:p><w:r><w:t>Перший текст.</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Розділ 2</w:t></w:r></w:p>
    <w:p><w:r><w:t>Другий текст.</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
    let styles = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
  </w:style>
</w:styles>"#;
    let core = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Тестовий DOCX</dc:title>
  <dc:creator>Укладач Тестовий</dc:creator>
  <dc:language>uk</dc:language>
</cp:coreProperties>"#;
    build_zip(&[
        ("word/document.xml", document.as_bytes()),
        ("word/styles.xml", styles.as_bytes()),
        ("docProps/core.xml", core.as_bytes()),
    ])
}

#[test]
fn imports_docx_with_heading_styles() {
    let bytes = docx_fixture();
    assert_eq!(
        speech_text::import::detect_format("book", &bytes),
        Some(BookFormat::Docx)
    );
    let book = import_book("book.docx", &bytes).unwrap();
    assert_eq!(book.title, "Тестовий DOCX");
    assert_eq!(book.author, "Укладач Тестовий");
    assert_eq!(book.chapters.len(), 3);
    assert_eq!(book.chapters[0].title, "Передмова");
    assert!(book.chapters[0].text.contains("Вступ до книги."));
    assert_eq!(book.chapters[1].title, "Розділ 1");
    assert!(book.chapters[1].text.contains("Перший текст."));
    assert_eq!(book.chapters[2].title, "Розділ 2");
}

#[test]
fn rejects_unsupported_binary_content() {
    assert_eq!(
        speech_text::import::detect_format("book.pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"),
        None
    );
    assert_eq!(
        speech_text::import::detect_format("book", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"),
        None
    );
    assert_eq!(
        speech_text::import::detect_format("book", b"\x7fELF\x02\x01\x01\0"),
        None
    );
    assert_eq!(
        speech_text::import::detect_format("book", b"PK\x03\x04not-a-real-zip"),
        None
    );
    assert_eq!(
        speech_text::import::detect_format("book", b"\x01\x02\x03\0\x04\x05"),
        None
    );
    assert!(matches!(
        import_book("book.pdf", b"%PDF-1.7\n"),
        Err(ImportError::UnsupportedFormat)
    ));
}

#[test]
fn sniffs_utf8_and_cp1251_plain_text() {
    assert_eq!(
        speech_text::import::detect_format("book", "Просто текст\n".as_bytes()),
        Some(BookFormat::Txt)
    );
    let cp1251 = [0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2, b'\n'];
    assert_eq!(
        speech_text::import::detect_format("book", &cp1251),
        Some(BookFormat::Txt)
    );
    let utf16 = [0xFF, 0xFE, b'T', 0x00, b'e', 0x00, b'x', 0x00, b't', 0x00];
    assert_eq!(
        speech_text::import::detect_format("book", &utf16),
        Some(BookFormat::Txt)
    );
}

#[test]
fn imports_plain_text_and_markdown() {
    let txt = "Передмова.\n\nРОЗДІЛ 1\n\nПерший текст.\n\nРОЗДІЛ 2\n\nДругий текст.\n";
    let book = import_book("book.txt", txt.as_bytes()).unwrap();
    assert_eq!(book.chapters.len(), 3);
    assert_eq!(book.chapters[1].title, "РОЗДІЛ 1");

    let md = "# Книга\n\nВступ.\n\n## Розділ 1\n\n**Жирний** текст.\n\n## Розділ 2\n\n- пункт\n";
    let book = import_book("book.md", md.as_bytes()).unwrap();
    assert_eq!(book.chapters.len(), 3);
    assert_eq!(book.chapters[1].title, "Розділ 1");
    assert!(book.chapters[1].text.contains("Жирний текст."));
    assert!(!book.chapters[1].text.contains("**"));
}

#[test]
fn rejects_path_traversal_in_zip() {
    let bytes = build_zip(&[
        ("META-INF/container.xml", CONTAINER.as_bytes()),
        ("../evil.xhtml", b"<html><body>evil</body></html>"),
    ]);
    let error = import_bytes(BookFormat::Epub, &bytes).unwrap_err();
    assert!(matches!(error, ImportError::Invalid(_)), "{error}");
}

#[test]
fn rejects_zip_with_too_many_entries() {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for index in 0..=speech_text::import::MAX_ZIP_ENTRIES {
        writer
            .start_file(format!("entry-{index}.txt"), options)
            .unwrap();
    }
    let bytes = writer.finish().unwrap().into_inner();
    let error = import_bytes(BookFormat::Epub, &bytes).unwrap_err();
    assert!(matches!(error, ImportError::LimitExceeded(_)), "{error}");
}

#[test]
fn rejects_docx_without_document() {
    let bytes = build_zip(&[("word/styles.xml", b"<styles/>")]);
    let error = import_bytes(BookFormat::Docx, &bytes).unwrap_err();
    assert!(matches!(error, ImportError::Invalid(_)), "{error}");
}
