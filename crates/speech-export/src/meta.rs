//! Спільні метадані для OGG/Opus та FLAC.
//!
//! OGG зберігає коментарі у `OpusTags`, FLAC — у Vorbis comment блоці
//! (тип 4). Набір ключів однаковий, щоб плеєри та аудіокнижкові каталоги
//! бачили ті самі назви, автора й глави.

use crate::{ExportInput, ExportedChapter};

/// Рядок енкодера, який потрапляє у vendor-поле та `ENCODER`.
pub(crate) const ENCODER: &str = concat!("Promin TTS ", env!("CARGO_PKG_VERSION"));

/// Коментарі книги без обкладинки (її формат відрізняється між контейнерами).
pub(crate) fn vorbis_comments(
    input: &ExportInput,
    chapters: &[ExportedChapter],
) -> Vec<(String, String)> {
    let metadata = &input.metadata;
    let album = if metadata.album.trim().is_empty() {
        &metadata.title
    } else {
        &metadata.album
    };

    let mut comments: Vec<(String, String)> = Vec::new();
    comments.push(("TITLE".to_string(), metadata.title.clone()));
    if !metadata.author.trim().is_empty() {
        comments.push(("ARTIST".to_string(), metadata.author.clone()));
        comments.push(("ALBUMARTIST".to_string(), metadata.author.clone()));
    }
    comments.push(("ALBUM".to_string(), album.clone()));
    comments.push(("GENRE".to_string(), "Audiobook".to_string()));
    comments.push(("ENCODER".to_string(), ENCODER.to_string()));
    if !metadata.language.trim().is_empty() {
        comments.push(("LANGUAGE".to_string(), metadata.language.clone()));
    }
    if let Some(narrator) = metadata
        .narrator
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        comments.push(("NARRATOR".to_string(), narrator.to_string()));
    }
    if let Some(year) = metadata
        .year
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        comments.push(("DATE".to_string(), year.to_string()));
    }
    if let Some(description) = metadata
        .description
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        comments.push(("DESCRIPTION".to_string(), description.to_string()));
    }
    for (position, chapter) in chapters.iter().enumerate() {
        let number = position + 1;
        comments.push((
            format!("CHAPTER{number:03}"),
            chapter_timestamp(chapter.start_ms),
        ));
        comments.push((format!("CHAPTER{number:03}NAME"), chapter.title.clone()));
    }
    comments
}

/// Vorbis comment block: vendor, кількість коментарів і `KEY=value` рядки.
pub(crate) fn vorbis_comment_block(comments: &[(String, String)]) -> Vec<u8> {
    let vendor = ENCODER.as_bytes();
    let mut block = Vec::new();
    block.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    block.extend_from_slice(vendor);
    block.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (key, value) in comments {
        let comment = format!("{key}={value}");
        block.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        block.extend_from_slice(comment.as_bytes());
    }
    block
}

/// `HH:MM:SS.mmm` — формат глав у Vorbis comments.
pub(crate) fn chapter_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms / 60_000) % 60;
    let seconds = (ms / 1000) % 60;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExportMetadata;

    fn input() -> ExportInput {
        ExportInput {
            format: crate::ExportFormat::Flac,
            bitrate_kbps: 0,
            metadata: ExportMetadata {
                title: "Книга".to_string(),
                author: "Автор".to_string(),
                album: String::new(),
                narrator: Some("Диктор".to_string()),
                year: Some("2026".to_string()),
                description: None,
                language: "uk".to_string(),
            },
            cover: None,
            chapters: Vec::new(),
        }
    }

    #[test]
    fn builds_vorbis_block_with_vendor_and_comments() {
        let comments = vec![
            ("TITLE".to_string(), "Книга".to_string()),
            ("ARTIST".to_string(), "Автор".to_string()),
        ];
        let block = vorbis_comment_block(&comments);
        let vendor_len = u32::from_le_bytes(block[..4].try_into().unwrap()) as usize;
        assert_eq!(&block[4..4 + vendor_len], ENCODER.as_bytes());
        let count_at = 4 + vendor_len;
        let count = u32::from_le_bytes(block[count_at..count_at + 4].try_into().unwrap());
        assert_eq!(count, 2);
    }

    #[test]
    fn comments_include_album_and_chapters() {
        let input = input();
        let chapters = vec![ExportedChapter {
            index: 0,
            title: "Розділ".to_string(),
            start_ms: 3_723_456,
            duration_ms: 1_000,
        }];
        let comments = vorbis_comments(&input, &chapters);
        assert!(comments.contains(&("ALBUM".to_string(), "Книга".to_string())));
        assert!(comments.contains(&("CHAPTER001".to_string(), "01:02:03.456".to_string())));
        assert!(comments.contains(&("CHAPTER001NAME".to_string(), "Розділ".to_string())));
    }
}
