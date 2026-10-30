//! MP3: безперервне кодування всієї книги з ID3v2.4 (глави, обкладинка).

use std::io::Write;

use id3_core::{
    AttachedPicture, Chapter as Id3Chapter, Comment, Frame, Language, Tag, TextEncoding, TextField,
    Toc,
};

use crate::{
    exported_chapters, read_wav, source_chapter_starts, EncodedBook, ExportError, ExportInput,
    ExportObserver, SAMPLE_RATE,
};

pub(crate) fn encode(
    input: &ExportInput,
    frame_counts: &[u64],
    bitrate_kbps: u32,
    cover: &crate::CoverImage,
    out: &mut dyn Write,
    observer: &mut dyn ExportObserver,
) -> Result<EncodedBook, ExportError> {
    let total_frames: u64 = frame_counts.iter().sum();
    let duration_ms = total_frames * 1000 / u64::from(SAMPLE_RATE);
    let starts = source_chapter_starts(frame_counts);
    let chapters = exported_chapters(input, &starts, duration_ms);

    let tag = build_tag(input, &chapters, cover)?;
    out.write_all(&tag)?;

    let mut encoder = rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
        bitrate_kbps,
        vbr_quality: None,
    });
    let total = input.chapters.len();
    for (position, chapter) in input.chapters.iter().enumerate() {
        if observer.is_cancelled() {
            return Err(ExportError::Cancelled);
        }
        observer.chapter_started(position, total);
        let pcm = read_wav(&chapter.wav_path)?;
        if pcm.len() as u64 != frame_counts[position] {
            return Err(ExportError::Invalid(format!(
                "chapter {} changed during export",
                chapter.index
            )));
        }
        encoder
            .push_pcm_s16(&pcm, 1, SAMPLE_RATE)
            .map_err(|error| ExportError::Mp3(error.to_string()))?;
        drain(&mut encoder, out)?;
        observer.chapter_finished(position, total);
    }
    encoder.finish();
    drain(&mut encoder, out)?;

    Ok(EncodedBook {
        duration_ms,
        chapters,
    })
}

fn drain(encoder: &mut rusty_mp3::Mp3Encoder, out: &mut dyn Write) -> Result<(), ExportError> {
    loop {
        match encoder.next_packet() {
            Ok(packet) => out.write_all(&packet)?,
            Err(rusty_mp3::Error::Eof | rusty_mp3::Error::Again) => return Ok(()),
            Err(error) => return Err(ExportError::Mp3(error.to_string())),
        }
    }
}

fn build_tag(
    input: &ExportInput,
    chapters: &[crate::ExportedChapter],
    cover: &crate::CoverImage,
) -> Result<Vec<u8>, ExportError> {
    let metadata = &input.metadata;
    let mut tag = Tag::new();
    tag.set_text(TextField::Title, TextEncoding::Utf8, &metadata.title);
    tag.set_text(TextField::Artist, TextEncoding::Utf8, &metadata.author);
    tag.set_text(TextField::AlbumArtist, TextEncoding::Utf8, &metadata.author);
    let album = if metadata.album.trim().is_empty() {
        &metadata.title
    } else {
        &metadata.album
    };
    tag.set_text(TextField::Album, TextEncoding::Utf8, album);
    tag.set_text(TextField::Genre, TextEncoding::Utf8, "Audiobook");
    if let Some(narrator) = metadata
        .narrator
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        tag.set_text(TextField::Composer, TextEncoding::Utf8, narrator);
    }
    if let Some(year) = metadata
        .year
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        tag.set_text(TextField::Year, TextEncoding::Utf8, year);
    }
    if let Some(description) = metadata
        .description
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        let language = Language::new(*b"ukr").unwrap_or(Language::new(*b"und").expect("valid"));
        tag.push(Frame::Comment(Comment::new(
            language,
            description,
            TextEncoding::Utf8,
        )));
    }
    tag.push(Frame::Picture(AttachedPicture::cover(cover.data.clone())));

    let mut toc = Toc::root("book");
    for (position, chapter) in chapters.iter().enumerate() {
        let id = format!("ch{position}");
        let start = u32::try_from(chapter.start_ms)
            .map_err(|_| ExportError::Mp3("chapter start exceeds ID3 range".to_string()))?;
        let end = u32::try_from(chapter.start_ms.saturating_add(chapter.duration_ms))
            .map_err(|_| ExportError::Mp3("chapter end exceeds ID3 range".to_string()))?;
        tag.push(Frame::Chapter(
            Id3Chapter::new(&id, start)
                .with_end(end)
                .with_title(&chapter.title),
        ));
        toc.push(&id);
    }
    tag.push(Frame::Toc(toc));
    tag.to_bytes()
        .map_err(|error| ExportError::Mp3(error.to_string()))
}
