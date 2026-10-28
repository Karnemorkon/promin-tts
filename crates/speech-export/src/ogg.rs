//! OGG Opus: безперервне кодування, глави та обкладинка в OpusTags.
//!
//! Вхід 24 кГц, але granule Opus завжди рахується в семплах 48 кГц
//! (`samples * 2`). Залишок неповного 20-мс кадру переноситься в наступну
//! главу, тому між главами немає штучної тиші; останній кадр доповнюється
//! нулями, а granule вказує на справжню довжину, тож декодер обрізає padding.

use std::io::Write;

use base64::Engine as _;
use rusty_opus::{Application, OpusEncoder};

use crate::{
    exported_chapters, read_wav, source_chapter_starts, EncodedBook, ExportError, ExportInput,
    ExportObserver, SAMPLE_RATE,
};

const OPUS_FRAME_SAMPLES: usize = 480; // 20 мс при 24 кГц
const OPUS_GRANULE_RATE: u64 = 48_000;
const OGG_SERIAL: u32 = 0x5052_4F4D; // "PROM"
const OPUS_PRE_SKIP: u16 = 120;

pub(crate) fn encode(
    input: &ExportInput,
    frame_counts: &[u64],
    bitrate_kbps: u32,
    cover: &crate::CoverImage,
    out: &mut dyn Write,
    observer: &mut dyn ExportObserver,
) -> Result<EncodedBook, ExportError> {
    let total_samples: u64 = frame_counts.iter().sum();
    let duration_ms = total_samples * 1000 / u64::from(SAMPLE_RATE);
    let starts = source_chapter_starts(frame_counts);
    let chapters = exported_chapters(input, &starts, duration_ms);

    let mut encoder = OpusEncoder::new(SAMPLE_RATE as i32, 1, Application::Audio)
        .map_err(|error| ExportError::Ogg(format!("Opus encoder: {error}")))?;
    encoder.bitrate_bps = bitrate_kbps.saturating_mul(1000) as i32;

    let mut writer = ogg::PacketWriter::new(out);
    writer
        .write_packet(opus_head(), OGG_SERIAL, ogg::PacketWriteEndInfo::EndPage, 0)
        .map_err(|error| ExportError::Ogg(error.to_string()))?;
    writer
        .write_packet(
            opus_tags(input, &chapters, cover)?,
            OGG_SERIAL,
            ogg::PacketWriteEndInfo::EndPage,
            0,
        )
        .map_err(|error| ExportError::Ogg(error.to_string()))?;

    // Один додатковий кадр тиші: pre-skip має бути присутнім у потоці,
    // інакше декодер відріже його з реального аудіо.
    let total_packets = total_samples.div_ceil(OPUS_FRAME_SAMPLES as u64) + 1;
    let mut packet_index = 0u64;
    let mut carry: Vec<i16> = Vec::with_capacity(OPUS_FRAME_SAMPLES);
    let total_chapters = input.chapters.len();

    for (position, chapter) in input.chapters.iter().enumerate() {
        if observer.is_cancelled() {
            return Err(ExportError::Cancelled);
        }
        observer.chapter_started(position, total_chapters);
        let pcm = read_wav(&chapter.wav_path)?;
        if pcm.len() as u64 != frame_counts[position] {
            return Err(ExportError::Invalid(format!(
                "chapter {} changed during export",
                chapter.index
            )));
        }

        let mut offset = 0usize;
        if !carry.is_empty() {
            let needed = OPUS_FRAME_SAMPLES - carry.len();
            let take = needed.min(pcm.len());
            carry.extend_from_slice(&pcm[..take]);
            offset = take;
            if carry.len() == OPUS_FRAME_SAMPLES {
                encode_packet(
                    &mut encoder,
                    &mut writer,
                    &carry,
                    total_samples,
                    total_packets,
                    &mut packet_index,
                )?;
                carry.clear();
            }
        }
        while offset + OPUS_FRAME_SAMPLES <= pcm.len() {
            encode_packet(
                &mut encoder,
                &mut writer,
                &pcm[offset..offset + OPUS_FRAME_SAMPLES],
                total_samples,
                total_packets,
                &mut packet_index,
            )?;
            offset += OPUS_FRAME_SAMPLES;
        }
        carry.extend_from_slice(&pcm[offset..]);
        observer.chapter_finished(position, total_chapters);
    }

    if !carry.is_empty() {
        let mut padded = carry.clone();
        padded.resize(OPUS_FRAME_SAMPLES, 0);
        encode_packet(
            &mut encoder,
            &mut writer,
            &padded,
            total_samples,
            total_packets,
            &mut packet_index,
        )?;
    }
    // Гарантований запас для pre-skip; granule останньої сторінки обріже його.
    encode_packet(
        &mut encoder,
        &mut writer,
        &[0i16; OPUS_FRAME_SAMPLES],
        total_samples,
        total_packets,
        &mut packet_index,
    )?;

    Ok(EncodedBook {
        duration_ms,
        chapters,
    })
}

fn encode_packet(
    encoder: &mut OpusEncoder,
    writer: &mut ogg::PacketWriter<'_, &mut dyn Write>,
    samples: &[i16],
    total_samples: u64,
    total_packets: u64,
    packet_index: &mut u64,
) -> Result<(), ExportError> {
    debug_assert_eq!(samples.len(), OPUS_FRAME_SAMPLES);
    let frame: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 32768.0).collect();
    let mut packet = [0u8; 1500];
    let length = encoder
        .encode(&frame, OPUS_FRAME_SAMPLES, &mut packet)
        .map_err(|error| ExportError::Ogg(format!("Opus encode: {error}")))?;

    let last = *packet_index + 1 == total_packets;
    let granule = u64::from(OPUS_PRE_SKIP)
        + if last {
            total_samples * OPUS_GRANULE_RATE / u64::from(SAMPLE_RATE)
        } else {
            (*packet_index + 1) * OPUS_FRAME_SAMPLES as u64 * OPUS_GRANULE_RATE
                / u64::from(SAMPLE_RATE)
        };
    let end = if last {
        ogg::PacketWriteEndInfo::EndStream
    } else {
        ogg::PacketWriteEndInfo::NormalPacket
    };
    writer
        .write_packet(packet[..length].to_vec(), OGG_SERIAL, end, granule)
        .map_err(|error| ExportError::Ogg(error.to_string()))?;
    *packet_index += 1;
    Ok(())
}

fn opus_head() -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1); // version
    head.push(1); // channels
    head.extend_from_slice(&OPUS_PRE_SKIP.to_le_bytes()); // pre-skip
    head.extend_from_slice(&SAMPLE_RATE.to_le_bytes()); // original sample rate
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family
    head
}

fn opus_tags(
    input: &ExportInput,
    chapters: &[crate::ExportedChapter],
    cover: &crate::CoverImage,
) -> Result<Vec<u8>, ExportError> {
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
    comments.push((
        "ENCODER".to_string(),
        concat!("Promin TTS ", env!("CARGO_PKG_VERSION")).to_string(),
    ));
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
        comments.push((format!("CHAPTER{number:03}"), timestamp(chapter.start_ms)));
        comments.push((format!("CHAPTER{number:03}NAME"), chapter.title.clone()));
    }
    comments.push((
        "METADATA_BLOCK_PICTURE".to_string(),
        base64::engine::general_purpose::STANDARD.encode(flac_picture_block(cover)),
    ));

    let vendor = concat!("Promin TTS ", env!("CARGO_PKG_VERSION")).as_bytes();
    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (key, value) in comments {
        let comment = format!("{key}={value}");
        tags.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        tags.extend_from_slice(comment.as_bytes());
    }
    Ok(tags)
}

fn flac_picture_block(cover: &crate::CoverImage) -> Vec<u8> {
    let (width, height, depth) = cover.dimensions();
    let mime = cover.mime.as_bytes();
    let mut block = Vec::new();
    block.extend_from_slice(&3u32.to_be_bytes()); // front cover
    block.extend_from_slice(&(mime.len() as u32).to_be_bytes());
    block.extend_from_slice(mime);
    block.extend_from_slice(&0u32.to_be_bytes()); // description
    block.extend_from_slice(&width.to_be_bytes());
    block.extend_from_slice(&height.to_be_bytes());
    block.extend_from_slice(&depth.to_be_bytes());
    block.extend_from_slice(&0u32.to_be_bytes()); // colors
    block.extend_from_slice(&(cover.data.len() as u32).to_be_bytes());
    block.extend_from_slice(&cover.data);
    block
}

fn timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms / 60_000) % 60;
    let seconds = (ms / 1000) % 60;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}
