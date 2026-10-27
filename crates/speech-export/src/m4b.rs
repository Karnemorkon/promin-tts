//! M4B: AAC-LC у MP4-контейнері з главами, метаданими та обкладинкою.
//!
//! Кожна глава кодується окремим енкодером (rusty_aac тримає весь PCM у
//! пам’яті до `finish()`), тому пікова пам’ять обмежена однією главою.
//! Перший priming-фрейм книги прибирається єдиним edit-list записом;
//! priming наступних глав лишається в медіатаймлайні, а маркери глав
//! ставляться на початок чутного аудіо.

use std::io::Write;

use mp4_core::chapter::{Chapter, ChapterList};
use mp4_core::metadata::atoms;
use mp4_core::{container, full_box_container_box, leaf, Box_};
use mp4io::{fourcc, Codec, EditEntry, InputSample, TrackParams, Writer, WriterConfig};

use crate::{
    exported_chapters, mp4_language, read_wav, EncodedBook, ExportError, ExportInput,
    ExportObserver, SAMPLE_RATE,
};

/// Кадр AAC-LC містить 1024 семпли.
const AAC_FRAME_SAMPLES: u64 = 1024;

pub(crate) fn encode(
    input: &ExportInput,
    frame_counts: &[u64],
    bitrate_kbps: u32,
    cover: &crate::CoverImage,
    out: &mut dyn Write,
    observer: &mut dyn ExportObserver,
) -> Result<EncodedBook, ExportError> {
    let encoded_counts: Vec<u64> = frame_counts
        .iter()
        .map(|&samples| samples.div_ceil(AAC_FRAME_SAMPLES) + 1)
        .collect();
    let total_media_samples: u64 = encoded_counts
        .iter()
        .map(|&frames| frames * AAC_FRAME_SAMPLES)
        .sum();
    if total_media_samples < AAC_FRAME_SAMPLES {
        return Err(ExportError::Invalid("book is too short".to_string()));
    }
    let duration_ms =
        (total_media_samples - AAC_FRAME_SAMPLES) * 1000 / u64::from(SAMPLE_RATE);

    let mut media_cursor = 0u64;
    let mut starts = Vec::with_capacity(encoded_counts.len());
    for &frames in &encoded_counts {
        starts.push(media_cursor * 1000 / u64::from(SAMPLE_RATE));
        media_cursor += frames * AAC_FRAME_SAMPLES;
    }

    let asc = rusty_aac::encode::audio_specific_config_bytes(SAMPLE_RATE, 1);
    let mut writer = Writer::new(WriterConfig {
        major_brand: fourcc(b"M4B "),
        minor_version: 0,
        compatible_brands: vec![
            fourcc(b"M4B "),
            fourcc(b"M4A "),
            fourcc(b"mp42"),
            fourcc(b"isom"),
        ],
        timescale: 1000,
    });
    let track = writer.add_track(
        TrackParams::audio(Codec::Aac, 1, SAMPLE_RATE)
            .aac_config(2, SAMPLE_RATE, 1, asc)
            .language(&mp4_language(&input.metadata.language))
            .edit_list(vec![EditEntry {
                segment_duration: duration_ms,
                // Прибрати priming-фрейм першої глави.
                media_time: AAC_FRAME_SAMPLES as i64,
                media_rate: 1.0,
            }]),
    );

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
        let packets = encode_aac(&pcm, bitrate_kbps.saturating_mul(1000))?;
        if packets.len() as u64 != encoded_counts[position] {
            return Err(ExportError::M4b(format!(
                "chapter {}: encoder produced {} frames, expected {}",
                chapter.index,
                packets.len(),
                encoded_counts[position]
            )));
        }
        for packet in &packets {
            writer
                .write_sample(
                    track,
                    InputSample {
                        data: packet,
                        duration: AAC_FRAME_SAMPLES as u32,
                        composition_offset: 0,
                        is_sync: true,
                    },
                )
                .map_err(|error| ExportError::M4b(error.to_string()))?;
        }
        observer.chapter_finished(position, total);
    }

    let mut mp4 = writer
        .finalize()
        .map_err(|error| ExportError::M4b(error.to_string()))?;
    let udta = build_udta(input, &starts, cover)?;
    insert_udta(&mut mp4, &udta)?;
    out.write_all(&mp4)?;

    Ok(EncodedBook {
        duration_ms,
        chapters: exported_chapters(input, &starts, duration_ms),
    })
}

fn encode_aac(pcm: &[i16], bitrate_bps: u32) -> Result<Vec<Vec<u8>>, ExportError> {
    let mut encoder = rusty_aac::encode::AacEncoder::new(rusty_aac::encode::AacEncoderConfig {
        bitrate_bps,
        ..Default::default()
    });
    let floats: Vec<f32> = pcm.iter().map(|&sample| f32::from(sample) / 32768.0).collect();
    encoder
        .push_pcm(&floats, 1, SAMPLE_RATE)
        .map_err(|error| ExportError::M4b(format!("AAC encode: {error}")))?;
    encoder.finish();
    let mut packets = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(packet) => packets.push(packet.data),
            Err(rusty_aac::Error::Eof | rusty_aac::Error::Again) => break,
            Err(error) => return Err(ExportError::M4b(format!("AAC encode: {error}"))),
        }
    }
    Ok(packets)
}

fn build_udta(
    input: &ExportInput,
    starts: &[u64],
    cover: &crate::CoverImage,
) -> Result<Vec<u8>, ExportError> {
    let metadata = &input.metadata;

    let mut items = vec![
        text_atom(atoms::TITLE, &metadata.title),
        text_atom(atoms::GENRE, "Audiobook"),
        text_atom(
            atoms::ENCODER,
            concat!("Promin TTS ", env!("CARGO_PKG_VERSION")),
        ),
    ];
    if !metadata.author.trim().is_empty() {
        items.push(text_atom(atoms::ARTIST, &metadata.author));
        items.push(text_atom(*b"aART", &metadata.author));
    }
    let album = if metadata.album.trim().is_empty() {
        &metadata.title
    } else {
        &metadata.album
    };
    items.push(text_atom(atoms::ALBUM, album));
    if let Some(narrator) = metadata.narrator.as_deref().filter(|value| !value.trim().is_empty()) {
        items.push(text_atom([0xA9, b'n', b'r', b't'], narrator));
    }
    if let Some(year) = metadata.year.as_deref().filter(|value| !value.trim().is_empty()) {
        items.push(text_atom(atoms::YEAR, year));
    }
    if let Some(description) = metadata
        .description
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        items.push(text_atom(atoms::DESCRIPTION, description));
    }
    items.push(container(atoms::MEDIA_KIND, vec![data_atom(21, vec![9])]));
    let cover_flag = if cover.is_png() { 14 } else { 13 };
    items.push(container(
        atoms::COVER,
        vec![data_atom(cover_flag, cover.data.clone())],
    ));

    let ilst = container(*b"ilst", items);
    let hdlr = leaf(*b"hdlr", {
        let mut payload = vec![0u8; 4]; // version + flags
        payload.extend_from_slice(&[0u8; 4]); // pre_defined
        payload.extend_from_slice(b"mdirappl");
        payload.extend(std::iter::repeat_n(0u8, 9));
        payload
    })
    .map_err(|error| ExportError::M4b(error.to_string()))?;
    let meta = full_box_container_box(*b"meta", vec![hdlr, ilst]);

    let chapters = ChapterList {
        chapters: input
            .chapters
            .iter()
            .enumerate()
            .map(|(position, chapter)| {
                Chapter::new(starts.get(position).copied().unwrap_or(0), &chapter.title)
            })
            .collect(),
    };
    let udta = container(*b"udta", vec![meta, chapters.to_chpl_box()]);
    Ok(udta.to_bytes())
}

fn text_atom(name: [u8; 4], value: &str) -> Box_ {
    container(name, vec![data_atom(1, value.as_bytes().to_vec())])
}

fn data_atom(flag: u32, payload: Vec<u8>) -> Box_ {
    let mut bytes = Vec::with_capacity(8 + payload.len());
    bytes.extend_from_slice(&flag.to_be_bytes());
    bytes.extend_from_slice(&0u32.to_be_bytes()); // locale
    bytes.extend_from_slice(&payload);
    leaf(*b"data", bytes).expect("data atom is valid")
}

/// Дописати `udta` у кінець `moov` (mp4io пише `moov` останнім).
fn insert_udta(mp4: &mut Vec<u8>, udta: &[u8]) -> Result<(), ExportError> {
    let mut position = 0usize;
    while position + 8 <= mp4.len() {
        let size32 = u32::from_be_bytes(
            mp4[position..position + 4]
                .try_into()
                .expect("four bytes checked above"),
        );
        let (size, header) = if size32 == 1 {
            if position + 16 > mp4.len() {
                return Err(ExportError::M4b("truncated box header".to_string()));
            }
            let size = u64::from_be_bytes(
                mp4[position + 8..position + 16]
                    .try_into()
                    .expect("eight bytes checked above"),
            ) as usize;
            (size, 16usize)
        } else if size32 == 0 {
            (mp4.len() - position, 8usize)
        } else {
            (size32 as usize, 8usize)
        };
        if size < header || position.saturating_add(size) > mp4.len() {
            return Err(ExportError::M4b("malformed MP4 box".to_string()));
        }
        if &mp4[position + 4..position + 8] == b"moov" {
            if position + size != mp4.len() {
                return Err(ExportError::M4b(
                    "moov box is not the last box; cannot append udta safely".to_string(),
                ));
            }
            let end = position + size;
            mp4.splice(end..end, udta.iter().copied());
            let new_size = size + udta.len();
            if size32 == 1 {
                mp4[position + 8..position + 16]
                    .copy_from_slice(&(new_size as u64).to_be_bytes());
            } else {
                let new_size = u32::try_from(new_size)
                    .map_err(|_| ExportError::M4b("moov box is too large".to_string()))?;
                mp4[position..position + 4].copy_from_slice(&new_size.to_be_bytes());
            }
            return Ok(());
        }
        position += size;
    }
    Err(ExportError::M4b("moov box not found".to_string()))
}
