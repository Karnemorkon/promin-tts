//! FLAC: lossless кодування всієї книги, Vorbis comments і нативний Picture.
//!
//! Джерело читає WAV-глави послідовно, тому в пам'яті не тримається весь
//! PCM-потік. `flacenc` накопичує лише стиснені фрейми (як M4B-контейнер).

use std::fs::File;
use std::io::{BufReader, Write};

use flacenc::bitsink::MemSink;
use flacenc::component::{BitRepr, MetadataBlockData};
use flacenc::config::Encoder;
use flacenc::error::{SourceError, Verify};
use flacenc::source::{Fill, Source};

use crate::{
    exported_chapters, source_chapter_starts, EncodedBook, ExportChapter, ExportError, ExportInput,
    ExportObserver, SAMPLE_RATE,
};

/// Розмір блоку FLAC; 4096 — типовий компроміс між стисненням і пам'яттю.
const BLOCK_SIZE: usize = 4096;

pub(crate) fn encode(
    input: &ExportInput,
    frame_counts: &[u64],
    cover: &crate::CoverImage,
    out: &mut dyn Write,
    observer: &mut dyn ExportObserver,
) -> Result<EncodedBook, ExportError> {
    let total_frames: u64 = frame_counts.iter().sum();
    let duration_ms = total_frames * 1000 / u64::from(SAMPLE_RATE);
    let starts = source_chapter_starts(frame_counts);
    let chapters = exported_chapters(input, &starts, duration_ms);

    let mut cancelled = false;
    let source = WavBookSource::new(
        &input.chapters,
        frame_counts,
        total_frames,
        observer,
        &mut cancelled,
    );
    let mut config = Encoder::default();
    // Один потік: експорт і так виконується під окремим semaphore, а
    // паралельний FLAC на 32-потоковому Xeon забирає всі ядра.
    config.multithread = false;
    let config = config
        .into_verified()
        .map_err(|(_, error)| ExportError::Flac(error.to_string()))?;
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, BLOCK_SIZE)
        .map_err(|error| ExportError::Flac(error.to_string()))?;
    if cancelled {
        return Err(ExportError::Cancelled);
    }

    let comments = crate::meta::vorbis_comments(input, &chapters);
    stream.add_metadata_block(
        MetadataBlockData::new_unknown(4, &crate::meta::vorbis_comment_block(&comments))
            .map_err(|error| ExportError::Flac(error.to_string()))?,
    );
    stream.add_metadata_block(
        MetadataBlockData::new_unknown(6, &crate::cover::picture_block(cover))
            .map_err(|error| ExportError::Flac(error.to_string()))?,
    );

    let mut sink = MemSink::<u8>::new();
    stream
        .write(&mut sink)
        .map_err(|error| ExportError::Flac(error.to_string()))?;
    out.write_all(sink.as_slice())?;

    Ok(EncodedBook {
        duration_ms,
        chapters,
    })
}

/// Потокове джерело PCM16: відкриває глави по одній і повідомляє observer.
struct WavBookSource<'a> {
    chapters: &'a [ExportChapter],
    frame_counts: &'a [u64],
    position: usize,
    reader: Option<hound::WavReader<BufReader<File>>>,
    remaining: u64,
    expected_total: u64,
    read_total: u64,
    buffer: Vec<i32>,
    observer: &'a mut dyn ExportObserver,
    cancelled: &'a mut bool,
}

impl<'a> WavBookSource<'a> {
    fn new(
        chapters: &'a [ExportChapter],
        frame_counts: &'a [u64],
        expected_total: u64,
        observer: &'a mut dyn ExportObserver,
        cancelled: &'a mut bool,
    ) -> Self {
        Self {
            chapters,
            frame_counts,
            position: 0,
            reader: None,
            remaining: 0,
            expected_total,
            read_total: 0,
            buffer: Vec::with_capacity(BLOCK_SIZE),
            observer,
            cancelled,
        }
    }

    /// Відкрити наступну непорожню главу; `false` — глави закінчилися.
    fn open_next(&mut self) -> Result<bool, SourceError> {
        while self.position < self.chapters.len() {
            let chapter = &self.chapters[self.position];
            let reader = hound::WavReader::open(&chapter.wav_path)
                .map_err(|error| SourceError::from_io_error(std::io::Error::other(error)))?;
            let spec = reader.spec();
            if spec.channels != 1
                || spec.sample_rate != SAMPLE_RATE
                || spec.bits_per_sample != 16
                || spec.sample_format != hound::SampleFormat::Int
            {
                return Err(SourceError::from_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "chapter {} WAV must be 24 kHz mono 16-bit PCM",
                        chapter.index
                    ),
                )));
            }
            let expected = self.frame_counts[self.position];
            if u64::from(reader.len()) != expected {
                return Err(SourceError::from_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("chapter {} changed during export", chapter.index),
                )));
            }
            self.remaining = expected;
            self.reader = Some(reader);
            self.observer
                .chapter_started(self.position, self.chapters.len());
            if self.remaining == 0 {
                self.observer
                    .chapter_finished(self.position, self.chapters.len());
                self.position += 1;
                self.reader = None;
                continue;
            }
            return Ok(true);
        }
        Ok(false)
    }
}

impl Source for WavBookSource<'_> {
    fn channels(&self) -> usize {
        1
    }

    fn bits_per_sample(&self) -> usize {
        16
    }

    fn sample_rate(&self) -> usize {
        SAMPLE_RATE as usize
    }

    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        if self.observer.is_cancelled() {
            *self.cancelled = true;
            return Err(SourceError::from_io_error(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "export cancelled",
            )));
        }

        self.buffer.clear();
        while self.buffer.len() < block_size {
            if self.reader.is_none() && !self.open_next()? {
                break;
            }
            let take = ((block_size - self.buffer.len()) as u64).min(self.remaining) as usize;
            let reader = self.reader.as_mut().expect("reader is open");
            let mut samples = reader.samples::<i16>();
            for _ in 0..take {
                let sample = samples
                    .next()
                    .transpose()
                    .map_err(|error| SourceError::from_io_error(std::io::Error::other(error)))?
                    .ok_or_else(|| {
                        SourceError::from_io_error(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "chapter WAV ended early",
                        ))
                    })?;
                self.buffer.push(i32::from(sample));
            }
            self.remaining -= take as u64;
            self.read_total += take as u64;
            if self.remaining == 0 {
                self.observer
                    .chapter_finished(self.position, self.chapters.len());
                self.position += 1;
                self.reader = None;
            }
        }

        if self.buffer.is_empty()
            && self.position >= self.chapters.len()
            && self.read_total != self.expected_total
        {
            return Err(SourceError::from_io_error(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chapter audio length changed during export",
            )));
        }

        dest.fill_interleaved(&self.buffer)?;
        Ok(self.buffer.len())
    }

    fn len_hint(&self) -> Option<usize> {
        usize::try_from(self.expected_total).ok()
    }
}
