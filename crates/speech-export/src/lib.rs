//! Експорт готових аудіокниг у M4B, MP3, OGG Opus та FLAC.
//!
//! Крейт не знає нічого про сервер чи проєкти: на вході — список глав із
//! шляхами до WAV (24 кГц mono PCM16), метадані та обкладинка; на виході —
//! аудіофайл, записаний у будь-який [`std::io::Write`], плюс таймлайн глав і
//! об’єднані SRT/VTT/ASS.
//!
//! Ключові особливості:
//! - M4B кодується по главах окремими AAC-енкодерами (обмеження пам’яті), а
//!   priming першої глави прибирається єдиним edit-list записом;
//! - MP3 та OGG кодуються безперервно на всю книгу, тому на межах глав немає
//!   артефактів і зайвої тиші;
//! - FLAC lossless читає глави потоково й вбудовує Vorbis comments та нативний
//!   Picture-блок;
//! - у кожен файл вбудовуються назва, автор, глави та обкладинка (типово —
//!   вбудована обкладинка Promin TTS);
//! - субтитри для M4B зсуваються на priming-фрейм кожної глави, для MP3/OGG —
//!   на вихідний таймлайн WAV; ASS містить послівну караоке-розмітку.

mod ass;
mod cover;
mod flac;
mod m4b;
mod meta;
mod mp3;
mod ogg;
pub mod subtitles;

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use ass::render_ass;
pub use cover::{default_cover, CoverImage};
pub use subtitles::{render_srt, render_vtt, Cue};

/// Частота дискретизації глав (як у `speech-audio`).
pub const SAMPLE_RATE: u32 = 24_000;

/// Типовий бітрейт M4B: rusty_aac на 96 кбіт/с дає помітно кращу якість за 64.
pub const DEFAULT_M4B_BITRATE_KBPS: u32 = 96;
/// Типовий бітрейт MP3 (MPEG-2 Layer III, 24 кГц).
pub const DEFAULT_MP3_BITRATE_KBPS: u32 = 64;
/// Типовий бітрейт OGG Opus.
pub const DEFAULT_OGG_BITRATE_KBPS: u32 = 48;

/// Формат експорту.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    M4b,
    Mp3,
    Ogg,
    Flac,
}

impl ExportFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "m4b" | "m4a" | "aac" => Some(Self::M4b),
            "mp3" => Some(Self::Mp3),
            "ogg" | "opus" | "oga" => Some(Self::Ogg),
            "flac" => Some(Self::Flac),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::M4b => "m4b",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
        }
    }

    pub fn extension(self) -> &'static str {
        self.as_str()
    }

    pub fn mime(self) -> &'static str {
        match self {
            Self::M4b => "audio/mp4",
            Self::Mp3 => "audio/mpeg",
            Self::Ogg => "audio/ogg",
            Self::Flac => "audio/flac",
        }
    }

    /// FLAC не має бітрейту: якість визначається лише lossless-кодуванням.
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Flac)
    }

    pub fn default_bitrate_kbps(self) -> u32 {
        match self {
            Self::M4b => DEFAULT_M4B_BITRATE_KBPS,
            Self::Mp3 => DEFAULT_MP3_BITRATE_KBPS,
            Self::Ogg => DEFAULT_OGG_BITRATE_KBPS,
            Self::Flac => 0,
        }
    }

    /// Допустимий діапазон бітрейту для формату.
    pub fn bitrate_range_kbps(self) -> (u32, u32) {
        match self {
            Self::M4b => (32, 256),
            Self::Mp3 => (32, 160),
            Self::Ogg => (16, 128),
            Self::Flac => (0, 0),
        }
    }

    /// Перевірити бітрейт; `0` означає типовий для формату.
    ///
    /// Для FLAC явний бітрейт ігнорується: це lossless-формат, і клієнти
    /// можуть надсилати спільне поле `bitrate_kbps` для всіх форматів.
    pub fn resolve_bitrate_kbps(self, requested: u32) -> Result<u32, ExportError> {
        if requested == 0 || self.is_lossless() {
            return Ok(self.default_bitrate_kbps());
        }
        let (min, max) = self.bitrate_range_kbps();
        if !(min..=max).contains(&requested) {
            return Err(ExportError::Invalid(format!(
                "bitrate {requested} kbps is outside the {min}..={max} kbps range for {}",
                self.as_str()
            )));
        }
        Ok(requested)
    }
}

/// Метадані книги.
#[derive(Debug, Clone, Default)]
pub struct ExportMetadata {
    pub title: String,
    pub author: String,
    /// Альбом; порожній означає назву книги.
    pub album: String,
    pub narrator: Option<String>,
    pub year: Option<String>,
    pub description: Option<String>,
    /// Мова книги (ISO 639-1 або 639-2).
    pub language: String,
}

/// Одна глава на вході експорту.
#[derive(Debug, Clone)]
pub struct ExportChapter {
    pub index: usize,
    pub title: String,
    /// WAV 24 кГц mono PCM16.
    pub wav_path: PathBuf,
    /// Таймінги чанків у межах глави (для субтитрів).
    pub cues: Vec<Cue>,
}

/// Параметри експорту.
#[derive(Debug, Clone)]
pub struct ExportInput {
    pub format: ExportFormat,
    /// `0` — типовий бітрейт для формату.
    pub bitrate_kbps: u32,
    pub metadata: ExportMetadata,
    /// `None` — вбудована обкладинка Promin TTS.
    pub cover: Option<CoverImage>,
    pub chapters: Vec<ExportChapter>,
}

/// Глава у підсумковому таймлайні.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedChapter {
    pub index: usize,
    pub title: String,
    pub start_ms: u64,
    pub duration_ms: u64,
}

/// Результат експорту.
#[derive(Debug, Clone)]
pub struct ExportSummary {
    pub format: ExportFormat,
    pub bitrate_kbps: u32,
    pub duration_ms: u64,
    pub chapters: Vec<ExportedChapter>,
    pub bytes_written: u64,
}

/// Спостерігач за прогресом експорту.
///
/// Сервер використовує його для оновлення запису job-а та перевірки
/// скасування між главами.
pub trait ExportObserver {
    fn chapter_started(&mut self, _index: usize, _total: usize) {}
    fn chapter_finished(&mut self, _index: usize, _total: usize) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Спостерігач, що нічого не робить.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopObserver;

impl ExportObserver for NoopObserver {}

/// Помилка експорту.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("export cancelled")]
    Cancelled,
    #[error("invalid export input: {0}")]
    Invalid(String),
    #[error("cannot read WAV {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("chapter WAV {path} must be 24 kHz mono 16-bit PCM")]
    WavFormat { path: PathBuf },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("M4B error: {0}")]
    M4b(String),
    #[error("MP3 error: {0}")]
    Mp3(String),
    #[error("OGG error: {0}")]
    Ogg(String),
    #[error("FLAC error: {0}")]
    Flac(String),
}

/// Експортувати книгу у `out`.
///
/// Аудіо пишеться послідовно; M4B через особливості контейнера збирається в
/// пам’яті (розмір стисненого аудіо), MP3/OGG стрімляться.
pub fn export_book(
    input: &ExportInput,
    out: &mut dyn Write,
    observer: &mut dyn ExportObserver,
) -> Result<ExportSummary, ExportError> {
    validate_input(input)?;
    let bitrate_kbps = input.format.resolve_bitrate_kbps(input.bitrate_kbps)?;
    let frame_counts = scan_chapters(&input.chapters)?;
    let total_frames: u64 = frame_counts.iter().sum();
    if total_frames == 0 {
        return Err(ExportError::Invalid(
            "all chapters are empty (no audio samples)".to_string(),
        ));
    }

    let mut counter = CountingWriter::new(out);
    let cover = input.cover.clone().unwrap_or_else(default_cover);
    let encoded = match input.format {
        ExportFormat::M4b => m4b::encode(
            input,
            &frame_counts,
            bitrate_kbps,
            &cover,
            &mut counter,
            observer,
        )?,
        ExportFormat::Mp3 => mp3::encode(
            input,
            &frame_counts,
            bitrate_kbps,
            &cover,
            &mut counter,
            observer,
        )?,
        ExportFormat::Ogg => ogg::encode(
            input,
            &frame_counts,
            bitrate_kbps,
            &cover,
            &mut counter,
            observer,
        )?,
        ExportFormat::Flac => flac::encode(input, &frame_counts, &cover, &mut counter, observer)?,
    };

    // Для lossless-формату фактичний бітрейт залежить від матеріалу, тому
    // повертаємо середній, а не нуль із запиту.
    let bitrate_kbps = if input.format.is_lossless() {
        effective_bitrate_kbps(counter.count, encoded.duration_ms)
    } else {
        bitrate_kbps
    };

    Ok(ExportSummary {
        format: input.format,
        bitrate_kbps,
        duration_ms: encoded.duration_ms,
        chapters: encoded.chapters,
        bytes_written: counter.count,
    })
}

fn effective_bitrate_kbps(bytes: u64, duration_ms: u64) -> u32 {
    if duration_ms == 0 {
        return 0;
    }
    let bits = bytes.saturating_mul(8);
    u32::try_from(bits / duration_ms.max(1))
        .unwrap_or(u32::MAX)
        .max(1)
}

/// Побудувати об’єднані SRT/VTT за таймлайном експорту.
///
/// Для M4B [`ExportedChapter::start_ms`] уже враховує priming-фрейм глави,
/// тому субтитри збігаються з чутним аудіо.
pub fn combined_subtitles(
    input: &[ExportChapter],
    chapters: &[ExportedChapter],
) -> (String, String) {
    let cues = combined_cues(input, chapters);
    (render_srt(&cues), render_vtt(&cues))
}

/// Побудувати об’єднаний ASS із послівною караоке-розміткою.
///
/// Точних послівних таймінгів модель поки не повертає, тому тривалість
/// кожного слова оцінюється пропорційно його довжині в межах речення.
pub fn combined_ass(input: &[ExportChapter], chapters: &[ExportedChapter], title: &str) -> String {
    render_ass(&combined_cues(input, chapters), title)
}

fn combined_cues(input: &[ExportChapter], chapters: &[ExportedChapter]) -> Vec<Cue> {
    let mut cues = Vec::new();
    for chapter in input {
        let Some(exported) = chapters.iter().find(|item| item.index == chapter.index) else {
            continue;
        };
        for cue in &chapter.cues {
            let start = exported.start_ms.saturating_add(cue.start_ms);
            let end = exported
                .start_ms
                .saturating_add(cue.end_ms.max(cue.start_ms));
            cues.push(Cue {
                start_ms: start,
                end_ms: end,
                text: cue.text.clone(),
            });
        }
    }
    cues
}

pub(crate) struct EncodedBook {
    pub duration_ms: u64,
    pub chapters: Vec<ExportedChapter>,
}

fn validate_input(input: &ExportInput) -> Result<(), ExportError> {
    if input.chapters.is_empty() {
        return Err(ExportError::Invalid(
            "a book must contain at least one chapter".to_string(),
        ));
    }
    if input.metadata.title.trim().is_empty() {
        return Err(ExportError::Invalid(
            "book title must not be empty".to_string(),
        ));
    }
    for chapter in &input.chapters {
        if chapter.wav_path.as_os_str().is_empty() {
            return Err(ExportError::Invalid(format!(
                "chapter {} has no WAV path",
                chapter.index
            )));
        }
    }
    Ok(())
}

fn scan_chapters(chapters: &[ExportChapter]) -> Result<Vec<u64>, ExportError> {
    let mut frames = Vec::with_capacity(chapters.len());
    for chapter in chapters {
        let reader =
            hound::WavReader::open(&chapter.wav_path).map_err(|source| ExportError::Wav {
                path: chapter.wav_path.clone(),
                source,
            })?;
        validate_wav_spec(&chapter.wav_path, reader.spec())?;
        frames.push(u64::from(reader.len()));
    }
    Ok(frames)
}

fn validate_wav_spec(path: &Path, spec: hound::WavSpec) -> Result<(), ExportError> {
    if spec.channels != 1
        || spec.sample_rate != SAMPLE_RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(ExportError::WavFormat {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

pub(crate) fn read_wav(path: &Path) -> Result<Vec<i16>, ExportError> {
    let mut reader = hound::WavReader::open(path).map_err(|source| ExportError::Wav {
        path: path.to_path_buf(),
        source,
    })?;
    validate_wav_spec(path, reader.spec())?;
    reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| ExportError::Wav {
            path: path.to_path_buf(),
            source,
        })
}

/// Мілісекунди з початку книги для кожного кадру глави.
pub(crate) fn source_chapter_starts(frames: &[u64]) -> Vec<u64> {
    let mut cursor = 0u64;
    let mut starts = Vec::with_capacity(frames.len());
    for &count in frames {
        starts.push(cursor * 1000 / u64::from(SAMPLE_RATE));
        cursor += count;
    }
    starts
}

pub(crate) fn exported_chapters(
    input: &ExportInput,
    starts: &[u64],
    duration_ms: u64,
) -> Vec<ExportedChapter> {
    input
        .chapters
        .iter()
        .enumerate()
        .map(|(position, chapter)| {
            let start_ms = starts.get(position).copied().unwrap_or(0);
            let end_ms = starts.get(position + 1).copied().unwrap_or(duration_ms);
            ExportedChapter {
                index: chapter.index,
                title: chapter.title.clone(),
                start_ms,
                duration_ms: end_ms.saturating_sub(start_ms),
            }
        })
        .collect()
}

/// ISO 639-1 → ISO 639-2/B для MP4 `language`.
pub(crate) fn mp4_language(language: &str) -> String {
    let code = language.trim().to_ascii_lowercase();
    let mapped = match code.as_str() {
        "uk" | "ukr" => "ukr",
        "en" | "eng" => "eng",
        "ru" | "rus" => "rus",
        "de" | "deu" | "ger" => "deu",
        "fr" | "fra" | "fre" => "fra",
        "pl" | "pol" => "pol",
        "es" | "spa" => "spa",
        "it" | "ita" => "ita",
        _ => "und",
    };
    mapped.to_string()
}

struct CountingWriter<'a> {
    inner: &'a mut dyn Write,
    count: u64,
}

impl<'a> CountingWriter<'a> {
    fn new(inner: &'a mut dyn Write) -> Self {
        Self { inner, count: 0 }
    }
}

impl Write for CountingWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.count += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
