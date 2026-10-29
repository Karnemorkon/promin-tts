use std::io::Cursor;
use std::path::{Path, PathBuf};

use speech_export::{
    combined_ass, combined_subtitles, export_book, Cue, ExportChapter, ExportFormat, ExportInput,
    ExportMetadata, NoopObserver,
};

fn tone(seconds: f32, frequency: f32) -> Vec<i16> {
    let rate = 24_000.0f32;
    let count = (rate * seconds) as usize;
    (0..count)
        .map(|index| {
            let time = index as f32 / rate;
            let envelope = (std::f32::consts::PI * (time * 1.5).min(1.0)).sin();
            let value = 0.4 * envelope * (2.0 * std::f32::consts::PI * frequency * time).sin();
            (value * 32767.0) as i16
        })
        .collect()
}

fn write_wav(path: &Path, pcm: &[i16]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &sample in pcm {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("speech-export-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn book(dir: &Path, format: ExportFormat) -> ExportInput {
    let first = dir.join("chapter-0.wav");
    let second = dir.join("chapter-1.wav");
    write_wav(&first, &tone(1.5, 220.0));
    write_wav(&second, &tone(1.0, 330.0));
    ExportInput {
        format,
        bitrate_kbps: 0,
        metadata: ExportMetadata {
            title: "Тестова книга".to_string(),
            author: "Автор".to_string(),
            album: String::new(),
            narrator: Some("Диктор".to_string()),
            year: Some("2026".to_string()),
            description: Some("Опис".to_string()),
            language: "uk".to_string(),
        },
        cover: None,
        chapters: vec![
            ExportChapter {
                index: 0,
                title: "Розділ 1".to_string(),
                wav_path: first,
                cues: vec![
                    Cue {
                        start_ms: 0,
                        end_ms: 700,
                        text: "Перше речення.".to_string(),
                    },
                    Cue {
                        start_ms: 700,
                        end_ms: 1_400,
                        text: "Друге речення.".to_string(),
                    },
                ],
            },
            ExportChapter {
                index: 1,
                title: "Розділ 2".to_string(),
                wav_path: second,
                cues: vec![Cue {
                    start_ms: 100,
                    end_ms: 900,
                    text: "Третє речення.".to_string(),
                }],
            },
        ],
    }
}

#[test]
fn exports_every_format_with_chapters_and_subtitles() {
    for format in [
        ExportFormat::M4b,
        ExportFormat::Mp3,
        ExportFormat::Ogg,
        ExportFormat::Flac,
    ] {
        let dir = temp_dir();
        let input = book(&dir, format);
        let mut output = Vec::new();
        let summary = export_book(&input, &mut output, &mut NoopObserver)
            .unwrap_or_else(|error| panic!("{format:?}: {error}"));

        assert_eq!(summary.format, format);
        assert_eq!(summary.bytes_written, output.len() as u64);
        // M4B додає priming-фрейм на кожну главу, тому довший за джерело.
        match format {
            ExportFormat::M4b => {
                assert!(summary.duration_ms >= 2_500 && summary.duration_ms <= 2_700)
            }
            _ => assert_eq!(summary.duration_ms, 2_500),
        }
        assert_eq!(summary.chapters.len(), 2);
        assert_eq!(summary.chapters[0].start_ms, 0);
        assert!(summary.chapters[1].start_ms >= 1_500);

        match format {
            ExportFormat::M4b => {
                assert_eq!(&output[4..8], b"ftyp");
                assert!(output.windows(4).any(|window| window == b"M4B "));
            }
            ExportFormat::Mp3 => {
                assert!(output.starts_with(b"ID3"));
                assert!(output.windows(4).any(|window| window == b"CHAP"));
                assert!(output.windows(4).any(|window| window == b"CTOC"));
            }
            ExportFormat::Ogg => {
                assert!(output.starts_with(b"OggS"));
                assert!(output.windows(8).any(|window| window == b"OpusHead"));
                assert!(output.windows(8).any(|window| window == b"OpusTags"));
                assert!(output.windows(10).any(|window| window == b"CHAPTER002"));
            }
            ExportFormat::Flac => {
                assert!(output.starts_with(b"fLaC"));
                assert!(output.windows(5).any(|window| window == b"TITLE"));
                assert!(output.windows(10).any(|window| window == b"CHAPTER002"));
                assert!(output.windows(3).any(|window| window == b"PNG"));
            }
        }

        let (srt, vtt) = combined_subtitles(&input.chapters, &summary.chapters);
        assert!(srt.contains("00:00:00,000 --> 00:00:00,700"));
        assert!(srt.contains("Третє речення."));
        assert!(vtt.starts_with("WEBVTT\n\n"));
        let third_start = summary.chapters[1].start_ms + 100;
        let expected = format!(
            "{:02}:{:02}:{:02}.{:03}",
            third_start / 3_600_000,
            (third_start / 60_000) % 60,
            (third_start / 1000) % 60,
            third_start % 1000
        );
        assert!(
            vtt.contains(&expected),
            "VTT should contain {expected}:\n{vtt}"
        );

        std::fs::write(dir.join(format!("book.{}", format.extension())), &output).unwrap();
    }
}

#[test]
fn rejects_wav_with_wrong_sample_rate() {
    let dir = temp_dir();
    let path = dir.join("wrong.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for _ in 0..1_000 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
    let input = ExportInput {
        format: ExportFormat::Mp3,
        bitrate_kbps: 0,
        metadata: ExportMetadata {
            title: "Книга".to_string(),
            ..Default::default()
        },
        cover: None,
        chapters: vec![ExportChapter {
            index: 0,
            title: "Розділ".to_string(),
            wav_path: path,
            cues: Vec::new(),
        }],
    };
    let error = export_book(&input, &mut Vec::new(), &mut NoopObserver).unwrap_err();
    assert!(matches!(
        error,
        speech_export::ExportError::WavFormat { .. }
    ));
}

#[test]
fn flac_roundtrip_is_lossless_and_keeps_metadata() {
    let dir = temp_dir();
    let input = book(&dir, ExportFormat::Flac);
    let mut output = Vec::new();
    let summary = export_book(&input, &mut output, &mut NoopObserver).unwrap();
    assert_eq!(summary.format, ExportFormat::Flac);
    assert_eq!(summary.duration_ms, 2_500);
    assert!(summary.bitrate_kbps > 0);

    let mut reader = claxon::FlacReader::new(Cursor::new(&output)).unwrap();
    let info = reader.streaminfo();
    assert_eq!(info.sample_rate, 24_000);
    assert_eq!(info.channels, 1);
    assert_eq!(info.bits_per_sample, 16);
    assert_eq!(reader.get_tag("TITLE").next(), Some("Тестова книга"));
    assert_eq!(reader.get_tag("CHAPTER002NAME").next(), Some("Розділ 2"));

    let decoded: Vec<i32> = reader.samples().map(|sample| sample.unwrap()).collect();
    let mut expected = Vec::new();
    for chapter in &input.chapters {
        let mut wav = hound::WavReader::open(&chapter.wav_path).unwrap();
        expected.extend(
            wav.samples::<i16>()
                .map(|sample| i32::from(sample.unwrap())),
        );
    }
    assert_eq!(decoded, expected);
}

#[test]
fn ass_subtitles_use_word_karaoke_and_book_timeline() {
    let dir = temp_dir();
    let input = book(&dir, ExportFormat::Flac);
    let mut output = Vec::new();
    let summary = export_book(&input, &mut output, &mut NoopObserver).unwrap();
    let ass = combined_ass(&input.chapters, &summary.chapters, "Тестова книга");
    assert!(ass.contains("Title: Тестова книга"));
    assert!(ass.contains("Dialogue: 0,0:00:00.00,0:00:00.70"));
    assert!(ass.contains("{\\k"));
    assert!(ass.contains("Перше"));
    let third_start = summary.chapters[1].start_ms + 100;
    let timestamp = format!(
        "{}:{:02}:{:02}.{:02}",
        third_start / 3_600_000,
        (third_start / 60_000) % 60,
        (third_start / 1_000) % 60,
        (third_start % 1_000) / 10
    );
    assert!(
        ass.contains(&timestamp),
        "ASS should contain {timestamp}:\n{ass}"
    );
}

#[test]
fn rejects_out_of_range_bitrate() {
    let dir = temp_dir();
    let mut input = book(&dir, ExportFormat::Ogg);
    input.bitrate_kbps = 256;
    let error = export_book(&input, &mut Vec::new(), &mut NoopObserver).unwrap_err();
    assert!(matches!(error, speech_export::ExportError::Invalid(_)));
}
