//! SRT/VTT із таймінгів синтезованих чанків.
//!
//! Кожен чанк ділиться на речення пропорційно кількості символів. Межі
//! чанків уже враховують кросфейд (див. `SynthChapterOutput::chunk_timings`).

use speech_core::projects::ChunkTiming;
use speech_text::segment::split_sentences;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Cue {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

/// SRT-файл (нумерація з 1, кома в мілісекундах).
pub fn render_srt(timings: &[ChunkTiming]) -> String {
    let mut out = String::new();
    for (index, cue) in cues(timings).iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(cue.start_ms, ','),
            timestamp(cue.end_ms, ',')
        ));
        out.push_str(&cue.text);
        out.push_str("\n\n");
    }
    out
}

/// WebVTT-файл (крапка в мілісекундах).
pub fn render_vtt(timings: &[ChunkTiming]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for cue in cues(timings) {
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(cue.start_ms, '.'),
            timestamp(cue.end_ms, '.')
        ));
        out.push_str(&cue.text);
        out.push_str("\n\n");
    }
    out
}

fn cues(timings: &[ChunkTiming]) -> Vec<Cue> {
    let mut cues = Vec::new();
    for timing in timings {
        if timing.end_ms <= timing.start_ms {
            continue;
        }
        cues.extend(split_chunk(timing));
    }
    cues
}

fn split_chunk(timing: &ChunkTiming) -> Vec<Cue> {
    let sentences: Vec<String> = split_sentences(&timing.text)
        .into_iter()
        .map(|sentence| normalize(&sentence))
        .filter(|sentence| !sentence.is_empty())
        .collect();
    if sentences.is_empty() {
        return Vec::new();
    }
    if sentences.len() == 1 {
        return vec![Cue {
            start_ms: timing.start_ms,
            end_ms: timing.end_ms,
            text: sentences.into_iter().next().expect("one sentence"),
        }];
    }

    let duration = timing.end_ms - timing.start_ms;
    if duration < sentences.len() as u64 {
        return vec![Cue {
            start_ms: timing.start_ms,
            end_ms: timing.end_ms,
            text: sentences.join(" "),
        }];
    }

    let weights: Vec<u64> = sentences
        .iter()
        .map(|sentence| sentence.chars().count().max(1) as u64)
        .collect();
    let total_weight: u64 = weights.iter().sum();
    let mut cursor = timing.start_ms;
    let mut cues = Vec::with_capacity(sentences.len());
    for (index, sentence) in sentences.iter().enumerate() {
        let remaining_cues = sentences.len() - index;
        let cue_end = if remaining_cues == 1 {
            timing.end_ms
        } else {
            let share = duration.saturating_mul(weights[index]) / total_weight;
            let latest = timing.end_ms - (remaining_cues as u64 - 1);
            (cursor + share.max(1)).min(latest)
        };
        if cue_end <= cursor {
            continue;
        }
        cues.push(Cue {
            start_ms: cursor,
            end_ms: cue_end,
            text: sentence.clone(),
        });
        cursor = cue_end;
    }
    cues
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn timestamp(ms: u64, separator: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms / 60_000) % 60;
    let seconds = (ms / 1_000) % 60;
    let millis = ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(start_ms: u64, end_ms: u64, text: &str) -> ChunkTiming {
        ChunkTiming {
            start_ms,
            end_ms,
            text: text.to_string(),
            role: None,
        }
    }

    #[test]
    fn splits_chunks_into_sentences() {
        let timings = vec![
            timing(0, 2_000, "Привіт. Як справи?"),
            timing(2_000, 3_000, "Добре!"),
        ];
        let cues = cues(&timings);
        assert_eq!(cues.len(), 3);
        assert_eq!(cues[0].start_ms, 0);
        assert_eq!(cues[0].text, "Привіт.");
        assert_eq!(cues[1].text, "Як справи?");
        assert_eq!(cues[1].end_ms, 2_000);
        assert_eq!(cues[2].start_ms, 2_000);
        assert_eq!(cues[2].end_ms, 3_000);
        assert_eq!(cues[2].text, "Добре!");
    }

    #[test]
    fn renders_srt_with_comma_and_numbering() {
        let timings = vec![timing(0, 1_500, "Перше речення. Друге речення.")];
        let srt = render_srt(&timings);
        assert!(srt.starts_with("1\n00:00:00,000 --> "));
        assert!(srt.contains("00:00:01,500\nДруге речення."));
        assert!(srt.contains("\n2\n"));
    }

    #[test]
    fn renders_vtt_with_header_and_dots() {
        let timings = vec![timing(65_500, 67_000, "Текст.")];
        let vtt = render_vtt(&timings);
        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:01:05.500 --> 00:01:07.000\nТекст."));
    }

    #[test]
    fn cleans_whitespace_and_skips_empty_chunks() {
        let timings = vec![
            timing(0, 1_000, "  Перше\n речення.  "),
            timing(1_000, 1_000, "Нульова тривалість."),
            timing(1_000, 2_000, "   "),
        ];
        let srt = render_srt(&timings);
        assert!(srt.contains("Перше речення."));
        assert!(!srt.contains("Нульова"));
        assert_eq!(srt.matches("-->").count(), 1);
    }

    #[test]
    fn empty_timings_render_empty_files() {
        assert_eq!(render_srt(&[]), "");
        assert_eq!(render_vtt(&[]), "WEBVTT\n\n");
    }

    #[test]
    fn very_short_chunks_merge_sentences() {
        let timings = vec![timing(0, 1, "Один. Два.")];
        let cues = cues(&timings);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Один. Два.");
        assert_eq!(cues[0].end_ms, 1);
    }
}
