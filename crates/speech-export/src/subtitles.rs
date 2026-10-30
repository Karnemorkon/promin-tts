//! Рендеринг SRT/VTT із таймінгів чанків.

/// Один рядок субтитрів у глобальному таймлайні.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// SRT-файл (нумерація з 1, кома в мілісекундах).
pub fn render_srt(cues: &[Cue]) -> String {
    let cues = normalized(cues);
    let mut out = String::new();
    for (index, cue) in cues.iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(cue.start_ms, ','),
            timestamp(cue.end_ms, ',')
        ));
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// WebVTT-файл (крапка в мілісекундах).
pub fn render_vtt(cues: &[Cue]) -> String {
    let cues = normalized(cues);
    let mut out = String::from("WEBVTT\n\n");
    for cue in &cues {
        out.push_str(&format!(
            "{} --> {}\n",
            timestamp(cue.start_ms, '.'),
            timestamp(cue.end_ms, '.')
        ));
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub(crate) fn normalized(cues: &[Cue]) -> Vec<Cue> {
    let mut cues: Vec<Cue> = cues
        .iter()
        .filter(|cue| !cue.text.trim().is_empty())
        .cloned()
        .collect();
    cues.sort_by_key(|cue| (cue.start_ms, cue.end_ms));
    for cue in &mut cues {
        if cue.end_ms <= cue.start_ms {
            cue.end_ms = cue.start_ms.saturating_add(1);
        }
    }
    cues
}

fn timestamp(ms: u64, separator: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms / 60_000) % 60;
    let seconds = (ms / 1000) % 60;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_srt_with_expected_timestamps() {
        let cues = vec![Cue {
            start_ms: 1_234,
            end_ms: 65_432,
            text: "Привіт".to_string(),
        }];
        let srt = render_srt(&cues);
        assert!(srt.starts_with("1\n00:00:01,234 --> 00:01:05,432\nПривіт\n\n"));
    }

    #[test]
    fn renders_vtt_header_and_dots() {
        let cues = vec![Cue {
            start_ms: 0,
            end_ms: 1_500,
            text: "Текст".to_string(),
        }];
        let vtt = render_vtt(&cues);
        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:01.500"));
    }

    #[test]
    fn sorts_and_skips_empty_cues() {
        let cues = vec![
            Cue {
                start_ms: 2_000,
                end_ms: 2_100,
                text: "другий".to_string(),
            },
            Cue {
                start_ms: 1_000,
                end_ms: 1_100,
                text: "перший".to_string(),
            },
            Cue {
                start_ms: 0,
                end_ms: 10,
                text: "  ".to_string(),
            },
        ];
        let srt = render_srt(&cues);
        assert!(srt.starts_with("1\n00:00:01,000"));
        assert!(srt.contains("2\n00:00:02,000"));
        assert_eq!(srt.matches("-->").count(), 2);
    }
}
