//! ASS (Advanced SubStation Alpha) із послівною караоке-розміткою.
//!
//! Модель поки не повертає точних послівних таймінгів, тому тривалість
//! кожного слова розподіляється пропорційно його довжині з невеликою
//! паузою після розділових знаків. Це дає придатну для відеомонтажу
//! караоке-доріжку, а після появи word-level alignment достатньо замінити
//! `karaoke_text` на реальні межі.

use crate::subtitles::{normalized, Cue};

/// Мінімальна тривалість караоке-слова в сантисекундах.
const MIN_WORD_CS: u64 = 1;

/// ASS-файл із заголовком, стилем `Default` і подіями `Dialogue`.
pub fn render_ass(cues: &[Cue], title: &str) -> String {
    let cues = normalized(cues);
    let mut out = header(title);
    for cue in &cues {
        out.push_str("Dialogue: 0,");
        out.push_str(&timestamp(cue.start_ms));
        out.push(',');
        out.push_str(&timestamp(cue.end_ms));
        out.push_str(",Default,,0,0,0,,");
        out.push_str(&karaoke_text(&cue.text, cue.start_ms, cue.end_ms));
        out.push('\n');
    }
    out
}

fn header(title: &str) -> String {
    let title = title.replace(['\r', '\n'], " ");
    format!(
        "[Script Info]\n\
         ; Promin TTS ASS subtitles\n\
         Title: {title}\n\
         ScriptType: v4.00+\n\
         WrapStyle: 0\n\
         ScaledBorderAndShadow: yes\n\
         YCbCr Matrix: TV.601\n\
         PlayResX: 1920\n\
         PlayResY: 1080\n\
         \n\
         [V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,80,80,90,1\n\
         \n\
         [Events]\n\
         Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
}

/// `{\k<cs>}слово` для кожного слова; сума `\k` дорівнює тривалості cue.
fn karaoke_text(text: &str, start_ms: u64, end_ms: u64) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return escape(text.trim());
    }
    let duration_cs = (end_ms.saturating_sub(start_ms) / 10).max(MIN_WORD_CS);
    if duration_cs < words.len() as u64 * MIN_WORD_CS {
        return format!("{{\\k{duration_cs}}}{}", escape(&words.join(" ")));
    }

    let weights: Vec<u64> = words
        .iter()
        .map(|word| {
            let base = word.chars().count().max(1) as u64;
            let pause = if word.ends_with(['.', '!', '?', '…']) {
                3
            } else if word.ends_with([',', ';', ':', '—', '–']) {
                2
            } else {
                0
            };
            base + pause
        })
        .collect();
    let total_weight: u64 = weights.iter().sum();

    let mut out = String::new();
    let mut elapsed = 0u64;
    let mut weight_before = 0u64;
    for (index, word) in words.iter().enumerate() {
        weight_before += weights[index];
        let duration = if index + 1 == words.len() {
            duration_cs.saturating_sub(elapsed)
        } else {
            let remaining_words = (words.len() - index - 1) as u64;
            let max_for_this = duration_cs
                .saturating_sub(elapsed)
                .saturating_sub(remaining_words);
            let target = duration_cs.saturating_mul(weight_before) / total_weight;
            target
                .saturating_sub(elapsed)
                .clamp(MIN_WORD_CS, max_for_this.max(MIN_WORD_CS))
        };
        out.push_str(&format!("{{\\k{duration}}}{}", escape(word)));
        if index + 1 != words.len() {
            out.push(' ');
        }
        elapsed += duration;
    }
    out
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '\n' => out.push_str("\\N"),
            '\r' => {}
            _ => out.push(character),
        }
    }
    out
}

fn timestamp(ms: u64) -> String {
    let centis = ms / 10;
    let hours = centis / 360_000;
    let minutes = (centis / 6_000) % 60;
    let seconds = (centis / 100) % 60;
    let centis = centis % 100;
    format!("{hours}:{minutes:02}:{seconds:02}.{centis:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start_ms: u64, end_ms: u64, text: &str) -> Cue {
        Cue {
            start_ms,
            end_ms,
            text: text.to_string(),
        }
    }

    fn karaoke_total(text: &str) -> u64 {
        let mut rest = text;
        let mut total = 0;
        while let Some(position) = rest.find("{\\k") {
            rest = &rest[position + 3..];
            let end = rest.find('}').expect("karaoke tag is closed");
            total += rest[..end].parse::<u64>().expect("karaoke duration");
            rest = &rest[end + 1..];
        }
        total
    }

    #[test]
    fn renders_header_and_dialogue() {
        let ass = render_ass(&[cue(0, 1_000, "Привіт світ")], "Книга");
        assert!(ass.starts_with("[Script Info]\n"));
        assert!(ass.contains("Title: Книга"));
        assert!(ass.contains("[V4+ Styles]"));
        assert!(ass.contains("Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\k"));
    }

    #[test]
    fn karaoke_covers_the_whole_cue() {
        let ass = render_ass(&[cue(1_000, 3_500, "Перше слово. Друге слово!")], "T");
        let line = ass
            .lines()
            .find(|line| line.starts_with("Dialogue:"))
            .unwrap();
        let text = line.split(",,,").last().unwrap();
        assert_eq!(karaoke_total(text), 250);
        assert!(text.contains("{\\k"));
        assert!(text.contains("Перше"));
    }

    #[test]
    fn escapes_ass_special_characters() {
        let ass = render_ass(&[cue(0, 1_000, "a {b} c\\d")], "T");
        let line = ass
            .lines()
            .find(|line| line.starts_with("Dialogue:"))
            .unwrap();
        assert!(line.contains("\\{b\\}"));
        assert!(line.contains("c\\\\d"));
    }

    #[test]
    fn empty_cues_render_header_only() {
        let ass = render_ass(&[], "T");
        assert!(ass.contains("[Events]"));
        assert!(!ass.contains("Dialogue:"));
    }

    #[test]
    fn very_short_cues_fall_back_to_one_tag() {
        let ass = render_ass(&[cue(0, 1, "Один. Два.")], "T");
        let line = ass
            .lines()
            .find(|line| line.starts_with("Dialogue:"))
            .unwrap();
        assert_eq!(karaoke_total(line.split(",,,").last().unwrap()), 1);
    }
}
