//! Транскрипція українського тексту в фонеми (IPA) — порт ipa_uk.cpp /
//! Python reference (ukrainian-word-stress / ipa-uk). Парність 1:1.

use crate::repl::to_fancy_template;
use fancy_regex::Regex;

const ACUTE: &str = "\u{02C8}"; // ˈ
const IN_ACUTE: &str = "\u{0301}";
const IN_GRAVE: &str = "\u{0300}";
const SENTINEL: &str = "\u{2040}"; // ⁀
const GRAVE_MARK: &str = "\u{02CC}"; // ˌ
const LENGTH: &str = "\u{02D0}"; // ː
const PALJ: &str = "\u{02B2}"; // ʲ
const SYL: &str = "\u{032F}"; // ̯

const VOWEL: &str = "a\u{025B}\u{026A}u\u{0254}\u{0250}o\u{028A}ei";
const CONS_NO_W: &str = "bdz\u{0292}\u{0261}\u{0266}mnlrpftskx\u{0283}j";
const CONS: &str =
    "bdz\u{0292}\u{0261}\u{0266}mnlrpftskx\u{0283}j\u{028B}\u{03B2}\u{031E}w\u{028D}";
const PALAT: &str = "tdsznlrbp\u{028B}f\u{0261}mk\u{0266}x\u{0283}\u{0292}";

fn re(p: &str) -> Regex {
    Regex::new(p).expect("ipa: invalid regex")
}

fn sub_all<P: AsRef<str>, R: AsRef<str>>(pattern: P, replacement: R, s: &str) -> String {
    let template = to_fancy_template(replacement.as_ref());
    re(pattern.as_ref())
        .replace_all(s, template.as_str())
        .into_owned()
}

fn utf8_tolower(s: &str) -> String {
    s.chars()
        .map(|c| char::from_u32(crate::utf8::cp_lower(c as u32)).unwrap_or(c))
        .collect()
}

fn in_set(set: &[&str], v: &str) -> bool {
    set.contains(&v)
}

/// Транскрипція. `check_accent=true` вмикає перевірку наявності наголосу
/// (AccentIsMissing) та моносилабічний сентинел; Phonemizer викликає з `false`.
pub fn transcribe_impl(text: &str, check_accent: bool) -> anyhow::Result<String> {
    const VOICING: [(&str, &str); 13] = [
        ("p", "b"),
        ("f", "v"),
        ("t", "d"),
        ("t\u{02B2}", "d\u{02B2}"),
        ("s", "z"),
        ("s\u{02B2}", "z\u{02B2}"),
        ("\u{0283}", "\u{0292}"),
        ("k", "\u{0261}"),
        ("x", "\u{0266}"),
        ("t\u{0361}s", "d\u{0361}z"),
        ("t\u{0361}s\u{02B2}", "d\u{0361}z\u{02B2}"),
        ("t\u{0361}\u{0283}", "d\u{0361}\u{0292}"),
        ("\u{0283}t\u{0361}\u{0283}", "\u{0292}d\u{0361}\u{0292}"),
    ];
    // Повний список onset з C++ (fl, xl теж є).
    const ONSET_FULL: [&str; 27] = [
        "spr",
        "str",
        "skr",
        "spl",
        "skl",
        "sp",
        "st",
        "sk",
        "sf",
        "sx",
        "pr",
        "br",
        "tr",
        "dr",
        "kr",
        "gr",
        "\u{0266}r",
        "fr",
        "xr",
        "pl",
        "bl",
        "kl",
        "gl",
        "\u{0266}l",
        "fl",
        "xl",
        "j",
    ];
    const ORTHO: [(&str, &str); 14] = [
        ("\u{043D}\u{0442}\u{0441}\u{044C}\u{043A}", "\u{043D}\u{044C}\u{0441}\u{044C}\u{043A}"),
        ("\u{0441}\u{0442}\u{0441}\u{044C}\u{043A}", "\u{0441}\u{044C}\u{043A}"),
        ("\u{043D}\u{0442}\u{0441}\u{0442}", "\u{043D}\u{0441}\u{0442}"),
        ("\u{0441}\u{0442}\u{0447}", "\u{0448}\u{0447}"),
        ("\u{0441}\u{0442}\u{0434}", "\u{0437}\u{0434}"),
        ("\u{0441}\u{0442}\u{0441}", "\u{0441}\u{02D0}"),
        ("#\u{0437}\u{0448}", "#\u{0448}\u{02D0}"),
        ("\u{0437}\u{0448}", "\u{0436}\u{0448}"),
        ("#\u{0437}\u{0447}", "#\u{0448}\u{0447}"),
        ("\u{0437}\u{0447}", "\u{0436}\u{0447}"),
        (
            "([\u{0431}\u{0432}\u{0433}\u{0491}\u{0434}])\\1",
            "$1\u{02D0}",
        ),
        ("([^\u{0434}]+)\u{0436}\u{0436}", "$1\u{0436}\u{02D0}"),
        ("([^\u{0434}]+)\u{0437}\u{0437}", "$1\u{0437}\u{02D0}"),
        (
            "([\u{0439}\u{043A}\u{043B}\u{043C}\u{043D}\u{043F}\u{0440}\u{0441}\u{0442}\u{0444}\u{0445}\u{0446}\u{0447}\u{0448}\u{0449}])\\1",
            "$1\u{02D0}",
        ),
    ];
    // + 2 правила "ддж"/"дзз" (з C++ після \\1-правил).
    const ORTHO_TAIL: [(&str, &str); 2] = [
        (
            "\u{0434}\u{0436}\u{0434}\u{0436}",
            "\u{0434}\u{0436}\u{02D0}",
        ),
        (
            "\u{0434}\u{0437}\u{0434}\u{0437}",
            "\u{0434}\u{0437}\u{02D0}",
        ),
    ];
    const MAP3: [(&str, &str); 2] = [
        ("\u{0434}\u{0437}\u{044C}", "d\u{0361}z\u{02B2}"),
        ("\u{0442}\u{044C}\u{0441}", "t\u{0361}s\u{02B2}\u{02D0}"),
    ];
    const MAP2: [(&str, &str); 10] = [
        ("\u{0434}\u{0436}", "d\u{0361}\u{0292}"),
        ("\u{0434}\u{0437}", "d\u{0361}z"),
        ("\u{0434}\u{0441}", "d\u{0361}zs"),
        ("\u{0434}\u{0448}", "d\u{0361}\u{0292}\u{0283}"),
        ("\u{0434}\u{0447}", "d\u{0361}\u{0292}t\u{0361}\u{0283}"),
        ("\u{0434}\u{0446}", "d\u{0361}zt\u{0361}s"),
        ("\u{0442}\u{0441}", "t\u{0361}s"),
        ("\u{0442}\u{0448}", "t\u{0361}\u{0283}\u{0283}"),
        ("\u{0442}\u{0447}", "t\u{0361}\u{0283}\u{02D0}"),
        ("\u{0442}\u{0446}", "t\u{0361}s\u{02D0}"),
    ];
    const MAP1: [(&str, &str); 36] = [
        ("\u{0430}", "a"),
        ("\u{0431}", "b"),
        ("\u{0432}", "\u{028B}"),
        ("\u{0433}", "\u{0266}"),
        ("\u{0491}", "\u{0261}"),
        ("\u{0434}", "d"),
        ("\u{0435}", "\u{025B}"),
        ("\u{0454}", "j\u{025B}"),
        ("\u{0436}", "\u{0292}"),
        ("\u{0437}", "z"),
        ("\u{0438}", "\u{026A}"),
        ("\u{0456}", "i"),
        ("\u{0457}", "ji"),
        ("\u{0439}", "j"),
        ("\u{043A}", "k"),
        ("\u{043B}", "l"),
        ("\u{043C}", "m"),
        ("\u{043D}", "n"),
        ("\u{043E}", "\u{0254}"),
        ("\u{043F}", "p"),
        ("\u{0440}", "r"),
        ("\u{0441}", "s"),
        ("\u{0442}", "t"),
        ("\u{0443}", "u"),
        ("\u{0444}", "f"),
        ("\u{0445}", "x"),
        ("\u{0446}", "t\u{0361}s"),
        ("\u{0447}", "t\u{0361}\u{0283}"),
        ("\u{0448}", "\u{0283}"),
        ("\u{0449}", "\u{0283}t\u{0361}\u{0283}"),
        ("\u{044C}", "\u{02B2}"),
        ("\u{044E}", "ju"),
        ("\u{044F}", "ja"),
        ("\u{2019}", "j"),
        (IN_ACUTE, ACUTE),
        (IN_GRAVE, GRAVE_MARK),
    ];

    let mut text = text.to_string();

    // Grave decomposition.
    for (from, to) in [
        ("\u{0450}", "e\u{0300}"),
        ("\u{0400}", "E\u{0300}"),
        ("\u{045D}", "i\u{0300}"),
        ("\u{040D}", "I\u{0300}"),
    ] {
        if text.contains(from) {
            text = text.replace(from, to);
        }
    }
    text = utf8_tolower(&text);

    if check_accent && !text.contains(IN_ACUTE) && !text.contains(IN_GRAVE) {
        let n = [
            "\u{0430}", "\u{0435}", "\u{0454}", "\u{0438}", "\u{0456}", "\u{0457}", "\u{043E}",
            "\u{0443}", "\u{044E}", "\u{044F}",
        ]
        .iter()
        .map(|v| text.matches(v).count())
        .sum::<usize>();
        if n > 1 {
            anyhow::bail!("AccentIsMissing");
        }
    }

    text = sub_all(r"\s+", " ", &text);

    let norm = sub_all(r"[\s\-]+", "\u{01}", &text);
    let parts: Vec<&str> = norm.split('\u{01}').collect();

    let stress_class = format!("[{ACUTE}{GRAVE_MARK}]");
    let stress_or_sent = format!("[{ACUTE}{GRAVE_MARK}{SENTINEL}]");
    let stress_neg = format!("[^{ACUTE}{GRAVE_MARK}{SENTINEL}]");
    let voiced_obstr = "[bdz\u{0292}\u{0261}\u{0266}]";
    let cons_c = format!("[{CONS}]");
    let cons_no_w_c = format!("[{CONS_NO_W}]");
    let vowel_c = format!("[{VOWEL}]");
    let palat_c = format!("[{PALAT}]");

    let mut pronuns: Vec<String> = Vec::with_capacity(parts.len());
    for part in parts {
        let mut ph = format!("#{part}#");

        for (p, x) in ORTHO {
            ph = sub_all(p, x, &ph);
        }
        for (p, x) in ORTHO_TAIL {
            ph = sub_all(p, x, &ph);
        }
        ph = sub_all(r"'", "%", &ph);
        for (p, x) in MAP3 {
            ph = sub_all(p, x, &ph);
        }
        for (p, x) in MAP2 {
            ph = sub_all(p, x, &ph);
        }
        for (p, x) in MAP1 {
            ph = sub_all(p, x, &ph);
        }

        // Перенесення наголосу перед голосну: ([aɛiɪuɔ])([ˈˌ]) -> $2$1
        ph = sub_all(
            format!("([a\u{025B}i\u{026A}u\u{0254}])({stress_class})"),
            "$2$1",
            &ph,
        );

        // Моносилабічний автонаголос (лише за check_accent).
        if check_accent {
            let cnt = re("[a\u{025B}i\u{026A}u\u{0254}]")
                .captures_iter(&ph)
                .filter_map(|c| c.ok())
                .count();
            if cnt == 1 {
                ph = sub_all(
                    "([a\u{025B}i\u{026A}u\u{0254}])",
                    format!("{SENTINEL}$1"),
                    &ph,
                );
            }
        }

        // Палаталізація перед i / j.
        ph = sub_all(
            format!("({palat_c})({LENGTH}?)({stress_or_sent}?)i"),
            format!("$1{PALJ}$2$3i"),
            &ph,
        );
        ph = sub_all(
            format!("({palat_c})({LENGTH}?)j"),
            format!("$1{PALJ}$2"),
            &ph,
        );
        ph = sub_all(format!("{PALJ}{LENGTH}j"), format!("{PALJ}{LENGTH}"), &ph);
        ph = sub_all(
            format!("st\u{0361}s{PALJ}({LENGTH}?)"),
            format!("s{PALJ}t\u{0361}s{PALJ}"),
            &ph,
        );

        // Озвучення (порядкове, регресивне).
        for (vl, vd) in VOICING {
            ph = sub_all(format!("{vl}({voiced_obstr}+)"), format!("{vd}$1"), &ph);
        }

        // Асиміляція м'яких приголосних.
        ph = sub_all(
            format!("([tdszn])(.){PALJ}"),
            format!("$1{PALJ}$2{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("([tdsznl])t\u{0361}s{PALJ}"),
            format!("$1{PALJ}t\u{0361}s{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("([tdsznl])d\u{0361}z{PALJ}"),
            format!("$1{PALJ}d\u{0361}z{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("t\u{0361}s(.){PALJ}"),
            format!("t\u{0361}s{PALJ}$1{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("d\u{0361}z(.){PALJ}"),
            format!("d\u{0361}z{PALJ}$1{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("d\u{0361}zt\u{0361}s{PALJ}"),
            format!("d\u{0361}z{PALJ}t\u{0361}s{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("t\u{0361}sd\u{0361}z{PALJ}"),
            format!("t\u{0361}s{PALJ}d\u{0361}z{PALJ}"),
            &ph,
        );

        // Асиміляція шиплячих + свистячих.
        ph = sub_all(
            format!("\u{0292}t\u{0361}s{PALJ}"),
            format!("z{PALJ}t\u{0361}s{PALJ}"),
            &ph,
        );
        ph = sub_all(
            format!("t\u{0361}\u{0283}t\u{0361}s{PALJ}"),
            format!("t\u{0361}s{PALJ}{LENGTH}"),
            &ph,
        );
        ph = sub_all(
            format!("\u{0283}t\u{0361}s{PALJ}"),
            format!("s{PALJ}t\u{0361}s{PALJ}"),
            &ph,
        );
        ph = sub_all(format!("\u{0283}s{PALJ}"), format!("s{PALJ}{LENGTH}"), &ph);

        // Свистячі перед шиплячими.
        ph = sub_all("z\u{0292}", format!("\u{0292}{LENGTH}"), &ph);
        ph = sub_all("s\u{0283}", format!("\u{0283}{LENGTH}"), &ph);
        ph = sub_all("zt\u{0361}\u{0283}", "\u{0292}t\u{0361}\u{0283}", &ph);
        ph = sub_all("zd\u{0361}\u{0292}", "\u{0292}d\u{0361}\u{0292}", &ph);
        ph = sub_all("t\u{0361}\u{0292}", "d\u{0361}\u{0292}", &ph);
        ph = sub_all("t\u{0361}z", "d\u{0361}z", &ph);

        // Прибирання зайвої палаталізації.
        ph = sub_all(
            format!(
                "([^a\u{025B}i\u{026A}u\u{0254}]+){PALJ}([^a\u{025B}i\u{026A}u\u{0254}]+){PALJ}([^a\u{025B}i\u{026A}u\u{0254}]+){PALJ}"
            ),
            format!("$1$2{PALJ}$3{PALJ}"),
            &ph,
        );

        // Аллофони.
        ph = sub_all(format!("({stress_neg})a"), "$1\u{0250}", &ph);
        ph = sub_all(format!("({stress_neg})u"), "$1\u{028A}", &ph);
        // Правило з C++ з квадратною дужкою всередині групи:
        ph = sub_all(
            format!(
                "\u{0254}([bdz\u{0292}\u{0261}\u{0266}mnlrpftskx{PALJ}\u{0283}\u{0361}]+)([{ACUTE}{GRAVE_MARK}{SENTINEL}][ui\u{028A}])"
            ),
            "o$1$2",
            &ph,
        );
        ph = sub_all(format!("({stress_neg})\u{025B}"), "$1e", &ph);

        ph = sub_all(SENTINEL, "", &ph);

        // Аллофони ʋ.
        ph = sub_all(
            format!("({vowel_c})\u{028B}([{CONS_NO_W}#])"),
            format!("$1u{SYL}$2"),
            &ph,
        );
        ph = sub_all(
            format!("\u{028B}([{ACUTE}{GRAVE_MARK}]?[\u{0254}uo\u{028A}bdz\u{0292}\u{0261}\u{0266}mnlr])"),
            "w$1",
            &ph,
        );
        ph = sub_all("\u{028B}([pftskx\u{0283}])", "\u{028D}$1", &ph);

        // Аллофони j.
        ph = sub_all(
            format!("({vowel_c})j([{CONS_NO_W}#])"),
            format!("$1i{SYL}$2"),
            &ph,
        );
        ph = sub_all(format!("#j({cons_no_w_c})"), format!("#i{SYL}$1"), &ph);

        ph = sub_all("%", "", &ph);

        // Перестановка наголосу.
        ph = sub_all(
            format!("([^#{VOWEL}]?[{PALJ}{LENGTH}]*)([{ACUTE}{GRAVE_MARK}])"),
            "$2$1",
            &ph,
        );
        ph = sub_all(
            format!("([^#{VOWEL}]\u{0361})([{ACUTE}{GRAVE_MARK}])"),
            "$2$1",
            &ph,
        );

        // Onset callback.
        {
            let pat = format!("(.)({PALJ}?)({cons_c})({PALJ}?)([{ACUTE}{GRAVE_MARK}])({cons_c})");
            let regex = re(&pat);
            let mut out = String::with_capacity(ph.len());
            let mut last = 0;
            let mut broken = false;
            for cap in regex.captures_iter(&ph) {
                let Ok(cap) = cap else {
                    broken = true;
                    break;
                };
                let Some(m) = cap.get(0) else {
                    broken = true;
                    break;
                };
                let (ms, me) = (m.start(), m.end());
                out.push_str(&ph[last..ms]);
                let g = |i: usize| cap.get(i).map(|m| m.as_str()).unwrap_or("");
                let (a, aj, b, bj, st, c) = (g(1), g(2), g(3), g(4), g(5), g(6));
                let cluster = format!("{a}{b}{c}");
                if in_set(&ONSET_FULL, &cluster) {
                    out.push_str(format!("{st}{a}{aj}{b}{bj}{c}").as_str());
                } else if in_set(&ONSET_FULL, format!("{b}{c}").as_str()) || c == "j" {
                    out.push_str(format!("{a}{aj}{st}{b}{bj}{c}").as_str());
                } else {
                    out.push_str(format!("{a}{aj}{b}{bj}{st}{c}").as_str());
                }
                last = me;
            }
            if !broken {
                out.push_str(&ph[last..]);
            }
            ph = out;
        }

        ph = sub_all(
            format!("([^#{VOWEL}]\u{0361})([{ACUTE}{GRAVE_MARK}])(.{PALJ}?j)"),
            "$2$1$3",
            &ph,
        );
        ph = sub_all(
            format!("([^#{VOWEL}]\u{0361})([{ACUTE}{GRAVE_MARK}])(.{PALJ}?)"),
            "$1$3$2",
            &ph,
        );
        ph = sub_all(
            format!("#([^#{VOWEL}]+)([{ACUTE}{GRAVE_MARK}])"),
            "#$2$1",
            &ph,
        );
        ph = sub_all(format!("#([ui]{SYL})([{ACUTE}{GRAVE_MARK}])"), "#$2$1", &ph);
        ph = sub_all(
            format!("{PALJ}?{LENGTH}{PALJ}"),
            format!("{PALJ}{LENGTH}"),
            &ph,
        );
        ph = sub_all(format!("l([^{PALJ}])"), "\u{026B}$1", &ph);

        pronuns.push(ph);
    }

    let joined = pronuns.join(" ");
    Ok(joined.replace('#', ""))
}

/// Транскрипція без перевірки наголосу (як у Phonemizer).
pub fn transcribe(text: &str) -> anyhow::Result<String> {
    transcribe_impl(text, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_simple_word() {
        let ph = transcribe("привіт.").expect("transcribe");
        assert!(!ph.is_empty());
        // Всі символи — з інвентарю IPA (жодної кирилиці).
        assert!(ph.chars().all(|c| !('\u{0400}'..='\u{04FF}').contains(&c)));
    }

    #[test]
    fn accent_missing_check() {
        assert!(transcribe_impl("привіт світе.", true).is_err());
    }

    // Golden-дані згенеровано C++ еталоном (holos-tts-core, ipa_uk.cpp)
    // через прямий виклик transcribe зі стрес-словником.
    #[test]
    fn golden_with_stress_markup() {
        assert_eq!(transcribe("приві́т.").unwrap(), "prɪˈʋʲit.");
        assert_eq!(transcribe("мі́сто.").unwrap(), "ˈmʲistɔ.");
        assert_eq!(transcribe("ща́стя.").unwrap(), "ˈʃt͡ʃasʲtʲɐ.");
        assert_eq!(transcribe("украї́нською.").unwrap(), "ʊkrɐˈjinʲsʲkɔjʊ.");
        assert_eq!(transcribe("Приві́т, сві́те!").unwrap(), "prɪˈʋʲit, ˈsʲʋʲite!");
        assert_eq!(transcribe("Сла́ва Украї́ні!").unwrap(), "ˈsɫaʋɐ ʊkrɐˈjinʲi!");
        assert_eq!(
            transcribe("Ґа́нок, їжа́к і солов'ї́.").unwrap(),
            "ˈɡanɔk, jiˈʒak i sɔɫɔˈʋji."
        );
    }
}
