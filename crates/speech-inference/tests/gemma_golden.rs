//! Golden-тест Gemma-вербалізатора (46 речень з holos-tts-core).
//!
//! Потрібні `model.onnx` + `tokenizer.model`:
//!
//! ```text
//! PROMIN_GEMMA_DIR=/root/models/verbalizer/gemma-int8 \
//! ORT_STRATEGY=system ORT_LIB_LOCATION=/opt/ort/current/lib \
//! cargo test -p speech-inference --test gemma_golden -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use speech_inference::GemmaVerbalizer;

/// `ʼ` (CB BC), `ʻ` (CB BB), `’` (E2 80 99) -> `'` (як у C++ golden-тесті).
fn normalize_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0xCB && i + 1 < bytes.len() && (bytes[i + 1] == 0xBC || bytes[i + 1] == 0xBB)
        {
            out.push('\'');
            i += 2;
            continue;
        }
        if bytes[i] == 0xE2 && i + 2 < bytes.len() && bytes[i + 1] == 0x80 && bytes[i + 2] == 0x99 {
            out.push('\'');
            i += 3;
            continue;
        }
        let ch = text[i..].chars().next().expect("valid utf-8");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn has_ascii_digits(text: &str) -> bool {
    text.bytes().any(|b| b.is_ascii_digit())
}

#[test]
#[ignore = "requires Gemma model: PROMIN_GEMMA_DIR + ORT 1.17"]
fn gemma_matches_cpp_golden() {
    let Some(dir) = std::env::var_os("PROMIN_GEMMA_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: PROMIN_GEMMA_DIR is not set");
        return;
    };
    if !dir.join("model.onnx").is_file() || !dir.join("tokenizer.model").is_file() {
        eprintln!("SKIP: no Gemma model in {}", dir.display());
        return;
    }

    let verbalizer = GemmaVerbalizer::from_dir(&dir, 4).expect("load Gemma verbalizer");
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/gemma_verbalizer.tsv");
    let golden = std::fs::read_to_string(&golden_path).expect("read golden TSV");

    let mut total = 0usize;
    let mut matched = 0usize;
    let mut broken = 0usize;
    for line in golden.lines() {
        if line.is_empty() {
            continue;
        }
        let Some((input, expected)) = line.split_once('\t') else {
            continue;
        };
        total += 1;
        let got = verbalizer.verbalize_text(input);
        if got.is_empty() || has_ascii_digits(&got) {
            broken += 1;
            eprintln!("[BROKEN] in : {input}\n         out: {got}");
        }
        if normalize_quotes(&got) == normalize_quotes(expected) {
            matched += 1;
        } else {
            eprintln!("[DIFF] in   : {input}\n       got  : {got}\n       want : {expected}");
        }
    }

    eprintln!(
        "match: {matched}/{total} ({:.0}%), broken: {broken}",
        if total == 0 {
            0.0
        } else {
            100.0 * matched as f64 / total as f64
        }
    );
    assert!(total > 0, "empty golden file");
    assert_eq!(broken, 0, "golden output contains digits or is empty");
    assert_eq!(matched, total, "golden mismatch: {matched}/{total}");
}
