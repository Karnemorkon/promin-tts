//! Golden-тест accentor-lite: 54 слова з C++ `holos::Accentor::stress`.
//!
//! ```text
//! PROMIN_ACCENTOR=/root/models/accentor/accentor-lite.onnx \
//! ORT_STRATEGY=system ORT_LIB_LOCATION=/opt/ort/current/lib \
//! cargo test -p speech-inference --test accentor_golden -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use speech_inference::Accentor;

#[test]
#[ignore = "requires accentor-lite.onnx: PROMIN_ACCENTOR + ORT 1.31"]
fn accentor_matches_cpp_golden() {
    let Some(path) = std::env::var_os("PROMIN_ACCENTOR").map(PathBuf::from) else {
        eprintln!("SKIP: PROMIN_ACCENTOR is not set");
        return;
    };
    if !path.is_file() {
        eprintln!("SKIP: {} not found", path.display());
        return;
    }

    let accentor = Accentor::from_file(&path).expect("load accentor");
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/accentor.tsv");
    let golden = std::fs::read_to_string(&golden_path).expect("read golden TSV");

    let mut total = 0usize;
    let mut matched = 0usize;
    for line in golden.lines() {
        if line.is_empty() {
            continue;
        }
        let Some((word, expected)) = line.split_once('\t') else {
            continue;
        };
        total += 1;
        let got = accentor
            .stress_word(word)
            .unwrap_or_else(|error| panic!("stress {word:?}: {error}"));
        if got == expected {
            matched += 1;
        } else {
            eprintln!("[DIFF] word: {word}\n       got : {got}\n       want: {expected}");
        }
    }

    eprintln!("accentor match: {matched}/{total}");
    assert!(total > 0, "empty golden file");
    assert_eq!(matched, total, "accentor mismatch: {matched}/{total}");
}
