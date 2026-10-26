//! Golden-тест rule-вербалізатора: вихід C++ `holos::verbalize`.

use std::path::PathBuf;

#[test]
fn rule_verbalizer_matches_cpp_golden() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/verbalizer_rule.tsv");
    let data =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let mut count = 0usize;
    for (index, line) in data.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (input, expected) = line
            .split_once('\t')
            .unwrap_or_else(|| panic!("line {}: missing tab", index + 1));
        assert_eq!(
            speech_text::verbalize::verbalize(input),
            expected,
            "line {}: input {input:?}",
            index + 1
        );
        count += 1;
    }
    assert!(count >= 60, "golden file is suspiciously small: {count}");
}
