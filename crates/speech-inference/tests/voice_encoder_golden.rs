//! Golden-тест voice encoder: C++ ORT vs Rust ORT на тій самій int8-моделі.
//!
//! Сигнал детермінований (та сама формула, що в `test_voice_encoder.cpp`).
//!
//! ```text
//! PROMIN_VOICE_ENCODER=/root/models/tts/voice_encoder.onnx \
//! ORT_STRATEGY=system ORT_LIB_LOCATION=/opt/ort/current/lib \
//! cargo test -p speech-inference --test voice_encoder_golden -- --ignored --nocapture
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;

use speech_inference::VoiceEncoder;

const SAMPLE_RATE: f64 = 24_000.0;

/// Та сама формула, що в C++ `make_signal`.
fn make_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            let signal = 0.35 * (2.0 * PI as f64 * 173.0 * t / SAMPLE_RATE).sin()
                + 0.2
                    * (2.0 * PI as f64 * 431.0 * t / SAMPLE_RATE).sin()
                    * (0.5 + 0.5 * (2.0 * PI as f64 * 2.7 * t / SAMPLE_RATE).sin());
            signal as f32
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

#[test]
#[ignore = "requires voice_encoder.onnx: PROMIN_VOICE_ENCODER + ORT 1.31"]
fn voice_encoder_matches_cpp_golden() {
    let Some(path) = std::env::var_os("PROMIN_VOICE_ENCODER").map(PathBuf::from) else {
        eprintln!("SKIP: PROMIN_VOICE_ENCODER is not set");
        return;
    };
    if !path.is_file() {
        eprintln!("SKIP: {} not found", path.display());
        return;
    }

    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/voice_encoder.txt");
    let golden: Vec<f32> = std::fs::read_to_string(&golden_path)
        .expect("read golden")
        .split_whitespace()
        .map(|value| value.parse::<f32>().expect("golden float"))
        .collect();
    assert_eq!(golden.len(), 256, "golden must contain 256 floats");

    let encoder = VoiceEncoder::from_file(&path, 4).expect("load encoder");
    let wave = make_signal(24_000 * 3);
    let first = encoder.encode(&wave).expect("encode");
    assert_eq!(first.len(), 256);

    let similarity = cosine(&first, &golden);
    eprintln!("voice encoder golden cos = {similarity:.9} (threshold 0.995)");
    assert!(similarity >= 0.995, "cos below threshold: {similarity}");

    let second = encoder.encode(&wave).expect("encode again");
    let determinism = cosine(&first, &second);
    eprintln!("determinism cos = {determinism:.9}");
    assert!(determinism >= 0.9999, "unstable output: {determinism}");
}
