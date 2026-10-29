//! Інтеграційні тести багатоетапного бандла HolosTTS.
//!
//! Потребують реальних ваг і ONNX Runtime 1.31:
//!
//! ```text
//! PROMIN_HOLOS_BUNDLE=/models/holos-onnx \
//! PROMIN_HOLOS_TEXT='pryvʲit svʲite' \
//! ORT_STRATEGY=system ORT_LIB_LOCATION=/opt/ort-1.31.0/lib \
//! cargo test -p speech-inference --test holos_pipeline -- --ignored
//! ```
//!
//! `PROMIN_HOLOS_EXPECTED` — необов'язковий raw float32-файл з еталонним
//! аудіо (Python: `audio.astype(np.float32).tofile("ref.f32")`).

use std::path::PathBuf;

use speech_inference::{Bundle, HolosPipeline};
use speech_text::tokenizer::CharTokenizer;

fn bundle_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("PROMIN_HOLOS_BUNDLE")
            .expect("set PROMIN_HOLOS_BUNDLE to a holos-onnx bundle directory"),
    )
}

fn text_and_ids() -> (String, Vec<i64>) {
    let text = std::env::var("PROMIN_HOLOS_TEXT")
        .expect("set PROMIN_HOLOS_TEXT to a phoneme string from the bundle vocabulary");
    let bundle = Bundle::open(bundle_dir(), speech_inference::VOICE_DIM).expect("open bundle");
    let tokenizer = CharTokenizer::from_vocab(bundle.vocab());
    let ids = tokenizer.encode(&text);
    assert!(
        !ids.is_empty(),
        "text has no tokens from the bundle vocabulary"
    );
    (text, ids)
}

fn voice(pipeline: &HolosPipeline) -> Vec<f32> {
    let name = std::env::var("PROMIN_HOLOS_VOICE").unwrap_or_else(|_| {
        pipeline
            .voice_names()
            .first()
            .copied()
            .expect("bundle has no voices")
            .to_string()
    });
    pipeline
        .voice(&name)
        .unwrap_or_else(|| panic!("unknown voice {name:?}"))
        .to_vec()
}

#[test]
#[ignore = "requires PROMIN_HOLOS_BUNDLE + ORT 1.31"]
fn holos_bundle_synthesizes() {
    let dir = bundle_dir();
    let (text, ids) = text_and_ids();
    let pipeline = HolosPipeline::open(&dir, 4, false).expect("load exact pipeline");
    let voice = voice(&pipeline);

    let audio = pipeline
        .synthesize(&ids, &voice, 1.0)
        .unwrap_or_else(|error| panic!("synthesize {text:?}: {error:#}"));

    assert!(!audio.is_empty(), "model returned no audio");
    assert!(
        audio.iter().all(|sample| sample.is_finite()),
        "audio contains NaN or Inf"
    );
    let step = pipeline.samples_per_frame();
    assert_eq!(audio.len() % step, 0, "audio is not frame-aligned");

    if let Ok(expected_path) = std::env::var("PROMIN_HOLOS_EXPECTED") {
        let bytes = std::fs::read(&expected_path).expect("read expected audio");
        assert_eq!(bytes.len() % 4, 0, "expected file is not float32");
        let expected: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(
            audio.len(),
            expected.len(),
            "sample count differs from the Python reference"
        );
        let max_error = audio
            .iter()
            .zip(&expected)
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 1e-4,
            "max sample error {max_error} exceeds 1e-4"
        );
    }
}

#[test]
#[ignore = "requires PROMIN_HOLOS_BUNDLE with streaming stages + ORT 1.31"]
fn holos_streaming_matches_exact() {
    let dir = bundle_dir();
    let (text, ids) = text_and_ids();
    let exact = HolosPipeline::open(&dir, 4, false).expect("load exact pipeline");
    let voice = voice(&exact);
    let expected = exact
        .synthesize(&ids, &voice, 1.0)
        .expect("exact synthesis");

    let streaming = HolosPipeline::open(&dir, 4, true).expect("load streaming pipeline");
    assert!(streaming.is_streaming());
    let chunks = streaming
        .synthesize_chunks(&ids, &voice, 1.0, None)
        .unwrap_or_else(|error| panic!("stream {text:?}: {error:#}"));
    assert!(chunks.len() > 1, "expected more than one streaming window");
    let actual: Vec<f32> = chunks.into_iter().flatten().collect();

    assert_eq!(
        actual.len(),
        expected.len(),
        "streaming audio length differs from exact decoding"
    );
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(streamed, exact)| (streamed - exact).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 1e-4,
        "streaming differs from exact decoding: max error {max_error}"
    );

    // Колбековий шлях мусить дати ті самі шматки, що й пакетний.
    let mut callback_chunks = Vec::new();
    let completed = streaming
        .synthesize_stream(&ids, &voice, 1.0, None, |chunk| {
            callback_chunks.push(chunk.to_vec());
            true
        })
        .expect("callback streaming");
    assert!(completed);
    let from_callback: Vec<f32> = callback_chunks.into_iter().flatten().collect();
    assert_eq!(
        from_callback.len(),
        actual.len(),
        "callback streaming length differs from synthesize_chunks"
    );
    let max_error = from_callback
        .iter()
        .zip(&actual)
        .map(|(callback, batch)| (callback - batch).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_error, 0.0, "callback streaming is not deterministic");
}
