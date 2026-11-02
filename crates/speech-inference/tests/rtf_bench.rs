//! RTF-бенчмарк синтезу (ТЗ 3.2.5 «Продуктивність»: 10x–15x RTF на GPU,
//! 2x–5x на AVX2 CPU, 1.2x–2x на AVX CPU).
//!
//! `RTF` = час синтезу / тривалість аудіо (менше краще);
//! «x разів» = тривалість аудіо / час синтезу (більше краще) — саме в
//! таких одиницях ТЗ подає цілі. Тест ігнорується типово і запускається
//! вручну на цільовому обладнанні:
//!
//! ```text
//! PROMIN_HOLOS_BUNDLE=/models/holos-onnx-int8 \
//! PROMIN_HOLOS_TEXT='pryvit svit' \
//! ORT_STRATEGY=system ORT_LIB_LOCATION=<ort>/lib \
//! cargo test -p speech-inference --test rtf_bench -- --ignored --nocapture
//! ```
//!
//! Додатково друкуються архітектура/рівень SIMD процесора, обраний
//! execution provider і квантування бандла — щоб результат можна було
//! інтерпретувати і порівняти з цілями ТЗ.

use std::path::Path;
use std::time::Instant;

use speech_inference::{Bundle, HolosPipeline};
use speech_text::tokenizer::CharTokenizer;

/// Найменша допустима швидкість (audio/time) — нижче цього синтез
/// фактично зламаний незалежно від машини.
const MIN_SPEED_FACTOR: f64 = 0.1;

fn bundle_dir() -> String {
    std::env::var("PROMIN_HOLOS_BUNDLE")
        .expect("set PROMIN_HOLOS_BUNDLE to a holos-onnx bundle directory")
}

fn text_and_ids(bundle: &Bundle) -> (String, Vec<i64>) {
    let text = std::env::var("PROMIN_HOLOS_TEXT").unwrap_or_else(|_| "pryvit svit".to_string());
    let tokenizer = CharTokenizer::from_vocab(bundle.vocab());
    let ids = tokenizer.encode(&text);
    assert!(
        !ids.is_empty(),
        "text has no tokens from the bundle vocab: {text:?}"
    );
    (text, ids)
}

fn is_quantized(dir: &Path) -> bool {
    std::fs::read(dir.join("holos_onnx.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .is_some_and(|value| {
            !matches!(
                value.get("quantization"),
                None | Some(serde_json::Value::Null)
            )
        })
}

/// Триавторський звіт: (етап, тривалість аудіо с, час синтезу с, швидкість).
fn report(label: &str, audio_seconds: f64, elapsed_seconds: f64) {
    let speed = audio_seconds / elapsed_seconds;
    let rtf = elapsed_seconds / audio_seconds;
    println!(
        "{label:<28} audio={audio_seconds:6.2}s  synth={elapsed_seconds:6.3}s  \
         speed={speed:5.2}x (audio/time)  rtf={rtf:5.3} (time/audio)"
    );
    assert!(
        speed >= MIN_SPEED_FACTOR,
        "{label}: швидкість {speed:.3}x нижча за мінімум {MIN_SPEED_FACTOR}x"
    );
}

#[test]
#[ignore = "requires PROMIN_HOLOS_BUNDLE + ORT 1.31: bench by hand on target hardware"]
fn measures_synthesis_rtf() {
    let dir = bundle_dir();
    let quantized = is_quantized(Path::new(&dir));
    let cpu = speech_inference::cpu::detect();

    let pipeline = HolosPipeline::open(&dir, 4, true).expect("load bundle with streaming");
    let (text, ids) = text_and_ids(&pipeline_bundle(&dir));
    let voice = pipeline
        .voice_names()
        .first()
        .copied()
        .expect("bundle has no voices")
        .to_string();
    let voice = pipeline.voice(&voice).expect("default voice").to_vec();
    let sample_rate = pipeline.sample_rate();

    println!("=== RTF benchmark ===");
    println!(
        "bundle={dir} quantization={} arch={} simd={} avx2={} avx512vnni={} ep={} threads=4",
        if quantized { "int8" } else { "fp32" },
        cpu.arch,
        cpu.simd,
        cpu.avx2,
        cpu.avx512vnni,
        speech_inference::runtime::execution_provider(),
    );
    println!("text={text:?} tokens={}", ids.len());

    // Прогрів: перший виклик завантажує графи/алгоритми ORT.
    let warmup = pipeline
        .synthesize(&ids, &voice, 1.0)
        .expect("warmup exact synthesis");
    let audio_seconds = warmup.len() as f64 / f64::from(sample_rate);
    println!("warmup ok, audio={audio_seconds:.2}s");

    // Exact-декодер: кілька ітерацій, звіт по кожній + медіана.
    let mut exact_speeds = Vec::new();
    for iteration in 1..=3 {
        let start = Instant::now();
        let audio = pipeline
            .synthesize(&ids, &voice, 1.0)
            .unwrap_or_else(|e| panic!("exact #{iteration}: {e:#}"));
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(audio.len() as f64 / f64::from(sample_rate), audio_seconds);
        report(&format!("exact #{iteration}"), audio_seconds, elapsed);
        exact_speeds.push(audio_seconds / elapsed);
    }

    // Стрімінговий (віконний) декодер — рідко буває повільшим за exact,
    // але міряємо окремо, бо це шлях SSE-синтезу.
    if pipeline.is_streaming() {
        let start = Instant::now();
        let chunks = pipeline
            .synthesize_chunks(&ids, &voice, 1.0, None)
            .unwrap_or_else(|e| panic!("streaming: {e:#}"));
        let elapsed = start.elapsed().as_secs_f64();
        let streamed: Vec<f32> = chunks.into_iter().flatten().collect();
        let streamed_seconds = streamed.len() as f64 / f64::from(sample_rate);
        report("streaming (windowed)", streamed_seconds, elapsed);
    } else {
        println!("streaming: stages not present in bundle, skipped");
    }

    exact_speeds.sort_by(|a, b| a.partial_cmp(b).expect("finite speeds"));
    let median = exact_speeds[exact_speeds.len() / 2];
    println!(
        "=== median exact speed: {median:.2}x (rtf {:.3}) ===",
        1.0 / median
    );
}

/// Відкрити бандл ще раз лише для токенізації (vocab живе у бандлі).
fn pipeline_bundle(dir: &str) -> Bundle {
    Bundle::open(dir, speech_inference::VOICE_DIM).expect("open bundle for vocab")
}
