//! Tokenizer parity з C++ `SentencePieceProcessor::Encode` (Gemma tokenizer).
//!
//! Очікувані ID згенеровані системною бібліотекою sentencepiece 0.2.0
//! (та сама, що використовує `holos-tts-core`). Запуск:
//!
//! ```text
//! PROMIN_GEMMA_TOKENIZER=/root/models/verbalizer/gemma-int8/tokenizer.model \
//! cargo test -p speech-inference --test gemma_tokenizer -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sentencepiece::SentencePieceProcessor;

#[test]
#[ignore = "requires tokenizer.model: PROMIN_GEMMA_TOKENIZER"]
fn token_ids_match_cpp_sentencepiece() {
    let Some(path) = std::env::var_os("PROMIN_GEMMA_TOKENIZER").map(PathBuf::from) else {
        eprintln!("SKIP: PROMIN_GEMMA_TOKENIZER is not set");
        return;
    };
    if !path.is_file() {
        eprintln!("SKIP: {} not found", path.display());
        return;
    }

    let tokenizer = SentencePieceProcessor::open(&path).expect("open tokenizer");
    // (текст, ID з C++ sp.Encode)
    let cases: &[(&str, &[u32])] = &[
        ("user\n", &[1771, 8]),
        ("model\n", &[2826, 8]),
        ("\n", &[8]),
        (
            "У 2026 році, о 14:30, температура сягне 25 градусів.",
            &[
                36178, 35941, 35976, 35969, 35976, 36016, 18006, 35962, 942, 35941, 35968, 36006,
                35985, 35997, 35969, 35962, 22214, 21332, 27531, 35941, 35976, 36004, 22539, 29753,
                35959,
            ],
        ),
        (
            "12345678",
            &[35968, 35976, 35997, 36006, 36004, 36016, 36020, 36018],
        ),
        (
            "Ціна 1250 грн.",
            &[
                36249, 36078, 789, 35941, 35968, 35976, 36004, 35969, 24276, 35959,
            ],
        ),
        ("Привіт, світе!", &[8114, 11367, 35962, 17168, 1101, 36045]),
        (
            "Потяг №123 відправляється о 18:40.",
            &[
                5340, 6698, 36056, 9017, 35968, 35976, 35997, 6058, 35647, 9259, 942, 35941, 35968,
                36018, 35985, 36006, 35969, 35959,
            ],
        ),
        (
            "support@shop.com.ua",
            &[2068, 512, 531, 36071, 4986, 35959, 741, 35959, 2268],
        ),
        (
            "25.07.2026",
            &[
                35976, 36004, 35959, 35969, 36020, 35959, 35976, 35969, 35976, 36016,
            ],
        ),
        (
            "Температура -5 градусів, тиск 750 мм рт. ст.",
            &[
                31737, 33185, 647, 36004, 22539, 29753, 35962, 6243, 4639, 35941, 36020, 36004,
                35969, 13076, 35941, 4456, 35959, 3199, 35959,
            ],
        ),
    ];

    for (text, expected) in cases {
        let ids: Vec<u32> = tokenizer
            .encode(text)
            .unwrap_or_else(|error| panic!("encode {text:?}: {error}"))
            .into_iter()
            .map(|piece| piece.id)
            .collect();
        assert_eq!(&ids, expected, "token IDs differ for {text:?}");
    }

    assert_eq!(tokenizer.len(), 38_651);
}
