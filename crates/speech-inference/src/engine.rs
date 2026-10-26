//! ONNX Runtime TTS generator (holos_cpu_int8.onnx).
//!
//! Порт `synth.cpp`: входи графа зв'язуються за сигнатурою (як Python
//! `_bind_inputs`), токени отримують BOS/EOS=0, вихід `audio` повертається
//! як float32-вектор.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Input, Session, SessionBuilder},
    tensor::TensorElementDataType,
    Value,
};

/// Розмірність вектора голосу (voices.json).
pub const VOICE_DIM: usize = 256;

/// BOS/EOS-токен у послідовності фонем.
const BOS_EOS: i64 = 0;

/// Роль входу ONNX-графа.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// tensor(string) — фонемний рядок (не підтримується у v0.1).
    Phonemes,
    /// int64 ранг 2 — послідовність токенів.
    Tokens,
    /// int64 ранг <= 1 — довжина послідовності.
    Lengths,
    /// float32 [*, 256] або ім'я містить voice/style.
    Voice,
    /// float32 ранг <= 1 — керуючі значення (speed тощо).
    Control { is_speed: bool },
}

#[derive(Debug, Clone)]
struct BoundInput {
    name: String,
    slot: Slot,
}

impl BoundInput {
    fn role(&self) -> &'static str {
        match self.slot {
            Slot::Phonemes => "phonemes",
            Slot::Tokens => "tokens",
            Slot::Lengths => "lengths",
            Slot::Voice => "voice",
            Slot::Control { .. } => "control",
        }
    }
}

/// Детерміноване зв'язування входів за сигнатурою (як `bind_inputs` у C++).
fn bind_inputs(inputs: &[Input]) -> Result<Vec<BoundInput>> {
    let mut bound = Vec::with_capacity(inputs.len());
    let (mut has_phonemes, mut has_tokens, mut has_voice) = (false, false, false);

    for input in inputs {
        let low = input.name.to_ascii_lowercase();
        let rank = input.dimensions.len();
        let last_dim = input.dimensions.last().copied().flatten();

        let slot = match input.input_type {
            TensorElementDataType::String => Slot::Phonemes,
            TensorElementDataType::Float32
                if last_dim == Some(VOICE_DIM as u32)
                    || low.contains("voice")
                    || low.contains("style") =>
            {
                Slot::Voice
            }
            TensorElementDataType::Int64 if rank <= 1 => Slot::Lengths,
            TensorElementDataType::Int64 => Slot::Tokens,
            TensorElementDataType::Float32 if rank <= 1 => Slot::Control {
                is_speed: low.contains("speed"),
            },
            other => bail!(
                "unknown ONNX input node {:?} (type {other:?}, shape {:?})",
                input.name,
                input.dimensions
            ),
        };

        has_phonemes |= slot == Slot::Phonemes;
        has_tokens |= slot == Slot::Tokens;
        has_voice |= slot == Slot::Voice;

        bound.push(BoundInput {
            name: input.name.clone(),
            slot,
        });
    }

    if !has_phonemes && (!has_tokens || !has_voice) {
        bail!("failed to bind ONNX inputs: tokens and voice are required");
    }
    Ok(bound)
}

/// ONNX-based TTS generator engine (holos_cpu_int8.onnx).
pub struct TtsEngine {
    /// Kept alive so that the session stays valid.
    _env: Arc<Environment>,
    session: Session,
    binding: Vec<BoundInput>,
    audio_output: usize,
    intra_threads: i16,
}

impl TtsEngine {
    pub fn from_file(model_path: impl AsRef<Path>, intra_threads: i16) -> Result<Self> {
        let model_path = model_path.as_ref();
        let intra_threads = intra_threads.max(1);

        let env = Arc::new(
            Environment::builder()
                .with_name("promin-tts")
                .build()
                .context("create ONNX Runtime environment")?,
        );
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(intra_threads)?
            .with_model_from_file(model_path)
            .with_context(|| format!("load ONNX model {}", model_path.display()))?;

        let binding = bind_inputs(&session.inputs)?;
        let audio_output = session
            .outputs
            .iter()
            .position(|o| o.name == "audio")
            .unwrap_or(0);

        let engine = Self {
            _env: env,
            session,
            binding,
            audio_output,
            intra_threads,
        };
        tracing::info!(
            model = %model_path.display(),
            threads = intra_threads,
            inputs = %engine.binding_summary(),
            outputs = ?engine.output_names(),
            "TTS model loaded"
        );
        Ok(engine)
    }

    /// Людиночитний опис зв'язування входів (для логів і діагностики).
    pub fn binding_summary(&self) -> String {
        self.binding
            .iter()
            .map(|b| format!("{}={}", b.name, b.role()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.session
            .inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect()
    }

    pub fn output_names(&self) -> Vec<&str> {
        self.session
            .outputs
            .iter()
            .map(|o| o.name.as_str())
            .collect()
    }

    pub fn intra_threads(&self) -> i16 {
        self.intra_threads
    }

    /// Синтез фонемних id у float32-аудіо (24 кГц mono).
    ///
    /// `voice` — вектор із 256 float (рядок voices.json), `speed` — темп
    /// (передається у вхід, ім'я якого містить "speed"; інакше 1.0).
    pub fn generate(&self, phoneme_ids: &[i64], voice: &[f32], speed: f32) -> Result<Vec<f32>> {
        if voice.len() != VOICE_DIM {
            bail!(
                "voice vector must have {VOICE_DIM} floats, got {}",
                voice.len()
            );
        }
        if !speed.is_finite() {
            bail!("speed must be finite, got {speed}");
        }

        // Токени: [1, N] з BOS/EOS=0 (як HolosTTS.generate). Якщо граф
        // приймає фонемний рядок — токени не потрібні.
        let uses_tokens = self.binding.iter().any(|b| b.slot == Slot::Tokens);
        let mut tokens = Vec::new();
        if uses_tokens {
            tokens.reserve(phoneme_ids.len() + 2);
            tokens.push(BOS_EOS);
            tokens.extend_from_slice(phoneme_ids);
            tokens.push(BOS_EOS);
        }
        let lengths = [tokens.len() as i64];

        // Буфери мусять жити до завершення run: Value не копіює дані.
        let tokens_arr = CowArray::from(Array::from_shape_vec(IxDyn(&[1, tokens.len()]), tokens)?);
        let lengths_arr = CowArray::from(Array::from_elem(IxDyn(&[1]), lengths[0]));
        let voice_arr = CowArray::from(Array::from_shape_vec(
            IxDyn(&[1, VOICE_DIM]),
            voice.to_vec(),
        )?);

        let control_values: Vec<f32> = self
            .binding
            .iter()
            .filter_map(|b| match b.slot {
                Slot::Control { is_speed } => Some(if is_speed { speed } else { 1.0 }),
                _ => None,
            })
            .collect();
        let control_arrays: Vec<CowArray<'_, f32, IxDyn>> = control_values
            .iter()
            .map(|&v| CowArray::from(Array::from_elem(IxDyn(&[1]), v)))
            .collect();

        // Порядок values мусить збігатися з порядком session.inputs.
        let mut values = Vec::with_capacity(self.binding.len());
        let mut control_idx = 0;
        for bound in &self.binding {
            let value = match bound.slot {
                Slot::Phonemes => {
                    bail!("phoneme string input {:?} is not supported yet", bound.name)
                }
                Slot::Tokens => Value::from_array(self.session.allocator(), &tokens_arr)?,
                Slot::Lengths => Value::from_array(self.session.allocator(), &lengths_arr)?,
                Slot::Voice => Value::from_array(self.session.allocator(), &voice_arr)?,
                Slot::Control { .. } => {
                    let array = &control_arrays[control_idx];
                    control_idx += 1;
                    Value::from_array(self.session.allocator(), array)?
                }
            };
            values.push(value);
        }

        let outputs = self
            .session
            .run(values)
            .context("ONNX session run (TTS generator)")?;
        let output = outputs
            .get(self.audio_output)
            .context("model returned no outputs")?;
        let tensor = output
            .try_extract::<f32>()
            .context("extract audio tensor as f32")?;
        let view = tensor.view();
        let slice = view.as_slice().context("audio tensor is not contiguous")?;
        Ok(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TtsEngine>();
    }

    #[test]
    #[ignore = "requires models + ORT 1.31: ORT_STRATEGY=system ORT_LIB_LOCATION=<ort>/lib PROMIN_MODELS_DIR=<dir>"]
    fn load_holos_model() {
        let dir = std::env::var("PROMIN_MODELS_DIR").expect("PROMIN_MODELS_DIR not set");
        let nested = Path::new(&dir).join("tts/holos_cpu_int8.onnx");
        let model = if nested.exists() {
            nested
        } else {
            Path::new(&dir).join("holos_cpu_int8.onnx")
        };
        let engine = TtsEngine::from_file(&model, 4).expect("load TTS model");
        println!("inputs: {}", engine.binding_summary());
        println!("outputs: {:?}", engine.output_names());
        assert!(!engine.input_names().is_empty());
        assert!(!engine.output_names().is_empty());
    }
}
