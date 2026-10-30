//! Gemma-3-270M INT8 вербалізатор (ONNX, decoder-only з KV-кешем).
//!
//! Порт `verbalizer_gemma.cpp`:
//! - детермінований препроцесинг (`gemma_prep`) для номерів карток/IBAN;
//! - речення з ASCII-цифрами проганяються через Gemma (промпт Gemma-3 chat);
//! - жадібний argmax із забороною токенів, чиї piece-рядки містять цифри;
//! - речення без цифр повертаються без змін, невдалі — теж (fallback).

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    GraphOptimizationLevel, Value,
};
use sentencepiece::SentencePieceProcessor;
use speech_text::gemma_prep;
use speech_text::neural::NeuralVerbalizer;
use speech_text::verbalize::{split_sentences, strip_ws};

/// Максимум нових токенів на одне речення (як `kMaxNewTokens` у C++).
const MAX_NEW_TOKENS: usize = 192;

/// Слот KV-кешу: вхід `past_key_values.N.*` та відповідний вихід `present.N.*`.
struct PastSlot {
    feed_pos: usize,
    /// `[1, kv_heads, 0, head_dim]` — порожній кеш на першому кроці.
    empty_shape: [i64; 4],
    out_idx: usize,
}

/// ONNX-сесія Gemma + токенізатор + мапа входів/виходів.
pub struct GemmaVerbalizer {
    /// Тримає Environment живим, поки живе сесія.
    _env: Arc<Environment>,
    session: Session,
    tokenizer: SentencePieceProcessor,
    input_names: Vec<String>,
    ids_pos: usize,
    mask_pos: usize,
    logits_idx: usize,
    past_slots: Vec<PastSlot>,
    vocab: usize,
    bos_id: i64,
    start_turn_id: i64,
    end_turn_id: i64,
    eos_ids: Vec<i64>,
    /// Токени, чиї piece-рядки містять ASCII-цифри (як `banned` у C++).
    banned: Vec<bool>,
}

impl GemmaVerbalizer {
    /// Завантажує `model.onnx` + `tokenizer.model` з каталогу.
    pub fn from_dir(dir: impl AsRef<Path>, intra_threads: i16) -> Result<Self> {
        let dir = dir.as_ref();
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.model");
        if !model_path.is_file() {
            bail!("missing {}", model_path.display());
        }
        if !tokenizer_path.is_file() {
            bail!("missing {}", tokenizer_path.display());
        }

        let tokenizer = SentencePieceProcessor::open(&tokenizer_path)
            .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
        let bos_id = special_id(&tokenizer, "<bos>")?;
        let start_turn_id = special_id(&tokenizer, "<start_of_turn>")?;
        let end_turn_id = special_id(&tokenizer, "<end_of_turn>")?;
        let eos_id = special_id(&tokenizer, "<eos>")?;
        let eos_ids = vec![eos_id, end_turn_id];

        let env = Arc::new(
            Environment::builder()
                .with_name("promin-tts-gemma")
                .build()
                .context("create ONNX Runtime environment (Gemma)")?,
        );
        let session = SessionBuilder::new(&env)?
            .with_intra_threads(intra_threads.max(1))?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(&model_path)
            .with_context(|| format!("load ONNX model {}", model_path.display()))?;

        let mut input_names = Vec::with_capacity(session.inputs.len());
        let mut ids_pos = None;
        let mut mask_pos = None;
        let mut past_slots: Vec<PastSlot> = Vec::new();
        for (pos, input) in session.inputs.iter().enumerate() {
            let name = input.name.clone();
            match name.as_str() {
                "input_ids" => ids_pos = Some(pos),
                "attention_mask" => mask_pos = Some(pos),
                _ if name.starts_with("past_key_values.") => {
                    let dims = &input.dimensions;
                    if dims.len() != 4 {
                        bail!("unsupported past shape for {name}: {dims:?}");
                    }
                    let kv_heads = dims[1].context("past_key_values head count is dynamic")?;
                    let head_dim = dims[3].context("past_key_values head_dim is dynamic")?;
                    if kv_heads == 0 || head_dim == 0 {
                        bail!("unsupported past shape for {name}: {dims:?}");
                    }
                    past_slots.push(PastSlot {
                        feed_pos: pos,
                        empty_shape: [1, i64::from(kv_heads), 0, i64::from(head_dim)],
                        out_idx: usize::MAX,
                    });
                }
                _ => bail!("unsupported model input: {name}"),
            }
            input_names.push(name);
        }
        let ids_pos = ids_pos.context("model misses input_ids")?;
        let mask_pos = mask_pos.context("model misses attention_mask")?;
        if past_slots.is_empty() {
            bail!("model misses past_key_values inputs");
        }

        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let logits_idx = output_names
            .iter()
            .position(|name| name == "logits")
            .context("model has no logits output")?;
        for slot in &mut past_slots {
            let past_name = &input_names[slot.feed_pos];
            let suffix = past_name
                .strip_prefix("past_key_values.")
                .expect("past slot name is checked above");
            let present_name = format!("present.{suffix}");
            slot.out_idx = output_names
                .iter()
                .position(|name| *name == present_name)
                .with_context(|| format!("model misses output {present_name}"))?;
        }

        let logits_dims = &session.outputs[logits_idx].dimensions;
        if logits_dims.len() != 3 {
            bail!("unexpected logits shape: {logits_dims:?}");
        }
        let vocab = logits_dims[2].context("logits vocab dimension is dynamic")? as usize;
        if vocab == 0 {
            bail!("unexpected logits shape: {logits_dims:?}");
        }

        let pieces = read_model_pieces(&tokenizer_path)?;
        if pieces.len() != tokenizer.len() {
            bail!(
                "tokenizer.model piece count mismatch: parsed {}, expected {}",
                pieces.len(),
                tokenizer.len()
            );
        }
        let mut banned = vec![false; vocab];
        for (id, piece) in pieces.iter().enumerate().take(vocab) {
            banned[id] = piece.bytes().any(|b| b.is_ascii_digit());
        }

        tracing::info!(
            model = %model_path.display(),
            vocab,
            past_slots = past_slots.len(),
            threads = intra_threads,
            "Gemma verbalizer loaded"
        );

        Ok(Self {
            _env: env,
            session,
            tokenizer,
            input_names,
            ids_pos,
            mask_pos,
            logits_idx,
            past_slots,
            vocab,
            bos_id,
            start_turn_id,
            end_turn_id,
            eos_ids,
            banned,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(self
            .tokenizer
            .encode(text)
            .with_context(|| format!("sentencepiece encode {text:?}"))?
            .into_iter()
            .map(|piece| i64::from(piece.id))
            .collect())
    }

    /// Промпт Gemma-3 chat: `<bos><start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n`.
    fn prompt_ids(&self, text: &str) -> Result<Vec<i64>> {
        let mut ids = Vec::new();
        ids.push(self.bos_id);
        ids.push(self.start_turn_id);
        ids.extend(self.encode("user\n")?);
        ids.extend(self.encode(&strip_ws(text))?);
        ids.push(self.end_turn_id);
        ids.extend(self.encode("\n")?);
        ids.push(self.start_turn_id);
        ids.extend(self.encode("model\n")?);
        Ok(ids)
    }

    /// Один крок сесії: `cur_ids` + повна маска + KV-кеш.
    fn run(&self, cur_ids: &[i64], mask: &[i64], past: &[Vec<f32>]) -> Result<Vec<Value<'static>>> {
        let ids_array = CowArray::from(Array::from_shape_vec(
            IxDyn(&[1, cur_ids.len()]),
            cur_ids.to_vec(),
        )?);
        let mask_array = CowArray::from(Array::from_shape_vec(
            IxDyn(&[1, mask.len()]),
            mask.to_vec(),
        )?);

        let mut past_arrays = Vec::with_capacity(self.past_slots.len());
        for (slot, buffer) in self.past_slots.iter().zip(past) {
            let kv_heads = slot.empty_shape[1] as usize;
            let head_dim = slot.empty_shape[3] as usize;
            let len = buffer.len() / (kv_heads * head_dim);
            past_arrays.push(CowArray::from(Array::from_shape_vec(
                IxDyn(&[1, kv_heads, len, head_dim]),
                buffer.clone(),
            )?));
        }

        let mut values = Vec::with_capacity(self.input_names.len());
        for pos in 0..self.input_names.len() {
            let value = if pos == self.ids_pos {
                Value::from_array(self.session.allocator(), &ids_array)?
            } else if pos == self.mask_pos {
                Value::from_array(self.session.allocator(), &mask_array)?
            } else {
                let slot_idx = self
                    .past_slots
                    .iter()
                    .position(|slot| slot.feed_pos == pos)
                    .with_context(|| {
                        format!(
                            "unmapped model input slot {pos} ({})",
                            self.input_names[pos]
                        )
                    })?;
                Value::from_array(self.session.allocator(), &past_arrays[slot_idx])?
            };
            values.push(value);
        }

        self.session
            .run(values)
            .context("ONNX session run (Gemma verbalizer)")
    }

    /// Копіює `present.*` у буфери KV-кешу для наступного кроку.
    fn copy_presents(&self, outputs: &[Value<'static>], past: &mut [Vec<f32>]) -> Result<()> {
        for (slot, buffer) in self.past_slots.iter().zip(past.iter_mut()) {
            let tensor = outputs[slot.out_idx]
                .try_extract::<f32>()
                .context("extract present tensor")?;
            let view = tensor.view();
            let slice = view
                .as_slice()
                .context("present tensor is not contiguous")?;
            buffer.clear();
            buffer.extend_from_slice(slice);
        }
        Ok(())
    }

    /// Жадібний argmax із пропуском заборонених токенів (перший максимум).
    fn argmax(&self, row: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (id, &value) in row.iter().enumerate() {
            if self.banned.get(id).copied().unwrap_or(false) {
                continue;
            }
            if value > best_value {
                best_value = value;
                best = id;
            }
        }
        best
    }

    /// Генерує нормалізоване речення (порожній рядок -> fallback у виклику).
    fn run_sentence(&self, text: &str) -> Result<String> {
        let ids = self.prompt_ids(text)?;
        let mut mask = vec![1i64; ids.len()];
        let mut past: Vec<Vec<f32>> = vec![Vec::new(); self.past_slots.len()];

        let mut outputs = self.run(&ids, &mask, &past)?;
        let mut generated: Vec<u32> = Vec::new();
        while generated.len() < MAX_NEW_TOKENS {
            let next = {
                let tensor = outputs[self.logits_idx]
                    .try_extract::<f32>()
                    .context("extract logits tensor")?;
                let view = tensor.view();
                let shape = view.shape();
                if shape.len() != 3 || shape[2] != self.vocab {
                    bail!("unexpected logits shape: {shape:?}");
                }
                let seq_len = shape[1];
                if seq_len == 0 {
                    bail!("empty logits sequence");
                }
                let slice = view.as_slice().context("logits tensor is not contiguous")?;
                let row = &slice[(seq_len - 1) * self.vocab..seq_len * self.vocab];
                self.argmax(row)
            };

            if self.eos_ids.contains(&(next as i64)) {
                break;
            }
            generated.push(next as u32);
            if generated.len() >= MAX_NEW_TOKENS {
                break;
            }

            self.copy_presents(&outputs, &mut past)?;
            mask.push(1);
            outputs = self.run(&[next as i64], &mask, &past)?;
        }

        if generated.is_empty() {
            return Ok(String::new());
        }
        let decoded = self
            .tokenizer
            .decode_piece_ids(&generated)
            .context("sentencepiece decode")?;
        Ok(strip_ws(&decoded))
    }

    /// Повний конвеєр C++ `GemmaVerbalizer::verbalize`.
    pub fn verbalize_text(&self, text: &str) -> String {
        let trimmed = strip_ws(text);
        if trimmed.is_empty() {
            return String::new();
        }
        if !gemma_prep::has_ascii_digits(&trimmed) {
            return text.to_string();
        }
        if let Some(digits) = gemma_prep::match_bare_digits(&trimmed) {
            return gemma_prep::spell_digits(&digits);
        }

        let sentences = split_sentences(&trimmed);
        let mut results = Vec::with_capacity(sentences.len());
        for sentence in sentences {
            if !gemma_prep::has_ascii_digits(&sentence) {
                results.push(sentence);
                continue;
            }
            let prepared = gemma_prep::add_spacing(&sentence);
            match self.run_sentence(&prepared) {
                Ok(out) if !out.is_empty() => results.push(out),
                Ok(_) => results.push(sentence),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        sentence = %sentence,
                        "gemma sentence failed, keeping original"
                    );
                    results.push(sentence);
                }
            }
        }
        results.join(" ")
    }
}

impl NeuralVerbalizer for GemmaVerbalizer {
    fn verbalize(&self, text: &str) -> Result<String> {
        Ok(self.verbalize_text(text))
    }
}

/// Спеціальний токен або помилка, якщо токенізатор його не знає.
fn special_id(tokenizer: &SentencePieceProcessor, piece: &str) -> Result<i64> {
    match tokenizer.piece_to_id(piece) {
        Ok(Some(id)) => Ok(i64::from(id)),
        Ok(None) => bail!("tokenizer.model lacks {piece}"),
        Err(error) => bail!("tokenizer.model piece lookup {piece}: {error}"),
    }
}

/// Мінімальний парсер SentencePiece `ModelProto` (id -> piece).
///
/// `sentencepiece` crate не має `id_to_piece`, тому читаємо repeated
/// `pieces` (field 1) напряму з protobuf: потрібні лише назви piece для
/// заборони цифрових токенів.
fn read_model_pieces(path: &Path) -> Result<Vec<String>> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut pieces = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let (key, next) = read_varint(&data, pos)?;
        pos = next;
        let field = key >> 3;
        let wire = (key & 7) as u8;
        if field == 1 && wire == 2 {
            let (len, next) = read_varint(&data, pos)?;
            pos = next;
            let end = pos
                .checked_add(len as usize)
                .filter(|&end| end <= data.len())
                .with_context(|| format!("truncated piece at offset {pos}"))?;
            pieces.push(read_piece_name(&data[pos..end])?);
            pos = end;
        } else {
            pos = skip_field(&data, pos, wire)?;
        }
    }
    Ok(pieces)
}

/// Витягує `piece` (field 1, string) із вкладеного `SentencePiece`.
fn read_piece_name(data: &[u8]) -> Result<String> {
    let mut pos = 0usize;
    while pos < data.len() {
        let (key, next) = read_varint(data, pos)?;
        pos = next;
        let field = key >> 3;
        let wire = (key & 7) as u8;
        if field == 1 && wire == 2 {
            let (len, next) = read_varint(data, pos)?;
            pos = next;
            let end = pos
                .checked_add(len as usize)
                .filter(|&end| end <= data.len())
                .with_context(|| format!("truncated piece name at offset {pos}"))?;
            return String::from_utf8(data[pos..end].to_vec()).context("piece is not valid UTF-8");
        }
        pos = skip_field(data, pos, wire)?;
    }
    bail!("SentencePiece entry without a piece name")
}

fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *data
            .get(pos)
            .with_context(|| format!("truncated varint at offset {pos}"))?;
        pos += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
        shift += 7;
        if shift >= 64 {
            bail!("varint overflow at offset {pos}");
        }
    }
}

fn skip_field(data: &[u8], pos: usize, wire: u8) -> Result<usize> {
    let end = match wire {
        0 => return Ok(read_varint(data, pos)?.1),
        1 => pos.checked_add(8),
        2 => {
            let (len, next) = read_varint(data, pos)?;
            next.checked_add(len as usize)
        }
        5 => pos.checked_add(4),
        other => bail!("unsupported protobuf wire type {other} at offset {pos}"),
    };
    end.filter(|&end| end <= data.len())
        .with_context(|| format!("truncated protobuf field at offset {pos}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tokenizer_path() -> PathBuf {
        std::env::var_os("PROMIN_GEMMA_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/opencode/gemma/tokenizer.model"))
    }

    #[test]
    fn parses_tokenizer_model_pieces() {
        let path = tokenizer_path();
        if !path.is_file() {
            eprintln!("skip: {} not found", path.display());
            return;
        }
        let tokenizer = SentencePieceProcessor::open(&path).expect("open tokenizer");
        let pieces = read_model_pieces(&path).expect("parse pieces");
        assert_eq!(pieces.len(), tokenizer.len());
        assert_eq!(pieces[0], "<pad>");
        assert_eq!(pieces[1], "<eos>");
        assert_eq!(pieces[2], "<bos>");
        assert!(pieces.iter().any(|piece| piece == "<start_of_turn>"));
        assert!(pieces.iter().any(|piece| piece == "0"));
        // Дзеркало C++ `has_ascii_digits(sp.IdToPiece(id))`.
        let banned = pieces
            .iter()
            .filter(|p| p.bytes().any(|b| b.is_ascii_digit()))
            .count();
        assert!(banned > 10);
    }

    #[test]
    fn special_tokens_match_gemma_config() {
        let path = tokenizer_path();
        if !path.is_file() {
            eprintln!("skip: {} not found", path.display());
            return;
        }
        let tokenizer = SentencePieceProcessor::open(&path).expect("open tokenizer");
        assert_eq!(special_id(&tokenizer, "<bos>").unwrap(), 2);
        assert_eq!(special_id(&tokenizer, "<start_of_turn>").unwrap(), 6);
        assert_eq!(special_id(&tokenizer, "<end_of_turn>").unwrap(), 7);
        assert_eq!(special_id(&tokenizer, "<eos>").unwrap(), 1);
        assert!(special_id(&tokenizer, "<definitely-missing>").is_err());
    }

    #[test]
    fn varint_roundtrip() {
        let data = [0xAC, 0x02, 0x00];
        assert_eq!(read_varint(&data, 0).unwrap(), (300, 2));
        assert_eq!(read_varint(&data, 2).unwrap(), (0, 3));
        assert!(read_varint(&[0x80], 0).is_err());
    }
}
