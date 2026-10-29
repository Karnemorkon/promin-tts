//! Багатоетапний рушій HolosTTS: `encoder -> prosody -> decoder`.
//!
//! Це формат `holos-onnx` v2: замість одного графа модель складається з
//! кількох ONNX-сесій, а між ними Rust рахує вирівнювання кадрів
//! (`frame_to_token`) і передає f0/energy. Стрімінговий варіант додає
//! `source` та віконний декодер, який віддає аудіо частинами.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ndarray::{Array, ArrayD, CowArray, IxDyn};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    GraphOptimizationLevel, Value,
};

use crate::bundle::Bundle;
use crate::VOICE_DIM;

/// Скільки семплів додаємо/зрізаємо на межах стрімінгових вікон.
const STREAM_SILENCE_MARGIN_SECONDS: f32 = 0.010;
/// Фейд-ін після зрізання стартової тиші (5 мс при 24 кГц).
const STREAM_FADE_IN_SAMPLES: usize = 120;

/// Вихід етапу `encoder`.
struct Encoded {
    voice: Vec<f32>,
    acoustic_tokens: ArrayD<f32>,
    frame_to_token: Vec<i64>,
    f0: Vec<f32>,
    energy: Vec<f32>,
    durations: Vec<i64>,
}

enum DecoderStage {
    Exact(Session),
    Streaming {
        source: Session,
        decoder: Session,
        chunk_frames: usize,
        context_frames: usize,
    },
}

/// Готовий до синтезу багатоетапний рушій.
pub struct HolosPipeline {
    _env: Arc<Environment>,
    bundle: Bundle,
    encoder: Session,
    prosody: Session,
    decoder: DecoderStage,
    encoder_outputs: EncoderOutputs,
    prosody_outputs: ProsodyOutputs,
    decoder_outputs: AudioOutputs,
    source_output: usize,
}

struct EncoderOutputs {
    acoustic: usize,
    prosodic: usize,
    durations: usize,
}

struct ProsodyOutputs {
    f0: usize,
    energy: usize,
}

struct AudioOutputs {
    audio: usize,
    lengths: Option<usize>,
}

impl HolosPipeline {
    /// Відкрити бандл. `streaming` вантажить віконний декодер замість exact.
    pub fn open(dir: impl AsRef<Path>, intra_threads: i16, streaming: bool) -> Result<Self> {
        let dir = dir.as_ref();
        let bundle = Bundle::open(dir, VOICE_DIM)
            .with_context(|| format!("open Holos bundle {}", dir.display()))?;
        let env = Arc::new(
            Environment::builder()
                .with_name("promin-tts-holos")
                .build()
                .context("create ONNX Runtime environment (holos bundle)")?,
        );
        let threads = intra_threads.max(1);

        let encoder = load_stage(&env, &bundle.stage(&bundle.manifest.encoder), threads)?;
        require_inputs(&encoder, &["tokens", "voice", "speed"], "encoder")?;
        let encoder_outputs = EncoderOutputs {
            acoustic: output_index(&encoder, "acoustic_tokens")?,
            prosodic: output_index(&encoder, "prosodic_tokens")?,
            durations: output_index(&encoder, "durations")?,
        };

        let prosody = load_stage(&env, &bundle.stage(&bundle.manifest.prosody), threads)?;
        require_inputs(
            &prosody,
            &["prosodic_tokens", "frame_to_token", "voice"],
            "prosody",
        )?;
        let prosody_outputs = ProsodyOutputs {
            f0: output_index(&prosody, "f0")?,
            energy: output_index(&prosody, "energy")?,
        };

        let (decoder, decoder_outputs, source_output) = if streaming {
            let config =
                bundle.manifest.streaming.as_ref().context(
                    "bundle has no streaming section; re-export the model to use streaming",
                )?;
            let source = load_stage(&env, &bundle.stage(&config.source), threads)?;
            require_inputs(&source, &["f0"], "source")?;
            let source_output = output_index(&source, "source")?;

            let decoder = load_stage(&env, &bundle.stage(&config.decoder), threads)?;
            require_inputs(
                &decoder,
                &[
                    "acoustic_tokens",
                    "frame_to_token",
                    "f0",
                    "energy",
                    "source",
                    "voice",
                ],
                "decoder_window",
            )?;
            let outputs = AudioOutputs {
                audio: output_index(&decoder, "audio")?,
                lengths: optional_output_index(&decoder, "audio_lengths"),
            };
            (
                DecoderStage::Streaming {
                    source,
                    decoder,
                    chunk_frames: config.chunk_frames,
                    context_frames: config.context_frames,
                },
                outputs,
                source_output,
            )
        } else {
            let decoder = load_stage(&env, &bundle.stage(&bundle.manifest.decoder), threads)?;
            require_inputs(
                &decoder,
                &["acoustic_tokens", "frame_to_token", "f0", "energy", "voice"],
                "decoder",
            )?;
            let outputs = AudioOutputs {
                audio: output_index(&decoder, "audio")?,
                lengths: optional_output_index(&decoder, "audio_lengths"),
            };
            (DecoderStage::Exact(decoder), outputs, 0)
        };

        tracing::info!(
            bundle = %dir.display(),
            model = %bundle.manifest.encoder,
            sample_rate = bundle.sample_rate(),
            samples_per_frame = bundle.samples_per_frame(),
            max_frames = bundle.max_frames(),
            voices = bundle.voices.len(),
            streaming,
            threads,
            "Holos bundle loaded"
        );

        Ok(Self {
            _env: env,
            bundle,
            encoder,
            prosody,
            decoder,
            encoder_outputs,
            prosody_outputs,
            decoder_outputs,
            source_output,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.bundle.sample_rate()
    }

    pub fn voice_dim(&self) -> usize {
        self.bundle.voice_dim()
    }

    pub fn max_frames(&self) -> usize {
        self.bundle.max_frames()
    }

    pub fn samples_per_frame(&self) -> usize {
        self.bundle.samples_per_frame()
    }

    pub fn voice_names(&self) -> Vec<&str> {
        self.bundle.voice_names()
    }

    pub fn voice(&self, name: &str) -> Option<&[f32]> {
        self.bundle.voices.get(name).map(Vec::as_slice)
    }

    pub fn is_streaming(&self) -> bool {
        matches!(self.decoder, DecoderStage::Streaming { .. })
    }

    /// Синтез фонемних id у float32-аудіо (моно, `sample_rate`).
    pub fn synthesize(&self, phoneme_ids: &[i64], voice: &[f32], speed: f32) -> Result<Vec<f32>> {
        let encoded = self.encode(phoneme_ids, voice, speed)?;
        match &self.decoder {
            DecoderStage::Exact(_) => self.decode_exact(&encoded),
            DecoderStage::Streaming { .. } => {
                let mut audio = Vec::new();
                for chunk in self.decode_windows(&encoded, None)? {
                    audio.extend_from_slice(&chunk);
                }
                Ok(audio)
            }
        }
    }

    /// Стрімінгові чанки; порожньо, якщо бандл без віконного декодера.
    ///
    /// Стартова тиша зрізається до 10 мс, після чого перші 5 мс отримують
    /// фейд-ін — так само, як `_trim_leading` у Python-рантаймі.
    pub fn synthesize_chunks(
        &self,
        phoneme_ids: &[i64],
        voice: &[f32],
        speed: f32,
        chunk_frames: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let encoded = self.encode(phoneme_ids, voice, speed)?;
        match &self.decoder {
            DecoderStage::Streaming { .. } => {
                let mut chunks = self.decode_windows(&encoded, chunk_frames)?;
                let skip = stream_silence_skip(
                    &encoded.durations,
                    self.bundle.samples_per_frame(),
                    self.bundle.sample_rate(),
                );
                let fade = if skip > 0 { STREAM_FADE_IN_SAMPLES } else { 0 };
                trim_leading_silence(&mut chunks, skip, fade);
                chunks.retain(|chunk| !chunk.is_empty());
                Ok(chunks)
            }
            DecoderStage::Exact(_) => Ok(vec![self.decode_exact(&encoded)?]),
        }
    }

    /// Стрімінг: колбек отримує кожен готовий шматок; `false` — клієнт відключився.
    ///
    /// Для exact-декодера буде рівно один шматок (без обрізання тиші, як у
    /// `generate`), для віконного — по одному на вікно з тим самим обрізанням
    /// стартової тиші, що й у `synthesize_chunks`.
    pub fn synthesize_stream<F>(
        &self,
        phoneme_ids: &[i64],
        voice: &[f32],
        speed: f32,
        chunk_frames: Option<usize>,
        mut on_chunk: F,
    ) -> Result<bool>
    where
        F: FnMut(&[f32]) -> bool,
    {
        let encoded = self.encode(phoneme_ids, voice, speed)?;
        match &self.decoder {
            DecoderStage::Exact(_) => {
                let audio = self.decode_exact(&encoded)?;
                Ok(on_chunk(&audio))
            }
            DecoderStage::Streaming { .. } => {
                let skip = stream_silence_skip(
                    &encoded.durations,
                    self.bundle.samples_per_frame(),
                    self.bundle.sample_rate(),
                );
                let fade = if skip > 0 { STREAM_FADE_IN_SAMPLES } else { 0 };
                let mut trimmer = LeadingTrimmer::new(skip, fade);
                self.decode_windows_with(&encoded, chunk_frames, &mut |chunk| match trimmer
                    .apply(chunk)
                {
                    Some(trimmed) => on_chunk(&trimmed),
                    None => true,
                })
            }
        }
    }

    fn encode(&self, phoneme_ids: &[i64], voice: &[f32], speed: f32) -> Result<Encoded> {
        if voice.len() != self.bundle.voice_dim() {
            bail!(
                "voice vector must have {} floats, got {}",
                self.bundle.voice_dim(),
                voice.len()
            );
        }
        if !speed.is_finite() || speed <= 0.0 {
            bail!("speed must be a positive finite number, got {speed}");
        }
        let vocab_len = self.bundle.vocab().len() as i64;
        if let Some(&token) = phoneme_ids
            .iter()
            .find(|&&token| token < 0 || token >= vocab_len)
        {
            bail!("phoneme id {token} is outside the bundle vocabulary (0..{vocab_len})");
        }

        let mut tokens = Vec::with_capacity(phoneme_ids.len() + 2);
        tokens.push(0);
        tokens.extend_from_slice(phoneme_ids);
        tokens.push(0);
        let token_count = tokens.len();

        let tokens_arr = Array::from_shape_vec(IxDyn(&[1, token_count]), tokens)?;
        let voice_arr = Array::from_shape_vec(IxDyn(&[1, voice.len()]), voice.to_vec())?;
        let speed_arr = Array::from_elem(IxDyn(&[1]), speed);
        let tokens_cow = CowArray::from(tokens_arr);
        let voice_cow = CowArray::from(voice_arr);
        let speed_cow = CowArray::from(speed_arr);
        let mut values = Vec::with_capacity(self.encoder.inputs.len());
        for input in &self.encoder.inputs {
            let value = match input.name.as_str() {
                "tokens" => Value::from_array(self.encoder.allocator(), &tokens_cow)?,
                "voice" => Value::from_array(self.encoder.allocator(), &voice_cow)?,
                "speed" => Value::from_array(self.encoder.allocator(), &speed_cow)?,
                other => bail!("unexpected encoder input {other:?}"),
            };
            values.push(value);
        }
        let outputs = self.encoder.run(values).context("encoder run")?;

        let acoustic_tokens =
            extract_f32(&outputs, self.encoder_outputs.acoustic, "acoustic_tokens")?;
        let prosodic_tokens =
            extract_f32(&outputs, self.encoder_outputs.prosodic, "prosodic_tokens")?;
        let durations = extract_i64(&outputs, self.encoder_outputs.durations, "durations")?;

        let acoustic_shape = acoustic_tokens.shape();
        if acoustic_shape.len() != 3 || acoustic_shape[0] != 1 || acoustic_shape[2] != token_count {
            bail!(
                "encoder acoustic_tokens has shape {acoustic_shape:?}, expected [1, dim, {token_count}]"
            );
        }
        let feature_dim = acoustic_shape[1];
        if prosodic_tokens.shape() != [1, token_count, feature_dim] {
            bail!(
                "encoder prosodic_tokens has shape {:?}, expected [1, {token_count}, {feature_dim}]",
                prosodic_tokens.shape()
            );
        }
        if durations.shape() != [1, token_count] {
            bail!(
                "encoder durations has shape {:?}, expected [1, {token_count}]",
                durations.shape()
            );
        }

        let durations: Vec<i64> = durations.iter().copied().collect();
        let frame_to_token = build_frame_to_token(&durations)?;
        if frame_to_token.len() > self.bundle.max_frames() {
            bail!(
                "predicted {} frames, exceeding max_frames={}",
                frame_to_token.len(),
                self.bundle.max_frames()
            );
        }

        let frames = frame_to_token.len();
        let frame_to_token_arr = Array::from_shape_vec(IxDyn(&[frames]), frame_to_token.clone())?;
        let prosodic_arr = Array::from_shape_vec(
            IxDyn(&[1, token_count, feature_dim]),
            prosodic_tokens.iter().copied().collect(),
        )?;
        let voice_arr = Array::from_shape_vec(IxDyn(&[1, voice.len()]), voice.to_vec())?;
        let frame_to_token_cow = CowArray::from(frame_to_token_arr);
        let prosodic_cow = CowArray::from(prosodic_arr);
        let prosody_voice_cow = CowArray::from(voice_arr);
        let mut values = Vec::with_capacity(self.prosody.inputs.len());
        for input in &self.prosody.inputs {
            let value = match input.name.as_str() {
                "prosodic_tokens" => Value::from_array(self.prosody.allocator(), &prosodic_cow)?,
                "frame_to_token" => {
                    Value::from_array(self.prosody.allocator(), &frame_to_token_cow)?
                }
                "voice" => Value::from_array(self.prosody.allocator(), &prosody_voice_cow)?,
                other => bail!("unexpected prosody input {other:?}"),
            };
            values.push(value);
        }
        let outputs = self.prosody.run(values).context("prosody run")?;
        let f0 = extract_f32(&outputs, self.prosody_outputs.f0, "f0")?;
        let energy = extract_f32(&outputs, self.prosody_outputs.energy, "energy")?;
        if f0.shape() != [1, 2 * frames] || energy.shape() != [1, 2 * frames] {
            bail!(
                "prosody outputs have shapes {:?}/{:?}, expected [1, {}]",
                f0.shape(),
                energy.shape(),
                2 * frames
            );
        }

        Ok(Encoded {
            voice: voice.to_vec(),
            acoustic_tokens,
            frame_to_token,
            f0: f0.iter().copied().collect(),
            energy: energy.iter().copied().collect(),
            durations,
        })
    }

    fn decode_exact(&self, encoded: &Encoded) -> Result<Vec<f32>> {
        let DecoderStage::Exact(session) = &self.decoder else {
            bail!("exact decoder is not loaded");
        };
        let frames = encoded.frame_to_token.len();
        let expected = frames * self.bundle.samples_per_frame();

        let acoustic_arr = CowArray::from(encoded.acoustic_tokens.clone());
        let frame_arr = Array::from_shape_vec(IxDyn(&[frames]), encoded.frame_to_token.clone())?;
        let f0_arr = Array::from_shape_vec(IxDyn(&[1, 2 * frames]), encoded.f0.clone())?;
        let energy_arr = Array::from_shape_vec(IxDyn(&[1, 2 * frames]), encoded.energy.clone())?;
        let voice_arr =
            Array::from_shape_vec(IxDyn(&[1, encoded.voice.len()]), encoded.voice.clone())?;
        let frame_cow = CowArray::from(frame_arr);
        let f0_cow = CowArray::from(f0_arr);
        let energy_cow = CowArray::from(energy_arr);
        let voice_cow = CowArray::from(voice_arr);

        let mut values = Vec::with_capacity(session.inputs.len());
        for input in &session.inputs {
            let value = match input.name.as_str() {
                "acoustic_tokens" => Value::from_array(session.allocator(), &acoustic_arr)?,
                "frame_to_token" => Value::from_array(session.allocator(), &frame_cow)?,
                "f0" => Value::from_array(session.allocator(), &f0_cow)?,
                "energy" => Value::from_array(session.allocator(), &energy_cow)?,
                "voice" => Value::from_array(session.allocator(), &voice_cow)?,
                other => bail!("unexpected decoder input {other:?}"),
            };
            values.push(value);
        }
        let outputs = session.run(values).context("decoder run")?;
        let audio = extract_f32(&outputs, self.decoder_outputs.audio, "audio")?;
        let audio_length = match self.decoder_outputs.lengths {
            Some(index) => extract_i64(&outputs, index, "audio_lengths")?
                .iter()
                .next()
                .copied()
                .unwrap_or(expected as i64),
            None => expected as i64,
        };
        if audio_length != expected as i64 {
            bail!("decoder returned {audio_length} samples, expected {expected}");
        }
        let flat: Vec<f32> = audio.iter().copied().collect();
        if flat.len() < expected {
            bail!(
                "decoder audio has {} samples, expected at least {expected}",
                flat.len()
            );
        }
        let audio = &flat[..expected];
        if !audio.iter().all(|sample| sample.is_finite()) {
            bail!("decoder produced NaN or Inf");
        }
        Ok(audio.to_vec())
    }

    fn decode_windows(
        &self,
        encoded: &Encoded,
        chunk_frames: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut chunks = Vec::new();
        self.decode_windows_with(encoded, chunk_frames, &mut |chunk| {
            chunks.push(chunk.to_vec());
            true
        })?;
        Ok(chunks)
    }

    /// Декодувати вікна, віддаючи кожне колбеку; `false` — зупинитись.
    fn decode_windows_with<F>(
        &self,
        encoded: &Encoded,
        chunk_frames: Option<usize>,
        on_chunk: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(&[f32]) -> bool,
    {
        let DecoderStage::Streaming {
            source,
            decoder,
            chunk_frames: max_chunk,
            context_frames,
        } = &self.decoder
        else {
            bail!("streaming decoder is not loaded");
        };
        let chunk = chunk_frames.unwrap_or(*max_chunk);
        if chunk == 0 || chunk > *max_chunk {
            bail!("chunk_frames must be between 1 and {max_chunk}");
        }
        let frames = encoded.frame_to_token.len();
        let step = self.bundle.samples_per_frame();

        let f0_arr = Array::from_shape_vec(IxDyn(&[1, 2 * frames]), encoded.f0.clone())?;
        let f0_cow = CowArray::from(f0_arr);
        let source_value = {
            let mut values = Vec::with_capacity(source.inputs.len());
            for input in &source.inputs {
                let value = match input.name.as_str() {
                    "f0" => Value::from_array(source.allocator(), &f0_cow)?,
                    other => bail!("unexpected source input {other:?}"),
                };
                values.push(value);
            }
            let outputs = source.run(values).context("source run")?;
            extract_f32(&outputs, self.source_output, "source")?
        };
        if source_value.shape() != [1, frames * step] {
            bail!(
                "source has shape {:?}, expected [1, {}]",
                source_value.shape(),
                frames * step
            );
        }
        let source: Vec<f32> = source_value.iter().copied().collect();

        for (start, end, window_start, window_end) in stream_windows(frames, chunk, *context_frames)
        {
            let window = window_end - window_start;
            let frame_arr = Array::from_shape_vec(
                IxDyn(&[window]),
                encoded.frame_to_token[window_start..window_end].to_vec(),
            )?;
            let f0_window = Array::from_shape_vec(
                IxDyn(&[1, 2 * window]),
                encoded.f0[2 * window_start..2 * window_end].to_vec(),
            )?;
            let energy_window = Array::from_shape_vec(
                IxDyn(&[1, 2 * window]),
                encoded.energy[2 * window_start..2 * window_end].to_vec(),
            )?;
            let source_window = Array::from_shape_vec(
                IxDyn(&[1, window * step]),
                source[window_start * step..window_end * step].to_vec(),
            )?;
            let acoustic_arr = CowArray::from(encoded.acoustic_tokens.clone());
            let voice_arr =
                Array::from_shape_vec(IxDyn(&[1, encoded.voice.len()]), encoded.voice.clone())?;
            let voice_cow = CowArray::from(voice_arr);
            let frame_cow = CowArray::from(frame_arr);
            let f0_cow = CowArray::from(f0_window);
            let energy_cow = CowArray::from(energy_window);
            let source_cow = CowArray::from(source_window);

            let mut values = Vec::with_capacity(decoder.inputs.len());
            for input in &decoder.inputs {
                let value = match input.name.as_str() {
                    "acoustic_tokens" => Value::from_array(decoder.allocator(), &acoustic_arr)?,
                    "frame_to_token" => Value::from_array(decoder.allocator(), &frame_cow)?,
                    "f0" => Value::from_array(decoder.allocator(), &f0_cow)?,
                    "energy" => Value::from_array(decoder.allocator(), &energy_cow)?,
                    "source" => Value::from_array(decoder.allocator(), &source_cow)?,
                    "voice" => Value::from_array(decoder.allocator(), &voice_cow)?,
                    other => bail!("unexpected decoder_window input {other:?}"),
                };
                values.push(value);
            }
            let outputs = decoder.run(values).context("decoder_window run")?;
            let audio = extract_f32(&outputs, self.decoder_outputs.audio, "audio")?;
            let flat: Vec<f32> = audio.iter().copied().collect();
            let offset = (start - window_start) * step;
            let length = (end - start) * step;
            if flat.len() < offset + length {
                bail!(
                    "decoder_window returned {} samples, expected at least {}",
                    flat.len(),
                    offset + length
                );
            }
            let chunk = &flat[offset..offset + length];
            if !chunk.iter().all(|sample| sample.is_finite()) {
                bail!("decoder_window produced NaN or Inf");
            }
            if !on_chunk(chunk) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn load_stage(env: &Arc<Environment>, path: &Path, threads: i16) -> Result<Session> {
    if !path.is_file() {
        bail!("ONNX stage not found: {}", path.display());
    }
    SessionBuilder::new(env)?
        .with_intra_threads(threads)?
        .with_inter_threads(1)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(path)
        .with_context(|| format!("load ONNX stage {}", path.display()))
}

fn require_inputs(session: &Session, expected: &[&str], stage: &str) -> Result<()> {
    let names: Vec<&str> = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect();
    for name in expected {
        if !names.contains(name) {
            bail!("{stage} has no {name:?} input; got {names:?}");
        }
    }
    Ok(())
}

fn output_index(session: &Session, name: &str) -> Result<usize> {
    session
        .outputs
        .iter()
        .position(|output| output.name == name)
        .with_context(|| format!("model has no {name:?} output"))
}

fn optional_output_index(session: &Session, name: &str) -> Option<usize> {
    session
        .outputs
        .iter()
        .position(|output| output.name == name)
}

fn extract_f32(outputs: &[Value<'static>], index: usize, name: &str) -> Result<ArrayD<f32>> {
    let tensor = outputs
        .get(index)
        .with_context(|| format!("model returned no {name:?} output"))?
        .try_extract::<f32>()
        .with_context(|| format!("extract {name} as f32"))?;
    Ok(tensor.view().to_owned())
}

fn extract_i64(outputs: &[Value<'static>], index: usize, name: &str) -> Result<ArrayD<i64>> {
    let tensor = outputs
        .get(index)
        .with_context(|| format!("model returned no {name:?} output"))?
        .try_extract::<i64>()
        .with_context(|| format!("extract {name} as i64"))?;
    Ok(tensor.view().to_owned())
}

/// `frame_to_token`: індекс токена для кожного кадру, без padding.
fn build_frame_to_token(durations: &[i64]) -> Result<Vec<i64>> {
    let mut frames = 0usize;
    for &duration in durations {
        if duration < 0 {
            bail!("durations must be non-negative, got {duration}");
        }
        frames = frames
            .checked_add(duration as usize)
            .context("frame count overflow")?;
    }
    if frames == 0 {
        bail!("model predicted zero frames");
    }
    let mut frame_to_token = Vec::with_capacity(frames);
    for (token, &duration) in durations.iter().enumerate() {
        for _ in 0..duration {
            frame_to_token.push(token as i64);
        }
    }
    Ok(frame_to_token)
}

/// Вікна стрімінгу: `(start, end, window_start, window_end)`.
fn stream_windows(
    frame_count: usize,
    chunk_frames: usize,
    context_frames: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let window = chunk_frames + 2 * context_frames;
    let mut windows = Vec::new();
    let mut start = 0;
    while start < frame_count {
        let end = (start + chunk_frames).min(frame_count);
        let window_start = start
            .saturating_sub(context_frames)
            .min(frame_count.saturating_sub(window));
        let window_end = (window_start + window).min(frame_count);
        windows.push((start, end, window_start, window_end));
        start += chunk_frames;
    }
    windows
}

/// Стан обрізання стартової тиші для потокових вікон.
struct LeadingTrimmer {
    skip: usize,
    fade: usize,
    faded: usize,
}

impl LeadingTrimmer {
    fn new(skip: usize, fade: usize) -> Self {
        Self {
            skip,
            fade,
            faded: 0,
        }
    }

    /// Обробити шматок; `None`, якщо після обрізання він порожній.
    fn apply(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        let mut start = 0;
        if self.skip > 0 {
            let dropped = self.skip.min(chunk.len());
            start = dropped;
            self.skip -= dropped;
        }
        let mut out = chunk[start..].to_vec();
        if self.faded < self.fade && !out.is_empty() {
            let count = (self.fade - self.faded).min(out.len());
            for (index, sample) in out.iter_mut().take(count).enumerate() {
                *sample *= (self.faded + index + 1) as f32 / self.fade as f32;
            }
            self.faded += count;
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

/// Зрізати стартову тишу та зробити фейд-ін (як `_trim_leading` у Python).
pub fn trim_leading_silence(chunks: &mut [Vec<f32>], skip: usize, fade: usize) {
    let mut remaining = skip;
    let mut faded = 0usize;
    for chunk in chunks.iter_mut() {
        if remaining > 0 {
            let dropped = remaining.min(chunk.len());
            chunk.drain(..dropped);
            remaining -= dropped;
        }
        if faded < fade && !chunk.is_empty() {
            let count = (fade - faded).min(chunk.len());
            for (index, sample) in chunk.iter_mut().take(count).enumerate() {
                let gain = (faded + index + 1) as f32 / fade as f32;
                *sample *= gain;
            }
            faded += count;
        }
    }
}

/// Скільки семплів стартової тиші зрізати для першого чанка.
pub fn stream_silence_skip(durations: &[i64], samples_per_frame: usize, sample_rate: u32) -> usize {
    let silence = durations.first().copied().unwrap_or(0).max(0) as usize * samples_per_frame;
    let margin = (STREAM_SILENCE_MARGIN_SECONDS * sample_rate as f32).round() as usize;
    silence.saturating_sub(margin)
}

/// Довжина фейд-іну для стрімінгу.
pub fn stream_fade_in() -> usize {
    STREAM_FADE_IN_SAMPLES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_frame_to_token_without_padding() {
        assert_eq!(
            build_frame_to_token(&[2, 0, 3]).unwrap(),
            vec![0, 0, 2, 2, 2]
        );
        assert!(build_frame_to_token(&[]).is_err());
        assert!(build_frame_to_token(&[0, 0]).is_err());
        assert!(build_frame_to_token(&[-1]).is_err());
    }

    #[test]
    fn stream_windows_cover_utterance_without_gaps() {
        let windows = stream_windows(100, 40, 20);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], (0, 40, 0, 80));
        assert_eq!(windows[1], (40, 80, 20, 100));
        assert_eq!(windows[2], (80, 100, 20, 100));
        for (start, end, _, _) in &windows {
            assert!(start < end);
        }
    }

    #[test]
    fn stream_windows_handle_short_utterance() {
        assert_eq!(stream_windows(5, 40, 20), vec![(0, 5, 0, 5)]);
    }

    #[test]
    fn trims_silence_across_chunks() {
        let mut chunks = vec![vec![1.0; 10], vec![1.0; 10]];
        trim_leading_silence(&mut chunks, 15, 4);
        assert_eq!(chunks[0].len(), 0);
        assert_eq!(chunks[1].len(), 5);
        assert!((chunks[1][0] - 0.25).abs() < 1e-6);
        assert!((chunks[1][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trimmer_matches_batch_trimming() {
        let mut batch = vec![vec![1.0; 10], vec![1.0; 10], vec![1.0; 10]];
        trim_leading_silence(&mut batch, 12, 4);
        let batch: Vec<Vec<f32>> = batch
            .into_iter()
            .filter(|chunk| !chunk.is_empty())
            .collect();

        let mut trimmer = LeadingTrimmer::new(12, 4);
        let streamed: Vec<Vec<f32>> = [vec![1.0; 10], vec![1.0; 10], vec![1.0; 10]]
            .iter()
            .filter_map(|chunk| trimmer.apply(chunk))
            .collect();

        assert_eq!(streamed, batch);
    }

    #[test]
    fn trimmer_without_skip_keeps_fade_disabled() {
        let mut trimmer = LeadingTrimmer::new(0, 0);
        assert_eq!(trimmer.apply(&[0.5, 0.5]), Some(vec![0.5, 0.5]));
        assert_eq!(trimmer.apply(&[]), None);
    }

    #[test]
    fn pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HolosPipeline>();
    }
}
