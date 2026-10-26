//! Wyoming TCP-протокол для Home Assistant.
//!
//! Підтримуються події `describe`, `synthesize`, `ping`; відповіді —
//! `info`, `audio-start`, `audio-chunk`, `audio-stop`, `pong`, `error`.
//! Кожна подія — JSON-рядок; якщо є `payload_length`, далі йдуть сирі байти.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use speech_audio::SAMPLE_RATE;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use crate::service::{StreamMeta, StreamSummary, SynthError, SynthService, VerbalizerMode};

/// Розмір семпла PCM16 (байти).
const SAMPLE_WIDTH: u16 = 2;
const CHANNELS: u16 = 1;

#[derive(Debug, Deserialize)]
struct WyomingEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    data: Option<Value>,
    #[serde(default)]
    payload_length: Option<u64>,
}

enum StreamEvent {
    Meta(StreamMeta),
    Chunk(Vec<u8>),
    Done(StreamSummary),
    Error(SynthError),
}

/// Запустити Wyoming-сервер; працює до помилки accept.
pub async fn serve(addr: SocketAddr, service: Arc<SynthService>) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind Wyoming TCP {addr}"))?;
    tracing::info!(%addr, "Wyoming TCP listening");

    loop {
        let (stream, peer) = listener.accept().await?;
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, service).await {
                tracing::debug!(%peer, error = %error, "Wyoming connection closed");
            }
        });
    }
}

async fn handle_connection(stream: TcpStream, service: Arc<SynthService>) -> Result<()> {
    let (read_half, mut writer) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader.read_line(&mut line).await?;
        if read == 0 {
            return Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event: WyomingEvent = match serde_json::from_str(trimmed) {
            Ok(event) => event,
            Err(error) => {
                write_error(&mut writer, &format!("invalid JSON: {error}")).await?;
                continue;
            }
        };

        // Будь-який payload (наприклад, аудіо для ASR) треба прочитати,
        // інакше він зіпсує наступний JSON-рядок.
        if let Some(length) = event.payload_length {
            if length > 0 {
                let mut payload = vec![0u8; length as usize];
                reader.read_exact(&mut payload).await?;
            }
        }

        match event.event_type.as_str() {
            "describe" => {
                write_event(&mut writer, "info", info_payload(&service), None).await?;
            }
            "synthesize" => {
                let data = event.data.unwrap_or(Value::Null);
                synthesize(&service, &data, &mut writer).await?;
            }
            "ping" => {
                write_event(&mut writer, "pong", json!({}), None).await?;
            }
            other => {
                write_error(&mut writer, &format!("unsupported event type: {other}")).await?;
            }
        }
    }
}

fn info_payload(service: &SynthService) -> Value {
    let mut voices: Vec<Value> = service
        .voice_names()
        .into_iter()
        .map(|name| {
            json!({
                "name": name,
                "description": name,
                "languages": ["uk"],
            })
        })
        .collect();
    voices.extend(service.custom_voice_names().into_iter().map(|name| {
        json!({
            "name": name,
            "description": format!("{name} (cloned)"),
            "languages": ["uk"],
        })
    }));

    json!({
        "asr": [],
        "tts": [{
            "name": "promin-tts",
            "description": "Promin TTS — Ukrainian neural text-to-speech",
            "version": env!("CARGO_PKG_VERSION"),
            "voices": voices,
            "supports_synthesize_streaming": true,
        }],
        "handle": [],
        "wake": [],
        "intent": [],
    })
}

fn voice_name(data: &Value) -> Option<String> {
    match data.get("voice") {
        Some(Value::String(name)) => Some(name.clone()),
        Some(Value::Object(object)) => object
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

async fn synthesize<W>(service: &Arc<SynthService>, data: &Value, writer: &mut W) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let Some(text) = data.get("text").and_then(Value::as_str) else {
        write_error(writer, "synthesize requires data.text").await?;
        return Ok(());
    };
    let text = text.to_string();
    let voice = voice_name(data);
    let speed = data.get("speed").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let raw_stress = data
        .get("raw_stress")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mode = VerbalizerMode::parse(data.get("auto_verbalize"));
    let max_chars = data
        .get("max_chars")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(1_200)
        .clamp(50, 5_000);

    let (tx, mut rx) = mpsc::channel::<StreamEvent>(4);
    let service = Arc::clone(service);
    tokio::task::spawn_blocking(move || {
        let result = service.synthesize_stream_pcm(
            &text,
            voice.as_deref(),
            None,
            speed,
            mode,
            raw_stress,
            max_chars,
            |meta| tx.blocking_send(StreamEvent::Meta(meta.clone())).is_ok(),
            |chunk| tx.blocking_send(StreamEvent::Chunk(chunk.to_vec())).is_ok(),
        );
        match result {
            Ok(summary) => {
                let _ = tx.blocking_send(StreamEvent::Done(summary));
            }
            Err(error) => {
                let _ = tx.blocking_send(StreamEvent::Error(error));
            }
        }
    });

    let first = match rx.recv().await {
        Some(event) => event,
        None => {
            write_error(writer, "synthesis task ended unexpectedly").await?;
            return Ok(());
        }
    };
    match first {
        StreamEvent::Meta(meta) => {
            tracing::debug!(
                verbalizer = meta.verbalizer,
                chunks = meta.chunks,
                "Wyoming synthesis started"
            );
        }
        StreamEvent::Error(error) => {
            write_error(writer, &error.to_string()).await?;
            return Ok(());
        }
        _ => {
            write_error(writer, "unexpected synthesis event").await?;
            return Ok(());
        }
    }

    let audio_format = json!({
        "rate": SAMPLE_RATE,
        "width": SAMPLE_WIDTH,
        "channels": CHANNELS,
    });
    write_event(writer, "audio-start", audio_format.clone(), None).await?;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Chunk(bytes) => {
                write_event(writer, "audio-chunk", audio_format.clone(), Some(&bytes)).await?;
            }
            StreamEvent::Done(summary) => {
                tracing::debug!(
                    chunks = summary.chunks,
                    audio_ms = summary.audio_ms,
                    verbalizer = summary.verbalizer,
                    "Wyoming synthesis finished"
                );
                write_event(writer, "audio-stop", json!({}), None).await?;
                break;
            }
            StreamEvent::Error(error) => {
                write_error(writer, &error.to_string()).await?;
                break;
            }
            StreamEvent::Meta(_) => {}
        }
    }

    Ok(())
}

async fn write_event<W>(
    writer: &mut W,
    event_type: &str,
    data: Value,
    payload: Option<&[u8]>,
) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let mut event = json!({
        "type": event_type,
        "data": data,
    });
    if let Some(payload) = payload {
        event["payload_length"] = json!(payload.len());
    }
    let mut encoded = serde_json::to_vec(&event)?;
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    if let Some(payload) = payload {
        writer.write_all(payload).await?;
    }
    writer.flush().await?;
    Ok(())
}

async fn write_error<W>(writer: &mut W, message: &str) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write_event(writer, "error", json!({ "text": message }), None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_voice_name_from_object_and_string() {
        assert_eq!(
            voice_name(&json!({ "voice": { "name": "Speaker_0" } })),
            Some("Speaker_0".to_string())
        );
        assert_eq!(
            voice_name(&json!({ "voice": "Тест Клон" })),
            Some("Тест Клон".to_string())
        );
        assert_eq!(voice_name(&json!({})), None);
    }

    #[tokio::test]
    async fn writes_json_line_and_payload() {
        let mut buffer = Vec::new();
        write_event(
            &mut buffer,
            "audio-chunk",
            json!({ "rate": 24_000 }),
            Some(b"pcm"),
        )
        .await
        .unwrap();
        let newline = buffer.iter().position(|byte| *byte == b'\n').unwrap();
        let event: Value = serde_json::from_slice(&buffer[..newline]).unwrap();
        assert_eq!(event["type"], "audio-chunk");
        assert_eq!(event["payload_length"], 3);
        assert_eq!(&buffer[newline + 1..], b"pcm");
    }
}
