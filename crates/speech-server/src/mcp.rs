//! MCP (Model Context Protocol) поверх HTTP JSON-RPC 2.0.
//!
//! Стандартний stateless Streamable HTTP endpoint: клієнт надсилає JSON-RPC
//! запити на `POST /mcp`, сервер повертає JSON-відповідь. Для клієнтів без
//! HTTP-транспорту є stdio-проксі `promin-mcp`.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde_json::{json, Value};
use speech_audio::{wav_bytes, SAMPLE_RATE};
use speech_core::custom_voices::VOICE_DIM;
use speech_core::jobs::JobRequest;

use crate::service::VerbalizerMode;
use crate::AppState;

const SERVER_NAME: &str = "promin-tts";
const DEFAULT_PROTOCOL_VERSION: &str = "2025-03-26";

/// `POST /mcp`: обробити один JSON-RPC запит.
pub async fn handle(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let id = request.get("id").cloned().filter(|value| !value.is_null());

    // JSON-RPC notifications (без id) не мають відповіді.
    let Some(id) = id else {
        return StatusCode::ACCEPTED.into_response();
    };

    match method.as_str() {
        "initialize" => {
            let requested = request
                .pointer("/params/protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_PROTOCOL_VERSION);
            let version = match requested {
                "2024-11-05" | "2025-03-26" | "2025-06-18" => requested,
                _ => DEFAULT_PROTOCOL_VERSION,
            };
            rpc_result(
                id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "instructions": "Promin TTS: Ukrainian text-to-speech, long-form jobs and voice cloning.",
                }),
            )
        }
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match call_tool(&state, &name, &arguments).await {
                Ok(result) => rpc_result(id, result),
                Err(error) => rpc_result(id, tool_error(error)),
            }
        }
        other => rpc_error(id, -32601, &format!("method not found: {other}")),
    }
}

fn rpc_result(id: Value, result: Value) -> axum::response::Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

fn rpc_error(id: Value, code: i64, message: &str) -> axum::response::Response {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    }))
    .into_response()
}

fn tool_text(text: impl Into<String>) -> Value {
    json!({
        "content": [{ "type": "text", "text": text.into() }],
    })
}

fn tool_error(message: impl Into<String>) -> Value {
    json!({
        "content": [{ "type": "text", "text": message.into() }],
        "isError": true,
    })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "synthesize",
            "description": "Synthesize short Ukrainian text (max 5000 chars) to WAV audio.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to speak" },
                    "voice": { "type": "string", "description": "Preset or cloned voice name" },
                    "speed": { "type": "number", "minimum": 0.5, "maximum": 2.0, "default": 1.0 },
                    "auto_verbalize": { "description": "true/false or neural/rule/off" },
                    "raw_stress": { "type": "boolean", "default": false },
                },
                "required": ["text"],
            },
        }),
        json!({
            "name": "synthesize_async",
            "description": "Queue long text for background synthesis. Returns a job record; poll get_job or subscribe to SSE.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "voice": { "type": "string" },
                    "speed": { "type": "number", "minimum": 0.5, "maximum": 2.0, "default": 1.0 },
                    "auto_verbalize": { "description": "true/false or neural/rule/off" },
                    "raw_stress": { "type": "boolean", "default": false },
                    "webhook_url": { "type": "string" },
                    "max_chars": { "type": "integer", "minimum": 50, "maximum": 5000 },
                    "crossfade_ms": { "type": "integer", "minimum": 0, "maximum": 200 },
                },
                "required": ["text"],
            },
        }),
        json!({
            "name": "get_job",
            "description": "Get status, progress and audio URL of an async synthesis job.",
            "inputSchema": {
                "type": "object",
                "properties": { "job_id": { "type": "string" } },
                "required": ["job_id"],
            },
        }),
        json!({
            "name": "list_jobs",
            "description": "List recent async synthesis jobs.",
            "inputSchema": {
                "type": "object",
                "properties": { "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 50 } },
            },
        }),
        json!({
            "name": "list_voices",
            "description": "List preset and cloned voices.",
            "inputSchema": { "type": "object", "properties": {} },
        }),
        json!({
            "name": "clone_voice",
            "description": "Create a custom voice from a base64 WAV reference (0.25–120 s).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "audio_base64": { "type": "string", "description": "Base64 WAV reference" },
                },
                "required": ["name", "audio_base64"],
            },
        }),
        json!({
            "name": "delete_voice",
            "description": "Delete a cloned voice by name.",
            "inputSchema": {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"],
            },
        }),
        json!({
            "name": "verbalize",
            "description": "Verbalize text (numbers, dates, times) without synthesis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "mode": { "type": "string", "enum": ["neural", "rule", "off"] },
                },
                "required": ["text"],
            },
        }),
    ]
}

async fn call_tool(state: &AppState, name: &str, arguments: &Value) -> Result<Value, String> {
    match name {
        "synthesize" => synthesize(state, arguments).await,
        "synthesize_async" => synthesize_async(state, arguments).await,
        "get_job" => {
            let id = required_str(arguments, "job_id")?;
            let job = state
                .jobs
                .get(id)
                .ok_or_else(|| format!("unknown job: {id}"))?;
            Ok(tool_text(pretty(&job)?))
        }
        "list_jobs" => {
            let limit = arguments
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(50)
                .clamp(1, 500) as usize;
            let jobs = state.jobs.list(limit);
            let summaries: Vec<crate::jobs::JobSummary> =
                jobs.iter().map(crate::jobs::JobSummary::from).collect();
            Ok(tool_text(pretty(&summaries)?))
        }
        "list_voices" => {
            let value = json!({
                "default": state.service.default_voice(),
                "sample_rate": SAMPLE_RATE,
                "voices": state.service.voice_names(),
                "custom": state.service.custom_voice_names(),
            });
            Ok(tool_text(pretty(&value)?))
        }
        "clone_voice" => {
            let name = required_str(arguments, "name")?.to_string();
            let audio = required_str(arguments, "audio_base64")?.to_string();
            let service = Arc::clone(&state.service);
            let task_name = name.clone();
            tokio::task::spawn_blocking(move || service.enroll_voice(&task_name, &audio))
                .await
                .map_err(|error| format!("enrollment task failed: {error}"))?
                .map_err(|error| error.to_string())?;
            Ok(tool_text(
                json!({ "name": name, "dim": VOICE_DIM }).to_string(),
            ))
        }
        "delete_voice" => {
            let name = required_str(arguments, "name")?.to_string();
            let service = Arc::clone(&state.service);
            let task_name = name.clone();
            let deleted = tokio::task::spawn_blocking(move || service.delete_voice(&task_name))
                .await
                .map_err(|error| format!("deletion task failed: {error}"))?
                .map_err(|error| error.to_string())?;
            if deleted {
                Ok(tool_text(format!("deleted voice: {name}")))
            } else {
                Err(format!("unknown custom voice: {name}"))
            }
        }
        "verbalize" => {
            let text = required_str(arguments, "text")?.to_string();
            let mode = VerbalizerMode::parse_mode(arguments.get("mode").and_then(Value::as_str));
            let service = Arc::clone(&state.service);
            let verbalized =
                tokio::task::spawn_blocking(move || service.verbalize_text(&text, mode))
                    .await
                    .map_err(|error| format!("verbalization task failed: {error}"))?
                    .map_err(|error| error.to_string())?;
            Ok(tool_text(verbalized))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

async fn synthesize(state: &AppState, arguments: &Value) -> Result<Value, String> {
    let text = required_str(arguments, "text")?.to_string();
    let voice = optional_str(arguments, "voice");
    let speed = optional_speed(arguments)?;
    let mode = VerbalizerMode::parse(arguments.get("auto_verbalize"));
    let raw_stress = arguments
        .get("raw_stress")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let service = Arc::clone(&state.service);
    let pcm = tokio::task::spawn_blocking(move || {
        service.synthesize_pcm(&text, voice.as_deref(), None, speed, mode, raw_stress)
    })
    .await
    .map_err(|error| format!("synthesis task failed: {error}"))?
    .map_err(|error| error.to_string())?;
    let wav = wav_bytes(&pcm.pcm, SAMPLE_RATE).map_err(|error| error.to_string())?;
    let audio = BASE64.encode(wav);
    Ok(json!({
        "content": [
            { "type": "audio", "data": audio, "mimeType": "audio/wav" },
            {
                "type": "text",
                "text": format!(
                    "verbalizer={}, chunks={}, duration_ms={}, sample_rate={}",
                    pcm.verbalizer,
                    pcm.chunks,
                    pcm.audio_ms(),
                    SAMPLE_RATE
                ),
            },
        ],
    }))
}

async fn synthesize_async(state: &AppState, arguments: &Value) -> Result<Value, String> {
    let text = required_str(arguments, "text")?.to_string();
    let request = JobRequest {
        text,
        voice: optional_str(arguments, "voice"),
        speed: optional_speed(arguments)?,
        auto_verbalize: arguments.get("auto_verbalize").cloned(),
        raw_stress: arguments
            .get("raw_stress")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        webhook_url: optional_str(arguments, "webhook_url"),
        max_chars: arguments
            .get("max_chars")
            .and_then(Value::as_u64)
            .map(|value| value as usize),
        crossfade_ms: arguments
            .get("crossfade_ms")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
    };
    let job = state.jobs.submit(request).await?;
    Ok(tool_text(pretty(&job)?))
}

fn required_str<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} is required"))
}

fn optional_str(arguments: &Value, key: &str) -> Option<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn optional_speed(arguments: &Value) -> Result<f32, String> {
    match arguments.get("speed") {
        None | Some(Value::Null) => Ok(1.0),
        Some(value) => value
            .as_f64()
            .map(|speed| speed as f32)
            .ok_or_else(|| "speed must be a number".to_string()),
    }
}

fn pretty<T: serde::Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_list_has_expected_names() {
        let names: Vec<String> = tool_definitions()
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        for expected in [
            "synthesize",
            "synthesize_async",
            "get_job",
            "list_voices",
            "clone_voice",
            "delete_voice",
            "verbalize",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn validates_arguments() {
        assert_eq!(
            required_str(&json!({ "text": " Привіт " }), "text"),
            Ok("Привіт")
        );
        assert!(required_str(&json!({}), "text").is_err());
        assert_eq!(optional_speed(&json!({})), Ok(1.0));
        assert!(optional_speed(&json!({ "speed": "fast" })).is_err());
        assert_eq!(
            optional_str(&json!({ "voice": " Speaker_0 " }), "voice"),
            Some("Speaker_0".to_string())
        );
    }

    #[test]
    fn error_result_is_flagged() {
        let value = tool_error("boom");
        assert_eq!(value["isError"], true);
        assert_eq!(value["content"][0]["text"], "boom");
    }
}
