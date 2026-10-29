use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, Query, Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post, put},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use speech_audio::{default_streaming_wav_header, pcm16_bytes, wav_bytes, SAMPLE_RATE};
use speech_core::jobs::{JobRequest, JobStatus};
use speech_core::projects::{LoudnessSettings, ProjectStore};
use tokio::sync::{broadcast, mpsc};

mod downloader;
mod exports;
mod jobs;
mod mcp;
mod projects;
mod pronunciation;
mod service;
mod subtitles;
mod wyoming;

use downloader::{ensure_models, DownloadOptions, ModelLayout};
use jobs::{CancelOutcome, JobManager};
use pronunciation::{
    PronunciationFile, PronunciationPayload, PronunciationStore, PronunciationStoreError,
};
use service::{ModelPaths, StreamMeta, StreamSummary, SynthError, SynthService, VerbalizerMode};

#[derive(Parser, Debug)]
#[command(name = "speech-server", version, about = "Promin TTS server")]
struct Args {
    /// HTTP listen address
    #[arg(long, env = "PROMIN_HTTP_ADDR", default_value = "0.0.0.0:8010")]
    http_addr: SocketAddr,

    /// Directory with TTS models (`tts/` + `stress/` or flat layout)
    #[arg(long, env = "PROMIN_MODELS_DIR", default_value = "models")]
    models_dir: PathBuf,

    /// Model format: `auto`, `legacy` (single graph) or `bundle` (holos-onnx)
    #[arg(long, env = "PROMIN_MODEL_FORMAT", default_value = "auto")]
    model_format: String,

    /// ONNX intra-op threads
    #[arg(long, env = "PROMIN_THREADS", default_value_t = 4)]
    threads: i16,

    /// Directory with cloned voices (`<name>.npy`); default `<models>/speakers`
    #[arg(long, env = "PROMIN_SPEAKERS_DIR")]
    speakers_dir: Option<PathBuf>,

    /// Directory for async job records and audio; default `<models>/jobs`
    #[arg(long, env = "PROMIN_JOBS_DIR")]
    jobs_dir: Option<PathBuf>,

    /// Number of async jobs processed concurrently
    #[arg(long, env = "PROMIN_JOBS_CONCURRENCY", default_value_t = 1)]
    jobs_concurrency: usize,

    /// Number of audiobook exports processed concurrently
    #[arg(long, env = "PROMIN_EXPORT_CONCURRENCY", default_value_t = 1)]
    export_concurrency: usize,

    /// Public base URL used in webhook payloads and job audio links
    #[arg(long, env = "PROMIN_PUBLIC_URL")]
    public_url: Option<String>,

    /// Comma-separated API keys; empty disables authentication
    #[arg(long, env = "PROMIN_API_KEYS", default_value = "")]
    api_keys: String,

    /// Wyoming TCP listen address for Home Assistant (`off` disables)
    #[arg(long, env = "PROMIN_WYOMING_ADDR", default_value = "0.0.0.0:10200")]
    wyoming_addr: String,

    /// Global pronunciation dictionary; default `<data>/pronunciation.json`
    #[arg(long, env = "PROMIN_PRONUNCIATION_FILE")]
    pronunciation_file: Option<PathBuf>,

    /// Directory for book projects; default `<models>/projects`
    #[arg(long, env = "PROMIN_PROJECTS_DIR")]
    projects_dir: Option<PathBuf>,

    /// Integrated loudness target (LUFS) for new projects
    #[arg(long, env = "PROMIN_LOUDNESS_TARGET", default_value_t = -19.0)]
    loudness_target: f64,

    /// True-peak ceiling (dBTP) for new projects
    #[arg(long, env = "PROMIN_TRUE_PEAK", default_value_t = -3.0)]
    true_peak: f64,
}

#[derive(Clone)]
struct AppState {
    service: Arc<SynthService>,
    jobs: Arc<JobManager>,
    projects: Arc<ProjectStore>,
    exports: Arc<exports::ExportManager>,
    project_lock: Arc<tokio::sync::Mutex<()>>,
    pronunciation: Arc<PronunciationStore>,
    default_loudness: LoudnessSettings,
    public_url: String,
    api_keys: Arc<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct SynthesizeRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    /// WAV у base64: клонування голосу з референсу (пріоритет над `voice`).
    #[serde(default)]
    audio_base64: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
    /// `true`/`false` або `"neural"`/`"rule"`/`"off"`; відсутнє -> neural.
    #[serde(default)]
    auto_verbalize: Option<serde_json::Value>,
    #[serde(default)]
    raw_stress: bool,
}

fn default_speed() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
struct VerbalizeRequest {
    text: String,
    /// `"neural"` (типово) або `"rule"`/`"off"`.
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Serialize)]
struct VerbalizeResponse {
    text: String,
}

#[derive(Debug, Serialize)]
struct VoicesResponse {
    voices: Vec<String>,
    /// Клоновані голоси (`<speakers>/<name>.npy`).
    custom: Vec<String>,
    default: String,
    sample_rate: u32,
}

#[derive(Debug, Deserialize)]
struct EnrollVoiceRequest {
    name: String,
    /// WAV у base64 (0.25–120 с, будь-яка частота; конвертується в 24 кГц).
    audio_base64: String,
}

#[derive(Debug, Serialize)]
struct EnrollVoiceResponse {
    name: String,
    dim: usize,
}

async fn healthz(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "engine": "promin-tts",
        "version": env!("CARGO_PKG_VERSION"),
        "model": state.service.model_identity(),
        "voices": state.service.voice_count(),
        "custom_voices": state.service.custom_voice_count(),
        "sample_rate": state.service.sample_rate(),
        "verbalizer": state.service.verbalizer_name(),
        "accentor": state.service.accentor_name(),
        "voice_encoder": state.service.voice_encoder_name(),
        "pronunciation_rules": state.pronunciation.rule_count(),
    }))
}

/// `GET /v1/model`: ідентичність завантаженої моделі.
async fn model(State(state): State<AppState>) -> Json<speech_core::model::ModelIdentity> {
    Json(state.service.model_identity().clone())
}

async fn voices(State(state): State<AppState>) -> Json<VoicesResponse> {
    Json(VoicesResponse {
        voices: state
            .service
            .voice_names()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        custom: state.service.custom_voice_names(),
        default: state.service.default_voice().to_owned(),
        sample_rate: state.service.sample_rate(),
    })
}

/// Помилка сервісу -> HTTP-відповідь (`400`/`503`/`500`).
fn synth_error_response(error: SynthError, context: &'static str) -> Response {
    match error {
        SynthError::BadRequest(message) => {
            (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
        }
        SynthError::Unavailable(message) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": message })),
        )
            .into_response(),
        SynthError::Cancelled => {
            (StatusCode::CONFLICT, Json(json!({ "error": "cancelled" }))).into_response()
        }
        SynthError::Internal(error) => {
            tracing::error!(error = %error, context, "request failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": error.to_string() })),
            )
                .into_response()
        }
    }
}

async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<SynthesizeRequest>,
) -> Response {
    let mode = VerbalizerMode::parse(request.auto_verbalize.as_ref());
    let service = Arc::clone(&state.service);
    let result = tokio::task::spawn_blocking(move || {
        service.synthesize_wav(
            &request.text,
            request.voice.as_deref(),
            request.audio_base64.as_deref(),
            request.speed,
            mode,
            request.raw_stress,
        )
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
            headers.insert(
                HeaderName::from_static("x-promin-verbalizer"),
                HeaderValue::from_static(output.verbalizer),
            );
            headers.insert(
                HeaderName::from_static("x-promin-vb-ms"),
                HeaderValue::from_str(&output.vb_ms.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );
            headers.insert(
                HeaderName::from_static("x-promin-chunks"),
                HeaderValue::from_str(&output.chunks.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );
            headers.insert(
                HeaderName::from_static("x-promin-audio-ms"),
                HeaderValue::from_str(&output.audio_ms.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );
            (headers, output.wav).into_response()
        }
        Ok(Err(error)) => synth_error_response(error, "synthesis"),
        Err(join_error) => {
            tracing::error!(error = %join_error, "synthesis task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "internal task error" })),
            )
                .into_response()
        }
    }
}

async fn verbalize(
    State(state): State<AppState>,
    Json(request): Json<VerbalizeRequest>,
) -> Response {
    let mode = VerbalizerMode::parse_mode(request.mode.as_deref());
    let service = Arc::clone(&state.service);
    let result =
        tokio::task::spawn_blocking(move || service.verbalize_text(&request.text, mode)).await;

    match result {
        Ok(Ok(text)) => Json(VerbalizeResponse { text }).into_response(),
        Ok(Err(error)) => synth_error_response(error, "verbalization"),
        Err(join_error) => {
            tracing::error!(error = %join_error, "verbalization task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "internal task error" })),
            )
                .into_response()
        }
    }
}

/// `POST /api/voices`: WAV-референс -> `<speakers>/<name>.npy`.
async fn enroll_voice(
    State(state): State<AppState>,
    Json(request): Json<EnrollVoiceRequest>,
) -> Response {
    let EnrollVoiceRequest { name, audio_base64 } = request;
    let task_name = name.clone();
    let service = Arc::clone(&state.service);
    let result =
        tokio::task::spawn_blocking(move || service.enroll_voice(&task_name, &audio_base64)).await;

    match result {
        Ok(Ok(())) => Json(EnrollVoiceResponse {
            name,
            dim: speech_core::custom_voices::VOICE_DIM,
        })
        .into_response(),
        Ok(Err(error)) => synth_error_response(error, "voice enrollment"),
        Err(join_error) => {
            tracing::error!(error = %join_error, "voice enrollment task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "internal task error" })),
            )
                .into_response()
        }
    }
}

/// `DELETE /api/voices/{name}`: 204, якщо видалено; 404, якщо голосу немає.
async fn delete_voice(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let service = Arc::clone(&state.service);
    let result = tokio::task::spawn_blocking(move || service.delete_voice(&name)).await;

    match result {
        Ok(Ok(true)) => StatusCode::NO_CONTENT.into_response(),
        Ok(Ok(false)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown custom voice" })),
        )
            .into_response(),
        Ok(Err(error)) => synth_error_response(error, "voice deletion"),
        Err(join_error) => {
            tracing::error!(error = %join_error, "voice deletion task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "internal task error" })),
            )
                .into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
struct AsyncJobRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default)]
    auto_verbalize: Option<serde_json::Value>,
    #[serde(default)]
    raw_stress: bool,
    #[serde(default)]
    webhook_url: Option<String>,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default)]
    crossfade_ms: Option<u32>,
}

impl AsyncJobRequest {
    fn into_job_request(self) -> JobRequest {
        JobRequest {
            text: self.text,
            voice: self.voice,
            speed: self.speed,
            auto_verbalize: self.auto_verbalize,
            raw_stress: self.raw_stress,
            webhook_url: self.webhook_url,
            max_chars: self.max_chars,
            crossfade_ms: self.crossfade_ms,
        }
    }
}

#[derive(Debug, Deserialize)]
struct JobListQuery {
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/projects/generate-async`: поставити довгий текст у чергу.
async fn generate_async(
    State(state): State<AppState>,
    Json(request): Json<AsyncJobRequest>,
) -> Response {
    match state.jobs.submit(request.into_job_request()).await {
        Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Err(error) => {
            let status = if error.contains("queue is closed") {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, Json(json!({ "error": error }))).into_response()
        }
    }
}

/// `GET /v1/jobs`: останні завдання.
async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobListQuery>,
) -> Json<serde_json::Value> {
    let jobs = state.jobs.list(query.limit.unwrap_or(50).clamp(1, 500));
    let summaries: Vec<jobs::JobSummary> = jobs.iter().map(jobs::JobSummary::from).collect();
    Json(json!({ "jobs": summaries }))
}

/// `GET /v1/jobs/{id}`: статус завдання.
async fn get_job(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.jobs.get(&id) {
        Some(job) => Json(job).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown job" })),
        )
            .into_response(),
    }
}

/// `GET /v1/jobs/{id}/audio`: готовий WAV.
async fn job_audio(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(job) = state.jobs.get(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown job" })),
        )
            .into_response();
    };
    if job.status != JobStatus::Completed {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "job is not completed",
                "status": job.status,
                "progress": job.progress,
            })),
        )
            .into_response();
    }
    match state.jobs.audio(&id) {
        Ok(Some(wav)) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
            headers.insert(
                header::CONTENT_DISPOSITION,
                HeaderValue::from_str(&format!("attachment; filename=\"job-{id}.wav\""))
                    .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
            );
            (headers, wav).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job audio is missing" })),
        )
            .into_response(),
        Err(error) => {
            tracing::error!(job_id = id, error = %error, "cannot read job audio");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "cannot read job audio" })),
            )
                .into_response()
        }
    }
}

/// `DELETE /v1/jobs/{id}`: скасувати завдання.
async fn cancel_job(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.jobs.cancel(&id) {
        CancelOutcome::Cancelled => StatusCode::NO_CONTENT.into_response(),
        CancelOutcome::NotFound => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown job" })),
        )
            .into_response(),
        CancelOutcome::AlreadyFinished => (
            StatusCode::CONFLICT,
            Json(json!({ "error": "job is already finished" })),
        )
            .into_response(),
    }
}

/// `GET /v1/jobs/{id}/events`: SSE з прогресом.
async fn job_events(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(job) = state.jobs.get(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown job" })),
        )
            .into_response();
    };
    let mut receiver = state.jobs.subscribe();
    let stream = async_stream::stream! {
        let initial = serde_json::to_string(&jobs::JobEvent::from_record(&job))
            .unwrap_or_else(|_| "{}".to_string());
        yield Ok::<Event, Infallible>(Event::default().event("job").data(initial));
        if job.status.is_terminal() {
            return;
        }
        loop {
            match receiver.recv().await {
                Ok(event) if event.job_id == id => {
                    let terminal = event.status.is_terminal();
                    let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("job").data(data));
                    if terminal {
                        break;
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(job_id = id, skipped, "SSE receiver lagged");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Debug, Deserialize)]
struct StreamRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    audio_base64: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default)]
    auto_verbalize: Option<serde_json::Value>,
    #[serde(default)]
    raw_stress: bool,
    #[serde(default)]
    max_chars: Option<usize>,
}

enum StreamEvent {
    Ready(StreamMeta),
    Chunk(Vec<u8>),
    Done(StreamSummary),
    Error(SynthError),
}

/// `POST /v1/tts/stream`: потоковий WAV (chunked transfer).
async fn synthesize_stream(
    State(state): State<AppState>,
    Json(request): Json<StreamRequest>,
) -> Response {
    let mode = VerbalizerMode::parse(request.auto_verbalize.as_ref());
    let max_chars = request.max_chars.unwrap_or(1_200).clamp(50, 5_000);
    let service = Arc::clone(&state.service);
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(4);

    tokio::task::spawn_blocking(move || {
        let result = service.synthesize_stream_pcm(
            &request.text,
            request.voice.as_deref(),
            request.audio_base64.as_deref(),
            request.speed,
            mode,
            request.raw_stress,
            max_chars,
            |meta| tx.blocking_send(StreamEvent::Ready(meta.clone())).is_ok(),
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

    let first = match tokio::time::timeout(Duration::from_secs(120), rx.recv()).await {
        Ok(Some(event)) => event,
        Ok(None) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "stream task ended unexpectedly" })),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(json!({ "error": "stream preparation timed out" })),
            )
                .into_response();
        }
    };

    let meta = match first {
        StreamEvent::Ready(meta) => meta,
        StreamEvent::Error(error) => return synth_error_response(error, "stream preparation"),
        _ => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "unexpected stream event" })),
            )
                .into_response();
        }
    };

    let header = default_streaming_wav_header();
    let stream = async_stream::stream! {
        yield Ok::<Bytes, std::io::Error>(Bytes::from(header));
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Chunk(bytes) => {
                    yield Ok(Bytes::from(bytes));
                }
                StreamEvent::Done(summary) => {
                    tracing::debug!(
                        chunks = summary.chunks,
                        audio_ms = summary.audio_ms,
                        verbalizer = summary.verbalizer,
                        vb_ms = summary.vb_ms,
                        "stream completed"
                    );
                    break;
                }
                StreamEvent::Error(error) => {
                    tracing::warn!(error = %error, "stream failed after headers");
                    break;
                }
                StreamEvent::Ready(_) => {}
            }
        }
    };

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    headers.insert(
        HeaderName::from_static("x-promin-verbalizer"),
        HeaderValue::from_static(meta.verbalizer),
    );
    headers.insert(
        HeaderName::from_static("x-promin-vb-ms"),
        HeaderValue::from_str(&meta.vb_ms.to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    headers.insert(
        HeaderName::from_static("x-promin-chunks"),
        HeaderValue::from_str(&meta.chunks.to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    (headers, Body::from_stream(stream)).into_response()
}

#[derive(Debug, Deserialize)]
struct OpenAiSpeechRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
}

fn is_openai_voice(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "alloy"
            | "echo"
            | "fable"
            | "onyx"
            | "nova"
            | "shimmer"
            | "ash"
            | "ballad"
            | "coral"
            | "sage"
            | "verse"
    )
}

/// `POST /v1/audio/speech`: OpenAI-сумісний синтез (WAV/PCM).
async fn openai_speech(
    State(state): State<AppState>,
    Json(request): Json<OpenAiSpeechRequest>,
) -> Response {
    let default_voice = state.service.default_voice().to_string();
    let voice = match request.voice.as_deref() {
        Some(name) if is_openai_voice(name) => default_voice,
        Some(name) => name.to_string(),
        None => default_voice,
    };
    let format = request
        .response_format
        .as_deref()
        .unwrap_or("wav")
        .to_ascii_lowercase();
    if !matches!(format.as_str(), "wav" | "pcm") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("unsupported response_format: {format}; use wav or pcm")
            })),
        )
            .into_response();
    }

    let service = Arc::clone(&state.service);
    let result = tokio::task::spawn_blocking(move || {
        service.synthesize_pcm(
            &request.input,
            Some(&voice),
            None,
            request.speed,
            VerbalizerMode::Neural,
            false,
        )
    })
    .await;

    match result {
        Ok(Ok(pcm)) => {
            let (body, content_type) = if format == "wav" {
                match wav_bytes(&pcm.pcm, SAMPLE_RATE) {
                    Ok(wav) => (wav, "audio/wav"),
                    Err(error) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": error.to_string() })),
                        )
                            .into_response();
                    }
                }
            } else {
                (pcm16_bytes(&pcm.pcm), "audio/pcm")
            };
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
            headers.insert(
                HeaderName::from_static("x-sample-rate"),
                HeaderValue::from_str(&SAMPLE_RATE.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("24000")),
            );
            headers.insert(
                HeaderName::from_static("x-promin-verbalizer"),
                HeaderValue::from_static(pcm.verbalizer),
            );
            headers.insert(
                HeaderName::from_static("x-promin-vb-ms"),
                HeaderValue::from_str(&pcm.vb_ms.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );
            (headers, body).into_response()
        }
        Ok(Err(error)) => synth_error_response(error, "OpenAI speech"),
        Err(join_error) => {
            tracing::error!(error = %join_error, "OpenAI speech task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "internal task error" })),
            )
                .into_response()
        }
    }
}

/// `GET /v1/pronunciation`: глобальний словник вимови.
async fn get_pronunciation(State(state): State<AppState>) -> Json<PronunciationFile> {
    Json(PronunciationFile {
        rules: state.pronunciation.rules(),
    })
}

/// `PUT /v1/pronunciation`: замінити глобальний словник.
async fn put_pronunciation(
    State(state): State<AppState>,
    Json(payload): Json<PronunciationPayload>,
) -> Response {
    match state.pronunciation.replace(payload.into_rules()) {
        Ok(rules) => Json(PronunciationFile { rules }).into_response(),
        Err(PronunciationStoreError::Invalid(message)) => {
            (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
        }
        Err(PronunciationStoreError::Io(error)) => {
            tracing::error!(error = %error, "save pronunciation dictionary");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": error.to_string() })),
            )
                .into_response()
        }
    }
}

/// `GET /v1/models`: мінімальний OpenAI-сумісний список моделей.
async fn openai_models() -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": "promin-tts",
            "object": "model",
            "created": 0,
            "owned_by": "promin",
        }],
    }))
}

/// `GET /v1/audio/voices`: пресети та клони у форматі, зручному для клієнтів.
async fn openai_voices(State(state): State<AppState>) -> Json<serde_json::Value> {
    let voices: Vec<serde_json::Value> = state
        .service
        .voice_names()
        .into_iter()
        .map(|name| json!({ "id": name, "name": name, "custom": false }))
        .chain(
            state
                .service
                .custom_voice_names()
                .into_iter()
                .map(|name| json!({ "id": name, "name": name, "custom": true })),
        )
        .collect();
    Json(json!({ "object": "list", "data": voices }))
}

/// Витягти токен із `Authorization: Bearer` або `X-API-Key`.
fn request_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
        })
}

/// Чи дозволений запит (порожній список ключів = авторизація вимкнена).
fn is_authorized(headers: &HeaderMap, keys: &[String]) -> bool {
    keys.is_empty()
        || request_token(headers)
            .map(|token| keys.iter().any(|key| key == token))
            .unwrap_or(false)
}

async fn require_api_key(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if request.uri().path() == "/healthz" || is_authorized(request.headers(), &state.api_keys) {
        return next.run(request).await;
    }
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "missing or invalid API key" })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!(error = %error, "cannot install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // Бандл не потребує старих обов'язкових файлів, тому не тягнемо їх
    // з Hugging Face даремно. Словник наголосів усе одно має бути локально.
    let use_bundle = match args.model_format.trim().to_ascii_lowercase().as_str() {
        "bundle" | "holos-onnx" | "onnx" => true,
        "legacy" | "holos-cpu" | "int8" => false,
        _ => ModelPaths::bundle_present(&args.models_dir),
    };

    let download_options = DownloadOptions::from_env();
    if download_options.enabled {
        let layout = if use_bundle {
            ModelLayout::Bundle
        } else {
            ModelLayout::Legacy
        };
        let missing = download_options.missing_required(&args.models_dir, layout);
        if !missing.is_empty() {
            tracing::info!(
                missing = missing.len(),
                repo = %download_options.repo,
                layout = ?layout,
                "required models are missing, downloading from Hugging Face"
            );
        }
        let report = ensure_models(&args.models_dir, &download_options, layout).await?;
        if !report.downloaded.is_empty() {
            tracing::info!(
                downloaded = report.downloaded.len(),
                "models downloaded: {}",
                report.downloaded.join(", ")
            );
        }
        if !report.failed_optional.is_empty() {
            tracing::warn!(
                failed = report.failed_optional.len(),
                "optional models unavailable: {}",
                report.failed_optional.join(", ")
            );
        }
    }

    let paths = if args.model_format.eq_ignore_ascii_case("auto") {
        ModelPaths::discover(&args.models_dir)?
    } else {
        ModelPaths::discover_with_format(&args.models_dir, &args.model_format)?
    };
    let speakers_dir = args
        .speakers_dir
        .unwrap_or_else(|| args.models_dir.join("speakers"));
    tracing::info!(
        models_dir = %args.models_dir.display(),
        speakers_dir = %speakers_dir.display(),
        "loading models"
    );
    let data_dir = args
        .models_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| StdPath::new("."));
    let pronunciation_path = args
        .pronunciation_file
        .clone()
        .unwrap_or_else(|| data_dir.join("pronunciation.json"));
    let pronunciation = Arc::new(PronunciationStore::load(&pronunciation_path)?);
    tracing::info!(
        path = %pronunciation.path().display(),
        rules = pronunciation.rule_count(),
        "pronunciation dictionary ready"
    );
    let mut service = SynthService::load(&paths, args.threads, speakers_dir)?;
    service.set_pronunciation_store(Arc::clone(&pronunciation));
    let service = Arc::new(service);

    let preferred_jobs = args
        .jobs_dir
        .unwrap_or_else(|| args.models_dir.join("jobs"));
    let jobs_dir = if writable_dir(&preferred_jobs) {
        preferred_jobs
    } else {
        let fallback = std::env::temp_dir().join("promin-tts-jobs");
        tracing::warn!(
            preferred = %preferred_jobs.display(),
            fallback = %fallback.display(),
            "jobs directory is not writable, using temporary directory"
        );
        fallback
    };
    let store = Arc::new(speech_core::jobs::JobStore::new(&jobs_dir)?);

    let default_loudness = LoudnessSettings {
        enabled: true,
        target_lufs: args.loudness_target,
        true_peak_db: args.true_peak,
    };
    default_loudness
        .validate()
        .context("invalid loudness settings")?;
    let preferred_projects = args
        .projects_dir
        .unwrap_or_else(|| args.models_dir.join("projects"));
    let projects_dir = if writable_dir(&preferred_projects) {
        preferred_projects
    } else {
        let fallback = std::env::temp_dir().join("promin-tts-projects");
        tracing::warn!(
            preferred = %preferred_projects.display(),
            fallback = %fallback.display(),
            "projects directory is not writable, using temporary directory"
        );
        fallback
    };
    let projects = Arc::new(ProjectStore::new(&projects_dir)?);
    let reconciled = projects::reconcile_projects(&projects)?;
    if reconciled > 0 {
        tracing::warn!(projects = reconciled, "reconciled projects after restart");
    }
    tracing::info!(
        projects_dir = %projects_dir.display(),
        reconciled,
        "book projects ready"
    );
    let project_lock = Arc::new(tokio::sync::Mutex::new(()));

    let public_url = args
        .public_url
        .unwrap_or_else(|| format!("http://127.0.0.1:{}", args.http_addr.port()));
    let jobs = JobManager::new(
        Arc::clone(&store),
        Arc::clone(&service),
        Arc::clone(&projects),
        Arc::clone(&pronunciation),
        Arc::clone(&project_lock),
        args.jobs_concurrency,
        public_url.clone(),
    )?;
    tracing::info!(
        jobs_dir = %jobs_dir.display(),
        public_url = %public_url,
        concurrency = args.jobs_concurrency.max(1),
        "async jobs ready"
    );
    let exports = exports::ExportManager::new(
        Arc::clone(&projects),
        service.model_identity().clone(),
        args.export_concurrency,
    )?;
    tracing::info!(
        concurrency = args.export_concurrency.max(1),
        "audiobook exports ready"
    );

    let api_keys: Arc<Vec<String>> = Arc::new(
        args.api_keys
            .split(',')
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .collect(),
    );
    if !api_keys.is_empty() {
        tracing::info!(keys = api_keys.len(), "API key authentication enabled");
    }

    let state = AppState {
        service,
        jobs,
        projects,
        exports,
        project_lock,
        pronunciation,
        default_loudness,
        public_url,
        api_keys,
    };

    // Wyoming для Home Assistant: окремий TCP-порт, не блокує HTTP.
    let wyoming_setting = args.wyoming_addr.trim().to_ascii_lowercase();
    if !matches!(
        wyoming_setting.as_str(),
        "" | "off" | "0" | "false" | "no" | "never"
    ) {
        match args.wyoming_addr.trim().parse::<SocketAddr>() {
            Ok(addr) => {
                let service = Arc::clone(&state.service);
                tokio::spawn(async move {
                    if let Err(error) = wyoming::serve(addr, service).await {
                        tracing::error!(error = %error, "Wyoming server stopped");
                    }
                });
            }
            Err(error) => tracing::warn!(
                addr = args.wyoming_addr,
                error = %error,
                "invalid PROMIN_WYOMING_ADDR, Wyoming disabled"
            ),
        }
    }

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/api/model", get(model))
        .route("/v1/model", get(model))
        .route("/api/voices", get(voices).post(enroll_voice))
        .route("/api/voices/{name}", delete(delete_voice))
        .route("/api/verbalize", post(verbalize))
        .route("/api/synthesize", post(synthesize))
        .route("/api/synthesize/stream", post(synthesize_stream))
        .route("/v1/tts/synthesize", post(synthesize))
        .route("/v1/tts/stream", post(synthesize_stream))
        .route("/v1/projects/generate-async", post(generate_async))
        .route(
            "/v1/projects",
            get(projects::list_projects).post(projects::create_project),
        )
        .route("/v1/projects/import", post(projects::import_project))
        .route(
            "/v1/projects/{id}",
            get(projects::get_project)
                .put(projects::update_project)
                .delete(projects::delete_project),
        )
        .route(
            "/v1/projects/{id}/pronunciation",
            put(projects::put_project_pronunciation),
        )
        .route(
            "/v1/projects/{id}/generate",
            post(projects::generate_project),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}",
            get(projects::get_chapter).put(projects::update_chapter),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/audio",
            get(projects::chapter_audio),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/subtitles",
            get(projects::chapter_subtitles),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes",
            get(projects::chapter_takes).post(projects::invalidate_chapter_takes),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes/units/{unit}",
            post(projects::regenerate_chapter_unit),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes/units/{unit}/takes/{take}/activate",
            post(projects::activate_chapter_unit_take),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes/units/{unit}/takes/{take}",
            delete(projects::delete_chapter_unit_take),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes/units/{unit}/takes/{take}/audio",
            get(projects::chapter_unit_take_audio),
        )
        .route(
            "/v1/projects/{id}/chapters/{index}/takes/rebuild",
            post(projects::rebuild_chapter_from_takes),
        )
        .route(
            "/v1/projects/{id}/manifest",
            get(projects::project_manifest),
        )
        .route(
            "/v1/projects/{id}/speechproj",
            get(projects::project_speechproj),
        )
        .route(
            "/v1/projects/{id}/exports",
            get(exports::list_exports).post(exports::create_export),
        )
        .route(
            "/v1/projects/{id}/exports/{export_id}",
            get(exports::get_export).delete(exports::delete_export),
        )
        .route(
            "/v1/projects/{id}/exports/{export_id}/audio",
            get(exports::export_audio),
        )
        .route(
            "/v1/projects/{id}/exports/{export_id}/subtitles",
            get(exports::export_subtitles),
        )
        .route("/v1/jobs", get(list_jobs))
        .route("/v1/jobs/{id}", get(get_job).delete(cancel_job))
        .route("/v1/jobs/{id}/audio", get(job_audio))
        .route("/v1/jobs/{id}/events", get(job_events))
        .route("/v1/audio/speech", post(openai_speech))
        .route("/v1/audio/voices", get(openai_voices))
        .route("/v1/models", get(openai_models))
        .route(
            "/v1/pronunciation",
            get(get_pronunciation).put(put_pronunciation),
        )
        .route("/mcp", post(mcp::handle))
        // `audio_base64` може містити WAV до 120 с (base64 ~30–60 МБ);
        // для multipart-імпорту 64 МБ файлу потрібен запас на межі частин.
        .layer(DefaultBodyLimit::max(80 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.http_addr).await?;
    tracing::info!("listening on http://{}", args.http_addr);

    // Після SIGTERM даємо активним HTTP-з'єднанням час закритися, але не
    // чекаємо на spawn_blocking синтез: інакше Docker stop висне до SIGKILL.
    let (shutdown_started_tx, mut shutdown_started_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        while !*shutdown_started_rx.borrow() {
            if shutdown_started_rx.changed().await.is_err() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
        tracing::warn!("graceful shutdown timed out; forcing exit");
        std::process::exit(0);
    });
    let shutdown = async move {
        shutdown_signal().await;
        let _ = shutdown_started_tx.send(true);
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    tracing::info!("HTTP server stopped");
    // Незавершені job-и лишаться на диску; після рестарту вони позначаються
    // failed через JobStore::recover_interrupted.
    tokio::time::sleep(Duration::from_millis(200)).await;
    std::process::exit(0);
}

/// Чи можна створити каталог і записати в нього файл.
fn writable_dir(dir: &std::path::Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".promin-write-test");
    if std::fs::write(&probe, b"ok").is_err() {
        return false;
    }
    let _ = std::fs::remove_file(probe);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> Vec<String> {
        vec!["secret".to_string(), "second".to_string()]
    }

    #[test]
    fn accepts_bearer_token() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert!(is_authorized(&headers, &keys()));
        assert_eq!(request_token(&headers), Some("secret"));
    }

    #[test]
    fn accepts_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_static("second"),
        );
        assert!(is_authorized(&headers, &keys()));
    }

    #[test]
    fn rejects_missing_or_wrong_key() {
        assert!(!is_authorized(&HeaderMap::new(), &keys()));
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer nope"),
        );
        assert!(!is_authorized(&headers, &keys()));
    }

    #[test]
    fn empty_key_list_disables_auth() {
        assert!(is_authorized(&HeaderMap::new(), &[]));
    }

    #[test]
    fn maps_openai_voice_names_case_insensitively() {
        for name in ["alloy", "NOVA", "Shimmer", "verse"] {
            assert!(is_openai_voice(name), "{name} should be an OpenAI voice");
        }
        assert!(!is_openai_voice("Speaker_0"));
        assert!(!is_openai_voice("Тест Клон"));
    }
}
