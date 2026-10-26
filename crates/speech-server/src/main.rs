use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;

mod service;

use service::{ModelPaths, SynthError, SynthService, VerbalizerMode};

#[derive(Parser, Debug)]
#[command(name = "speech-server", version, about = "Promin TTS server")]
struct Args {
    /// HTTP listen address
    #[arg(long, env = "PROMIN_HTTP_ADDR", default_value = "0.0.0.0:8010")]
    http_addr: SocketAddr,

    /// Directory with TTS models (`tts/` + `stress/` or flat layout)
    #[arg(long, env = "PROMIN_MODELS_DIR", default_value = "models")]
    models_dir: PathBuf,

    /// ONNX intra-op threads
    #[arg(long, env = "PROMIN_THREADS", default_value_t = 4)]
    threads: i16,

    /// Directory with cloned voices (`<name>.npy`); default `<models>/speakers`
    #[arg(long, env = "PROMIN_SPEAKERS_DIR")]
    speakers_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    service: Arc<SynthService>,
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
        "voices": state.service.voice_count(),
        "custom_voices": state.service.custom_voice_count(),
        "sample_rate": state.service.sample_rate(),
        "verbalizer": state.service.verbalizer_name(),
        "accentor": state.service.accentor_name(),
        "voice_encoder": state.service.voice_encoder_name(),
    }))
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

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
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

    let paths = ModelPaths::discover(&args.models_dir)?;
    let speakers_dir = args
        .speakers_dir
        .unwrap_or_else(|| args.models_dir.join("speakers"));
    tracing::info!(
        models_dir = %args.models_dir.display(),
        speakers_dir = %speakers_dir.display(),
        "loading models"
    );
    let service = Arc::new(SynthService::load(&paths, args.threads, speakers_dir)?);

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/api/voices", get(voices).post(enroll_voice))
        .route("/api/voices/{name}", delete(delete_voice))
        .route("/api/verbalize", post(verbalize))
        .route("/api/synthesize", post(synthesize))
        .route("/v1/tts/synthesize", post(synthesize))
        // `audio_base64` може містити WAV до 120 с (base64 ~30–60 МБ).
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(AppState { service });

    let listener = tokio::net::TcpListener::bind(args.http_addr).await?;
    tracing::info!("listening on http://{}", args.http_addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
