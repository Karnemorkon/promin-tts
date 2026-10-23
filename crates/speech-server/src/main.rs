use std::net::SocketAddr;

use axum::{routing::get, Json, Router};
use clap::Parser;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(name = "speech-server", version, about = "Promin TTS server")]
struct Args {
    /// HTTP listen address
    #[arg(long, env = "PROMIN_HTTP_ADDR", default_value = "0.0.0.0:8010")]
    http_addr: SocketAddr,
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "engine": "promin-tts",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let app = Router::new().route("/healthz", get(healthz));

    let listener = tokio::net::TcpListener::bind(args.http_addr).await?;
    tracing::info!("listening on http://{}", args.http_addr);
    axum::serve(listener, app).await?;
    Ok(())
}
