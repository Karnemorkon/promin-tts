//! Обслуговування вбудованого веб-інтерфейсу (SPA).
//!
//! Статичні файли (HTML/JS/CSS/зображення) вбудовуються в бінарний файл
//! за допомогою `rust-embed` під час збірки. Axum віддає їх за маршрутами
//! `/` і `/{*path}`. Для React Router усі невідомі шляхи повертають
//! `index.html` (SPA fallback).

use axum::{
    body::Body,
    extract::Path,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;
use std::path::{Path as StdPath, PathBuf};

/// Вбудована директорія `web/dist` (заповнюється під час `cargo build`).
#[derive(RustEmbed)]
#[folder = "web/dist/"]
struct WebAssets;

/// MIME-фолбек, якщо вгадати тип не вдалося.
const FALLBACK_MIME: &str = "application/octet-stream";

/// Просте визначення MIME-типу за розширенням файлу.
fn mime_type(path: &StdPath) -> &'static str {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_ascii_lowercase(),
        None => return FALLBACK_MIME,
    };
    match ext.as_str() {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" | "jsx" => "application/javascript",
        "ts" | "tsx" => "application/typescript",
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "webp" => "image/webp",
        "wasm" => "application/wasm",
        "pdf" => "application/pdf",
        "txt" => "text/plain; charset=utf-8",
        _ => FALLBACK_MIME,
    }
}

/// `GET /` та `GET /{*path}` — віддати статичний файл або index.html (SPA fallback).
///
/// Для `/` екстрактор `Path` відсутній, тому параметр опційний.
pub async fn serve(path: Option<Path<PathBuf>>) -> Response {
    let path = path.map(|Path(path)| path).unwrap_or_default();
    let path_str = path.to_string_lossy();
    let rel = path_str.trim_start_matches('/');

    // Порожній шлях або index.html — головна сторінка. Невідомий шлях —
    // SPA fallback на index.html (React Router).
    let (entry, served_name) = if rel.is_empty() || rel == "index.html" {
        (WebAssets::get("index.html"), "index.html")
    } else {
        match WebAssets::get(rel) {
            Some(file) => (Some(file), rel),
            None => (WebAssets::get("index.html"), "index.html"),
        }
    };

    let Some(file) = entry else {
        return (
            StatusCode::NOT_FOUND,
            [(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            "web UI not built: run `npm run build` in crates/speech-server/web",
        )
            .into_response();
    };

    let mime = mime_type(StdPath::new(served_name));
    // Хешовані ассети Vite можна кешувати надовго; index.html — ні.
    let cache = if served_name == "index.html" {
        "no-cache"
    } else {
        "public, max-age=31536000, immutable"
    };

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, header::HeaderValue::from_static(mime)),
            (
                header::CACHE_CONTROL,
                header::HeaderValue::from_static(cache),
            ),
        ],
        Body::from(file.data.into_owned()),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::Router;

    async fn spawn_app() -> String {
        let app = Router::new()
            .route("/", get(serve))
            .route("/{*path}", get(serve));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn serves_index_html() {
        let base = spawn_app().await;
        let response = reqwest::get(format!("{base}/")).await.expect("GET /");
        assert_eq!(response.status(), 200);
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(content_type.starts_with("text/html"));
        let body = response.text().await.expect("body");
        assert!(body.contains("Promin TTS"));
    }

    #[tokio::test]
    async fn spa_fallback_for_unknown_routes() {
        let base = spawn_app().await;
        let response = reqwest::get(format!("{base}/projects/123"))
            .await
            .expect("GET /projects/123");
        assert_eq!(response.status(), 200);
        let body = response.text().await.expect("body");
        assert!(body.contains("Promin TTS"));
    }

    #[test]
    fn mime_types() {
        assert_eq!(
            mime_type(StdPath::new("index.html")),
            "text/html; charset=utf-8"
        );
        assert_eq!(mime_type(StdPath::new("app.js")), "application/javascript");
        assert_eq!(
            mime_type(StdPath::new("app.css")),
            "text/css; charset=utf-8"
        );
        assert_eq!(
            mime_type(StdPath::new("unknown.bin")),
            "application/octet-stream"
        );
    }
}
