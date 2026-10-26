//! Автозавантаження моделей із Hugging Face (`Morkon/holos-native-models`).
//!
//! Завантаження потокове, з `.part`-файлом і докачкою через HTTP Range.
//! Обов'язкові файли (TTS + словник) без яких сервер не стартує; решта —
//! опційні (accentor, voice encoder, Gemma-вербалізатор).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

/// Обов'язкові файли для базового синтезу.
pub const REQUIRED_FILES: &[&str] = &[
    "tts/holos_cpu_int8.onnx",
    "tts/vocab.json",
    "tts/voices.json",
    "stress/stress.trie",
];

/// Опційні моделі (клонування, neural-наголос, Gemma-вербалізатор).
pub const OPTIONAL_FILES: &[&str] = &[
    "accentor/accentor-lite.onnx",
    "tts/voice_encoder.onnx",
    "verbalizer/gemma-int8/model.onnx",
    "verbalizer/gemma-int8/tokenizer.model",
    "verbalizer/gemma-int8/config.json",
    "verbalizer/gemma-int8/GEMMA-NOTICE.txt",
];

/// Налаштування автозавантаження.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// `false` — не завантажувати; `true` — завантажити відсутнє.
    pub enabled: bool,
    /// Завантажувати опційні моделі.
    pub optional: bool,
    pub repo: String,
    pub revision: String,
    pub token: Option<String>,
    pub base_url: String,
    pub timeout: Duration,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            optional: true,
            repo: "Morkon/holos-native-models".to_string(),
            revision: "main".to_string(),
            token: None,
            base_url: "https://huggingface.co".to_string(),
            timeout: Duration::from_secs(3600),
        }
    }
}

impl DownloadOptions {
    /// Прочитати `PROMIN_AUTO_DOWNLOAD`, `PROMIN_DOWNLOAD_OPTIONAL`,
    /// `PROMIN_HF_REPO`, `PROMIN_HF_REVISION`, `PROMIN_HF_TOKEN`,
    /// `PROMIN_HF_BASE_URL`.
    pub fn from_env() -> Self {
        let mode = std::env::var("PROMIN_AUTO_DOWNLOAD")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase();
        let enabled = !matches!(mode.as_str(), "0" | "false" | "off" | "no" | "never");
        let optional = std::env::var("PROMIN_DOWNLOAD_OPTIONAL")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true);
        Self {
            enabled,
            optional,
            repo: std::env::var("PROMIN_HF_REPO")
                .unwrap_or_else(|_| "Morkon/holos-native-models".to_string()),
            revision: std::env::var("PROMIN_HF_REVISION").unwrap_or_else(|_| "main".to_string()),
            token: std::env::var("PROMIN_HF_TOKEN")
                .ok()
                .filter(|token| !token.is_empty()),
            base_url: std::env::var("PROMIN_HF_BASE_URL")
                .unwrap_or_else(|_| "https://huggingface.co".to_string())
                .trim_end_matches('/')
                .to_string(),
            timeout: std::env::var("PROMIN_DOWNLOAD_TIMEOUT")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(3600)),
        }
    }

    /// Чи потрібно завантажувати хоч один обов'язковий файл.
    pub fn missing_required(&self, models_dir: &Path) -> Vec<String> {
        REQUIRED_FILES
            .iter()
            .filter(|file| !file_present(models_dir, file))
            .map(|file| (*file).to_string())
            .collect()
    }
}

/// Плоский відповідник вкладеного шляху (підтримка `models/*` layout).
fn flat_equivalent(file: &str) -> Option<&'static str> {
    Some(match file {
        "tts/holos_cpu_int8.onnx" => "holos_cpu_int8.onnx",
        "tts/vocab.json" => "vocab.json",
        "tts/voices.json" => "voices.json",
        "tts/voice_encoder.onnx" => "voice_encoder.onnx",
        "stress/stress.trie" => "stress.trie",
        "accentor/accentor-lite.onnx" => "accentor-lite.onnx",
        "verbalizer/gemma-int8/model.onnx" => "model.onnx",
        "verbalizer/gemma-int8/tokenizer.model" => "tokenizer.model",
        "verbalizer/gemma-int8/config.json" => "config.json",
        "verbalizer/gemma-int8/GEMMA-NOTICE.txt" => "GEMMA-NOTICE.txt",
        _ => return None,
    })
}

/// Файл є або у вкладеній, або у плоскій розкладці.
fn file_present(models_dir: &Path, file: &str) -> bool {
    let non_empty = |path: &Path| {
        path.is_file()
            && path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
    };
    if non_empty(&models_dir.join(file)) {
        return true;
    }
    flat_equivalent(file)
        .map(|flat| non_empty(&models_dir.join(flat)))
        .unwrap_or(false)
}

/// Підсумок автозавантаження.
#[derive(Debug, Default)]
pub struct DownloadReport {
    pub downloaded: Vec<String>,
    pub skipped: Vec<String>,
    pub failed_optional: Vec<String>,
}

/// Перевірити/завантажити моделі. Повертає помилку лише для обов'язкових файлів.
pub async fn ensure_models(models_dir: &Path, options: &DownloadOptions) -> Result<DownloadReport> {
    let mut report = DownloadReport::default();
    if !options.enabled {
        return Ok(report);
    }
    if !writable_dir(models_dir) {
        let missing = options.missing_required(models_dir);
        if missing.is_empty() {
            return Ok(report);
        }
        bail!(
            "models are missing ({}) and {} is not writable; mount a writable models volume or set PROMIN_AUTO_DOWNLOAD=0",
            missing.join(", "),
            models_dir.display()
        );
    }

    let client = reqwest::Client::builder()
        .timeout(options.timeout)
        .user_agent("promin-tts-downloader/0.5")
        .build()
        .context("build downloader HTTP client")?;

    for file in REQUIRED_FILES {
        if file_present(models_dir, file) {
            report.skipped.push((*file).to_string());
            continue;
        }
        download_file(&client, options, models_dir, file)
            .await
            .with_context(|| format!("download required model {file}"))?;
        report.downloaded.push((*file).to_string());
    }

    if options.optional {
        for file in OPTIONAL_FILES {
            if file_present(models_dir, file) {
                report.skipped.push((*file).to_string());
                continue;
            }
            match download_file(&client, options, models_dir, file).await {
                Ok(()) => report.downloaded.push((*file).to_string()),
                Err(error) => {
                    tracing::warn!(file, error = %error, "optional model download failed");
                    report.failed_optional.push((*file).to_string());
                }
            }
        }
    }

    Ok(report)
}

fn writable_dir(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".promin-download-test");
    if std::fs::write(&probe, b"ok").is_err() {
        return false;
    }
    let _ = std::fs::remove_file(probe);
    true
}

fn resolve_url(options: &DownloadOptions, file: &str) -> String {
    format!(
        "{}/{}/resolve/{}/{}?download=true",
        options.base_url, options.repo, options.revision, file
    )
}

/// Завантажити один файл з докачкою.
pub async fn download_file(
    client: &reqwest::Client,
    options: &DownloadOptions,
    models_dir: &Path,
    file: &str,
) -> Result<()> {
    let final_path = models_dir.join(file);
    if let Some(parent) = final_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let part_path = part_path(models_dir, file);

    let mut offset = tokio::fs::metadata(&part_path)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let url = resolve_url(options, file);
    let mut request = client.get(&url);
    if let Some(token) = &options.token {
        request = request.bearer_auth(token);
    }
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }

    tracing::info!(file, offset, "downloading model");
    let response = request.send().await.with_context(|| format!("GET {url}"))?;
    let status = response.status();

    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && offset > 0 {
        // `.part` вже містить повний файл (або сервер не підтримує Range).
        tokio::fs::rename(&part_path, &final_path)
            .await
            .with_context(|| format!("finalize {}", final_path.display()))?;
        return Ok(());
    }
    if !status.is_success() {
        bail!("HTTP {status} for {url}");
    }

    let append = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !append {
        offset = 0;
    }
    let total = response.content_length().map(|len| len + offset);
    let mut output = if append {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&part_path)
            .await
            .with_context(|| format!("open {}", part_path.display()))?
    } else {
        tokio::fs::File::create(&part_path)
            .await
            .with_context(|| format!("create {}", part_path.display()))?
    };

    let mut downloaded = offset;
    let mut last_log = Instant::now();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read body of {url}"))?;
        output
            .write_all(&chunk)
            .await
            .with_context(|| format!("write {}", part_path.display()))?;
        downloaded += chunk.len() as u64;
        if last_log.elapsed() >= Duration::from_secs(5) {
            match total {
                Some(total) if total > 0 => {
                    tracing::info!(
                        file,
                        downloaded_mb = downloaded / 1_048_576,
                        total_mb = total / 1_048_576,
                        percent = downloaded * 100 / total,
                        "download progress"
                    );
                }
                _ => {
                    tracing::info!(
                        file,
                        downloaded_mb = downloaded / 1_048_576,
                        "download progress"
                    );
                }
            }
            last_log = Instant::now();
        }
    }
    output.flush().await?;
    drop(output);

    if let Some(total) = total {
        if downloaded != total {
            bail!(
                "incomplete download for {file}: got {downloaded} of {total} bytes (will resume)"
            );
        }
    }
    if downloaded == 0 {
        bail!("downloaded empty file: {file}");
    }
    tokio::fs::rename(&part_path, &final_path)
        .await
        .with_context(|| format!("finalize {}", final_path.display()))?;
    tracing::info!(file, size_mb = downloaded / 1_048_576, "model downloaded");
    Ok(())
}

/// Шлях до `.part`-файлу (для тестів/діагностики).
pub fn part_path(models_dir: &Path, file: &str) -> PathBuf {
    let final_path = models_dir.join(file);
    final_path.with_extension(format!(
        "{}.part",
        final_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("bin")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve_once(body: Vec<u8>, status: String, extra_headers: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra_headers}\r\n",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn options(base_url: String) -> DownloadOptions {
        DownloadOptions {
            enabled: true,
            optional: false,
            repo: "test/repo".into(),
            revision: "main".into(),
            token: None,
            base_url,
            timeout: Duration::from_secs(10),
        }
    }

    #[tokio::test]
    async fn downloads_file() {
        let body = b"model-bytes".to_vec();
        let base = serve_once(body.clone(), "200 OK".to_string(), String::new()).await;
        let dir = std::env::temp_dir().join(format!("promin-dl-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let client = reqwest::Client::new();
        download_file(&client, &options(base), &dir, "tts/vocab.json")
            .await
            .unwrap();
        let downloaded = tokio::fs::read(dir.join("tts/vocab.json")).await.unwrap();
        assert_eq!(downloaded, body);
        tokio::fs::remove_dir_all(dir).await.unwrap();
    }

    #[tokio::test]
    async fn resumes_partial_file() {
        let dir = std::env::temp_dir().join(format!("promin-dl-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(dir.join("tts")).await.unwrap();
        tokio::fs::write(part_path(&dir, "tts/vocab.json"), b"model-")
            .await
            .unwrap();
        let base = serve_once(
            b"bytes".to_vec(),
            "206 Partial Content".to_string(),
            "Content-Range: bytes 6-10/11\r\n".to_string(),
        )
        .await;
        let client = reqwest::Client::new();
        download_file(&client, &options(base), &dir, "tts/vocab.json")
            .await
            .unwrap();
        let downloaded = tokio::fs::read(dir.join("tts/vocab.json")).await.unwrap();
        assert_eq!(downloaded, b"model-bytes");
        tokio::fs::remove_dir_all(dir).await.unwrap();
    }

    #[test]
    fn manifest_url_and_missing() {
        let dir = std::env::temp_dir().join(format!("promin-dl-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let options = DownloadOptions::default();
        assert_eq!(options.missing_required(&dir).len(), REQUIRED_FILES.len());
        assert!(resolve_url(&options, "tts/vocab.json")
            .contains("Morkon/holos-native-models/resolve/main/tts/vocab.json"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn detects_flat_layout() {
        let dir = std::env::temp_dir().join(format!("promin-dl-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in [
            "holos_cpu_int8.onnx",
            "vocab.json",
            "voices.json",
            "stress.trie",
        ] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
        let options = DownloadOptions::default();
        assert!(options.missing_required(&dir).is_empty());
        assert!(!file_present(&dir, "tts/voice_encoder.onnx"));
        std::fs::write(dir.join("voice_encoder.onnx"), b"x").unwrap();
        assert!(file_present(&dir, "tts/voice_encoder.onnx"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
