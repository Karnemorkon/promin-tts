//! Ліниве завантаження української ASR-моделі для Back-ASR.
//!
//! Модель не входить до Docker-образу: перший QC-запит або завантажує
//! `speech-uk/w2v-bert-v3-onnx` у `<models>/asr/w2v-bert-v3`, або повертає
//! зрозумілу помилку, якщо автозавантаження вимкнено. ONNX-сесія створюється
//! у `spawn_blocking`, тому довге завантаження/ініціалізація не блокують
//! HTTP-рантайм.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use speech_asr::model::{default_model_dir, missing_model_files, AsrModel};
use speech_asr::AsrTranscript;
use tokio::sync::Mutex as AsyncMutex;

use crate::downloader::{download_file_to, DownloadOptions};

/// Налаштування ASR-сервісу.
#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub threads: i16,
    pub download: DownloadOptions,
}

impl AsrOptions {
    /// Зібрати параметри з CLI/env. `PROMIN_ASR_AUTO_DOWNLOAD` має
    /// пріоритет над загальним `PROMIN_AUTO_DOWNLOAD`.
    pub fn new(
        models_dir: &Path,
        model_dir: Option<PathBuf>,
        model_id: impl Into<String>,
        repo: impl Into<String>,
        revision: impl Into<String>,
        threads: i16,
    ) -> Self {
        let model_id = model_id.into();
        let repo = repo.into();
        let revision = revision.into();
        let mut download = DownloadOptions::from_env();
        download.repo = if repo.trim().is_empty() {
            model_id.clone()
        } else {
            repo
        };
        download.revision = revision;
        download.bundle_prefix = String::new();
        if let Ok(mode) = std::env::var("PROMIN_ASR_AUTO_DOWNLOAD") {
            download.enabled = parse_enabled(&mode);
        }
        let model_dir = model_dir.unwrap_or_else(|| default_model_dir(models_dir));
        Self {
            model_dir,
            model_id,
            threads: threads.max(1),
            download,
        }
    }
}

/// Стан ASR для `/healthz`.
#[derive(Debug, Clone, Serialize)]
pub struct AsrStatus {
    pub model: String,
    pub model_dir: String,
    pub loaded: bool,
    pub files_present: bool,
    pub missing: Vec<String>,
    pub auto_download: bool,
    pub threads: i16,
}

/// Лінивий ASR-сервіс.
pub struct AsrService {
    options: AsrOptions,
    model: StdMutex<Option<Arc<AsrModel>>>,
    load_lock: AsyncMutex<()>,
}

impl AsrService {
    pub fn new(options: AsrOptions) -> Self {
        Self {
            options,
            model: StdMutex::new(None),
            load_lock: AsyncMutex::new(()),
        }
    }

    /// Завантажити модель, якщо вона ще не завантажена.
    pub async fn ensure_loaded(&self) -> Result<Arc<AsrModel>> {
        if let Some(model) = self.loaded() {
            return Ok(model);
        }
        let _guard = self.load_lock.lock().await;
        if let Some(model) = self.loaded() {
            return Ok(model);
        }

        self.ensure_files().await?;
        let model_dir = self.options.model_dir.clone();
        let model_id = self.options.model_id.clone();
        let threads = self.options.threads;
        let model =
            tokio::task::spawn_blocking(move || AsrModel::load(&model_dir, &model_id, threads))
                .await
                .context("ASR load task panicked")??;
        let model = Arc::new(model);
        *self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::clone(&model));
        tracing::info!(
            model = %self.options.model_id,
            threads = self.options.threads,
            "Back-ASR model loaded"
        );
        Ok(model)
    }

    /// Розпізнати аудіо (блокуючий ONNX-виклик винесено у `spawn_blocking`).
    pub async fn transcribe(&self, samples: Vec<f32>, sample_rate: u32) -> Result<AsrTranscript> {
        let model = self.ensure_loaded().await?;
        tokio::task::spawn_blocking(move || model.transcribe(&samples, sample_rate))
            .await
            .context("ASR transcribe task panicked")?
    }

    /// Перевірити, чи модель доступна локально або може бути завантажена.
    pub fn availability(&self) -> Result<()> {
        let missing = missing_model_files(&self.options.model_dir);
        if missing.is_empty() {
            return Ok(());
        }
        if self.options.download.enabled {
            return Ok(());
        }
        bail!(
            "ASR model is missing ({}); set PROMIN_ASR_AUTO_DOWNLOAD=auto or mount {}",
            missing.join(", "),
            self.options.model_dir.display()
        )
    }

    pub fn model_id(&self) -> &str {
        &self.options.model_id
    }

    pub fn model_dir(&self) -> &Path {
        &self.options.model_dir
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded().is_some()
    }

    pub fn status(&self) -> AsrStatus {
        let missing = missing_model_files(&self.options.model_dir);
        AsrStatus {
            model: self.options.model_id.clone(),
            model_dir: self.options.model_dir.display().to_string(),
            loaded: self.is_loaded(),
            files_present: missing.is_empty(),
            missing: missing.into_iter().map(str::to_string).collect(),
            auto_download: self.options.download.enabled,
            threads: self.options.threads,
        }
    }

    fn loaded(&self) -> Option<Arc<AsrModel>> {
        self.model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    async fn ensure_files(&self) -> Result<()> {
        let missing = missing_model_files(&self.options.model_dir);
        if missing.is_empty() {
            return Ok(());
        }
        if !self.options.download.enabled {
            bail!(
                "ASR model is missing ({}); set PROMIN_ASR_AUTO_DOWNLOAD=auto or mount {}",
                missing.join(", "),
                self.options.model_dir.display()
            );
        }
        tokio::fs::create_dir_all(&self.options.model_dir)
            .await
            .with_context(|| {
                format!(
                    "create ASR model directory {}",
                    self.options.model_dir.display()
                )
            })?;
        let client = reqwest::Client::builder()
            .timeout(self.options.download.timeout)
            .user_agent("promin-tts-asr/0.12")
            .build()
            .context("build ASR downloader HTTP client")?;
        tracing::info!(
            missing = missing.len(),
            repo = %self.options.download.repo,
            dir = %self.options.model_dir.display(),
            "downloading Back-ASR model from Hugging Face"
        );
        for file in &missing {
            download_file_to(
                &client,
                &self.options.download,
                &self.options.model_dir,
                file,
                file,
            )
            .await
            .with_context(|| format!("download ASR file {file}"))?;
        }
        let still_missing = missing_model_files(&self.options.model_dir);
        if !still_missing.is_empty() {
            bail!(
                "ASR model is incomplete after download: {}",
                still_missing.join(", ")
            );
        }
        Ok(())
    }
}

fn parse_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no" | "never"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_download_modes() {
        assert!(parse_enabled("auto"));
        assert!(parse_enabled("1"));
        assert!(!parse_enabled("off"));
        assert!(!parse_enabled("false"));
    }

    #[test]
    fn options_default_to_external_model() {
        let options = AsrOptions::new(
            Path::new("/models"),
            None,
            speech_asr::DEFAULT_MODEL_ID,
            "",
            "main",
            4,
        );
        assert_eq!(options.model_dir, Path::new("/models/asr/w2v-bert-v3"));
        assert_eq!(options.download.repo, speech_asr::DEFAULT_MODEL_ID);
        assert!(options.download.bundle_prefix.is_empty());
        assert!(speech_asr::model::EXTERNAL_DATA_FILE.ends_with("onnx_data"));
    }
}
