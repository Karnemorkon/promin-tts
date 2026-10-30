//! Back-ASR контроль якості готових глав.
//!
//! Перевірка асинхронна: `POST .../verify` створює job, який зберігається у
//! `<project>/qc/<job>.json`, виконується під `PROMIN_QC_CONCURRENCY` і
//! переживає рестарт (активні job-и позначаються `failed`). Аудіо ріжеться на
//! вікна ~25 с за `ChunkTiming`, кожне вікно розпізнається W2V-BERT, після
//! чого WER/CER і помилки на рівні слів зіставляються зі скриптом.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use speech_asr::diff::{compare, DiffReport};
use speech_audio::diagnostics::{analyze as analyze_audio, AudioDiagnostics, Severity};
use speech_core::projects::{now_rfc3339, valid_project_id, Chapter, ChapterStatus, ProjectStore};
use speech_core::takes::{sha256_hex, TakeManifest};
use speech_text::verbalize::{split_sentences, verbalize};
use tokio::sync::{mpsc, Mutex as AsyncMutex, Semaphore};

use crate::asr::AsrService;
use crate::AppState;

/// Версія схеми звіту/запису QC.
pub const QC_SCHEMA_VERSION: u32 = 1;
/// Максимальна довжина id job-а.
const MAX_QC_ID_LEN: usize = 64;

fn default_schema() -> u32 {
    QC_SCHEMA_VERSION
}

/// Статус QC-job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QcStatus {
    #[default]
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl QcStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

/// Пороги, за якими QC формує попередження/критичні знахідки.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QcThresholds {
    pub wer_warn: f32,
    pub wer_fail: f32,
    pub cer_warn: f32,
    pub cer_fail: f32,
}

impl Default for QcThresholds {
    fn default() -> Self {
        Self {
            wer_warn: 0.10,
            wer_fail: 0.25,
            cer_warn: 0.05,
            cer_fail: 0.15,
        }
    }
}

impl QcThresholds {
    fn validate(self) -> Result<()> {
        for (name, value) in [
            ("wer_warn", self.wer_warn),
            ("wer_fail", self.wer_fail),
            ("cer_warn", self.cer_warn),
            ("cer_fail", self.cer_fail),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("{name} must be between 0 and 1");
            }
        }
        if self.wer_warn > self.wer_fail {
            bail!("wer_warn must not exceed wer_fail");
        }
        if self.cer_warn > self.cer_fail {
            bail!("cer_warn must not exceed cer_fail");
        }
        Ok(())
    }
}

/// Одна знахідка QC (ASR або аудіо-діагностика).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
}

/// Порівняння одного вікна.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcSegment {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Оригінальний текст для субтитрів.
    pub script: String,
    /// Текст, який фактично пішов у синтез (тейк або rule-вербалізація).
    pub reference: String,
    pub recognized: String,
    pub diff: DiffReport,
}

/// Повний звіт QC глави.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcReport {
    #[serde(default = "default_schema")]
    pub schema: u32,
    pub reference_source: String,
    pub model: String,
    pub duration_ms: u64,
    pub segments_total: usize,
    pub segments_with_errors: usize,
    pub diff: DiffReport,
    pub diagnostics: AudioDiagnostics,
    pub segments: Vec<QcSegment>,
    pub issues: Vec<QcIssue>,
    pub thresholds: QcThresholds,
    pub elapsed_ms: u64,
    pub asr_ms: u64,
    pub rtf: f32,
}

/// Запис QC-job (серіалізується на диск).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcJob {
    #[serde(default = "default_schema")]
    pub version: u32,
    pub id: String,
    pub project_id: String,
    pub chapter_index: usize,
    pub status: QcStatus,
    pub stage: String,
    pub progress: u8,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_source: Option<String>,
    pub audio_sha256: String,
    pub audio_bytes: u64,
    pub duration_ms: u64,
    pub segments_total: usize,
    pub segments_done: usize,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<QcReport>,
}

/// Помилка QC API.
#[derive(Debug)]
pub enum QcApiError {
    NotFound(&'static str),
    BadRequest(String),
    Conflict(String),
    Unavailable(String),
    Internal(String),
}

impl QcApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(context: &'static str, error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, context, "QC request failed");
        Self::Internal(error.to_string())
    }
}

impl IntoResponse for QcApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message.to_string()),
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::Conflict(message) => (StatusCode::CONFLICT, message),
            Self::Unavailable(message) => (StatusCode::SERVICE_UNAVAILABLE, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

/// План одного вікна до розпізнавання.
#[derive(Debug, Clone)]
struct PlannedSegment {
    start_ms: u64,
    end_ms: u64,
    script: String,
    reference: String,
}

/// План усіх вікон глави.
#[derive(Debug, Clone)]
struct SegmentPlan {
    source: &'static str,
    segments: Vec<PlannedSegment>,
}

/// Черга Back-ASR перевірок.
pub struct QcManager {
    projects: Arc<ProjectStore>,
    asr: Arc<AsrService>,
    thresholds: QcThresholds,
    max_segment_ms: u64,
    state: Arc<Mutex<HashMap<String, QcJob>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    submit_lock: AsyncMutex<()>,
    sender: mpsc::Sender<String>,
}

impl QcManager {
    /// Створити чергу, відновити записи й позначити перервані job-и.
    pub fn new(
        projects: Arc<ProjectStore>,
        asr: Arc<AsrService>,
        concurrency: usize,
        thresholds: QcThresholds,
        max_segment_ms: u64,
    ) -> Result<Arc<Self>> {
        thresholds.validate()?;
        let reconciled = reconcile_jobs(&projects)?;
        if reconciled > 0 {
            tracing::warn!(reconciled, "marked interrupted quality checks as failed");
        }
        let mut state = HashMap::new();
        for job in load_all_jobs(&projects)? {
            state.insert(job.id.clone(), job);
        }

        let (sender, mut receiver) = mpsc::channel::<String>(256);
        let manager = Arc::new(Self {
            projects,
            asr,
            thresholds,
            max_segment_ms: max_segment_ms.max(1_000),
            state: Arc::new(Mutex::new(state)),
            cancel: Arc::new(Mutex::new(HashSet::new())),
            submit_lock: AsyncMutex::new(()),
            sender,
        });

        let dispatcher = Arc::clone(&manager);
        let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
        tokio::spawn(async move {
            while let Some(id) = receiver.recv().await {
                let permit = match Arc::clone(&semaphore).acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => break,
                };
                let manager = Arc::clone(&dispatcher);
                tokio::spawn(async move {
                    manager.run_job(&id).await;
                    drop(permit);
                });
            }
        });

        Ok(manager)
    }

    /// Поставити перевірку глави в чергу.
    ///
    /// Якщо для глави вже є активний job або завершений job із тим самим
    /// аудіо й моделлю, повертається він (`created = false`).
    pub async fn submit(
        &self,
        project_id: &str,
        chapter_index: usize,
        force: bool,
    ) -> Result<(QcJob, bool), QcApiError> {
        let _guard = self.submit_lock.lock().await;
        if !valid_project_id(project_id) {
            return Err(QcApiError::bad("invalid project id"));
        }
        self.asr
            .availability()
            .map_err(|error| QcApiError::Unavailable(error.to_string()))?;
        let project = self
            .projects
            .load(project_id)
            .map_err(|error| QcApiError::internal("load project", error))?
            .ok_or(QcApiError::NotFound("unknown project"))?;
        let chapter = project
            .chapters
            .get(chapter_index)
            .ok_or(QcApiError::NotFound("unknown chapter"))?;
        if chapter.status != ChapterStatus::Completed {
            return Err(QcApiError::Conflict(format!(
                "quality check requires a completed chapter (status: {})",
                chapter.status.as_str()
            )));
        }
        let projects = Arc::clone(&self.projects);
        let audio_project = project_id.to_string();
        let bytes = tokio::task::spawn_blocking(move || {
            projects.load_chapter_audio(&audio_project, chapter_index)
        })
        .await
        .map_err(|error| QcApiError::internal("read chapter audio task", error))?
        .map_err(|error| QcApiError::internal("read chapter audio", error))?
        .ok_or(QcApiError::Conflict("chapter audio is missing".into()))?;
        let audio_sha256 = sha256_hex(&bytes);
        let duration_ms = chapter
            .duration_ms
            .or_else(|| chapter.chunk_timings.last().map(|timing| timing.end_ms))
            .unwrap_or(0);
        let manifest = match self.projects.load_takes(project_id, chapter_index) {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::warn!(project_id, chapter_index, error = %error, "cannot load takes for QC; using script text");
                None
            }
        };
        let plan = plan_segments(chapter, manifest.as_ref(), self.max_segment_ms, duration_ms);
        if plan.segments.is_empty() {
            return Err(QcApiError::Conflict(
                "chapter has no speakable segments".into(),
            ));
        }

        let existing = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state
                .values()
                .filter(|job| job.project_id == project_id && job.chapter_index == chapter_index)
                .max_by(|left, right| left.created_at.cmp(&right.created_at))
                .cloned()
        };
        if let Some(job) = existing {
            if job.status.is_active() {
                if !force {
                    return Ok((job, false));
                }
                self.cancel
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(job.id.clone());
            } else if !force
                && job.status == QcStatus::Completed
                && job.audio_sha256 == audio_sha256
                && job.model == self.asr.model_id()
            {
                return Ok((job, false));
            }
        }

        let job = QcJob {
            version: QC_SCHEMA_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            chapter_index,
            status: QcStatus::Queued,
            stage: "queued".to_string(),
            progress: 0,
            model: self.asr.model_id().to_string(),
            reference_source: Some(plan.source.to_string()),
            audio_sha256,
            audio_bytes: bytes.len() as u64,
            duration_ms,
            segments_total: plan.segments.len(),
            segments_done: 0,
            created_at: now_rfc3339(),
            started_at: None,
            finished_at: None,
            elapsed_ms: None,
            error: None,
            report: None,
        };
        let dir = self
            .projects
            .qc_dir(project_id)
            .map_err(|error| QcApiError::internal("resolve QC directory", error))?;
        save_job(&dir, &job).map_err(|error| QcApiError::internal("save QC job", error))?;
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job.id.clone(), job.clone());
        if self.sender.send(job.id.clone()).await.is_err() {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&job.id);
            return Err(QcApiError::internal(
                "enqueue quality check",
                "QC queue is closed",
            ));
        }
        tracing::info!(
            job_id = %job.id,
            project_id,
            chapter_index,
            segments = job.segments_total,
            source = job.reference_source.as_deref().unwrap_or("script"),
            "quality check queued"
        );
        Ok((job, true))
    }

    /// Отримати job за id.
    pub fn get(&self, project_id: &str, job_id: &str) -> Result<QcJob, QcApiError> {
        if !valid_project_id(project_id) {
            return Err(QcApiError::bad("invalid project id"));
        }
        if !valid_qc_id(job_id) {
            return Err(QcApiError::bad("invalid quality check id"));
        }
        if !self.projects.exists(project_id) {
            return Err(QcApiError::NotFound("unknown project"));
        }
        if let Some(job) = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(job_id)
            .filter(|job| job.project_id == project_id)
        {
            return Ok(job.clone());
        }
        let dir = self
            .projects
            .qc_dir(project_id)
            .map_err(|error| QcApiError::internal("resolve QC directory", error))?;
        let path = dir.join(format!("{job_id}.json"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(QcApiError::NotFound("unknown quality check"))
            }
            Err(error) => return Err(QcApiError::internal("read QC job", error)),
        };
        let job: QcJob = serde_json::from_slice(&bytes)
            .map_err(|error| QcApiError::internal("parse QC job", error))?;
        if job.project_id != project_id {
            return Err(QcApiError::NotFound("unknown quality check"));
        }
        Ok(job)
    }

    /// Останній job глави.
    pub fn latest(&self, project_id: &str, chapter_index: usize) -> Result<QcJob, QcApiError> {
        if !valid_project_id(project_id) {
            return Err(QcApiError::bad("invalid project id"));
        }
        if !self.projects.exists(project_id) {
            return Err(QcApiError::NotFound("unknown project"));
        }
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state
            .values()
            .filter(|job| job.project_id == project_id && job.chapter_index == chapter_index)
            .max_by(|left, right| left.created_at.cmp(&right.created_at))
            .cloned()
            .ok_or(QcApiError::NotFound("no quality check for this chapter"))
    }

    /// Список job-ів проєкту (без повних звітів).
    pub fn list(&self, project_id: &str) -> Result<Vec<QcJob>, QcApiError> {
        if !valid_project_id(project_id) {
            return Err(QcApiError::bad("invalid project id"));
        }
        if !self.projects.exists(project_id) {
            return Err(QcApiError::NotFound("unknown project"));
        }
        let mut jobs: Vec<QcJob> = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|job| job.project_id == project_id)
            .cloned()
            .collect();
        jobs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        for job in &mut jobs {
            job.report = None;
        }
        Ok(jobs)
    }

    /// Скасувати активний job (кооперативно).
    pub fn cancel(&self, project_id: &str, job_id: &str) -> Result<QcJob, QcApiError> {
        let mut job = self.get(project_id, job_id)?;
        if job.status.is_active() {
            self.cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(job.id.clone());
            if job.status == QcStatus::Queued {
                job.status = QcStatus::Cancelled;
                job.stage = "cancelled".to_string();
                job.finished_at = Some(now_rfc3339());
                self.persist(&job);
            }
        }
        Ok(job)
    }

    /// Скасувати всі активні перевірки проєкту (перед видаленням).
    pub fn cancel_project(&self, project_id: &str) {
        let active: Vec<QcJob> = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|job| job.project_id == project_id && job.status.is_active())
            .cloned()
            .collect();
        for mut job in active {
            self.cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(job.id.clone());
            if job.status == QcStatus::Queued {
                job.status = QcStatus::Cancelled;
                job.stage = "cancelled".to_string();
                job.finished_at = Some(now_rfc3339());
                self.persist(&job);
            }
        }
    }

    /// Кількість активних job-ів (для healthz).
    pub fn active_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|job| job.status.is_active())
            .count()
    }

    async fn run_job(&self, id: &str) {
        let Some(mut job) = self.snapshot(id) else {
            return;
        };
        if job.status != QcStatus::Queued {
            return;
        }
        if self.is_cancelled(id) {
            self.finish_cancelled(&mut job);
            return;
        }
        job.status = QcStatus::Running;
        job.stage = "loading_model".to_string();
        job.progress = 2;
        job.started_at = Some(now_rfc3339());
        self.persist(&job);
        let started = Instant::now();

        if let Err(error) = self.asr.availability() {
            self.fail(&mut job, error.to_string());
            return;
        }
        if let Err(error) = self.asr.ensure_loaded().await {
            self.fail(&mut job, format!("cannot load ASR model: {error}"));
            return;
        }

        let project = match self.projects.load(&job.project_id) {
            Ok(Some(project)) => project,
            Ok(None) => {
                self.fail(&mut job, "project was deleted during quality check");
                return;
            }
            Err(error) => {
                self.fail(&mut job, format!("cannot load project: {error}"));
                return;
            }
        };
        let chapter = match project.chapters.get(job.chapter_index) {
            Some(chapter) => chapter,
            None => {
                self.fail(&mut job, "chapter was deleted during quality check");
                return;
            }
        };
        if chapter.status != ChapterStatus::Completed {
            self.fail(
                &mut job,
                format!(
                    "chapter is no longer completed (status: {})",
                    chapter.status.as_str()
                ),
            );
            return;
        }
        let projects = Arc::clone(&self.projects);
        let audio_project = job.project_id.clone();
        let audio_chapter = job.chapter_index;
        let audio = tokio::task::spawn_blocking(
            move || -> Result<(String, Vec<f32>, u32, AudioDiagnostics), String> {
                let bytes = projects
                    .load_chapter_audio(&audio_project, audio_chapter)
                    .map_err(|error| format!("cannot read chapter audio: {error}"))?
                    .ok_or_else(|| "chapter audio is missing".to_string())?;
                let sha256 = sha256_hex(&bytes);
                let (samples, sample_rate) = speech_audio::decode_wav_f32(&bytes)
                    .map_err(|error| format!("cannot decode chapter audio: {error}"))?;
                let diagnostics = analyze_audio(&samples, sample_rate)
                    .map_err(|error| format!("audio diagnostics failed: {error}"))?;
                Ok((sha256, samples, sample_rate, diagnostics))
            },
        )
        .await;
        let (audio_sha256, samples, sample_rate, diagnostics) = match audio {
            Ok(Ok(audio)) => audio,
            Ok(Err(error)) => {
                self.fail(&mut job, error);
                return;
            }
            Err(error) => {
                self.fail(&mut job, format!("audio processing task failed: {error}"));
                return;
            }
        };
        if audio_sha256 != job.audio_sha256 {
            self.fail(
                &mut job,
                "chapter audio changed during quality check; rerun verification",
            );
            return;
        }
        let duration_ms = samples.len() as u64 * 1000 / u64::from(sample_rate);
        let manifest = match self.projects.load_takes(&job.project_id, job.chapter_index) {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::warn!(job_id = %job.id, error = %error, "cannot load takes for QC; using script text");
                None
            }
        };
        let plan = plan_segments(chapter, manifest.as_ref(), self.max_segment_ms, duration_ms);
        if plan.segments.is_empty() {
            self.fail(&mut job, "chapter has no speakable segments");
            return;
        }
        job.segments_total = plan.segments.len();
        job.reference_source = Some(plan.source.to_string());
        job.duration_ms = duration_ms;
        job.stage = "transcribing".to_string();
        job.progress = 5;
        self.persist(&job);

        let mut segments = Vec::with_capacity(plan.segments.len());
        let mut asr_ms = 0u64;
        for (index, planned) in plan.segments.iter().enumerate() {
            if self.is_cancelled(id) {
                self.finish_cancelled(&mut job);
                return;
            }
            let audio = slice_ms(&samples, sample_rate, planned.start_ms, planned.end_ms);
            let (recognized, inference_ms) = if audio.is_empty() {
                (String::new(), 0)
            } else {
                match self.asr.transcribe(audio.to_vec(), sample_rate).await {
                    Ok(transcript) => (transcript.text, transcript.inference_ms),
                    Err(error) => {
                        self.fail(
                            &mut job,
                            format!(
                                "ASR failed on segment {index} ({}-{} ms): {error}",
                                planned.start_ms, planned.end_ms
                            ),
                        );
                        return;
                    }
                }
            };
            asr_ms = asr_ms.saturating_add(inference_ms);
            let diff = compare(&planned.reference, &recognized);
            segments.push(QcSegment {
                index,
                start_ms: planned.start_ms,
                end_ms: planned.end_ms,
                script: planned.script.clone(),
                reference: planned.reference.clone(),
                recognized,
                diff,
            });
            job.segments_done = index + 1;
            job.progress = (5 + (job.segments_done as u64 * 90 / job.segments_total.max(1) as u64))
                .min(95) as u8;
            self.persist(&job);
        }
        if self.is_cancelled(id) {
            self.finish_cancelled(&mut job);
            return;
        }

        let diff = aggregate_diff(&segments);
        let segments_with_errors = segments
            .iter()
            .filter(|segment| segment.diff.has_errors())
            .count();
        let issues = build_issues(&diff, &diagnostics, self.thresholds);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let report = QcReport {
            schema: QC_SCHEMA_VERSION,
            reference_source: plan.source.to_string(),
            model: job.model.clone(),
            duration_ms,
            segments_total: segments.len(),
            segments_with_errors,
            diff,
            diagnostics,
            segments,
            issues,
            thresholds: self.thresholds,
            elapsed_ms,
            asr_ms,
            rtf: elapsed_ms as f32 / duration_ms.max(1) as f32,
        };
        job.status = QcStatus::Completed;
        job.stage = "completed".to_string();
        job.progress = 100;
        job.finished_at = Some(now_rfc3339());
        job.elapsed_ms = Some(elapsed_ms);
        job.error = None;
        job.report = Some(report);
        self.persist(&job);
        tracing::info!(
            job_id = %job.id,
            project_id = %job.project_id,
            chapter_index = job.chapter_index,
            segments = job.segments_total,
            elapsed_ms,
            asr_ms,
            "quality check completed"
        );
    }

    fn snapshot(&self, id: &str) -> Option<QcJob> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned()
    }

    fn is_cancelled(&self, id: &str) -> bool {
        self.cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(id)
    }

    fn persist(&self, job: &QcJob) {
        if let Ok(dir) = self.projects.qc_dir(&job.project_id) {
            if let Err(error) = save_job(&dir, job) {
                tracing::error!(job_id = %job.id, error = %error, "cannot save QC job");
            }
        }
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job.id.clone(), job.clone());
    }

    fn fail(&self, job: &mut QcJob, error: impl Into<String>) {
        let error = error.into();
        tracing::warn!(job_id = %job.id, error = %error, "quality check failed");
        job.status = QcStatus::Failed;
        job.stage = "failed".to_string();
        job.progress = job.progress.min(99);
        job.finished_at = Some(now_rfc3339());
        job.error = Some(error);
        self.persist(job);
    }

    fn finish_cancelled(&self, job: &mut QcJob) {
        job.status = QcStatus::Cancelled;
        job.stage = "cancelled".to_string();
        job.finished_at = Some(now_rfc3339());
        job.error = None;
        self.persist(job);
    }
}

/// Запит `POST .../verify?force=true`.
#[derive(Debug, Default, Deserialize)]
pub struct VerifyQuery {
    #[serde(default)]
    pub force: bool,
}

/// `POST /v1/projects/{id}/chapters/{index}/verify`: запустити перевірку.
pub async fn start_verify(
    State(state): State<AppState>,
    AxumPath((id, index)): AxumPath<(String, usize)>,
    Query(query): Query<VerifyQuery>,
) -> Result<Response, QcApiError> {
    let (job, created) = state.qc.submit(&id, index, query.force).await?;
    let status = if created {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(job)).into_response())
}

/// `GET /v1/projects/{id}/chapters/{index}/verify`: останній звіт глави.
pub async fn latest_verify(
    State(state): State<AppState>,
    AxumPath((id, index)): AxumPath<(String, usize)>,
) -> Result<Json<QcJob>, QcApiError> {
    Ok(Json(state.qc.latest(&id, index)?))
}

/// `GET /v1/projects/{id}/qc`: історія перевірок проєкту.
pub async fn list_jobs(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, QcApiError> {
    let jobs = state.qc.list(&id)?;
    Ok(Json(json!({ "project_id": id, "jobs": jobs })))
}

/// `GET /v1/projects/{id}/qc/{job_id}`: повний звіт job-а.
pub async fn get_job(
    State(state): State<AppState>,
    AxumPath((id, job_id)): AxumPath<(String, String)>,
) -> Result<Json<QcJob>, QcApiError> {
    Ok(Json(state.qc.get(&id, &job_id)?))
}

/// `DELETE /v1/projects/{id}/qc/{job_id}`: скасувати активну перевірку.
pub async fn cancel_job(
    State(state): State<AppState>,
    AxumPath((id, job_id)): AxumPath<(String, String)>,
) -> Result<Json<QcJob>, QcApiError> {
    Ok(Json(state.qc.cancel(&id, &job_id)?))
}

fn plan_segments(
    chapter: &Chapter,
    manifest: Option<&TakeManifest>,
    max_segment_ms: u64,
    duration_ms: u64,
) -> SegmentPlan {
    let max_segment_ms = max_segment_ms.max(1_000);
    let spoken = spoken_texts(chapter, manifest);
    if !chapter.chunk_timings.is_empty() {
        let source = if spoken.is_some() { "takes" } else { "script" };
        let mut items = Vec::with_capacity(chapter.chunk_timings.len());
        let mut previous_end = 0u64;
        for (index, timing) in chapter.chunk_timings.iter().enumerate() {
            let start = timing.start_ms.max(previous_end);
            let end = timing.end_ms.max(start + 1);
            previous_end = end;
            let script = timing.text.trim().to_string();
            let reference = spoken
                .as_ref()
                .and_then(|texts| texts.get(index))
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| verbalize(&script));
            let reference = if reference.trim().is_empty() {
                script.clone()
            } else {
                reference
            };
            if script.is_empty() && reference.is_empty() {
                continue;
            }
            items.push(PlannedSegment {
                start_ms: start,
                end_ms: end,
                script,
                reference,
            });
        }
        let final_end = duration_ms.max(previous_end);
        return SegmentPlan {
            source,
            segments: group_segments(items, max_segment_ms, final_end),
        };
    }

    let text = chapter.text.trim();
    if text.is_empty() {
        return SegmentPlan {
            source: "script",
            segments: Vec::new(),
        };
    }
    let sentences = split_sentences(text);
    let sentences = if sentences.is_empty() {
        vec![text.to_string()]
    } else {
        sentences
    };
    let total_chars: u64 = sentences
        .iter()
        .map(|sentence| sentence.chars().count().max(1) as u64)
        .sum();
    let duration = if duration_ms > 0 {
        duration_ms
    } else {
        (total_chars * 1000 / 14).max(1_000)
    };
    let mut cursor = 0u64;
    let mut items = Vec::with_capacity(sentences.len());
    for (index, sentence) in sentences.iter().enumerate() {
        let chars = sentence.chars().count().max(1) as u64;
        let end = if index + 1 == sentences.len() {
            duration
        } else {
            cursor + duration.saturating_mul(chars) / total_chars.max(1)
        };
        let end = end.max(cursor + 1);
        let script = sentence.trim().to_string();
        let reference = verbalize(&script);
        items.push(PlannedSegment {
            start_ms: cursor,
            end_ms: end,
            script,
            reference: if reference.trim().is_empty() {
                sentence.clone()
            } else {
                reference
            },
        });
        cursor = end;
    }
    SegmentPlan {
        source: "script",
        segments: group_segments(items, max_segment_ms, duration),
    }
}

fn spoken_texts(chapter: &Chapter, manifest: Option<&TakeManifest>) -> Option<Vec<String>> {
    let manifest = manifest?;
    if manifest.units.len() != chapter.chunk_timings.len() {
        return None;
    }
    let mut texts = Vec::with_capacity(manifest.units.len());
    for (index, unit) in manifest.units.iter().enumerate() {
        if unit.index != index {
            return None;
        }
        let text = unit
            .active()
            .map(|take| take.spoken.trim())
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| unit.text.trim());
        if text.is_empty() {
            return None;
        }
        texts.push(text.to_string());
    }
    Some(texts)
}

fn group_segments(
    items: Vec<PlannedSegment>,
    max_segment_ms: u64,
    final_end_ms: u64,
) -> Vec<PlannedSegment> {
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut current: Option<PlannedSegment> = None;
    for item in items {
        let item_end = item.end_ms.max(item.start_ms + 1);
        match current.take() {
            None => current = Some(item),
            Some(mut segment) => {
                let boundary = item.start_ms.max(segment.start_ms + 1);
                if boundary.saturating_sub(segment.start_ms) > max_segment_ms {
                    segment.end_ms = boundary.max(segment.start_ms + 1);
                    segments.push(segment);
                    current = Some(item);
                } else {
                    segment.end_ms = segment.end_ms.max(item_end);
                    append_text(&mut segment.script, &item.script);
                    append_text(&mut segment.reference, &item.reference);
                    current = Some(segment);
                }
            }
        }
    }
    if let Some(mut segment) = current {
        if final_end_ms > segment.end_ms {
            segment.end_ms = final_end_ms;
        }
        segments.push(segment);
    }
    segments
}

fn append_text(target: &mut String, extra: &str) {
    let extra = extra.trim();
    if extra.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push(' ');
    }
    target.push_str(extra);
}

fn slice_ms(samples: &[f32], sample_rate: u32, start_ms: u64, end_ms: u64) -> &[f32] {
    let len = samples.len();
    let start = ((start_ms as u128 * sample_rate as u128) / 1000).min(len as u128) as usize;
    let end = ((end_ms as u128 * sample_rate as u128) / 1000).min(len as u128) as usize;
    &samples[start..end.max(start)]
}

fn aggregate_diff(segments: &[QcSegment]) -> DiffReport {
    let mut total = DiffReport::default();
    for segment in segments {
        total.reference_words += segment.diff.reference_words;
        total.hypothesis_words += segment.diff.hypothesis_words;
        total.correct += segment.diff.correct;
        total.substitutions += segment.diff.substitutions;
        total.deletions += segment.diff.deletions;
        total.insertions += segment.diff.insertions;
        total.reference_chars += segment.diff.reference_chars;
        total.hypothesis_chars += segment.diff.hypothesis_chars;
        total.char_errors += segment.diff.char_errors;
        total.errors.extend(segment.diff.errors.iter().cloned());
    }
    let word_errors = total.substitutions + total.deletions + total.insertions;
    total.wer = rate(word_errors, total.reference_words);
    total.cer = rate(total.char_errors, total.reference_chars);
    total
}

fn rate(errors: usize, reference: usize) -> f32 {
    if reference == 0 {
        if errors == 0 {
            0.0
        } else {
            1.0
        }
    } else {
        errors as f32 / reference as f32
    }
}

fn build_issues(
    diff: &DiffReport,
    diagnostics: &AudioDiagnostics,
    thresholds: QcThresholds,
) -> Vec<QcIssue> {
    let mut issues = Vec::new();
    if diff.wer >= thresholds.wer_fail {
        issues.push(QcIssue {
            code: "wer_above_fail".into(),
            severity: Severity::Critical,
            message: format!(
                "WER {:.2}% exceeds the failure threshold {:.2}%",
                diff.wer * 100.0,
                thresholds.wer_fail * 100.0
            ),
        });
    } else if diff.wer >= thresholds.wer_warn {
        issues.push(QcIssue {
            code: "wer_above_warn".into(),
            severity: Severity::Warning,
            message: format!(
                "WER {:.2}% exceeds the warning threshold {:.2}%",
                diff.wer * 100.0,
                thresholds.wer_warn * 100.0
            ),
        });
    }
    if diff.cer >= thresholds.cer_fail {
        issues.push(QcIssue {
            code: "cer_above_fail".into(),
            severity: Severity::Critical,
            message: format!(
                "CER {:.2}% exceeds the failure threshold {:.2}%",
                diff.cer * 100.0,
                thresholds.cer_fail * 100.0
            ),
        });
    } else if diff.cer >= thresholds.cer_warn {
        issues.push(QcIssue {
            code: "cer_above_warn".into(),
            severity: Severity::Warning,
            message: format!(
                "CER {:.2}% exceeds the warning threshold {:.2}%",
                diff.cer * 100.0,
                thresholds.cer_warn * 100.0
            ),
        });
    }
    for issue in &diagnostics.issues {
        issues.push(QcIssue {
            code: issue.code.clone(),
            severity: issue.severity,
            message: issue.message.clone(),
        });
    }
    issues
}

fn reconcile_jobs(projects: &ProjectStore) -> Result<usize> {
    let mut reconciled = 0;
    for (project_id, dir) in project_qc_dirs(projects)? {
        for path in json_files(&dir)? {
            let mut job: QcJob = match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            {
                Some(job) => job,
                None => {
                    tracing::warn!(path = %path.display(), "skipping malformed QC job");
                    continue;
                }
            };
            if job.project_id != project_id {
                tracing::warn!(path = %path.display(), "QC job project mismatch");
                continue;
            }
            if job.status.is_active() {
                job.status = QcStatus::Failed;
                job.stage = "failed".to_string();
                job.error = Some("quality check interrupted by server restart".to_string());
                job.finished_at = Some(now_rfc3339());
                save_job(&dir, &job)
                    .with_context(|| format!("save reconciled QC job {}", path.display()))?;
                reconciled += 1;
            }
        }
    }
    Ok(reconciled)
}

fn load_all_jobs(projects: &ProjectStore) -> Result<Vec<QcJob>> {
    let mut jobs = Vec::new();
    for (project_id, dir) in project_qc_dirs(projects)? {
        for path in json_files(&dir)? {
            match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<QcJob>(&bytes).ok())
            {
                Some(job) if job.project_id == project_id => jobs.push(job),
                Some(_) => {
                    tracing::warn!(path = %path.display(), "QC job project mismatch");
                }
                None => {
                    tracing::warn!(path = %path.display(), "skipping malformed QC job");
                }
            }
        }
    }
    Ok(jobs)
}

fn project_qc_dirs(projects: &ProjectStore) -> Result<Vec<(String, PathBuf)>> {
    let mut dirs = Vec::new();
    if !projects.dir().is_dir() {
        return Ok(dirs);
    }
    for entry in std::fs::read_dir(projects.dir())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let project_id = entry.file_name().to_string_lossy().into_owned();
        if !valid_project_id(&project_id) {
            continue;
        }
        let dir = projects.qc_dir(&project_id)?;
        if dir.is_dir() {
            dirs.push((project_id, dir));
        }
    }
    Ok(dirs)
}

fn json_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !dir.is_dir() {
        return Ok(paths);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn save_job(dir: &Path, job: &QcJob) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", job.id));
    let tmp = dir.join(format!("{}.json.tmp", job.id));
    let bytes =
        serde_json::to_vec_pretty(job).map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn valid_qc_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_QC_ID_LEN
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use speech_core::projects::{Chapter, ChunkTiming};
    use speech_core::takes::{Take, TakeManifest, UnitTakes};

    fn chapter_with_timings() -> Chapter {
        let mut chapter = Chapter::new(0, "Розділ", "Привіт, світе! Друге речення.");
        chapter.chunk_timings = vec![
            ChunkTiming {
                start_ms: 0,
                end_ms: 1_000,
                text: "Привіт, світе!".into(),
                role: None,
            },
            ChunkTiming {
                start_ms: 1_500,
                end_ms: 3_000,
                text: "Друге речення.".into(),
                role: None,
            },
        ];
        chapter
    }

    fn manifest() -> TakeManifest {
        TakeManifest::new(
            "fingerprint",
            vec![
                UnitTakes {
                    index: 0,
                    text: "Привіт, світе!".into(),
                    role: None,
                    active_take: 1,
                    takes: vec![Take {
                        id: 1,
                        voice: None,
                        speed: 1.0,
                        spoken: "привіт світе".into(),
                        duration_ms: 1_000,
                        bytes: 1,
                        sha256: "a".repeat(64),
                        created_at: now_rfc3339(),
                    }],
                },
                UnitTakes {
                    index: 1,
                    text: "Друге речення.".into(),
                    role: None,
                    active_take: 1,
                    takes: vec![Take {
                        id: 1,
                        voice: None,
                        speed: 1.0,
                        spoken: "друге речення".into(),
                        duration_ms: 1_500,
                        bytes: 1,
                        sha256: "b".repeat(64),
                        created_at: now_rfc3339(),
                    }],
                },
            ],
        )
    }

    #[test]
    fn plan_uses_spoken_takes_and_groups_windows() {
        let chapter = chapter_with_timings();
        let plan = plan_segments(&chapter, Some(&manifest()), 1_000, 3_000);
        assert_eq!(plan.source, "takes");
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].reference, "привіт світе");
        assert_eq!(plan.segments[0].start_ms, 0);
        assert_eq!(plan.segments[0].end_ms, 1_500);
        assert_eq!(plan.segments[1].reference, "друге речення");
        assert_eq!(plan.segments[1].end_ms, 3_000);
    }

    #[test]
    fn plan_falls_back_to_script() {
        let chapter = chapter_with_timings();
        let plan = plan_segments(&chapter, None, 25_000, 3_000);
        assert_eq!(plan.source, "script");
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].reference, "Привіт, світе! Друге речення.");
        assert_eq!(plan.segments[0].end_ms, 3_000);
    }

    #[test]
    fn plan_without_timings_splits_sentences() {
        let mut chapter = Chapter::new(0, "Розділ", "Перше речення. Друге речення.");
        chapter.chunk_timings.clear();
        let plan = plan_segments(&chapter, None, 1_000, 2_000);
        assert_eq!(plan.source, "script");
        assert!(!plan.segments.is_empty());
        assert_eq!(plan.segments.last().unwrap().end_ms, 2_000);
    }

    #[test]
    fn slices_audio_by_milliseconds() {
        let samples = vec![0.0f32; 48_000];
        let slice = slice_ms(&samples, 24_000, 500, 1_000);
        assert_eq!(slice.len(), 12_000);
        let slice = slice_ms(&samples, 24_000, 0, 10_000);
        assert_eq!(slice.len(), 48_000);
    }

    #[test]
    fn aggregates_segments() {
        let segments = vec![
            QcSegment {
                index: 0,
                start_ms: 0,
                end_ms: 1_000,
                script: "кіт".into(),
                reference: "кіт".into(),
                recognized: "кот".into(),
                diff: compare("кіт", "кот"),
            },
            QcSegment {
                index: 1,
                start_ms: 1_000,
                end_ms: 2_000,
                script: "пес".into(),
                reference: "пес".into(),
                recognized: "пес".into(),
                diff: compare("пес", "пес"),
            },
        ];
        let total = aggregate_diff(&segments);
        assert_eq!(total.reference_words, 2);
        assert_eq!(total.substitutions, 1);
        assert_eq!(total.wer, 0.5);
    }
}
