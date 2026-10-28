//! Асинхронний експорт готових аудіокниг (M4B/MP3/OGG).
//!
//! Записи експорту зберігаються як `<projects>/<id>/exports/<export>.json`,
//! аудіо — `<export>.<ext>`, субтитри — `<export>.srt|.vtt`. Експорт
//! виконується у `spawn_blocking` під `PROMIN_EXPORT_CONCURRENCY`, прогрес
//! оновлюється після кожної глави, а після рестарту незавершені записи
//! позначаються `failed` (як і job-и синтезу).

use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::json;
use speech_core::projects::{
    now_rfc3339, valid_project_id, ChapterStatus, Project, ProjectStore,
};
use speech_export::{
    combined_subtitles, export_book, CoverImage, Cue, ExportChapter, ExportError,
    ExportFormat, ExportInput, ExportMetadata, ExportObserver, ExportSummary,
};
use tokio::sync::{mpsc, Semaphore};
use tokio_util::io::ReaderStream;

use crate::AppState;

/// Максимальний розмір обкладинки після base64-декодування.
pub const MAX_COVER_BYTES: usize = 4 * 1024 * 1024;
/// Максимальна довжина id експорту.
const MAX_EXPORT_ID_LEN: usize = 64;

/// Статус експорту.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ExportStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

/// Запис експорту (серіалізується на диск).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    pub id: String,
    pub project_id: String,
    pub format: ExportFormat,
    pub status: ExportStatus,
    pub progress: u8,
    pub chapters_total: usize,
    pub chapters_done: usize,
    pub bitrate_kbps: u32,
    /// `updated_at` проєкту на момент запуску (для прапорця `stale`).
    #[serde(default)]
    pub project_updated_at: String,
    pub created_at: String,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub finished_at: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub audio_bytes: Option<u64>,
    #[serde(default)]
    pub audio_file: Option<String>,
    #[serde(default)]
    pub subtitles_srt_file: Option<String>,
    #[serde(default)]
    pub subtitles_vtt_file: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Запит на створення експорту.
#[derive(Debug, Deserialize)]
pub struct CreateExportRequest {
    pub format: String,
    #[serde(default)]
    pub bitrate_kbps: Option<u32>,
    /// PNG/JPEG у base64 (можна з data-URI префіксом).
    #[serde(default)]
    pub cover_base64: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub album: Option<String>,
    #[serde(default)]
    pub narrator: Option<String>,
    #[serde(default)]
    pub year: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Помилка API експорту.
#[derive(Debug)]
pub enum ExportApiError {
    NotFound(&'static str),
    BadRequest(String),
    Conflict {
        message: String,
        pending_chapters: Vec<usize>,
    },
    NotReady(String),
    Internal(String),
}

impl ExportApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(context: &str, error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, context, "export API error");
        Self::Internal(context.to_string())
    }

    fn conflict(message: impl Into<String>, pending_chapters: Vec<usize>) -> Self {
        Self::Conflict {
            message: message.into(),
            pending_chapters,
        }
    }
}

impl IntoResponse for ExportApiError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            Self::NotFound(message) => (StatusCode::NOT_FOUND, json!({ "error": message })),
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, json!({ "error": message })),
            Self::Conflict {
                message,
                pending_chapters,
            } => (
                StatusCode::CONFLICT,
                json!({ "error": message, "pending_chapters": pending_chapters }),
            ),
            Self::NotReady(message) => (StatusCode::CONFLICT, json!({ "error": message })),
            Self::Internal(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": message }),
            ),
        };
        (status, Json(body)).into_response()
    }
}

/// Параметри глави, зняті на момент запуску (не серіалізуються).
#[derive(Debug, Clone)]
struct ExportSpec {
    metadata: ExportMetadata,
    cover: Option<CoverImage>,
}

/// Черга експорту.
pub struct ExportManager {
    projects: Arc<ProjectStore>,
    state: Arc<Mutex<HashMap<String, ExportRecord>>>,
    specs: Arc<Mutex<HashMap<String, ExportSpec>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    sender: mpsc::Sender<String>,
}

impl ExportManager {
    /// Створити чергу, прибрати недописані файли та відновити записи.
    pub fn new(projects: Arc<ProjectStore>, concurrency: usize) -> anyhow::Result<Arc<Self>> {
        let reconciled = reconcile_exports(&projects)?;
        if reconciled > 0 {
            tracing::warn!(reconciled, "marked interrupted exports as failed");
        }

        let mut state = HashMap::new();
        for record in load_all_records(&projects)? {
            state.insert(record.id.clone(), record);
        }

        let (sender, mut receiver) = mpsc::channel::<String>(256);
        let manager = Arc::new(Self {
            projects,
            state: Arc::new(Mutex::new(state)),
            specs: Arc::new(Mutex::new(HashMap::new())),
            cancel: Arc::new(Mutex::new(HashSet::new())),
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
                    manager.run_export(&id).await;
                    drop(permit);
                });
            }
        });

        Ok(manager)
    }

    /// Поставити експорт у чергу.
    pub async fn submit(
        &self,
        project_id: &str,
        request: CreateExportRequest,
    ) -> Result<ExportRecord, ExportApiError> {
        if !valid_project_id(project_id) {
            return Err(ExportApiError::bad("invalid project id"));
        }
        let format = ExportFormat::parse(&request.format)
            .ok_or_else(|| ExportApiError::bad("format must be m4b, mp3 or ogg"))?;
        let bitrate_kbps = format
            .resolve_bitrate_kbps(request.bitrate_kbps.unwrap_or(0))
            .map_err(|error| ExportApiError::bad(error.to_string()))?;
        let project = self
            .projects
            .load(project_id)
            .map_err(|error| ExportApiError::internal("load project", error))?
            .ok_or(ExportApiError::NotFound("unknown project"))?;

        if project.chapters.is_empty() {
            return Err(ExportApiError::conflict(
                "project has no chapters",
                Vec::new(),
            ));
        }
        let pending: Vec<usize> = project
            .chapters
            .iter()
            .filter(|chapter| {
                chapter.status != ChapterStatus::Completed
                    || !self
                        .projects
                        .chapter_audio_exists(project_id, chapter.index)
            })
            .map(|chapter| chapter.index)
            .collect();
        if !pending.is_empty() {
            return Err(ExportApiError::conflict(
                "all chapters must be completed before export",
                pending,
            ));
        }

        let cover = decode_cover(request.cover_base64)?;
        let title = request
            .title
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| project.title.clone());
        if title.trim().is_empty() {
            return Err(ExportApiError::bad("book title must not be empty"));
        }
        let author = request.author.unwrap_or_else(|| project.author.clone());
        let album = request
            .album
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| project.title.clone());
        let year = request
            .year
            .filter(|value| !value.trim().is_empty())
            .or_else(|| now_rfc3339().get(..4).map(str::to_string));
        let metadata = ExportMetadata {
            title,
            author,
            album,
            narrator: request
                .narrator
                .filter(|value| !value.trim().is_empty()),
            year,
            description: request
                .description
                .filter(|value| !value.trim().is_empty()),
            language: project.language.clone(),
        };

        let now = now_rfc3339();
        let record = ExportRecord {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            format,
            status: ExportStatus::Queued,
            progress: 0,
            chapters_total: project.chapters.len(),
            chapters_done: 0,
            bitrate_kbps,
            project_updated_at: project.updated_at.clone(),
            created_at: now,
            started_at: None,
            finished_at: None,
            duration_ms: None,
            audio_bytes: None,
            audio_file: None,
            subtitles_srt_file: None,
            subtitles_vtt_file: None,
            error: None,
        };

        let dir = self.export_dir(project_id)?;
        std::fs::create_dir_all(&dir)
            .map_err(|error| ExportApiError::internal("create exports directory", error))?;
        save_record(&dir, &record)
            .map_err(|error| ExportApiError::internal("save export record", error))?;
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(record.id.clone(), record.clone());
        self.specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                record.id.clone(),
                ExportSpec {
                    metadata,
                    cover,
                },
            );

        if self.sender.send(record.id.clone()).await.is_err() {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&record.id);
            self.specs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&record.id);
            return Err(ExportApiError::internal(
                "enqueue export",
                "export queue is closed",
            ));
        }
        tracing::info!(
            export_id = %record.id,
            project_id,
            format = format.as_str(),
            bitrate_kbps,
            chapters = record.chapters_total,
            "export queued"
        );
        Ok(record)
    }

    /// Отримати запис експорту.
    pub fn get(
        &self,
        project_id: &str,
        export_id: &str,
    ) -> Result<ExportRecord, ExportApiError> {
        if !valid_project_id(project_id) {
            return Err(ExportApiError::bad("invalid project id"));
        }
        if !valid_export_id(export_id) {
            return Err(ExportApiError::bad("invalid export id"));
        }
        if let Some(record) = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(export_id)
            .filter(|record| record.project_id == project_id)
        {
            return Ok(record.clone());
        }
        let path = self
            .export_dir(project_id)?
            .join(format!("{export_id}.json"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ExportApiError::NotFound("unknown export"))
            }
            Err(error) => {
                return Err(ExportApiError::internal("read export record", error))
            }
        };
        serde_json::from_slice(&bytes)
            .map_err(|error| ExportApiError::internal("parse export record", error))
    }

    /// Список експортів проєкту (новіші першими).
    pub fn list(&self, project_id: &str) -> Result<Vec<ExportRecord>, ExportApiError> {
        if !valid_project_id(project_id) {
            return Err(ExportApiError::bad("invalid project id"));
        }
        if !self.projects.exists(project_id) {
            return Err(ExportApiError::NotFound("unknown project"));
        }
        let dir = self.export_dir(project_id)?;
        let mut records = Vec::new();
        if dir.is_dir() {
            let entries = std::fs::read_dir(&dir)
                .map_err(|error| ExportApiError::internal("list exports", error))?;
            for entry in entries {
                let entry =
                    entry.map_err(|error| ExportApiError::internal("list exports", error))?;
                let path = entry.path();
                if path.extension().and_then(|value| value.to_str()) != Some("json") {
                    continue;
                }
                match std::fs::read(&path)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<ExportRecord>(&bytes).ok())
                {
                    Some(record) => records.push(record),
                    None => tracing::warn!(path = %path.display(), "skipping malformed export record"),
                }
            }
        }
        let live = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for record in &mut records {
            if let Some(current) = live.get(&record.id) {
                *record = current.clone();
            }
        }
        records.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(records)
    }

    /// Видалити запис і файли; активний експорт скасовується.
    pub fn delete(&self, project_id: &str, export_id: &str) -> Result<(), ExportApiError> {
        let record = self.get(project_id, export_id)?;
        if record.status.is_active() {
            self.cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(export_id.to_string());
        }
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(export_id);
        self.specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(export_id);

        let dir = self.export_dir(project_id)?;
        remove_file_if_exists(&dir.join(format!("{export_id}.json")));
        remove_file_if_exists(&dir.join(format!("{export_id}.part")));
        if let Some(name) = record.audio_file.as_deref() {
            remove_file_if_exists(&dir.join(name));
        }
        if let Some(name) = record.subtitles_srt_file.as_deref() {
            remove_file_if_exists(&dir.join(name));
        }
        if let Some(name) = record.subtitles_vtt_file.as_deref() {
            remove_file_if_exists(&dir.join(name));
        }
        Ok(())
    }

    /// Скасувати всі активні експорти проєкту (перед видаленням).
    pub fn cancel_project(&self, project_id: &str) {
        let ids: Vec<String> = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let ids: Vec<String> = state
                .values()
                .filter(|record| record.project_id == project_id && record.status.is_active())
                .map(|record| record.id.clone())
                .collect();
            for id in &ids {
                state.remove(id);
            }
            ids
        };
        if ids.is_empty() {
            return;
        }
        let mut cancel = self
            .cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut specs = self
            .specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for id in ids {
            cancel.insert(id.clone());
            specs.remove(&id);
        }
    }

    /// Шлях до файлу експорту (тільки ім’я, без шляхів).
    pub fn file_path(
        &self,
        project_id: &str,
        name: &str,
    ) -> Result<PathBuf, ExportApiError> {
        if !safe_file_name(name) {
            return Err(ExportApiError::internal(
                "invalid export file name",
                name,
            ));
        }
        Ok(self.export_dir(project_id)?.join(name))
    }

    fn export_dir(&self, project_id: &str) -> Result<PathBuf, ExportApiError> {
        if !valid_project_id(project_id) {
            return Err(ExportApiError::bad("invalid project id"));
        }
        Ok(self.projects.dir().join(project_id).join("exports"))
    }

    fn record(&self, id: &str) -> Option<ExportRecord> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned()
    }

    fn persist(&self, record: &ExportRecord) {
        if let Ok(dir) = self.export_dir(&record.project_id) {
            if let Err(error) = save_record(&dir, record) {
                tracing::error!(
                    export_id = %record.id,
                    error = %error,
                    "cannot save export record"
                );
            }
        }
    }

    async fn run_export(&self, id: &str) {
        let Some(mut record) = self.record(id) else {
            self.cleanup_task_state(id);
            return;
        };
        if record.status != ExportStatus::Queued {
            self.cleanup_task_state(id);
            return;
        }
        let Some(spec) = self
            .specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned()
        else {
            self.fail_record(id, "export request is missing");
            return;
        };

        record.status = ExportStatus::Running;
        record.started_at = Some(now_rfc3339());
        record.progress = 0;
        record.error = None;
        self.persist(&record);
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.insert(id.to_string(), record.clone());
        }

        let project = match self.projects.load(&record.project_id) {
            Ok(Some(project)) => project,
            Ok(None) => {
                self.fail_record(id, "project does not exist");
                return;
            }
            Err(error) => {
                self.fail_record(id, format!("cannot load project: {error}"));
                return;
            }
        };
        let input = match build_input(&self.projects, &project, &record, spec) {
            Ok(input) => input,
            Err(message) => {
                self.fail_record(id, message);
                return;
            }
        };
        let dir = match self.export_dir(&record.project_id) {
            Ok(dir) => dir,
            Err(error) => {
                self.fail_record(id, format!("{error:?}"));
                return;
            }
        };
        if let Err(error) = std::fs::create_dir_all(&dir) {
            self.fail_record(id, format!("cannot create exports directory: {error}"));
            return;
        }
        let part = dir.join(format!("{id}.part"));
        let observer = ProgressObserver {
            id: id.to_string(),
            state: Arc::clone(&self.state),
            cancel: Arc::clone(&self.cancel),
            dir: dir.clone(),
        };

        let part_for_task = part.clone();
        let result = tokio::task::spawn_blocking(
            move || -> Result<(ExportSummary, String, String), ExportError> {
                let file = std::fs::File::create(&part_for_task)?;
                let mut writer = BufWriter::new(file);
                let mut observer = observer;
                let summary = export_book(&input, &mut writer, &mut observer)?;
                writer.flush()?;
                let (srt, vtt) = combined_subtitles(&input.chapters, &summary.chapters);
                Ok((summary, srt, vtt))
            },
        )
        .await;

        match result {
            Ok(Ok((summary, srt, vtt))) => self.complete(id, &summary, srt, vtt),
            Ok(Err(ExportError::Cancelled)) => self.mark_cancelled(id),
            Ok(Err(error)) => self.fail_record(id, error.to_string()),
            Err(join_error) => self.fail_record(id, format!("export task failed: {join_error}")),
        }
    }

    fn complete(&self, id: &str, summary: &ExportSummary, srt: String, vtt: String) {
        let Some(record) = self.record(id) else {
            remove_file_if_exists(&self.part_path(id));
            self.cleanup_task_state(id);
            return;
        };
        if record.status != ExportStatus::Running {
            remove_file_if_exists(&self.part_path(id));
            self.cleanup_task_state(id);
            return;
        }
        let dir = match self.export_dir(&record.project_id) {
            Ok(dir) => dir,
            Err(error) => {
                self.fail_record(id, format!("{error:?}"));
                return;
            }
        };
        let part = dir.join(format!("{id}.part"));
        let audio_file = format!("{id}.{}", record.format.extension());
        let audio_path = dir.join(&audio_file);
        if let Err(error) = std::fs::rename(&part, &audio_path) {
            self.fail_record(id, format!("cannot finalize export: {error}"));
            return;
        }
        let srt_file = format!("{id}.srt");
        let vtt_file = format!("{id}.vtt");
        if let Err(error) = std::fs::write(dir.join(&srt_file), srt.as_bytes()) {
            remove_file_if_exists(&audio_path);
            self.fail_record(id, format!("cannot write subtitles: {error}"));
            return;
        }
        if let Err(error) = std::fs::write(dir.join(&vtt_file), vtt.as_bytes()) {
            remove_file_if_exists(&audio_path);
            remove_file_if_exists(&dir.join(&srt_file));
            self.fail_record(id, format!("cannot write subtitles: {error}"));
            return;
        }

        let mut record = record;
        record.status = ExportStatus::Completed;
        record.progress = 100;
        record.chapters_done = record.chapters_total;
        record.duration_ms = Some(summary.duration_ms);
        record.audio_bytes = Some(summary.bytes_written);
        record.audio_file = Some(audio_file);
        record.subtitles_srt_file = Some(srt_file);
        record.subtitles_vtt_file = Some(vtt_file);
        record.finished_at = Some(now_rfc3339());
        record.error = None;
        self.persist(&record);
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.insert(id.to_string(), record.clone());
        }
        self.cleanup_task_state(id);
        tracing::info!(
            export_id = id,
            project_id = %record.project_id,
            format = record.format.as_str(),
            duration_ms = summary.duration_ms,
            audio_bytes = summary.bytes_written,
            "export completed"
        );
    }

    fn fail_record(&self, id: &str, message: impl Into<String>) {
        let message = message.into();
        let record = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(record) = state.get_mut(id) else {
                drop(state);
                self.cleanup_task_state(id);
                return;
            };
            if record.status == ExportStatus::Cancelled {
                drop(state);
                self.cleanup_task_state(id);
                return;
            }
            record.status = ExportStatus::Failed;
            record.error = Some(message.clone());
            record.finished_at = Some(now_rfc3339());
            record.progress = record.progress.min(99);
            record.clone()
        };
        self.persist(&record);
        remove_file_if_exists(&self.part_path(id));
        self.cleanup_task_state(id);
        tracing::error!(
            export_id = id,
            project_id = %record.project_id,
            error = %message,
            "export failed"
        );
    }

    fn mark_cancelled(&self, id: &str) {
        let record = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(record) = state.get_mut(id) else {
                drop(state);
                self.cleanup_task_state(id);
                return;
            };
            record.status = ExportStatus::Cancelled;
            record.error = Some("cancelled".to_string());
            record.finished_at = Some(now_rfc3339());
            record.clone()
        };
        self.persist(&record);
        remove_file_if_exists(&self.part_path(id));
        self.cleanup_task_state(id);
        tracing::info!(export_id = id, "export cancelled");
    }

    fn part_path(&self, id: &str) -> PathBuf {
        let project_id = self
            .record(id)
            .map(|record| record.project_id)
            .unwrap_or_default();
        self.projects
            .dir()
            .join(project_id)
            .join("exports")
            .join(format!("{id}.part"))
    }

    fn cleanup_task_state(&self, id: &str) {
        self.specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
        self.cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
    }
}

/// Побудувати вхід експорту з актуального проєкту.
fn build_input(
    projects: &ProjectStore,
    project: &Project,
    record: &ExportRecord,
    spec: ExportSpec,
) -> Result<ExportInput, String> {
    let mut chapters = Vec::with_capacity(project.chapters.len());
    for chapter in &project.chapters {
        if chapter.status != ChapterStatus::Completed {
            return Err(format!(
                "chapter {} is not completed (status: {})",
                chapter.index,
                chapter.status.as_str()
            ));
        }
        let wav_path = projects
            .chapter_audio_path(&project.id, chapter.index)
            .map_err(|error| format!("cannot resolve chapter audio: {error}"))?;
        if !wav_path.is_file() {
            return Err(format!("chapter {} audio is missing", chapter.index));
        }
        chapters.push(ExportChapter {
            index: chapter.index,
            title: chapter.title.clone(),
            wav_path,
            cues: chapter
                .chunk_timings
                .iter()
                .map(|timing| Cue {
                    start_ms: timing.start_ms,
                    end_ms: timing.end_ms,
                    text: timing.text.clone(),
                })
                .collect(),
        });
    }
    Ok(ExportInput {
        format: record.format,
        bitrate_kbps: record.bitrate_kbps,
        metadata: spec.metadata,
        cover: spec.cover,
        chapters,
    })
}

/// Оновлює прогрес у пам’яті та на диску між главами.
struct ProgressObserver {
    id: String,
    state: Arc<Mutex<HashMap<String, ExportRecord>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    dir: PathBuf,
}

impl ProgressObserver {
    fn update(&self, index: usize, total: usize, finished: bool) {
        let record = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(record) = state.get_mut(&self.id) else {
                return;
            };
            if record.status != ExportStatus::Running {
                return;
            }
            record.chapters_done = if finished { index + 1 } else { index };
            record.progress = record
                .chapters_done
                .saturating_mul(100)
                .checked_div(total)
                .unwrap_or(0)
                .min(99) as u8;
            record.clone()
        };
        if let Err(error) = save_record(&self.dir, &record) {
            tracing::warn!(
                export_id = %self.id,
                error = %error,
                "cannot update export progress"
            );
        }
    }
}

impl ExportObserver for ProgressObserver {
    fn chapter_started(&mut self, index: usize, total: usize) {
        self.update(index, total, false);
    }

    fn chapter_finished(&mut self, index: usize, total: usize) {
        self.update(index, total, true);
    }

    fn is_cancelled(&self) -> bool {
        self.cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&self.id)
    }
}

fn decode_cover(raw: Option<String>) -> Result<Option<CoverImage>, ExportApiError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let payload = raw
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(',').map(|(_, data)| data))
        .unwrap_or(raw);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|error| ExportApiError::bad(format!("invalid cover_base64: {error}")))?;
    if bytes.len() > MAX_COVER_BYTES {
        return Err(ExportApiError::bad(format!(
            "cover is too large: {} bytes (max {MAX_COVER_BYTES})",
            bytes.len()
        )));
    }
    CoverImage::new(bytes)
        .map(Some)
        .map_err(|error| ExportApiError::bad(error.to_string()))
}

fn valid_export_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_EXPORT_ID_LEN
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.contains('/')
        && !name.contains('\\')
        && Path::new(name).file_name().and_then(|value| value.to_str()) == Some(name)
}

fn save_record(dir: &Path, record: &ExportRecord) -> std::io::Result<()> {
    let path = dir.join(format!("{}.json", record.id));
    let tmp = dir.join(format!("{}.json.tmp", record.id));
    let bytes = serde_json::to_vec_pretty(record)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)
}

fn remove_file_if_exists(path: &Path) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), error = %error, "cannot remove export file");
        }
    }
}

/// Позначити незавершені експорти failed і прибрати `.part` після рестарту.
pub fn reconcile_exports(projects: &ProjectStore) -> anyhow::Result<usize> {
    let mut changed = 0;
    for entry in std::fs::read_dir(projects.dir())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let project_id = entry.file_name().to_string_lossy().into_owned();
        if !valid_project_id(&project_id) {
            continue;
        }
        let exports_dir = entry.path().join("exports");
        if !exports_dir.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&exports_dir)? {
            let file = file?;
            let path = file.path();
            if path.extension().and_then(|value| value.to_str()) == Some("part") {
                remove_file_if_exists(&path);
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(mut record) = serde_json::from_slice::<ExportRecord>(&bytes) else {
                tracing::warn!(path = %path.display(), "skipping malformed export record");
                continue;
            };
            let mut dirty = false;
            if record.status.is_active() {
                record.status = ExportStatus::Failed;
                record.error = Some("export interrupted by server restart".to_string());
                record.finished_at = Some(now_rfc3339());
                record.progress = record.progress.min(99);
                dirty = true;
            } else if record.status == ExportStatus::Completed {
                let audio_missing = record
                    .audio_file
                    .as_deref()
                    .map(|name| !exports_dir.join(name).is_file())
                    .unwrap_or(true);
                if audio_missing {
                    record.status = ExportStatus::Failed;
                    record.error = Some("export file is missing".to_string());
                    dirty = true;
                }
            }
            if dirty {
                save_record(&exports_dir, &record)?;
                changed += 1;
            }
        }
    }
    Ok(changed)
}

fn load_all_records(projects: &ProjectStore) -> anyhow::Result<Vec<ExportRecord>> {
    let mut records = Vec::new();
    for entry in std::fs::read_dir(projects.dir())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let project_id = entry.file_name().to_string_lossy().into_owned();
        if !valid_project_id(&project_id) {
            continue;
        }
        let exports_dir = entry.path().join("exports");
        if !exports_dir.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&exports_dir)? {
            let file = file?;
            let path = file.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<ExportRecord>(&bytes).ok())
            {
                Some(record) if record.project_id == project_id => records.push(record),
                Some(record) => tracing::warn!(
                    path = %path.display(),
                    record_project = %record.project_id,
                    directory_project = %project_id,
                    "skipping export record in the wrong project directory"
                ),
                None => tracing::warn!(path = %path.display(), "skipping malformed export record"),
            }
        }
    }
    Ok(records)
}

/// JSON-подання запису з URL-посиланнями та прапорцем `stale`.
fn export_view(
    record: &ExportRecord,
    project_updated_at: &str,
    base: &str,
) -> serde_json::Value {
    let mut value = serde_json::to_value(record).unwrap_or_else(|_| json!({}));
    let Some(object) = value.as_object_mut() else {
        return value;
    };
    object.insert(
        "stale".to_string(),
        json!(record.project_updated_at != project_updated_at),
    );
    if record.status == ExportStatus::Completed {
        let base = base.trim_end_matches('/');
        object.insert(
            "audio_url".to_string(),
            json!(format!(
                "{base}/v1/projects/{}/exports/{}/audio",
                record.project_id, record.id
            )),
        );
        object.insert(
            "subtitles_srt_url".to_string(),
            json!(format!(
                "{base}/v1/projects/{}/exports/{}/subtitles?format=srt",
                record.project_id, record.id
            )),
        );
        object.insert(
            "subtitles_vtt_url".to_string(),
            json!(format!(
                "{base}/v1/projects/{}/exports/{}/subtitles?format=vtt",
                record.project_id, record.id
            )),
        );
    }
    value
}

/// `POST /v1/projects/{id}/exports`
pub async fn create_export(
    State(state): State<AppState>,
    AxumPath(project_id): AxumPath<String>,
    Json(request): Json<CreateExportRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ExportApiError> {
    let record = state.exports.submit(&project_id, request).await?;
    let project = state
        .projects
        .load(&project_id)
        .map_err(|error| ExportApiError::internal("load project", error))?
        .ok_or(ExportApiError::NotFound("unknown project"))?;
    let view = export_view(&record, &project.updated_at, &state.public_url);
    Ok((StatusCode::ACCEPTED, Json(view)))
}

/// `GET /v1/projects/{id}/exports`
pub async fn list_exports(
    State(state): State<AppState>,
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ExportApiError> {
    let project = state
        .projects
        .load(&project_id)
        .map_err(|error| ExportApiError::internal("load project", error))?
        .ok_or(ExportApiError::NotFound("unknown project"))?;
    let records = state.exports.list(&project_id)?;
    let exports: Vec<serde_json::Value> = records
        .iter()
        .map(|record| export_view(record, &project.updated_at, &state.public_url))
        .collect();
    Ok(Json(json!({ "exports": exports })))
}

/// `GET /v1/projects/{id}/exports/{export_id}`
pub async fn get_export(
    State(state): State<AppState>,
    AxumPath((project_id, export_id)): AxumPath<(String, String)>,
) -> Result<Json<serde_json::Value>, ExportApiError> {
    let record = state.exports.get(&project_id, &export_id)?;
    let project = state
        .projects
        .load(&project_id)
        .map_err(|error| ExportApiError::internal("load project", error))?
        .ok_or(ExportApiError::NotFound("unknown project"))?;
    Ok(Json(export_view(
        &record,
        &project.updated_at,
        &state.public_url,
    )))
}

/// `DELETE /v1/projects/{id}/exports/{export_id}`
pub async fn delete_export(
    State(state): State<AppState>,
    AxumPath((project_id, export_id)): AxumPath<(String, String)>,
) -> Result<StatusCode, ExportApiError> {
    state.exports.delete(&project_id, &export_id)?;
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /v1/projects/{id}/exports/{export_id}/audio`
pub async fn export_audio(
    State(state): State<AppState>,
    AxumPath((project_id, export_id)): AxumPath<(String, String)>,
) -> Result<Response, ExportApiError> {
    let record = state.exports.get(&project_id, &export_id)?;
    if record.status != ExportStatus::Completed {
        return Err(ExportApiError::NotReady(format!(
            "export is not completed (status: {})",
            record.status.as_str()
        )));
    }
    let name = record
        .audio_file
        .as_deref()
        .ok_or_else(|| ExportApiError::internal("export audio", "audio file is not recorded"))?;
    let path = state.exports.file_path(&project_id, name)?;
    let file = tokio::fs::File::open(&path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ExportApiError::NotFound("export audio is missing")
        } else {
            ExportApiError::internal("open export audio", error)
        }
    })?;
    let metadata = file
        .metadata()
        .await
        .map_err(|error| ExportApiError::internal("read export audio metadata", error))?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(record.format.mime()),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"promin-{}.{}\"",
            record.id,
            record.format.extension()
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    if let Ok(length) = HeaderValue::from_str(&metadata.len().to_string()) {
        headers.insert(header::CONTENT_LENGTH, length);
    }
    let stream = ReaderStream::new(file);
    Ok((headers, Body::from_stream(stream)).into_response())
}

/// Параметри субтитрів експорту.
#[derive(Debug, Deserialize)]
pub struct ExportSubtitleQuery {
    #[serde(default)]
    pub format: Option<String>,
}

/// `GET /v1/projects/{id}/exports/{export_id}/subtitles?format=srt|vtt`
pub async fn export_subtitles(
    State(state): State<AppState>,
    AxumPath((project_id, export_id)): AxumPath<(String, String)>,
    Query(query): Query<ExportSubtitleQuery>,
) -> Result<Response, ExportApiError> {
    let record = state.exports.get(&project_id, &export_id)?;
    if record.status != ExportStatus::Completed {
        return Err(ExportApiError::NotReady(format!(
            "export is not completed (status: {})",
            record.status.as_str()
        )));
    }
    let format = query.format.unwrap_or_else(|| "srt".to_string());
    let format = format.trim().to_ascii_lowercase();
    let (name, content_type) = match format.as_str() {
        "srt" => (
            record.subtitles_srt_file.as_deref(),
            "application/x-subrip; charset=utf-8",
        ),
        "vtt" | "webvtt" => (
            record.subtitles_vtt_file.as_deref(),
            "text/vtt; charset=utf-8",
        ),
        other => {
            return Err(ExportApiError::bad(format!(
                "unsupported subtitle format: {other}"
            )))
        }
    };
    let name = name.ok_or(ExportApiError::NotFound(
        "subtitles are missing for this export",
    ))?;
    let path = state.exports.file_path(&project_id, name)?;
    let bytes = tokio::fs::read(&path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ExportApiError::NotFound("subtitles are missing for this export")
        } else {
            ExportApiError::internal("read export subtitles", error)
        }
    })?;
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"promin-{}.{}\"",
            record.id,
            if format == "srt" { "srt" } else { "vtt" }
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    Ok((headers, bytes).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use speech_core::projects::Chapter;
    use speech_export::default_cover;

    fn temp_projects() -> ProjectStore {
        let dir = std::env::temp_dir().join(format!(
            "promin-export-api-{}",
            uuid::Uuid::new_v4()
        ));
        ProjectStore::new(dir).unwrap()
    }

    #[test]
    fn export_id_validation() {
        assert!(valid_export_id("7f0c7f9a-6a4a-4c3e-9a2e-4f8f0b1c2d3e"));
        assert!(!valid_export_id(""));
        assert!(!valid_export_id("../escape"));
        assert!(!valid_export_id("with space"));
    }

    #[test]
    fn safe_file_names_reject_paths() {
        assert!(safe_file_name("export.m4b"));
        assert!(!safe_file_name("../export.m4b"));
        assert!(!safe_file_name("dir/export.m4b"));
        assert!(!safe_file_name(""));
    }

    #[test]
    fn reconcile_marks_interrupted_exports_failed() {
        let store = temp_projects();
        let mut project = Project::new("Книга");
        project.chapters.push(Chapter::new(0, "Розділ", "Текст"));
        project.normalize();
        store.create(&project).unwrap();
        let exports = store.dir().join(&project.id).join("exports");
        std::fs::create_dir_all(&exports).unwrap();
        let record = ExportRecord {
            id: "11111111-1111-1111-1111-111111111111".to_string(),
            project_id: project.id.clone(),
            format: ExportFormat::M4b,
            status: ExportStatus::Running,
            progress: 40,
            chapters_total: 2,
            chapters_done: 1,
            bitrate_kbps: 96,
            project_updated_at: project.updated_at.clone(),
            created_at: now_rfc3339(),
            started_at: Some(now_rfc3339()),
            finished_at: None,
            duration_ms: None,
            audio_bytes: None,
            audio_file: None,
            subtitles_srt_file: None,
            subtitles_vtt_file: None,
            error: None,
        };
        save_record(&exports, &record).unwrap();
        std::fs::write(exports.join(format!("{}.part", record.id)), b"partial").unwrap();

        let changed = reconcile_exports(&store).unwrap();
        assert_eq!(changed, 1);
        let bytes = std::fs::read(exports.join(format!("{}.json", record.id))).unwrap();
        let restored: ExportRecord = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.status, ExportStatus::Failed);
        assert!(restored.error.unwrap().contains("restart"));
        assert!(!exports.join(format!("{}.part", record.id)).exists());
    }

    #[test]
    fn cover_decoding_accepts_data_uri() {
        let cover = default_cover();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&cover.data);
        let decoded = decode_cover(Some(format!("data:image/png;base64,{encoded}")))
            .unwrap()
            .unwrap();
        assert_eq!(decoded.mime, "image/png");
        assert_eq!(decoded.data, cover.data);
    }

    fn write_test_wav(path: &Path, samples: usize) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for index in 0..samples {
            let value = ((index as f32 * 0.05).sin() * 12_000.0) as i16;
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn completed_project(store: &ProjectStore, id: &str) -> Project {
        let mut project = Project::new("Тестова книга");
        project.id = id.to_string();
        project.author = "Автор".to_string();
        project.chapters = vec![
            Chapter::new(0, "Розділ 1", "Перший текст"),
            Chapter::new(1, "Розділ 2", "Другий текст"),
        ];
        project.normalize();
        for chapter in &mut project.chapters {
            let text = chapter.text.clone();
            chapter.mark_completed(
                48_000,
                1_000,
                None,
                vec![speech_core::projects::ChunkTiming {
                    start_ms: 0,
                    end_ms: 900,
                    text,
                }],
            );
        }
        store.create(&project).unwrap();
        for index in 0..project.chapters.len() {
            let path = store.chapter_audio_path(id, index).unwrap();
            write_test_wav(&path, 24_000);
        }
        project
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_incomplete_chapters() {
        let store = Arc::new(temp_projects());
        let id = uuid::Uuid::new_v4().to_string();
        let mut project = Project::new("Неповна книга");
        project.id = id.clone();
        project.chapters = vec![
            Chapter::new(0, "Розділ 1", "Текст"),
            Chapter::new(1, "Розділ 2", "Текст"),
        ];
        project.normalize();
        let text = project.chapters[0].text.clone();
        project.chapters[0].mark_completed(
            24_000,
            1_000,
            None,
            vec![speech_core::projects::ChunkTiming {
                start_ms: 0,
                end_ms: 900,
                text,
            }],
        );
        store.create(&project).unwrap();
        write_test_wav(&store.chapter_audio_path(&id, 0).unwrap(), 24_000);

        let manager = ExportManager::new(Arc::clone(&store), 1).unwrap();
        let error = manager
            .submit(
                &id,
                CreateExportRequest {
                    format: "mp3".to_string(),
                    bitrate_kbps: None,
                    cover_base64: None,
                    title: None,
                    author: None,
                    album: None,
                    narrator: None,
                    year: None,
                    description: None,
                },
            )
            .await
            .unwrap_err();
        match error {
            ExportApiError::Conflict {
                pending_chapters, ..
            } => assert_eq!(pending_chapters, vec![1]),
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(manager.list(&id).unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_manager_runs_full_cycle() {
        let store = Arc::new(temp_projects());
        let id = uuid::Uuid::new_v4().to_string();
        completed_project(&store, &id);
        let manager = ExportManager::new(Arc::clone(&store), 1).unwrap();
        let record = manager
            .submit(
                &id,
                CreateExportRequest {
                    format: "m4b".to_string(),
                    bitrate_kbps: None,
                    cover_base64: None,
                    title: None,
                    author: None,
                    album: None,
                    narrator: Some("Диктор".to_string()),
                    year: None,
                    description: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(record.status, ExportStatus::Queued);

        let mut finished = None;
        for _ in 0..200 {
            let current = manager.get(&id, &record.id).unwrap();
            if !current.status.is_active() {
                finished = Some(current);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        let finished = finished.unwrap_or_else(|| {
            let current = manager.get(&id, &record.id).unwrap();
            panic!("export did not finish in time: {current:?}")
        });
        assert_eq!(finished.status, ExportStatus::Completed, "{:?}", finished.error);
        assert_eq!(finished.progress, 100);
        assert_eq!(finished.chapters_done, 2);
        assert!(finished.audio_bytes.unwrap() > 0);
        assert!(finished.duration_ms.unwrap() >= 2_000);

        let dir = store.dir().join(&id).join("exports");
        let audio = dir.join(finished.audio_file.clone().unwrap());
        let srt = dir.join(finished.subtitles_srt_file.clone().unwrap());
        let vtt = dir.join(finished.subtitles_vtt_file.clone().unwrap());
        assert!(audio.is_file());
        assert!(srt.is_file());
        assert!(vtt.is_file());
        let srt_text = std::fs::read_to_string(&srt).unwrap();
        assert!(srt_text.contains("Перший текст"));
        assert!(srt_text.contains("Другий текст"));

        manager.delete(&id, &record.id).unwrap();
        assert!(!audio.exists());
        assert!(!srt.exists());
        assert!(!vtt.exists());
    }
}
