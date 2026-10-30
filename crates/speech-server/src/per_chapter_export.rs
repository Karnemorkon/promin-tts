//! Асинхронний експорт окремих глав книги у власні аудіофайли.
//!
//! Кожна глава записується як `<job-id>-chapter-<n>.<ext>` (M4B/MP3/OGG/FLAC)
//! у `<projects>/<id>/exports/per-chapter/`. За бажанням генеруються SRT/VTT/ASS
//! субтитри з `chunk_timings` глави. Job-и асинхронні, зберігаються на диск,
//! переживають рестарт і кооперативно скасовуються.
//!
//! Використовує `speech_export::export_book` з однією главою на вході, тому
//! кодування (включаючи M4B edit-list, FLAC Vorbis comments і Picture-блок)
//! повторює логіку повного експорту аудіокниги.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use speech_core::projects::{now_rfc3339, valid_project_id, ChapterStatus, ProjectStore};
use speech_export::{
    combined_ass, combined_subtitles, Cue, ExportChapter, ExportFormat, ExportInput, ExportMetadata,
};
use tokio::sync::{mpsc, Semaphore};
use tokio_util::io::ReaderStream;

use crate::AppState;

/// Максимальна довжина id job-а.
const MAX_PER_CHAPTER_ID_LEN: usize = 64;
/// Каталог записів експорту по главах: `<project>/exports/per-chapter/`.
const PER_CHAPTER_SUBDIR: &str = "per-chapter";

/// Статус job-а експорту по главах.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Результат експорту однієї глави.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterExportResult {
    pub chapter_index: usize,
    pub status: JobStatus,
    pub audio_file: Option<String>,
    pub subtitles_srt_file: Option<String>,
    pub subtitles_vtt_file: Option<String>,
    pub subtitles_ass_file: Option<String>,
    pub duration_ms: Option<u64>,
    pub bytes: Option<u64>,
    pub error: Option<String>,
}

/// Запис job-а експорту по главах (серіалізується на диск).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerChapterExportJob {
    pub id: String,
    pub project_id: String,
    pub format: ExportFormat,
    pub bitrate_kbps: u32,
    /// Порожньо — всі завершені глави проєкту.
    pub chapters: Vec<usize>,
    pub include_subtitles: bool,
    pub status: JobStatus,
    pub progress: u8,
    pub chapters_total: usize,
    pub chapters_done: usize,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<ChapterExportResult>,
}

/// Запит `POST /v1/projects/{id}/exports/per-chapter`.
#[derive(Debug, Deserialize)]
pub struct CreatePerChapterExportRequest {
    pub format: String,
    #[serde(default)]
    pub bitrate_kbps: Option<u32>,
    /// Індекси глави для експорту; порожньо — всі завершені глави.
    #[serde(default)]
    pub chapters: Option<Vec<usize>>,
    #[serde(default)]
    pub include_subtitles: bool,
}

/// Помилка API експорту по главах.
#[derive(Debug)]
pub enum PerChapterExportApiError {
    NotFound(&'static str),
    BadRequest(String),
    Conflict(String),
    NotReady(String),
    Internal(String),
}

impl PerChapterExportApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(context: &'static str, error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, context, "per-chapter export API error");
        Self::Internal(error.to_string())
    }
}

impl IntoResponse for PerChapterExportApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message.to_string()),
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::Conflict(message) => (StatusCode::CONFLICT, message),
            Self::NotReady(message) => (StatusCode::CONFLICT, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

/// Параметри глави, зняті на момент запуску (не серіалізуються).
#[derive(Clone)]
struct PerChapterExportSpec {
    metadata: ExportMetadata,
}

/// Черга експорту по главах.
pub struct PerChapterExportManager {
    projects: Arc<ProjectStore>,
    state: Arc<Mutex<HashMap<String, PerChapterExportJob>>>,
    specs: Arc<Mutex<HashMap<String, PerChapterExportSpec>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    /// Job-и, які видалено або чий проєкт видалено: забороняємо повторний persist.
    discard: Arc<Mutex<HashSet<String>>>,
    sender: mpsc::Sender<String>,
}

impl PerChapterExportManager {
    /// Створити чергу, відновити записи й позначити перервані job-и.
    pub fn new(projects: Arc<ProjectStore>, concurrency: usize) -> anyhow::Result<Arc<Self>> {
        let reconciled = reconcile_jobs(&projects)?;
        if reconciled > 0 {
            tracing::warn!(
                reconciled,
                "marked interrupted per-chapter exports as failed"
            );
        }

        let mut state = HashMap::new();
        for job in load_all_jobs(&projects)? {
            state.insert(job.id.clone(), job);
        }

        let (sender, mut receiver) = mpsc::channel::<String>(256);
        let manager = Arc::new(Self {
            projects,
            state: Arc::new(Mutex::new(state)),
            specs: Arc::new(Mutex::new(HashMap::new())),
            cancel: Arc::new(Mutex::new(HashSet::new())),
            discard: Arc::new(Mutex::new(HashSet::new())),
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

    /// Поставити експорт глави(ей) у чергу.
    pub async fn submit(
        &self,
        project_id: &str,
        request: CreatePerChapterExportRequest,
    ) -> Result<PerChapterExportJob, PerChapterExportApiError> {
        if !valid_project_id(project_id) {
            return Err(PerChapterExportApiError::bad("invalid project id"));
        }
        let format = ExportFormat::parse(&request.format)
            .ok_or_else(|| PerChapterExportApiError::bad("format must be m4b, mp3, ogg or flac"))?;
        let bitrate_kbps = format
            .resolve_bitrate_kbps(request.bitrate_kbps.unwrap_or(0))
            .map_err(|error| PerChapterExportApiError::bad(error.to_string()))?;
        let project = self
            .projects
            .load(project_id)
            .map_err(|error| PerChapterExportApiError::internal("load project", error))?
            .ok_or(PerChapterExportApiError::NotFound("unknown project"))?;

        let chapters: Vec<usize> = match request.chapters {
            Some(chapters) if !chapters.is_empty() => {
                let invalid: Vec<usize> = chapters
                    .iter()
                    .filter(|&&idx| idx >= project.chapters.len())
                    .copied()
                    .collect();
                if !invalid.is_empty() {
                    return Err(PerChapterExportApiError::bad(format!(
                        "chapter index(es) out of range: {:?}",
                        invalid
                    )));
                }
                let mut unique = Vec::with_capacity(chapters.len());
                for index in chapters {
                    if !unique.contains(&index) {
                        unique.push(index);
                    }
                }
                let pending: Vec<usize> = unique
                    .iter()
                    .filter(|&&idx| {
                        project.chapters[idx].status != ChapterStatus::Completed
                            || !self.projects.chapter_audio_exists(project_id, idx)
                    })
                    .copied()
                    .collect();
                if !pending.is_empty() {
                    return Err(PerChapterExportApiError::Conflict(format!(
                        "chapter(s) must be completed before export: {pending:?}"
                    )));
                }
                unique
            }
            None | Some(_) => {
                let completed: Vec<usize> = project
                    .chapters
                    .iter()
                    .filter(|c| c.status == ChapterStatus::Completed)
                    .map(|c| c.index)
                    .collect();
                let missing_audio: Vec<usize> = completed
                    .iter()
                    .copied()
                    .filter(|&idx| !self.projects.chapter_audio_exists(project_id, idx))
                    .collect();
                if !missing_audio.is_empty() {
                    return Err(PerChapterExportApiError::Conflict(format!(
                        "chapter audio is missing: {missing_audio:?}"
                    )));
                }
                completed
            }
        };
        if chapters.is_empty() {
            return Err(PerChapterExportApiError::bad(
                "no completed chapters to export",
            ));
        }

        let now = now_rfc3339();
        let job = PerChapterExportJob {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            format,
            bitrate_kbps,
            chapters: chapters.clone(),
            include_subtitles: request.include_subtitles,
            status: JobStatus::Queued,
            progress: 0,
            chapters_total: chapters.len(),
            chapters_done: 0,
            created_at: now.clone(),
            started_at: None,
            finished_at: None,
            error: None,
            results: Vec::new(),
        };

        let dir = self.job_dir(project_id)?;
        std::fs::create_dir_all(&dir)
            .map_err(|error| PerChapterExportApiError::internal("create job directory", error))?;
        save_job(&dir, &job).map_err(|error| {
            PerChapterExportApiError::internal("save per-chapter export job", error)
        })?;
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job.id.clone(), job.clone());
        self.specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                job.id.clone(),
                PerChapterExportSpec {
                    metadata: ExportMetadata {
                        title: project.title.clone(),
                        author: project.author.clone(),
                        album: project.title.clone(),
                        narrator: None,
                        year: now.get(..4).map(str::to_string),
                        description: None,
                        language: project.language.clone(),
                    },
                },
            );

        if self.sender.send(job.id.clone()).await.is_err() {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&job.id);
            self.specs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&job.id);
            remove_file_if_exists(&dir.join(format!("{}.json", job.id)));
            return Err(PerChapterExportApiError::internal(
                "enqueue per-chapter export",
                "per-chapter export queue is closed",
            ));
        }
        tracing::info!(
            job_id = %job.id,
            project_id,
            format = format.as_str(),
            chapters = job.chapters_total,
            "per-chapter export queued"
        );
        Ok(job)
    }

    /// Отримати job за id.
    pub fn get(
        &self,
        project_id: &str,
        job_id: &str,
    ) -> Result<PerChapterExportJob, PerChapterExportApiError> {
        if !valid_project_id(project_id) {
            return Err(PerChapterExportApiError::bad("invalid project id"));
        }
        if !valid_per_chapter_id(job_id) {
            return Err(PerChapterExportApiError::bad(
                "invalid per-chapter export id",
            ));
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
        let dir = self.job_dir(project_id)?;
        let path = dir.join(format!("{job_id}.json"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(PerChapterExportApiError::NotFound(
                    "unknown per-chapter export",
                ))
            }
            Err(error) => {
                return Err(PerChapterExportApiError::internal(
                    "read per-chapter export job",
                    error,
                ))
            }
        };
        let job: PerChapterExportJob = serde_json::from_slice(&bytes).map_err(|error| {
            PerChapterExportApiError::internal("parse per-chapter export job", error)
        })?;
        if job.project_id != project_id {
            return Err(PerChapterExportApiError::NotFound(
                "unknown per-chapter export",
            ));
        }
        Ok(job)
    }

    /// Список job-ів проєкту (новіші першими).
    pub fn list(
        &self,
        project_id: &str,
    ) -> Result<Vec<PerChapterExportJob>, PerChapterExportApiError> {
        if !valid_project_id(project_id) {
            return Err(PerChapterExportApiError::bad("invalid project id"));
        }
        if !self.projects.exists(project_id) {
            return Err(PerChapterExportApiError::NotFound("unknown project"));
        }
        let dir = self.job_dir(project_id)?;
        let mut jobs: Vec<PerChapterExportJob> = Vec::new();
        if dir.is_dir() {
            let entries = std::fs::read_dir(&dir).map_err(|error| {
                PerChapterExportApiError::internal("list per-chapter exports", error)
            })?;
            for entry in entries {
                let entry = entry.map_err(|error| {
                    PerChapterExportApiError::internal("list per-chapter exports", error)
                })?;
                let path = entry.path();
                if path.extension().and_then(|value| value.to_str()) != Some("json") {
                    continue;
                }
                match std::fs::read(&path)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<PerChapterExportJob>(&bytes).ok())
                {
                    Some(job) => jobs.push(job),
                    None => {
                        tracing::warn!(path = %path.display(), "skipping malformed per-chapter export job")
                    }
                }
            }
        }
        let live = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for job in &mut jobs {
            if let Some(current) = live.get(&job.id) {
                *job = current.clone();
            }
        }
        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(jobs)
    }

    /// Скасувати всі job-и проєкту перед його видаленням.
    pub fn cancel_project(&self, project_id: &str) {
        let jobs: Vec<PerChapterExportJob> = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|job| job.project_id == project_id)
            .cloned()
            .collect();
        {
            let mut cancel = self
                .cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut discard = self
                .discard
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for job in &jobs {
                cancel.insert(job.id.clone());
                discard.insert(job.id.clone());
            }
        }
        // Каталог проєкту буде видалено: прибираємо job-и й spec-и з пам'яті,
        // не намагаючись нічого дописати на диск.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|_, job| job.project_id != project_id);
        let mut specs = self
            .specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for job in &jobs {
            specs.remove(&job.id);
        }
    }

    /// Видалити job і всі його файли; активний job скасовується.
    pub fn delete(&self, project_id: &str, job_id: &str) -> Result<(), PerChapterExportApiError> {
        let record = self.get(project_id, job_id)?;
        if record.status.is_active() {
            self.cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(job_id.to_string());
        }
        self.discard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job_id.to_string());
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
        self.specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);

        let dir = self.job_dir(project_id)?;
        remove_file_if_exists(&dir.join(format!("{job_id}.json")));
        remove_file_if_exists(&dir.join(format!("{job_id}.json.tmp")));
        for result in &record.results {
            remove_result_files(&dir, result);
        }
        // Прибираємо також недописані `.part` та осиротілі файли цього job-а.
        let prefix = format!("{job_id}-chapter-");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&prefix) && safe_file_name(&name) {
                    remove_file_if_exists(&entry.path());
                }
            }
        }
        Ok(())
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
        if job.status != JobStatus::Queued || self.is_discarded(id) {
            return;
        }
        if self.is_cancelled(id) {
            self.finish_cancelled(&mut job);
            return;
        }
        let dir = match self.job_dir(&job.project_id) {
            Ok(dir) => dir,
            Err(error) => {
                self.fail(&mut job, format!("cannot resolve job directory: {error:?}"));
                return;
            }
        };
        job.status = JobStatus::Running;
        job.started_at = Some(now_rfc3339());
        job.progress = 2;
        self.persist(&job);

        let project = match self.projects.load(&job.project_id) {
            Ok(Some(project)) => project,
            Ok(None) => {
                self.fail(&mut job, "project was deleted during per-chapter export");
                return;
            }
            Err(error) => {
                self.fail(&mut job, format!("cannot load project: {error}"));
                return;
            }
        };

        let spec = self
            .specs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned();

        let chapters_to_export: Vec<&speech_core::projects::Chapter> = if job.chapters.is_empty() {
            project
                .chapters
                .iter()
                .filter(|c| c.status == ChapterStatus::Completed)
                .collect()
        } else {
            let unavailable: Vec<usize> = job
                .chapters
                .iter()
                .copied()
                .filter(|&idx| {
                    project
                        .chapters
                        .get(idx)
                        .map(|chapter| chapter.status != ChapterStatus::Completed)
                        .unwrap_or(true)
                })
                .collect();
            if !unavailable.is_empty() {
                self.fail(
                    &mut job,
                    format!("chapter(s) are no longer completed: {unavailable:?}"),
                );
                return;
            }
            job.chapters
                .iter()
                .filter_map(|&idx| project.chapters.get(idx))
                .collect()
        };

        job.chapters_total = chapters_to_export.len();
        if job.chapters_total == 0 {
            self.fail(&mut job, "no completed chapters to export");
            return;
        }
        self.persist(&job);

        for (i, chapter) in chapters_to_export.iter().enumerate() {
            if self.is_discarded(id) {
                return;
            }
            if self.is_cancelled(id) {
                self.finish_cancelled(&mut job);
                return;
            }
            let chapter_index = chapter.index;
            let job_id = job.id.clone();
            let spec_clone = spec.clone();
            let dir_clone = dir.clone();
            let project_id_owned = job.project_id.clone();
            let projects_clone = Arc::clone(&self.projects);
            let format = job.format;
            let bitrate_kbps = job.bitrate_kbps;
            let include_subtitles = job.include_subtitles;

            let result = match tokio::task::spawn_blocking(move || -> ChapterExportResult {
                let chapter_result = export_single_chapter(
                    &projects_clone,
                    &project_id_owned,
                    &job_id,
                    chapter_index,
                    format,
                    bitrate_kbps,
                    include_subtitles,
                    spec_clone.as_ref(),
                    &dir_clone,
                );
                match chapter_result {
                    Ok(result) => result,
                    Err(error) => ChapterExportResult {
                        chapter_index,
                        status: JobStatus::Failed,
                        audio_file: None,
                        subtitles_srt_file: None,
                        subtitles_vtt_file: None,
                        subtitles_ass_file: None,
                        duration_ms: None,
                        bytes: None,
                        error: Some(error),
                    },
                }
            })
            .await
            {
                Ok(result) => result,
                Err(error) => ChapterExportResult {
                    chapter_index,
                    status: JobStatus::Failed,
                    audio_file: None,
                    subtitles_srt_file: None,
                    subtitles_vtt_file: None,
                    subtitles_ass_file: None,
                    duration_ms: None,
                    bytes: None,
                    error: Some(format!("chapter export task failed: {error}")),
                },
            };

            if self.is_discarded(id) {
                remove_result_files(&dir, &result);
                return;
            }
            job.results.push(result);
            job.chapters_done = i + 1;
            job.progress = (2 + (job.chapters_done as u64 * 95 / job.chapters_total.max(1) as u64))
                .min(98) as u8;
            self.persist(&job);
        }

        if self.is_discarded(id) {
            return;
        }
        if self.is_cancelled(id) {
            self.finish_cancelled(&mut job);
            return;
        }

        let failed: Vec<usize> = job
            .results
            .iter()
            .filter(|result| result.status != JobStatus::Completed)
            .map(|result| result.chapter_index)
            .collect();
        job.finished_at = Some(now_rfc3339());
        job.progress = 100;
        if failed.is_empty() {
            job.status = JobStatus::Completed;
            self.persist(&job);
            tracing::info!(
                job_id = %job.id,
                project_id = %job.project_id,
                chapters = job.chapters_total,
                "per-chapter export completed"
            );
        } else {
            job.status = JobStatus::Failed;
            job.error = Some(format!(
                "{} of {} chapter(s) failed: {:?}",
                failed.len(),
                job.chapters_total,
                failed
            ));
            self.persist(&job);
            tracing::warn!(
                job_id = %job.id,
                project_id = %job.project_id,
                failed = failed.len(),
                "per-chapter export finished with failures"
            );
        }
    }

    fn snapshot(&self, id: &str) -> Option<PerChapterExportJob> {
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

    fn is_discarded(&self, id: &str) -> bool {
        self.discard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(id)
    }

    fn persist(&self, job: &PerChapterExportJob) {
        // Тримаємо `discard` під час запису, щоб DELETE не міг змагатися з
        // persist і «воскресити» вже видалений job.
        let discard = self
            .discard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if discard.contains(&job.id) {
            return;
        }
        if let Ok(dir) = self.job_dir(&job.project_id) {
            if let Err(error) = save_job(&dir, job) {
                tracing::error!(job_id = %job.id, error = %error, "cannot save per-chapter export job");
            }
        }
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job.id.clone(), job.clone());
    }

    fn fail(&self, job: &mut PerChapterExportJob, error: impl Into<String>) {
        let error = error.into();
        tracing::warn!(job_id = %job.id, error = %error, "per-chapter export failed");
        job.status = JobStatus::Failed;
        job.finished_at = Some(now_rfc3339());
        job.error = Some(error);
        self.persist(job);
    }

    fn finish_cancelled(&self, job: &mut PerChapterExportJob) {
        job.status = JobStatus::Cancelled;
        job.finished_at = Some(now_rfc3339());
        self.persist(job);
    }

    fn job_dir(&self, project_id: &str) -> Result<PathBuf, PerChapterExportApiError> {
        if !valid_project_id(project_id) {
            return Err(PerChapterExportApiError::bad("invalid project id"));
        }
        Ok(self
            .projects
            .dir()
            .join(project_id)
            .join("exports")
            .join(PER_CHAPTER_SUBDIR))
    }
}

/// Експортувати одну главу у окремий аудіо- та (опційно) субтитрові файли.
#[allow(clippy::too_many_arguments)]
fn export_single_chapter(
    projects: &ProjectStore,
    project_id: &str,
    job_id: &str,
    chapter_index: usize,
    format: ExportFormat,
    bitrate_kbps: u32,
    include_subtitles: bool,
    spec: Option<&PerChapterExportSpec>,
    dir: &Path,
) -> Result<ChapterExportResult, String> {
    let project = projects
        .load(project_id)
        .map_err(|error| format!("cannot load project: {error}"))?
        .ok_or_else(|| "project does not exist".to_string())?;
    let chapter = project
        .chapters
        .get(chapter_index)
        .ok_or_else(|| format!("chapter {chapter_index} does not exist"))?;
    if chapter.status != ChapterStatus::Completed {
        return Err(format!(
            "chapter {chapter_index} is not completed (status: {})",
            chapter.status.as_str()
        ));
    }
    let wav_path = projects
        .chapter_audio_path(project_id, chapter_index)
        .map_err(|error| format!("cannot resolve chapter audio: {error}"))?;
    if !wav_path.is_file() {
        return Err(format!("chapter {chapter_index} audio is missing"));
    }

    let prefix = format!("{job_id}-chapter-{chapter_index}");
    let audio_name = format!("{prefix}.{}", format.extension());
    let audio_path = dir.join(&audio_name);
    let part_path = dir.join(format!("{audio_name}.part"));

    let input = ExportInput {
        format,
        bitrate_kbps,
        metadata: spec.map(|s| s.metadata.clone()).unwrap_or_default(),
        cover: None,
        chapters: vec![ExportChapter {
            index: chapter_index,
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
        }],
    };

    let summary = (|| -> Result<speech_export::ExportSummary, String> {
        let file = std::fs::File::create(&part_path)
            .map_err(|error| format!("cannot create audio file: {error}"))?;
        let mut writer = std::io::BufWriter::new(file);
        let summary =
            speech_export::export_book(&input, &mut writer, &mut speech_export::NoopObserver)
                .map_err(|error| format!("export error: {error}"))?;
        writer
            .flush()
            .map_err(|error| format!("cannot flush audio file: {error}"))?;
        drop(writer);
        std::fs::rename(&part_path, &audio_path)
            .map_err(|error| format!("cannot finalize audio file: {error}"))?;
        Ok(summary)
    })();
    let summary = match summary {
        Ok(summary) => summary,
        Err(error) => {
            remove_file_if_exists(&part_path);
            return Err(error);
        }
    };

    let mut result = ChapterExportResult {
        chapter_index,
        status: JobStatus::Completed,
        audio_file: Some(audio_name.clone()),
        subtitles_srt_file: None,
        subtitles_vtt_file: None,
        subtitles_ass_file: None,
        duration_ms: Some(summary.duration_ms),
        bytes: Some(summary.bytes_written),
        error: None,
    };

    let write_subtitles = (|| -> Result<(), String> {
        if !include_subtitles || chapter.chunk_timings.is_empty() {
            return Ok(());
        }
        // Використовуємо таймлайн експорту: для M4B він уже враховує priming.
        let (srt, vtt) = combined_subtitles(&input.chapters, &summary.chapters);
        let srt_name = format!("{prefix}.srt");
        std::fs::write(dir.join(&srt_name), srt)
            .map_err(|error| format!("cannot write SRT subtitles: {error}"))?;
        result.subtitles_srt_file = Some(srt_name);

        let vtt_name = format!("{prefix}.vtt");
        std::fs::write(dir.join(&vtt_name), vtt)
            .map_err(|error| format!("cannot write VTT subtitles: {error}"))?;
        result.subtitles_vtt_file = Some(vtt_name);

        let ass_name = format!("{prefix}.ass");
        let ass = combined_ass(&input.chapters, &summary.chapters, &chapter.title);
        std::fs::write(dir.join(&ass_name), ass)
            .map_err(|error| format!("cannot write ASS subtitles: {error}"))?;
        result.subtitles_ass_file = Some(ass_name);
        Ok(())
    })();

    if let Err(error) = write_subtitles {
        remove_result_files(dir, &result);
        return Err(error);
    }

    Ok(result)
}

/// Записати job на диск (атомарно).
fn save_job(dir: &Path, job: &PerChapterExportJob) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", job.id));
    let tmp = dir.join(format!("{}.json.tmp", job.id));
    let bytes =
        serde_json::to_vec_pretty(job).map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn valid_per_chapter_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_PER_CHAPTER_ID_LEN
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn reconcile_jobs(projects: &ProjectStore) -> anyhow::Result<usize> {
    let mut reconciled = 0;
    for project_id in project_ids(projects)? {
        let dir = match projects
            .dir()
            .join(&project_id)
            .join("exports")
            .join(PER_CHAPTER_SUBDIR)
        {
            d if d.is_dir() => d,
            _ => continue,
        };
        for path in json_files(&dir)? {
            let mut job: PerChapterExportJob = match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            {
                Some(job) => job,
                None => {
                    tracing::warn!(path = %path.display(), "skipping malformed per-chapter export job");
                    continue;
                }
            };
            if job.project_id != project_id {
                tracing::warn!(path = %path.display(), "per-chapter export job project mismatch");
                continue;
            }
            if job.status.is_active() {
                job.status = JobStatus::Failed;
                job.finished_at = Some(now_rfc3339());
                job.error = Some("per-chapter export interrupted by server restart".to_string());
                save_job(&dir, &job).with_context(|| {
                    format!("save reconciled per-chapter export job {}", path.display())
                })?;
                reconciled += 1;
            }
        }
    }
    Ok(reconciled)
}

fn load_all_jobs(projects: &ProjectStore) -> anyhow::Result<Vec<PerChapterExportJob>> {
    let mut jobs = Vec::new();
    for project_id in project_ids(projects)? {
        let dir = projects
            .dir()
            .join(&project_id)
            .join("exports")
            .join(PER_CHAPTER_SUBDIR);
        for path in json_files(&dir)? {
            match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<PerChapterExportJob>(&bytes).ok())
            {
                Some(job) if job.project_id == project_id => jobs.push(job),
                Some(_) => {
                    tracing::warn!(path = %path.display(), "per-chapter export job project mismatch");
                }
                None => {
                    tracing::warn!(path = %path.display(), "skipping malformed per-chapter export job");
                }
            }
        }
    }
    Ok(jobs)
}

fn project_ids(projects: &ProjectStore) -> anyhow::Result<Vec<String>> {
    let mut ids = Vec::new();
    if !projects.dir().is_dir() {
        return Ok(ids);
    }
    for entry in std::fs::read_dir(projects.dir())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let project_id = entry.file_name().to_string_lossy().into_owned();
        if valid_project_id(&project_id) {
            ids.push(project_id);
        }
    }
    Ok(ids)
}

fn json_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
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

// ========================================================================
// Axum route handlers
// ========================================================================

/// `POST /v1/projects/{id}/exports/per-chapter`: запустити експорт глави(ей).
pub async fn create_per_chapter_export(
    State(state): State<AppState>,
    AxumPath(project_id): AxumPath<String>,
    Json(request): Json<CreatePerChapterExportRequest>,
) -> Result<(StatusCode, Json<PerChapterExportJob>), PerChapterExportApiError> {
    let job = state
        .per_chapter_exports
        .submit(&project_id, request)
        .await?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

/// `GET /v1/projects/{id}/exports/per-chapter`: історія job-ів проєкту.
pub async fn list_per_chapter_exports(
    State(state): State<AppState>,
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Vec<PerChapterExportJob>>, PerChapterExportApiError> {
    Ok(Json(state.per_chapter_exports.list(&project_id)?))
}

/// `GET /v1/projects/{id}/exports/per-chapter/{job_id}`: детальний звіт job-а.
pub async fn get_per_chapter_export(
    State(state): State<AppState>,
    AxumPath((project_id, job_id)): AxumPath<(String, String)>,
) -> Result<Json<PerChapterExportJob>, PerChapterExportApiError> {
    Ok(Json(state.per_chapter_exports.get(&project_id, &job_id)?))
}

/// `GET /v1/projects/{id}/exports/per-chapter/{job_id}/chapter/{index}/audio`.
pub async fn get_per_chapter_audio(
    State(state): State<AppState>,
    AxumPath((project_id, job_id, chapter_index)): AxumPath<(String, String, usize)>,
) -> Result<Response, PerChapterExportApiError> {
    let job = state.per_chapter_exports.get(&project_id, &job_id)?;
    let result = match job
        .results
        .iter()
        .find(|r| r.chapter_index == chapter_index)
    {
        Some(result) => result,
        None if job.status.is_active() => {
            return Err(PerChapterExportApiError::NotReady(format!(
                "chapter {chapter_index} has not been exported yet (status: {})",
                job.status.as_str()
            )));
        }
        None => {
            return Err(PerChapterExportApiError::NotFound(
                "chapter not found in this export job",
            ));
        }
    };
    if result.status != JobStatus::Completed {
        return Err(PerChapterExportApiError::NotReady(format!(
            "chapter {chapter_index} export did not complete: {}",
            result.error.as_deref().unwrap_or("unknown error")
        )));
    }
    let name = result.audio_file.as_deref().ok_or_else(|| {
        PerChapterExportApiError::internal("per-chapter audio", "audio file is not recorded")
    })?;
    if !safe_file_name(name) {
        return Err(PerChapterExportApiError::internal(
            "per-chapter audio",
            "invalid audio file name",
        ));
    }
    let dir = state.per_chapter_exports.job_dir(&project_id)?;
    let path = dir.join(name);
    let file = tokio::fs::File::open(&path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PerChapterExportApiError::NotFound("per-chapter audio is missing")
        } else {
            PerChapterExportApiError::internal("open per-chapter audio", error)
        }
    })?;
    let metadata = file.metadata().await.map_err(|error| {
        PerChapterExportApiError::internal("read per-chapter audio metadata", error)
    })?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(job.format.mime()),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"promin-{}-chapter-{}.{}\"",
            job.id,
            chapter_index,
            job.format.extension()
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    if let Ok(length) = HeaderValue::from_str(&metadata.len().to_string()) {
        headers.insert(header::CONTENT_LENGTH, length);
    }
    let stream = ReaderStream::new(file);
    Ok((headers, Body::from_stream(stream)).into_response())
}

/// Параметри субтитрів експорту по главах.
#[derive(Debug, Deserialize)]
pub struct PerChapterExportSubtitleQuery {
    #[serde(default)]
    pub format: Option<String>,
}

/// `GET /v1/projects/{id}/exports/per-chapter/{job_id}/chapter/{index}/subtitles?format=srt|vtt|ass`.
pub async fn get_per_chapter_subtitles(
    State(state): State<AppState>,
    AxumPath((project_id, job_id, chapter_index)): AxumPath<(String, String, usize)>,
    Query(query): Query<PerChapterExportSubtitleQuery>,
) -> Result<Response, PerChapterExportApiError> {
    let job = state.per_chapter_exports.get(&project_id, &job_id)?;
    let result = match job
        .results
        .iter()
        .find(|r| r.chapter_index == chapter_index)
    {
        Some(result) => result,
        None if job.status.is_active() => {
            return Err(PerChapterExportApiError::NotReady(format!(
                "chapter {chapter_index} has not been exported yet (status: {})",
                job.status.as_str()
            )));
        }
        None => {
            return Err(PerChapterExportApiError::NotFound(
                "chapter not found in this export job",
            ));
        }
    };
    if result.status != JobStatus::Completed {
        return Err(PerChapterExportApiError::NotReady(format!(
            "chapter {chapter_index} export did not complete: {}",
            result.error.as_deref().unwrap_or("unknown error")
        )));
    }

    let format = query.format.unwrap_or_else(|| "srt".to_string());
    let format = format.trim().to_ascii_lowercase();
    let (name, content_type) = match format.as_str() {
        "srt" => (
            result.subtitles_srt_file.as_deref(),
            "application/x-subrip; charset=utf-8",
        ),
        "vtt" | "webvtt" => (
            result.subtitles_vtt_file.as_deref(),
            "text/vtt; charset=utf-8",
        ),
        "ass" | "ssa" => (
            result.subtitles_ass_file.as_deref(),
            "text/x-ssa; charset=utf-8",
        ),
        other => {
            return Err(PerChapterExportApiError::bad(format!(
                "unsupported subtitle format: {other}"
            )));
        }
    };
    let name = name.ok_or_else(|| {
        PerChapterExportApiError::bad("subtitles were not included in this export job")
    })?;
    if !safe_file_name(name) {
        return Err(PerChapterExportApiError::internal(
            "per-chapter subtitles",
            "invalid subtitle file name",
        ));
    }
    let dir = state.per_chapter_exports.job_dir(&project_id)?;
    let path = dir.join(name);
    let file = tokio::fs::File::open(&path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PerChapterExportApiError::NotFound("per-chapter subtitles are missing")
        } else {
            PerChapterExportApiError::internal("open per-chapter subtitles", error)
        }
    })?;
    let metadata = file.metadata().await.map_err(|error| {
        PerChapterExportApiError::internal("read per-chapter subtitles metadata", error)
    })?;
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"promin-{}-chapter-{}.{}\"",
            job.id,
            chapter_index,
            name.rsplit('.').next().unwrap_or(&format)
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    if let Ok(length) = HeaderValue::from_str(&metadata.len().to_string()) {
        headers.insert(header::CONTENT_LENGTH, length);
    }
    let stream = ReaderStream::new(file);
    Ok((headers, Body::from_stream(stream)).into_response())
}

/// `DELETE /v1/projects/{id}/exports/per-chapter/{job_id}`: скасувати/видалити job.
pub async fn delete_per_chapter_export(
    State(state): State<AppState>,
    AxumPath((project_id, job_id)): AxumPath<(String, String)>,
) -> Result<StatusCode, PerChapterExportApiError> {
    state.per_chapter_exports.delete(&project_id, &job_id)?;
    Ok(StatusCode::NO_CONTENT)
}

fn remove_result_files(dir: &Path, result: &ChapterExportResult) {
    for name in [
        result.audio_file.as_deref(),
        result.subtitles_srt_file.as_deref(),
        result.subtitles_vtt_file.as_deref(),
        result.subtitles_ass_file.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if safe_file_name(name) {
            remove_file_if_exists(&dir.join(name));
        }
    }
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.contains('/')
        && !name.contains('\\')
        && Path::new(name).file_name().and_then(|value| value.to_str()) == Some(name)
}

fn remove_file_if_exists(path: &Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_per_chapter_id() {
        assert!(valid_per_chapter_id("abc-123_def"));
        assert!(!valid_per_chapter_id(""));
        assert!(!valid_per_chapter_id("bad/id"));
        assert!(!valid_per_chapter_id(&"a".repeat(65)));
    }

    #[test]
    fn job_status_is_active() {
        assert!(JobStatus::Queued.is_active());
        assert!(JobStatus::Running.is_active());
        assert!(!JobStatus::Completed.is_active());
        assert!(!JobStatus::Failed.is_active());
        assert!(!JobStatus::Cancelled.is_active());
    }

    #[test]
    fn safe_file_names_reject_paths() {
        assert!(safe_file_name("abc-chapter-0.mp3"));
        assert!(!safe_file_name("../abc.mp3"));
        assert!(!safe_file_name("dir/abc.mp3"));
        assert!(!safe_file_name("dir\\abc.mp3"));
        assert!(!safe_file_name(""));
    }

    #[test]
    fn job_round_trips_through_disk() {
        let dir =
            std::env::temp_dir().join(format!("promin-per-chapter-test-{}", uuid::Uuid::new_v4()));
        let job = PerChapterExportJob {
            id: "job-1".to_string(),
            project_id: "project-1".to_string(),
            format: ExportFormat::Mp3,
            bitrate_kbps: 128,
            chapters: vec![0, 1],
            include_subtitles: true,
            status: JobStatus::Completed,
            progress: 100,
            chapters_total: 2,
            chapters_done: 2,
            created_at: now_rfc3339(),
            started_at: None,
            finished_at: None,
            error: None,
            results: vec![ChapterExportResult {
                chapter_index: 0,
                status: JobStatus::Completed,
                audio_file: Some("job-1-chapter-0.mp3".to_string()),
                subtitles_srt_file: Some("job-1-chapter-0.srt".to_string()),
                subtitles_vtt_file: None,
                subtitles_ass_file: None,
                duration_ms: Some(1_000),
                bytes: Some(16_000),
                error: None,
            }],
        };
        save_job(&dir, &job).unwrap();
        let bytes = std::fs::read(dir.join("job-1.json")).unwrap();
        let parsed: PerChapterExportJob = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.id, job.id);
        assert_eq!(parsed.format, ExportFormat::Mp3);
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(
            parsed.results[0].audio_file.as_deref(),
            Some("job-1-chapter-0.mp3")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
