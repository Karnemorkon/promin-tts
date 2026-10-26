//! Асинхронна черга завдань синтезу та SSE-події.
//!
//! Важкий ONNX-синтез виконується в `spawn_blocking`, а прогрес оновлюється
//! після кожного чанка. Одночасність обмежується `PROMIN_JOBS_CONCURRENCY`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use speech_audio::loudness::LoudnessTarget;
use speech_core::jobs::{AudioLocation, JobRecord, JobRequest, JobStatus, JobStore};
use speech_core::projects::{valid_project_id, ProjectSettings, ProjectStore, PronunciationRule};
use tokio::sync::{broadcast, mpsc, Semaphore};

use crate::pronunciation::PronunciationStore;
use crate::service::{SynthError, SynthService, VerbalizerMode, MAX_JOB_TEXT_CHARS, SPEED_RANGE};

/// Компактна подія для SSE/WebSocket (без повного тексту запиту).
#[derive(Debug, Clone, Serialize)]
pub struct JobEvent {
    pub job_id: String,
    pub status: JobStatus,
    pub progress: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbalizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chapter_index: Option<usize>,
}

impl JobEvent {
    pub fn from_record(job: &JobRecord) -> Self {
        Self {
            job_id: job.id.clone(),
            status: job.status,
            progress: job.progress,
            error: job.error.clone(),
            audio_url: job.audio_url.clone(),
            audio_bytes: job.audio_bytes,
            duration_ms: job.duration_ms,
            chunks: job.chunks,
            verbalizer: job.verbalizer.clone(),
            project_id: job.project_id().map(str::to_owned),
            chapter_index: job.chapter_index(),
        }
    }
}

/// Результат скасування.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    NotFound,
    Cancelled,
    AlreadyFinished,
}

/// Компактний запис для `GET /v1/jobs` (без повного тексту).
#[derive(Debug, Clone, Serialize)]
pub struct JobSummary {
    pub id: String,
    pub status: JobStatus,
    pub progress: u8,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub voice: Option<String>,
    pub speed: f32,
    pub text_chars: usize,
    pub error: Option<String>,
    pub audio_url: Option<String>,
    pub audio_bytes: Option<u64>,
    pub duration_ms: Option<u64>,
    pub chunks: Option<usize>,
    pub verbalizer: Option<String>,
    pub project_id: Option<String>,
    pub chapter_index: Option<usize>,
}

impl From<&JobRecord> for JobSummary {
    fn from(job: &JobRecord) -> Self {
        Self {
            id: job.id.clone(),
            status: job.status,
            progress: job.progress,
            created_at: job.created_at.clone(),
            started_at: job.started_at.clone(),
            finished_at: job.finished_at.clone(),
            voice: job.request.voice.clone(),
            speed: job.request.speed,
            text_chars: job.request.text.chars().count(),
            error: job.error.clone(),
            audio_url: job.audio_url.clone(),
            audio_bytes: job.audio_bytes,
            duration_ms: job.duration_ms,
            chunks: job.chunks,
            verbalizer: job.verbalizer.clone(),
            project_id: job.project_id().map(str::to_owned),
            chapter_index: job.chapter_index(),
        }
    }
}

/// Черга завдань.
pub struct JobManager {
    store: Arc<JobStore>,
    service: Arc<SynthService>,
    projects: Arc<ProjectStore>,
    pronunciation: Arc<PronunciationStore>,
    project_lock: Arc<tokio::sync::Mutex<()>>,
    sender: mpsc::Sender<String>,
    state: Arc<Mutex<HashMap<String, JobRecord>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    events: broadcast::Sender<JobEvent>,
    public_url: String,
    http: reqwest::Client,
}

impl JobManager {
    /// Створити чергу, відновити записи з диска та запустити воркерів.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<JobStore>,
        service: Arc<SynthService>,
        projects: Arc<ProjectStore>,
        pronunciation: Arc<PronunciationStore>,
        project_lock: Arc<tokio::sync::Mutex<()>>,
        concurrency: usize,
        public_url: String,
    ) -> Result<Arc<Self>> {
        let recovered = store.recover_interrupted()?;
        if recovered > 0 {
            tracing::warn!(recovered, "marked interrupted jobs as failed");
        }

        let mut state = HashMap::new();
        for job in store.load_all()? {
            state.insert(job.id.clone(), job);
        }

        let (sender, receiver) = mpsc::channel::<String>(1024);
        let (events, _) = broadcast::channel::<JobEvent>(256);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("promin-tts-webhook/0.5")
            .build()
            .context("build webhook HTTP client")?;

        let manager = Arc::new(Self {
            store,
            service,
            projects,
            pronunciation,
            project_lock,
            sender,
            state: Arc::new(Mutex::new(state)),
            cancel: Arc::new(Mutex::new(HashSet::new())),
            events,
            public_url: public_url.trim_end_matches('/').to_string(),
            http,
        });

        let dispatcher = Arc::clone(&manager);
        let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
        let mut receiver = receiver;
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

    /// Додати завдання в чергу.
    pub async fn submit(&self, request: JobRequest) -> Result<JobRecord, String> {
        validate_job_request(&request)?;
        let job = JobRecord::new(uuid::Uuid::new_v4().to_string(), request);
        self.enqueue(job).await
    }

    /// Додати завдання генерації глави проєкту із заздалегідь
    /// зарезервованим id.
    ///
    /// Дозволяє спершу зберегти `queued`-статус глави під `project_lock`,
    /// відпустити lock і лише потім поставити job-и в чергу (інакше воркер
    /// може заблокуватися на оновленні глави, поки черга переповнена).
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_chapter_with_id(
        &self,
        job_id: String,
        project_id: &str,
        chapter_index: usize,
        request: JobRequest,
        project_settings: Option<ProjectSettings>,
        pronunciation: Vec<PronunciationRule>,
    ) -> Result<JobRecord, String> {
        validate_job_request(&request)?;
        if !valid_project_id(project_id) {
            return Err("invalid project id".into());
        }
        let job = JobRecord::for_project_chapter(
            job_id,
            request,
            project_id,
            chapter_index,
            project_settings,
            pronunciation,
        );
        self.enqueue(job).await
    }

    async fn enqueue(&self, job: JobRecord) -> Result<JobRecord, String> {
        let id = job.id.clone();
        self.store.save(&job).map_err(|error| error.to_string())?;
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(id.clone(), job.clone());
        self.sender
            .send(id)
            .await
            .map_err(|_| "job queue is closed".to_string())?;
        let _ = self.events.send(JobEvent::from_record(&job));
        Ok(job)
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        if let Some(job) = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned()
        {
            return Some(job);
        }
        match self.store.load(id) {
            Ok(Some(job)) => {
                self.state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(id.to_string(), job.clone());
                Some(job)
            }
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(job_id = id, error = %error, "cannot load job");
                None
            }
        }
    }

    pub fn list(&self, limit: usize) -> Vec<JobRecord> {
        let mut jobs: Vec<JobRecord> = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .collect();
        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        jobs.truncate(limit.max(1));
        jobs
    }

    pub fn audio(&self, id: &str) -> Result<Option<Vec<u8>>> {
        self.store.load_audio(id)
    }

    pub fn cancel(&self, id: &str) -> CancelOutcome {
        let Some(mut job) = self.get(id) else {
            return CancelOutcome::NotFound;
        };
        if job.status.is_terminal() {
            return CancelOutcome::AlreadyFinished;
        }
        match job.status {
            JobStatus::Queued => {
                job.mark_cancelled();
                self.persist(&job);
                self.broadcast(&job);
                CancelOutcome::Cancelled
            }
            JobStatus::Running => {
                self.cancel
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(id.to_string());
                // Статус оновить воркер після поточного чанка.
                CancelOutcome::Cancelled
            }
            _ => CancelOutcome::AlreadyFinished,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<JobEvent> {
        self.events.subscribe()
    }

    /// Скасувати всі активні job-и проєкту (перед видаленням каталогу).
    pub fn cancel_project_jobs(&self, project_id: &str) -> usize {
        let ids: Vec<String> = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|job| job.project_id() == Some(project_id) && !job.status.is_terminal())
            .map(|job| job.id.clone())
            .collect();
        ids.iter()
            .filter(|id| self.cancel(id) == CancelOutcome::Cancelled)
            .count()
    }

    /// Зберегти запис на диск і оновити in-memory стан (єдине джерело для GET).
    fn persist(&self, job: &JobRecord) {
        persist_job(&self.state, &self.store, job);
    }

    fn broadcast(&self, job: &JobRecord) {
        let _ = self.events.send(JobEvent::from_record(job));
    }

    fn is_cancelled(&self, id: &str) -> bool {
        self.cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(id)
    }

    async fn run_job(&self, id: &str) {
        let Some(mut job) = self.get(id) else {
            tracing::warn!(job_id = id, "queued job disappeared");
            return;
        };
        if job.status != JobStatus::Queued {
            return;
        }
        if self.is_cancelled(id) {
            job.mark_cancelled();
            self.persist(&job);
            self.broadcast(&job);
            return;
        }

        job.mark_running();
        self.persist(&job);
        self.broadcast(&job);

        match job.audio_location.clone() {
            AudioLocation::Job => self.run_regular_job(job).await,
            AudioLocation::ProjectChapter {
                project_id,
                chapter_index,
            } => {
                self.run_project_chapter(job, &project_id, chapter_index)
                    .await;
            }
        }
    }

    async fn run_regular_job(&self, job: JobRecord) {
        let id = job.id.clone();
        let request = job.request.clone();
        let mode = VerbalizerMode::parse(request.auto_verbalize.as_ref());
        let max_chars = request.max_chars.unwrap_or(1_200).clamp(50, 5_000);
        let crossfade_ms = request.crossfade_ms.unwrap_or(40).min(200);

        let service = Arc::clone(&self.service);
        let state = Arc::clone(&self.state);
        let store = Arc::clone(&self.store);
        let events = self.events.clone();
        let cancel = Arc::clone(&self.cancel);
        let job_id = id.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut progress = progress_callback(state, store, events, cancel, job_id);

            service.synthesize_long_pcm(
                &request.text,
                request.voice.as_deref(),
                None,
                request.speed,
                mode,
                request.raw_stress,
                max_chars,
                crossfade_ms,
                Some(&mut progress),
            )
        })
        .await;

        let mut job = match self.get(&id) {
            Some(job) => job,
            None => return,
        };

        match result {
            Ok(Ok(pcm)) => {
                let wav = match speech_audio::wav_bytes(&pcm.pcm, speech_audio::SAMPLE_RATE) {
                    Ok(wav) => wav,
                    Err(error) => {
                        job.mark_failed(format!("WAV export failed: {error}"));
                        self.persist(&job);
                        self.broadcast(&job);
                        self.fire_webhook(&job);
                        return;
                    }
                };
                if let Err(error) = self.store.save_audio(&id, &wav) {
                    job.mark_failed(format!("cannot save audio: {error}"));
                    self.persist(&job);
                    self.broadcast(&job);
                    self.fire_webhook(&job);
                    return;
                }
                let audio_url = format!("{}/v1/jobs/{id}/audio", self.public_url);
                job.mark_completed(
                    audio_url,
                    wav.len() as u64,
                    pcm.audio_ms(),
                    pcm.chunks,
                    pcm.verbalizer.to_string(),
                    pcm.vb_ms,
                );
                self.persist(&job);
                self.broadcast(&job);
                self.fire_webhook(&job);
            }
            Ok(Err(SynthError::Cancelled)) => {
                job.mark_cancelled();
                self.persist(&job);
                self.broadcast(&job);
            }
            Ok(Err(error)) => {
                job.mark_failed(error.to_string());
                self.persist(&job);
                self.broadcast(&job);
                self.fire_webhook(&job);
            }
            Err(join_error) => {
                job.mark_failed(format!("synthesis task failed: {join_error}"));
                self.persist(&job);
                self.broadcast(&job);
                self.fire_webhook(&job);
            }
        }
    }

    async fn run_project_chapter(
        &self,
        mut job: JobRecord,
        project_id: &str,
        chapter_index: usize,
    ) {
        let id = job.id.clone();
        let project = match self.projects.load(project_id) {
            Ok(Some(project)) => project,
            Ok(None) => {
                self.fail_job(&mut job, "project does not exist");
                return;
            }
            Err(error) => {
                self.fail_job(&mut job, format!("cannot load project: {error}"));
                return;
            }
        };
        let Some(chapter) = project.chapter(chapter_index) else {
            self.fail_job(&mut job, "chapter does not exist");
            return;
        };
        if chapter.text != job.request.text {
            self.update_project_chapter(project_id, chapter_index, |chapter| chapter.mark_stale())
                .await;
            self.fail_job(&mut job, "chapter text changed after job was queued");
            return;
        }

        self.update_project_chapter(project_id, chapter_index, |chapter| chapter.mark_running())
            .await;

        let pronouncer = match self.pronunciation.merged_pronouncer(&job.pronunciation) {
            Ok(pronouncer) => pronouncer,
            Err(error) => {
                let message = format!("pronunciation dictionary: {error}");
                self.fail_job(&mut job, message.clone());
                self.update_project_chapter(project_id, chapter_index, |chapter| {
                    chapter.mark_failed(message)
                })
                .await;
                return;
            }
        };

        let loudness_target = job.project_settings.as_ref().and_then(|settings| {
            settings.loudness.enabled.then(|| {
                LoudnessTarget::new(
                    settings.loudness.target_lufs,
                    settings.loudness.true_peak_db,
                )
            })
        });
        let request = job.request.clone();
        let mode = VerbalizerMode::parse(request.auto_verbalize.as_ref());
        let max_chars = request.max_chars.unwrap_or(1_200).clamp(50, 5_000);
        let crossfade_ms = request.crossfade_ms.unwrap_or(40).min(200);

        let service = Arc::clone(&self.service);
        let state = Arc::clone(&self.state);
        let store = Arc::clone(&self.store);
        let events = self.events.clone();
        let cancel = Arc::clone(&self.cancel);
        let job_id = id.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut progress = progress_callback(state, store, events, cancel, job_id);
            service.synthesize_chapter_pcm(
                &request.text,
                request.voice.as_deref(),
                None,
                request.speed,
                mode,
                request.raw_stress,
                max_chars,
                crossfade_ms,
                Some(pronouncer.as_ref()),
                loudness_target,
                Some(&mut progress),
            )
        })
        .await;

        let mut job = match self.get(&id) {
            Some(job) => job,
            None => return,
        };
        if self.is_cancelled(&id) {
            job.mark_cancelled();
            self.persist(&job);
            self.broadcast(&job);
            self.update_project_chapter(project_id, chapter_index, |chapter| {
                chapter.mark_cancelled()
            })
            .await;
            return;
        }

        match result {
            Ok(Ok(output)) => {
                let wav = match speech_audio::wav_bytes(&output.pcm, speech_audio::SAMPLE_RATE) {
                    Ok(wav) => wav,
                    Err(error) => {
                        let message = format!("WAV export failed: {error}");
                        self.fail_job(&mut job, message.clone());
                        self.update_project_chapter(project_id, chapter_index, |chapter| {
                            chapter.mark_failed(message)
                        })
                        .await;
                        return;
                    }
                };
                match self
                    .complete_project_chapter(&job, project_id, chapter_index, &wav, &output)
                    .await
                {
                    Ok(()) => {
                        let audio_url = format!(
                            "{}/v1/projects/{project_id}/chapters/{chapter_index}/audio",
                            self.public_url
                        );
                        job.mark_completed(
                            audio_url,
                            wav.len() as u64,
                            output.duration_ms,
                            output.chunks,
                            output.verbalizer.to_string(),
                            output.vb_ms,
                        );
                        self.persist(&job);
                        self.broadcast(&job);
                        self.fire_webhook(&job);
                    }
                    Err(error) => {
                        if self.is_cancelled(&id) {
                            job.mark_cancelled();
                            self.persist(&job);
                            self.broadcast(&job);
                            return;
                        }
                        let message = format!("cannot save chapter: {error}");
                        self.fail_job(&mut job, message.clone());
                        self.update_project_chapter(project_id, chapter_index, |chapter| {
                            chapter.mark_failed(message)
                        })
                        .await;
                    }
                }
            }
            Ok(Err(SynthError::Cancelled)) => {
                job.mark_cancelled();
                self.persist(&job);
                self.broadcast(&job);
                self.update_project_chapter(project_id, chapter_index, |chapter| {
                    chapter.mark_cancelled()
                })
                .await;
            }
            Ok(Err(error)) => {
                let message = error.to_string();
                self.fail_job(&mut job, message.clone());
                self.update_project_chapter(project_id, chapter_index, |chapter| {
                    chapter.mark_failed(message)
                })
                .await;
            }
            Err(join_error) => {
                let message = format!("synthesis task failed: {join_error}");
                self.fail_job(&mut job, message.clone());
                self.update_project_chapter(project_id, chapter_index, |chapter| {
                    chapter.mark_failed(message)
                })
                .await;
            }
        }
    }

    /// Зберегти аудіо глави та оновити `project.json` під тим самим lock,
    /// що й видалення проєкту (щоб не відтворити видалений каталог).
    async fn complete_project_chapter(
        &self,
        job: &JobRecord,
        project_id: &str,
        chapter_index: usize,
        wav: &[u8],
        output: &crate::service::SynthChapterOutput,
    ) -> Result<(), String> {
        let _guard = self.project_lock.lock().await;
        let mut project = self
            .projects
            .load(project_id)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "project does not exist".to_string())?;
        let Some(chapter) = project.chapter(chapter_index) else {
            return Err("chapter does not exist".to_string());
        };
        let text_changed = chapter.text != job.request.text;
        let settings_changed = job.project_settings.as_ref() != Some(&project.settings);
        let pronunciation_changed = job.pronunciation != project.pronunciation;

        self.projects
            .save_chapter_audio(project_id, chapter_index, wav)
            .map_err(|error| error.to_string())?;

        let chapter = project
            .chapter_mut(chapter_index)
            .expect("chapter existence checked above");
        chapter.mark_completed(
            wav.len() as u64,
            output.duration_ms,
            output.loudness.clone(),
            output.chunk_timings.clone(),
        );
        if text_changed || settings_changed || pronunciation_changed {
            chapter.mark_stale();
            if text_changed {
                chapter.error = Some("chapter text changed after generation".to_string());
            }
        }
        self.projects
            .save(&mut project)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    async fn update_project_chapter<F>(&self, project_id: &str, chapter_index: usize, update: F)
    where
        F: FnOnce(&mut speech_core::projects::Chapter),
    {
        let _guard = self.project_lock.lock().await;
        match self.projects.load(project_id) {
            Ok(Some(mut project)) => {
                let Some(chapter) = project.chapter_mut(chapter_index) else {
                    return;
                };
                update(chapter);
                if let Err(error) = self.projects.save(&mut project) {
                    tracing::error!(project_id, error = %error, "cannot save project chapter");
                }
            }
            Ok(None) => {
                tracing::warn!(project_id, "project disappeared while updating chapter");
            }
            Err(error) => {
                tracing::error!(project_id, error = %error, "cannot load project");
            }
        }
    }

    fn fail_job(&self, job: &mut JobRecord, error: impl Into<String>) {
        job.mark_failed(error);
        self.persist(job);
        self.broadcast(job);
        self.fire_webhook(job);
    }

    fn fire_webhook(&self, job: &JobRecord) {
        let Some(url) = job.request.webhook_url.clone() else {
            return;
        };
        if !valid_webhook_url(&url) {
            tracing::warn!(job_id = %job.id, %url, "invalid webhook URL");
            return;
        }
        let payload = serde_json::json!({
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "audio_url": job.audio_url,
            "audio_bytes": job.audio_bytes,
            "duration_ms": job.duration_ms,
            "chunks": job.chunks,
            "verbalizer": job.verbalizer,
            "error": job.error,
            "finished_at": job.finished_at,
        });
        let client = self.http.clone();
        let job_id = job.id.clone();
        tokio::spawn(async move {
            match client.post(&url).json(&payload).send().await {
                Ok(response) if response.status().is_success() => {
                    tracing::info!(job_id = %job_id, %url, "webhook delivered");
                }
                Ok(response) => {
                    tracing::warn!(job_id = %job_id, %url, status = %response.status(), "webhook rejected");
                }
                Err(error) => {
                    tracing::warn!(job_id = %job_id, %url, error = %error, "webhook failed");
                }
            }
        });
    }
}

/// Колбек прогресу для `spawn_blocking`: оновлює job-запис і шле SSE.
fn progress_callback(
    state: Arc<Mutex<HashMap<String, JobRecord>>>,
    store: Arc<JobStore>,
    events: broadcast::Sender<JobEvent>,
    cancel: Arc<Mutex<HashSet<String>>>,
    job_id: String,
) -> impl FnMut(usize, usize) -> bool {
    move |done: usize, total: usize| -> bool {
        if cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&job_id)
        {
            return false;
        }
        let progress = done
            .saturating_mul(100)
            .checked_div(total)
            .unwrap_or(0)
            .min(99) as u8;
        let job = {
            let mut state = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(job) = state.get_mut(&job_id) else {
                return false;
            };
            job.set_progress(progress);
            job.clone()
        };
        if let Err(error) = store.save(&job) {
            tracing::error!(job_id = %job_id, error = %error, "cannot save progress");
        }
        let _ = events.send(JobEvent::from_record(&job));
        true
    }
}

/// Оновити in-memory стан і зберегти запис на диск.
fn persist_job(state: &Mutex<HashMap<String, JobRecord>>, store: &JobStore, job: &JobRecord) {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(job.id.clone(), job.clone());
    if let Err(error) = store.save(job) {
        tracing::error!(job_id = %job.id, error = %error, "cannot save job record");
    }
}

/// Базова валідація до постановки в чергу.
pub fn validate_job_request(request: &JobRequest) -> Result<(), String> {
    let text = request.text.trim();
    if text.is_empty() {
        return Err("text must not be empty".into());
    }
    if text.chars().count() > MAX_JOB_TEXT_CHARS {
        return Err(format!(
            "text is too long: max {MAX_JOB_TEXT_CHARS} characters"
        ));
    }
    if !request.speed.is_finite() || !SPEED_RANGE.contains(&request.speed) {
        return Err(format!(
            "speed must be in {}..={}",
            SPEED_RANGE.start(),
            SPEED_RANGE.end()
        ));
    }
    if let Some(url) = &request.webhook_url {
        if !valid_webhook_url(url) {
            return Err("webhook_url must be an absolute http(s) URL".into());
        }
    }
    if let Some(max_chars) = request.max_chars {
        if !(50..=5_000).contains(&max_chars) {
            return Err("max_chars must be in 50..=5000".into());
        }
    }
    if let Some(crossfade_ms) = request.crossfade_ms {
        if crossfade_ms > 200 {
            return Err("crossfade_ms must be 0..=200".into());
        }
    }
    Ok(())
}

fn valid_webhook_url(url: &str) -> bool {
    url::Url::parse(url)
        .map(|parsed| matches!(parsed.scheme(), "http" | "https"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> JobRequest {
        JobRequest {
            text: "Привіт".into(),
            voice: None,
            speed: 1.0,
            auto_verbalize: None,
            raw_stress: false,
            webhook_url: None,
            max_chars: Some(500),
            crossfade_ms: Some(40),
        }
    }

    #[test]
    fn validates_text_and_speed() {
        let mut req = request();
        assert!(validate_job_request(&req).is_ok());
        req.text = "   ".into();
        assert!(validate_job_request(&req).is_err());
        req.text = "Привіт".into();
        req.speed = 3.0;
        assert!(validate_job_request(&req).is_err());
    }

    #[test]
    fn validates_webhook_scheme() {
        let mut req = request();
        req.webhook_url = Some("ftp://example.com/hook".into());
        assert!(validate_job_request(&req).is_err());
        req.webhook_url = Some("https://example.com/hook".into());
        assert!(validate_job_request(&req).is_ok());
    }

    #[test]
    fn validates_limits() {
        let mut req = request();
        req.max_chars = Some(10);
        assert!(validate_job_request(&req).is_err());
        req.max_chars = Some(500);
        req.crossfade_ms = Some(500);
        assert!(validate_job_request(&req).is_err());
    }

    #[test]
    fn summary_omits_full_text() {
        let job = JobRecord::new("job-1".into(), request());
        let summary = JobSummary::from(&job);
        assert_eq!(summary.text_chars, "Привіт".chars().count());
        let json = serde_json::to_string(&summary).unwrap();
        assert!(!json.contains("Привіт"));
        assert!(json.contains("\"text_chars\":6"));
    }
    #[test]
    fn persist_updates_in_memory_state() {
        let dir = std::env::temp_dir().join(format!("promin-jobs-state-{}", uuid::Uuid::new_v4()));
        let store = JobStore::new(&dir).unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let state = Mutex::new(HashMap::new());
        state
            .lock()
            .unwrap()
            .insert(id.clone(), JobRecord::new(id.clone(), request()));
        let mut job = state.lock().unwrap().get(&id).unwrap().clone();
        job.mark_running();
        persist_job(&state, &store, &job);
        assert_eq!(
            state.lock().unwrap().get(&id).unwrap().status,
            JobStatus::Running
        );
        assert_eq!(store.load(&id).unwrap().unwrap().status, JobStatus::Running);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
