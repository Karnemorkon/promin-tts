//! Асинхронна черга завдань синтезу та SSE-події.
//!
//! Важкий ONNX-синтез виконується в `spawn_blocking`, а прогрес оновлюється
//! після кожного чанка. Одночасність обмежується `PROMIN_JOBS_CONCURRENCY`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use speech_core::jobs::{JobRecord, JobRequest, JobStatus, JobStore};
use tokio::sync::{broadcast, mpsc, Semaphore};

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
        }
    }
}

/// Черга завдань.
pub struct JobManager {
    store: Arc<JobStore>,
    service: Arc<SynthService>,
    sender: mpsc::Sender<String>,
    state: Arc<Mutex<HashMap<String, JobRecord>>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    events: broadcast::Sender<JobEvent>,
    public_url: String,
    http: reqwest::Client,
}

impl JobManager {
    /// Створити чергу, відновити записи з диска та запустити воркерів.
    pub fn new(
        store: Arc<JobStore>,
        service: Arc<SynthService>,
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
        let id = uuid::Uuid::new_v4().to_string();
        let job = JobRecord::new(id.clone(), request);
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

        let request = job.request.clone();
        let mode = VerbalizerMode::parse(request.auto_verbalize.as_ref());
        let max_chars = request.max_chars.unwrap_or(1_200).clamp(50, 5_000);
        let crossfade_ms = request.crossfade_ms.unwrap_or(40).min(200);

        let service = Arc::clone(&self.service);
        let state = Arc::clone(&self.state);
        let store = Arc::clone(&self.store);
        let events = self.events.clone();
        let cancel = Arc::clone(&self.cancel);
        let job_id = id.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let mut progress = |done: usize, total: usize| -> bool {
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
            };

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

        let mut job = match self.get(id) {
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
                if let Err(error) = self.store.save_audio(id, &wav) {
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
