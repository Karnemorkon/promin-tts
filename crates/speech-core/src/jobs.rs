//! Черга та сховище завдань генерації.
//!
//! Кожне завдання зберігається як `<jobs>/<id>.json`, готове аудіо —
//! `<jobs>/<id>.wav`. Записи пишуться атомарно (`.tmp` + rename), тому
//! перерваний процес не залишає напівфайлів.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::projects::{ProjectSettings, PronunciationRule, Role};

/// Стан завдання.
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
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

fn default_speed() -> f32 {
    1.0
}

/// Де зберігається аудіо завдання.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioLocation {
    /// Звичайне завдання: `<jobs>/<id>.wav`.
    #[default]
    Job,
    /// Глава проєкту: `<projects>/<project_id>/chapters/<index>.wav`.
    ProjectChapter {
        project_id: String,
        chapter_index: usize,
    },
}

impl AudioLocation {
    pub fn project_id(&self) -> Option<&str> {
        match self {
            Self::Job => None,
            Self::ProjectChapter { project_id, .. } => Some(project_id),
        }
    }

    pub fn chapter_index(&self) -> Option<usize> {
        match self {
            Self::Job => None,
            Self::ProjectChapter { chapter_index, .. } => Some(*chapter_index),
        }
    }
}

/// Параметри одного завдання (зберігаються разом із записом).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// `true`/`false` або `"neural"`/`"rule"`/`"off"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_verbalize: Option<serde_json::Value>,
    #[serde(default)]
    pub raw_stress: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_url: Option<String>,
    /// Максимальна довжина чанка в символах (0 — не різати).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    /// Кросфейд між чанками, мс (0–200).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfade_ms: Option<u32>,
}

/// Повний запис завдання, який віддається API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: JobStatus,
    /// 0–100.
    pub progress: u8,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    pub request: JobRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunks: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbalizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vb_ms: Option<u64>,
    /// Куди писати аудіо (job або глава проєкту).
    #[serde(default)]
    pub audio_location: AudioLocation,
    /// Знімок налаштувань проєкту на момент постановки.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_settings: Option<ProjectSettings>,
    /// Знімок проєктних правил вимови на момент постановки.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pronunciation: Vec<PronunciationRule>,
    /// Знімок ролей діалогу на момент постановки.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<Role>,
}

impl JobRecord {
    pub fn new(id: String, request: JobRequest) -> Self {
        Self {
            id,
            status: JobStatus::Queued,
            progress: 0,
            created_at: now_rfc3339(),
            started_at: None,
            finished_at: None,
            request,
            error: None,
            audio_url: None,
            audio_bytes: None,
            duration_ms: None,
            chunks: None,
            verbalizer: None,
            vb_ms: None,
            audio_location: AudioLocation::Job,
            project_settings: None,
            pronunciation: Vec::new(),
            roles: Vec::new(),
        }
    }

    /// Завдання для глави проєкту зі знімком налаштувань.
    pub fn for_project_chapter(
        id: String,
        request: JobRequest,
        project_id: impl Into<String>,
        chapter_index: usize,
        project_settings: Option<ProjectSettings>,
        pronunciation: Vec<PronunciationRule>,
        roles: Vec<Role>,
    ) -> Self {
        let mut job = Self::new(id, request);
        job.audio_location = AudioLocation::ProjectChapter {
            project_id: project_id.into(),
            chapter_index,
        };
        job.project_settings = project_settings;
        job.pronunciation = pronunciation;
        job.roles = roles;
        job
    }

    pub fn project_id(&self) -> Option<&str> {
        self.audio_location.project_id()
    }

    pub fn chapter_index(&self) -> Option<usize> {
        self.audio_location.chapter_index()
    }

    pub fn mark_running(&mut self) {
        self.status = JobStatus::Running;
        self.started_at = Some(now_rfc3339());
        self.progress = 1;
    }

    pub fn set_progress(&mut self, progress: u8) {
        self.progress = progress.min(99);
    }

    pub fn mark_completed(
        &mut self,
        audio_url: String,
        audio_bytes: u64,
        duration_ms: u64,
        chunks: usize,
        verbalizer: String,
        vb_ms: u64,
    ) {
        self.status = JobStatus::Completed;
        self.progress = 100;
        self.finished_at = Some(now_rfc3339());
        self.audio_url = Some(audio_url);
        self.audio_bytes = Some(audio_bytes);
        self.duration_ms = Some(duration_ms);
        self.chunks = Some(chunks);
        self.verbalizer = Some(verbalizer);
        self.vb_ms = Some(vb_ms);
    }

    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = JobStatus::Failed;
        self.finished_at = Some(now_rfc3339());
        self.error = Some(error.into());
    }

    pub fn mark_cancelled(&mut self) {
        self.status = JobStatus::Cancelled;
        self.finished_at = Some(now_rfc3339());
        self.error = Some("cancelled".to_string());
    }
}

/// Поточний час у RFC3339 (UTC, мілісекунди).
pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Чи безпечний ідентифікатор для використання як ім'я файлу.
pub fn valid_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

/// Файлове сховище завдань.
#[derive(Debug, Clone)]
pub struct JobStore {
    dir: PathBuf,
}

impl JobStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create jobs dir {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn record_path(&self, id: &str) -> Result<PathBuf> {
        if !valid_job_id(id) {
            bail!("invalid job id");
        }
        Ok(self.dir.join(format!("{id}.json")))
    }

    /// Шлях до WAV-файлу завдання (без перевірки існування).
    pub fn audio_path(&self, id: &str) -> Result<PathBuf> {
        if !valid_job_id(id) {
            bail!("invalid job id");
        }
        Ok(self.dir.join(format!("{id}.wav")))
    }

    pub fn save(&self, job: &JobRecord) -> Result<()> {
        let path = self.record_path(&job.id)?;
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec_pretty(job).context("serialize job")?;
        std::fs::write(&tmp, json).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Option<JobRecord>> {
        let path = self.record_path(id)?;
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let job = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse job {}", path.display()))?;
        Ok(Some(job))
    }

    /// Усі записи, найновіші першими; пошкоджені файли пропускаються.
    pub fn load_all(&self) -> Result<Vec<JobRecord>> {
        let mut jobs = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("read jobs dir {}", self.dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<JobRecord>(&bytes).ok())
            {
                Some(job) => jobs.push(job),
                None => tracing::warn!(path = %path.display(), "skipping malformed job record"),
            }
        }
        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(jobs)
    }

    pub fn save_audio(&self, id: &str, bytes: &[u8]) -> Result<PathBuf> {
        let path = self.audio_path(id)?;
        let tmp = path.with_extension("wav.tmp");
        std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(path)
    }

    pub fn load_audio(&self, id: &str) -> Result<Option<Vec<u8>>> {
        let path = self.audio_path(id)?;
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        Ok(Some(bytes))
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let record = self.record_path(id)?;
        let audio = self.audio_path(id)?;
        let mut removed = false;
        for path in [&record, &audio] {
            if path.is_file() {
                std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
                removed = true;
            }
        }
        Ok(removed)
    }

    /// Позначити незавершені після рестарту завдання як failed.
    pub fn recover_interrupted(&self) -> Result<usize> {
        let mut count = 0;
        for mut job in self.load_all()? {
            if matches!(job.status, JobStatus::Queued | JobStatus::Running) {
                job.mark_failed("server restarted before completion");
                self.save(&job)?;
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> JobStore {
        let dir = std::env::temp_dir().join(format!("promin-jobs-test-{}", uuid::Uuid::new_v4()));
        JobStore::new(dir).unwrap()
    }

    fn sample_request() -> JobRequest {
        JobRequest {
            text: "Привіт, світе!".into(),
            voice: Some("Speaker_0".into()),
            speed: 1.0,
            auto_verbalize: None,
            raw_stress: false,
            webhook_url: None,
            max_chars: Some(500),
            crossfade_ms: Some(40),
        }
    }

    #[test]
    fn save_load_and_delete() {
        let store = temp_store();
        let id = uuid::Uuid::new_v4().to_string();
        let mut job = JobRecord::new(id.clone(), sample_request());
        job.mark_running();
        job.set_progress(50);
        store.save(&job).unwrap();

        let loaded = store.load(&id).unwrap().unwrap();
        assert_eq!(loaded.status, JobStatus::Running);
        assert_eq!(loaded.progress, 50);

        store.save_audio(&id, b"RIFF").unwrap();
        assert_eq!(store.load_audio(&id).unwrap().unwrap(), b"RIFF");

        assert!(store.delete(&id).unwrap());
        assert!(store.load(&id).unwrap().is_none());
        assert!(store.load_audio(&id).unwrap().is_none());
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn rejects_path_traversal() {
        let store = temp_store();
        assert!(store.load("../etc/passwd").is_err());
        assert!(store.audio_path("a/b").is_err());
        assert!(!valid_job_id(""));
        assert!(!valid_job_id("a b"));
        assert!(valid_job_id("abc-123_XYZ"));
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn recovers_interrupted_jobs() {
        let store = temp_store();
        let id = uuid::Uuid::new_v4().to_string();
        let mut job = JobRecord::new(id.clone(), sample_request());
        job.mark_running();
        store.save(&job).unwrap();
        assert_eq!(store.recover_interrupted().unwrap(), 1);
        let loaded = store.load(&id).unwrap().unwrap();
        assert_eq!(loaded.status, JobStatus::Failed);
        assert!(loaded.error.unwrap().contains("restarted"));
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn terminal_states() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
    }

    #[test]
    fn project_chapter_jobs_keep_location_and_snapshot() {
        let job = JobRecord::for_project_chapter(
            "job-1".into(),
            sample_request(),
            "book-1",
            3,
            Some(ProjectSettings::default()),
            vec![PronunciationRule {
                pattern: "ІТ".into(),
                replacement: "ай-ті".into(),
                regex: false,
                case_sensitive: true,
            }],
            vec![Role::with_voice("Іван", "Speaker_1")],
        );
        assert_eq!(job.project_id(), Some("book-1"));
        assert_eq!(job.chapter_index(), Some(3));
        assert!(job.project_settings.is_some());
        assert_eq!(job.pronunciation.len(), 1);
        assert_eq!(job.roles.len(), 1);

        let json = serde_json::to_string(&job).unwrap();
        assert!(json.contains("project_chapter"));
        let parsed: JobRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.project_id(), Some("book-1"));
        assert_eq!(parsed.chapter_index(), Some(3));
        assert_eq!(parsed.pronunciation.len(), 1);
        assert_eq!(parsed.roles.len(), 1);
        assert_eq!(parsed.audio_location, job.audio_location);

        let plain = JobRecord::new("job-2".into(), sample_request());
        assert_eq!(plain.audio_location, AudioLocation::Job);
        assert_eq!(plain.project_id(), None);
    }
}
