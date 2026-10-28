//! Проєкти аудіокниг: модель, файлове сховище та формат `.speechproj`.
//!
//! Кожен проєкт зберігається як `<projects>/<id>/project.json`, аудіо глав —
//! `<projects>/<id>/chapters/<index>.wav`. Записи пишуться атомарно
//! (унікальний `.tmp` + rename), тому перерваний процес не залишає
//! напівфайлів.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

/// Версія схеми проєкту.
///
/// v2 додає ролі діалогів (`Project::roles`); проєкти v1 читаються без змін.
pub const PROJECT_SCHEMA_VERSION: u32 = 2;
/// Максимальна кількість глав у проєкті.
pub const MAX_CHAPTERS: usize = 2_000;
/// Максимальна довжина тексту однієї глави (символів).
pub const MAX_CHAPTER_CHARS: usize = 200_000;
/// Максимальна сумарна довжина тексту проєкту (символів).
pub const MAX_TOTAL_CHARS: usize = 2_000_000;
/// Максимальна кількість правил вимови.
pub const MAX_PRONUNCIATION_RULES: usize = 1_000;
/// Максимальна кількість ролей діалогу в проєкті.
pub const MAX_ROLES: usize = 100;
/// Максимальна довжина назви ролі.
pub const MAX_ROLE_NAME_CHARS: usize = 100;
/// Максимальна довжина назви проєкту/глави.
pub const MAX_TITLE_CHARS: usize = 500;
/// Максимальна довжина id проєкту.
pub const MAX_PROJECT_ID_LEN: usize = 64;

/// Поточний час у RFC3339 (UTC, мілісекунди).
pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Чи безпечний ідентифікатор для використання як ім'я каталогу.
pub fn valid_project_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_PROJECT_ID_LEN
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

/// Стан глави.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChapterStatus {
    /// Текст є, аудіо ще не генерувалося.
    #[default]
    Empty,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    /// Текст або налаштування змінилися після генерації.
    Stale,
}

impl ChapterStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Stale => "stale",
        }
    }

    /// Чи виконується зараз генерація.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    /// Чи можна вважати главу завершеною (аудіо актуальне).
    pub fn is_ready(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Правило вимови: заміна підрядка або regex до вербалізації.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationRule {
    pub pattern: String,
    pub replacement: String,
    /// `pattern` — регулярний вираз (`regex` crate).
    #[serde(default)]
    pub regex: bool,
    /// Враховувати регістр (для звичайного підрядка).
    #[serde(default)]
    pub case_sensitive: bool,
}

impl PronunciationRule {
    pub fn validate(&self) -> Result<()> {
        if self.pattern.trim().is_empty() {
            bail!("pronunciation pattern must not be empty");
        }
        if self.pattern.chars().count() > 1_000 {
            bail!("pronunciation pattern is too long");
        }
        if self.replacement.chars().count() > 1_000 {
            bail!("pronunciation replacement is too long");
        }
        Ok(())
    }
}

/// Роль діалогу: ім'я в тексті та голос, яким її озвучувати.
///
/// У главі репліка позначається як `Ім'я: текст` або `[Ім'я] текст`.
/// Якщо `voice` не задано, використовується голос проєкту.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

impl Role {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            voice: None,
        }
    }

    pub fn with_voice(name: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            voice: Some(voice.into()),
        }
    }

    /// Обрізати пробіли в імені та голосі; порожній голос стає `None`.
    pub fn normalize(&mut self) {
        self.name = self.name.trim().to_string();
        if let Some(voice) = &mut self.voice {
            *voice = voice.trim().to_string();
            if voice.is_empty() {
                self.voice = None;
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("role name must not be empty");
        }
        if self.name != self.name.trim() {
            bail!("role name must not have leading or trailing whitespace");
        }
        if self.name.chars().count() > MAX_ROLE_NAME_CHARS {
            bail!("role name is too long");
        }
        if self.name.chars().any(char::is_control) {
            bail!("role name contains control characters");
        }
        if self.name.contains([':', '[', ']']) {
            bail!("role name must not contain ':', '[' or ']'");
        }
        if let Some(voice) = &self.voice {
            if voice.trim().is_empty() {
                bail!("role voice must not be empty");
            }
            if voice != voice.trim() {
                bail!("role voice must not have leading or trailing whitespace");
            }
            if voice.chars().count() > 100 {
                bail!("role voice name is too long");
            }
        }
        Ok(())
    }
}

/// Перевірити набір ролей: ліміти, коректність імен і унікальність.
pub fn validate_roles(roles: &[Role]) -> Result<()> {
    if roles.len() > MAX_ROLES {
        bail!("too many roles: {}", roles.len());
    }
    let mut seen = HashSet::with_capacity(roles.len());
    for role in roles {
        role.validate()?;
        if !seen.insert(role.name.to_lowercase()) {
            bail!("duplicate role name: {}", role.name);
        }
    }
    Ok(())
}

fn default_true() -> bool {
    true
}

fn default_target_lufs() -> f64 {
    -19.0
}

fn default_true_peak_db() -> f64 {
    -3.0
}

/// Налаштування нормалізації гучності (типово ACX).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoudnessSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_target_lufs")]
    pub target_lufs: f64,
    #[serde(default = "default_true_peak_db")]
    pub true_peak_db: f64,
}

impl Default for LoudnessSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            target_lufs: default_target_lufs(),
            true_peak_db: default_true_peak_db(),
        }
    }
}

impl LoudnessSettings {
    pub fn validate(&self) -> Result<()> {
        if !self.target_lufs.is_finite() || !(-40.0..=-5.0).contains(&self.target_lufs) {
            bail!("loudness target_lufs must be in -40..=-5");
        }
        if !self.true_peak_db.is_finite() || !(-9.0..=0.0).contains(&self.true_peak_db) {
            bail!("loudness true_peak_db must be in -9..=0");
        }
        Ok(())
    }
}

fn default_speed() -> f32 {
    1.0
}

fn default_max_chars() -> usize {
    1_200
}

fn default_crossfade_ms() -> u32 {
    40
}

/// Налаштування синтезу проєкту.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default = "default_crossfade_ms")]
    pub crossfade_ms: u32,
    /// `true`/`false` або `"neural"`/`"rule"`/`"off"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_verbalize: Option<serde_json::Value>,
    #[serde(default)]
    pub raw_stress: bool,
    #[serde(default)]
    pub loudness: LoudnessSettings,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            voice: None,
            speed: default_speed(),
            max_chars: default_max_chars(),
            crossfade_ms: default_crossfade_ms(),
            auto_verbalize: None,
            raw_stress: false,
            loudness: LoudnessSettings::default(),
        }
    }
}

impl ProjectSettings {
    pub fn validate(&self) -> Result<()> {
        if !self.speed.is_finite() || !(0.5..=2.0).contains(&self.speed) {
            bail!("speed must be in 0.5..=2.0");
        }
        if !(50..=5_000).contains(&self.max_chars) {
            bail!("max_chars must be in 50..=5000");
        }
        if self.crossfade_ms > 200 {
            bail!("crossfade_ms must be in 0..=200");
        }
        if let Some(voice) = &self.voice {
            if voice.chars().count() > 100 {
                bail!("voice name is too long");
            }
        }
        self.loudness.validate()
    }
}

fn default_language() -> String {
    "uk".to_string()
}

fn default_schema_version() -> u32 {
    PROJECT_SCHEMA_VERSION
}

/// Звідки імпортовано проєкт.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_at: Option<String>,
}

/// Результат нормалізації гучності глави.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoudnessReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub input_true_peak_db: f64,
    pub output_true_peak_db: f64,
    pub gain_db: f64,
    pub target_lufs: f64,
    pub target_true_peak_db: f64,
    #[serde(default)]
    pub limited: bool,
}

/// Межі одного синтезованого чанка (для субтитрів).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkTiming {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Роль діалогу, якій належить чанк (`None` — нарація).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

impl ChunkTiming {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Глава проєкту.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub index: usize,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub status: ChapterStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loudness: Option<LoudnessReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunk_timings: Vec<ChunkTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

impl Chapter {
    pub fn new(index: usize, title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            index,
            title: title.into(),
            text: text.into(),
            status: ChapterStatus::Empty,
            job_id: None,
            audio_bytes: None,
            duration_ms: None,
            loudness: None,
            chunk_timings: Vec::new(),
            error: None,
            updated_at: None,
        }
    }

    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    pub fn mark_queued(&mut self, job_id: impl Into<String>) {
        self.status = ChapterStatus::Queued;
        self.job_id = Some(job_id.into());
        self.error = None;
        self.updated_at = Some(now_rfc3339());
    }

    pub fn mark_running(&mut self) {
        self.status = ChapterStatus::Running;
        self.error = None;
        self.updated_at = Some(now_rfc3339());
    }

    pub fn mark_completed(
        &mut self,
        audio_bytes: u64,
        duration_ms: u64,
        loudness: Option<LoudnessReport>,
        chunk_timings: Vec<ChunkTiming>,
    ) {
        self.status = ChapterStatus::Completed;
        self.audio_bytes = Some(audio_bytes);
        self.duration_ms = Some(duration_ms);
        self.loudness = loudness;
        self.chunk_timings = chunk_timings;
        self.error = None;
        self.updated_at = Some(now_rfc3339());
    }

    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = ChapterStatus::Failed;
        self.error = Some(error.into());
        self.updated_at = Some(now_rfc3339());
    }

    pub fn mark_cancelled(&mut self) {
        self.status = ChapterStatus::Cancelled;
        self.error = Some("cancelled".to_string());
        self.updated_at = Some(now_rfc3339());
    }

    /// Позначити главу застарілою: текст або налаштування змінилися.
    ///
    /// Активні глави теж стають `stale` — job, що вже виконується, побачить
    /// зміну тексту/знімка налаштувань і не перезапише актуальний стан.
    pub fn mark_stale(&mut self) {
        if self.status != ChapterStatus::Empty {
            self.status = ChapterStatus::Stale;
        }
        self.error = None;
        self.updated_at = Some(now_rfc3339());
    }

    /// Скинути стан генерації (наприклад, під час імпорту `.speechproj`).
    pub fn reset_generation(&mut self) {
        self.status = ChapterStatus::Empty;
        self.job_id = None;
        self.audio_bytes = None;
        self.duration_ms = None;
        self.loudness = None;
        self.chunk_timings.clear();
        self.error = None;
        self.updated_at = None;
    }
}

/// Проєкт аудіокниги.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    #[serde(default = "default_schema_version")]
    pub version: u32,
    pub title: String,
    #[serde(default)]
    pub author: String,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub source: SourceInfo,
    #[serde(default)]
    pub settings: ProjectSettings,
    /// Ролі діалогу: ім'я в тексті -> голос.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<Role>,
    #[serde(default)]
    pub chapters: Vec<Chapter>,
    #[serde(default)]
    pub pronunciation: Vec<PronunciationRule>,
    pub created_at: String,
    pub updated_at: String,
}

impl Project {
    pub fn new(title: impl Into<String>) -> Self {
        let now = now_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            version: PROJECT_SCHEMA_VERSION,
            title: title.into(),
            author: String::new(),
            language: default_language(),
            source: SourceInfo::default(),
            settings: ProjectSettings::default(),
            roles: Vec::new(),
            chapters: Vec::new(),
            pronunciation: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn from_chapters(title: impl Into<String>, chapters: Vec<Chapter>) -> Self {
        let mut project = Self::new(title);
        project.chapters = chapters;
        project.normalize();
        project
    }

    pub fn touch(&mut self) {
        self.updated_at = now_rfc3339();
    }

    /// Привести індекси глав до порядку, назви ролей — до канонічного вигляду
    /// та оновити версію.
    pub fn normalize(&mut self) {
        self.version = PROJECT_SCHEMA_VERSION;
        for role in &mut self.roles {
            role.normalize();
        }
        for (position, chapter) in self.chapters.iter_mut().enumerate() {
            chapter.index = position;
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !valid_project_id(&self.id) {
            bail!("invalid project id");
        }
        if self.version == 0 || self.version > PROJECT_SCHEMA_VERSION {
            bail!(
                "unsupported project schema version {} (max {})",
                self.version,
                PROJECT_SCHEMA_VERSION
            );
        }
        let title = self.title.trim();
        if title.is_empty() {
            bail!("project title must not be empty");
        }
        if title.chars().count() > MAX_TITLE_CHARS {
            bail!("project title is too long");
        }
        if self.chapters.len() > MAX_CHAPTERS {
            bail!("project has too many chapters: {}", self.chapters.len());
        }
        let mut total_chars = 0usize;
        for (position, chapter) in self.chapters.iter().enumerate() {
            if chapter.index != position {
                bail!(
                    "chapter index mismatch at position {position}: {}",
                    chapter.index
                );
            }
            if chapter.title.chars().count() > MAX_TITLE_CHARS {
                bail!("chapter {position} title is too long");
            }
            let chars = chapter.char_count();
            if chars > MAX_CHAPTER_CHARS {
                bail!("chapter {position} is too long: {chars} chars");
            }
            total_chars = total_chars.saturating_add(chars);
        }
        if total_chars > MAX_TOTAL_CHARS {
            bail!("project text is too long: {total_chars} chars");
        }
        if self.pronunciation.len() > MAX_PRONUNCIATION_RULES {
            bail!("too many pronunciation rules: {}", self.pronunciation.len());
        }
        for rule in &self.pronunciation {
            rule.validate()?;
        }
        validate_roles(&self.roles)?;
        self.settings.validate()?;
        Ok(())
    }

    pub fn total_chars(&self) -> usize {
        self.chapters.iter().map(Chapter::char_count).sum()
    }

    pub fn chapter(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }

    pub fn chapter_mut(&mut self, index: usize) -> Option<&mut Chapter> {
        self.chapters.get_mut(index)
    }

    /// Позначити всі готові глави застарілими (змінилися налаштування).
    pub fn mark_all_stale(&mut self) {
        for chapter in &mut self.chapters {
            chapter.mark_stale();
        }
        self.touch();
    }

    /// Підготувати імпортований `.speechproj` до створення нового проєкту.
    pub fn reset_for_import(&mut self) {
        self.id = uuid::Uuid::new_v4().to_string();
        let now = now_rfc3339();
        self.created_at = now.clone();
        self.updated_at = now;
        self.version = PROJECT_SCHEMA_VERSION;
        self.normalize();
        for chapter in &mut self.chapters {
            chapter.reset_generation();
        }
    }

    pub fn summary(&self) -> ProjectSummary {
        let completed = self
            .chapters
            .iter()
            .filter(|chapter| chapter.status.is_ready())
            .count();
        let active = self.chapters.iter().any(|chapter| chapter.is_active());
        let failed = self
            .chapters
            .iter()
            .any(|chapter| matches!(chapter.status, ChapterStatus::Failed));
        let status = if active {
            "running"
        } else if !self.chapters.is_empty() && completed == self.chapters.len() {
            "completed"
        } else if failed {
            "partial"
        } else {
            "draft"
        };
        ProjectSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            author: self.author.clone(),
            language: self.language.clone(),
            chapters: self.chapters.len(),
            completed_chapters: completed,
            total_chars: self.total_chars(),
            total_duration_ms: self
                .chapters
                .iter()
                .filter_map(|chapter| chapter.duration_ms)
                .sum(),
            status: status.to_string(),
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
        }
    }
}

/// Короткий опис проєкту для списку.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub id: String,
    pub title: String,
    pub author: String,
    pub language: String,
    pub chapters: usize,
    pub completed_chapters: usize,
    pub total_chars: usize,
    pub total_duration_ms: u64,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Файлове сховище проєктів.
#[derive(Debug, Clone)]
pub struct ProjectStore {
    dir: PathBuf,
}

impl ProjectStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create projects dir {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn project_dir(&self, id: &str) -> Result<PathBuf> {
        if !valid_project_id(id) {
            bail!("invalid project id");
        }
        Ok(self.dir.join(id))
    }

    fn project_file(&self, id: &str) -> Result<PathBuf> {
        Ok(self.project_dir(id)?.join("project.json"))
    }

    pub fn exists(&self, id: &str) -> bool {
        self.project_dir(id)
            .map(|dir| dir.join("project.json").is_file())
            .unwrap_or(false)
    }

    /// Створити новий проєкт; помилка, якщо каталог уже існує.
    pub fn create(&self, project: &Project) -> Result<()> {
        project.validate()?;
        let dir = self.project_dir(&project.id)?;
        if dir.exists() {
            bail!("project {} already exists", project.id);
        }
        std::fs::create_dir_all(dir.join("chapters"))
            .with_context(|| format!("create project dir {}", dir.display()))?;
        self.write_project(project)
    }

    /// Зберегти проєкт, оновивши `updated_at`.
    pub fn save(&self, project: &mut Project) -> Result<()> {
        project.normalize();
        project.touch();
        project.validate()?;
        self.write_project(project)
    }

    fn write_project(&self, project: &Project) -> Result<()> {
        let path = self.project_file(&project.id)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create project dir {}", parent.display()))?;
        }
        let json = serde_json::to_vec_pretty(project).context("serialize project")?;
        write_atomic(&path, &json)
    }

    pub fn load(&self, id: &str) -> Result<Option<Project>> {
        let path = self.project_file(id)?;
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let mut project: Project = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse project {}", path.display()))?;
        project.normalize();
        project.validate()?;
        Ok(Some(project))
    }

    /// Усі проєкти, найновіші першими; пошкоджені пропускаються.
    pub fn list(&self) -> Result<Vec<ProjectSummary>> {
        let mut projects = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("read projects dir {}", self.dir.display()))?
        {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            if !valid_project_id(&id) {
                continue;
            }
            match self.load(&id) {
                Ok(Some(project)) => projects.push(project.summary()),
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(project = %id, error = %error, "skipping malformed project");
                }
            }
        }
        projects.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(projects)
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let dir = self.project_dir(id)?;
        if !dir.exists() {
            return Ok(false);
        }
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("remove project dir {}", dir.display()))?;
        Ok(true)
    }

    pub fn chapters_dir(&self, id: &str) -> Result<PathBuf> {
        Ok(self.project_dir(id)?.join("chapters"))
    }

    pub fn chapter_audio_path(&self, id: &str, index: usize) -> Result<PathBuf> {
        if index >= MAX_CHAPTERS {
            bail!("invalid chapter index");
        }
        Ok(self.chapters_dir(id)?.join(format!("{index}.wav")))
    }

    pub fn save_chapter_audio(&self, id: &str, index: usize, bytes: &[u8]) -> Result<PathBuf> {
        let dir = self.project_dir(id)?;
        if !dir.join("project.json").is_file() {
            bail!("project {id} does not exist");
        }
        let path = self.chapter_audio_path(id, index)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create chapters dir {}", parent.display()))?;
        }
        write_atomic(&path, bytes)?;
        Ok(path)
    }

    pub fn load_chapter_audio(&self, id: &str, index: usize) -> Result<Option<Vec<u8>>> {
        let path = self.chapter_audio_path(id, index)?;
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        Ok(Some(bytes))
    }

    pub fn chapter_audio_exists(&self, id: &str, index: usize) -> bool {
        self.chapter_audio_path(id, index)
            .map(|path| path.is_file())
            .unwrap_or(false)
    }

    pub fn delete_chapter_audio(&self, id: &str, index: usize) -> Result<bool> {
        let path = self.chapter_audio_path(id, index)?;
        if !path.is_file() {
            return Ok(false);
        }
        std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        Ok(true)
    }

    /// Серіалізувати `.speechproj` (JSON-манифест без аудіо).
    pub fn speechproj_bytes(project: &Project) -> Result<Vec<u8>> {
        project.validate()?;
        serde_json::to_vec_pretty(project).context("serialize .speechproj")
    }

    /// Розібрати `.speechproj`; id/статуси не змінюються (див. `reset_for_import`).
    pub fn parse_speechproj(bytes: &[u8]) -> Result<Project> {
        let mut project: Project =
            serde_json::from_slice(bytes).context("parse .speechproj JSON")?;
        project.normalize();
        project.validate()?;
        Ok(project)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    let tmp = path.with_file_name(format!(".{file_name}.tmp-{}", uuid::Uuid::new_v4()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> ProjectStore {
        let dir =
            std::env::temp_dir().join(format!("promin-projects-test-{}", uuid::Uuid::new_v4()));
        ProjectStore::new(dir).unwrap()
    }

    fn sample_project() -> Project {
        Project::from_chapters(
            "Тестова книга",
            vec![
                Chapter::new(0, "Розділ 1", "Привіт, світе!"),
                Chapter::new(1, "Розділ 2", "Другий розділ."),
            ],
        )
    }

    #[test]
    fn create_load_save_list_delete() {
        let store = temp_store();
        let mut project = sample_project();
        store.create(&project).unwrap();

        let loaded = store.load(&project.id).unwrap().unwrap();
        assert_eq!(loaded.title, "Тестова книга");
        assert_eq!(loaded.chapters.len(), 2);
        assert_eq!(loaded.chapters[1].index, 1);

        project.chapters[0].mark_completed(
            100,
            1_000,
            None,
            vec![ChunkTiming {
                start_ms: 0,
                end_ms: 1_000,
                text: "Привіт, світе!".into(),
                role: None,
            }],
        );
        store.save(&mut project).unwrap();
        let loaded = store.load(&project.id).unwrap().unwrap();
        assert_eq!(loaded.chapters[0].status, ChapterStatus::Completed);
        assert_eq!(loaded.chapters[0].chunk_timings.len(), 1);

        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].completed_chapters, 1);

        assert!(store.delete(&project.id).unwrap());
        assert!(store.load(&project.id).unwrap().is_none());
        assert!(!store.delete(&project.id).unwrap());
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn rejects_path_traversal_and_bad_ids() {
        let store = temp_store();
        assert!(store.load("../etc/passwd").is_err());
        assert!(store.chapter_audio_path("a/b", 0).is_err());
        assert!(store.chapter_audio_path("valid-id", MAX_CHAPTERS).is_err());
        assert!(!valid_project_id(""));
        assert!(!valid_project_id("a b"));
        assert!(valid_project_id("abc-123_XYZ"));
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn validates_limits() {
        let mut project = sample_project();
        project.title = " ".into();
        assert!(project.validate().is_err());

        let mut project = sample_project();
        project.chapters[0].text = "а".repeat(MAX_CHAPTER_CHARS + 1);
        assert!(project.validate().is_err());

        let mut project = sample_project();
        project.settings.speed = 9.0;
        assert!(project.validate().is_err());

        let mut project = sample_project();
        project.pronunciation.push(PronunciationRule {
            pattern: " ".into(),
            replacement: "x".into(),
            regex: false,
            case_sensitive: false,
        });
        assert!(project.validate().is_err());
    }

    #[test]
    fn chapter_status_transitions() {
        let mut chapter = Chapter::new(0, "Глава", "Текст");
        assert_eq!(chapter.status, ChapterStatus::Empty);

        chapter.mark_queued("job-1");
        assert!(chapter.is_active());
        chapter.mark_running();
        assert_eq!(chapter.status, ChapterStatus::Running);
        chapter.mark_completed(10, 500, None, Vec::new());
        assert!(chapter.status.is_ready());
        assert_eq!(chapter.audio_bytes, Some(10));

        chapter.mark_stale();
        assert_eq!(chapter.status, ChapterStatus::Stale);

        chapter.mark_queued("job-2");
        chapter.mark_stale();
        assert_eq!(chapter.status, ChapterStatus::Stale);
        chapter.mark_running();
        chapter.mark_stale();
        assert_eq!(chapter.status, ChapterStatus::Stale);
        chapter.mark_failed("boom");
        assert_eq!(chapter.status, ChapterStatus::Failed);
        chapter.mark_stale();
        assert_eq!(chapter.status, ChapterStatus::Stale);

        chapter.reset_generation();
        assert_eq!(chapter.status, ChapterStatus::Empty);
        assert!(chapter.job_id.is_none());
        assert!(chapter.chunk_timings.is_empty());
    }

    #[test]
    fn speechproj_roundtrip_and_import_reset() {
        let mut project = sample_project();
        project.chapters[0].mark_completed(42, 700, None, Vec::new());
        let bytes = ProjectStore::speechproj_bytes(&project).unwrap();
        let parsed = ProjectStore::parse_speechproj(&bytes).unwrap();
        assert_eq!(parsed.id, project.id);
        assert_eq!(parsed.chapters[0].status, ChapterStatus::Completed);

        let mut imported = parsed;
        let old_id = imported.id.clone();
        imported.reset_for_import();
        assert_ne!(imported.id, old_id);
        assert_eq!(imported.chapters[0].status, ChapterStatus::Empty);
        assert!(imported.chapters[0].job_id.is_none());
        assert!(imported.chapters[0].audio_bytes.is_none());
    }

    #[test]
    fn chapter_audio_roundtrip() {
        let store = temp_store();
        let project = sample_project();
        store.create(&project).unwrap();

        assert!(!store.chapter_audio_exists(&project.id, 0));
        store.save_chapter_audio(&project.id, 0, b"RIFF").unwrap();
        assert!(store.chapter_audio_exists(&project.id, 0));
        assert_eq!(
            store.load_chapter_audio(&project.id, 0).unwrap().unwrap(),
            b"RIFF"
        );
        assert!(store.delete_chapter_audio(&project.id, 0).unwrap());
        assert!(!store.chapter_audio_exists(&project.id, 0));
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn summary_counts() {
        let mut project = sample_project();
        project.chapters[0].mark_completed(1, 1_500, None, Vec::new());
        let summary = project.summary();
        assert_eq!(summary.chapters, 2);
        assert_eq!(summary.completed_chapters, 1);
        assert_eq!(summary.total_duration_ms, 1_500);
        assert_eq!(summary.status, "draft");
        assert_eq!(summary.total_chars, project.total_chars());
    }

    #[test]
    fn roles_roundtrip_and_normalize() {
        let store = temp_store();
        let mut project = sample_project();
        project.roles = vec![Role::with_voice(" Іван ", " Speaker_1 ")];
        project.normalize();
        store.create(&project).unwrap();

        let loaded = store.load(&project.id).unwrap().unwrap();
        assert_eq!(loaded.roles, vec![Role::with_voice("Іван", "Speaker_1")]);
        assert_eq!(loaded.version, PROJECT_SCHEMA_VERSION);
        std::fs::remove_dir_all(store.dir()).unwrap();
    }

    #[test]
    fn rejects_invalid_roles() {
        let mut project = sample_project();

        project.roles = vec![Role::new("")];
        assert!(project.validate().is_err());

        project.roles = vec![Role::new("Іван"), Role::new("іван")];
        assert!(project.validate().is_err());

        project.roles = vec![Role::new("Іван:")];
        assert!(project.validate().is_err());

        project.roles = vec![Role::new("Іван"), Role::new("Марія")];
        assert!(project.validate().is_ok());
    }

    #[test]
    fn legacy_v1_project_without_roles_loads() {
        let json = r#"{
            "id": "legacy",
            "version": 1,
            "title": "Книга",
            "created_at": "2026-01-01T00:00:00.000Z",
            "updated_at": "2026-01-01T00:00:00.000Z"
        }"#;
        let project: Project = serde_json::from_str(json).unwrap();
        assert!(project.roles.is_empty());
        assert!(project.validate().is_ok());
    }
}
