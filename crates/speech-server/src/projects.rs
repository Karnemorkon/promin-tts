//! REST API проєктів аудіокниг (v0.8).
//!
//! Проєкт — це набір глав із текстом і налаштуваннями синтезу. Аудіо глав
//! генерується через `JobManager` (job-и з `AudioLocation::ProjectChapter`),
//! а `.speechproj` є JSON-манифестом без аудіо.

use std::path::Path as StdPath;

use axum::{
    extract::{Multipart, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;
use speech_core::jobs::JobRequest;
use speech_core::projects::{
    now_rfc3339, valid_project_id, validate_roles, Chapter, ChapterStatus, LoudnessSettings,
    Project, ProjectSettings, ProjectStore, PronunciationRule, Role, MAX_TITLE_CHARS,
};
use speech_text::import::{import_book, ImportError, ImportedBook};

use crate::pronunciation::PronunciationPayload;
use crate::subtitles::{render_srt, render_vtt};
use crate::AppState;

/// Помилка API проєктів.
#[derive(Debug)]
pub enum ProjectApiError {
    NotFound(&'static str),
    BadRequest(String),
    PayloadTooLarge(String),
    Conflict(String),
    Internal(String),
}

impl ProjectApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self::Conflict(message.into())
    }

    fn internal(context: &'static str, error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, context, "project request failed");
        Self::Internal(error.to_string())
    }
}

impl IntoResponse for ProjectApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message.to_string()),
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::PayloadTooLarge(message) => (StatusCode::PAYLOAD_TOO_LARGE, message),
            Self::Conflict(message) => (StatusCode::CONFLICT, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

/// Налаштування за замовчуванням для нових проєктів (з env-цілей LUFS).
fn default_settings(loudness: &LoudnessSettings) -> ProjectSettings {
    ProjectSettings {
        loudness: loudness.clone(),
        ..ProjectSettings::default()
    }
}

fn load_project(store: &ProjectStore, id: &str) -> Result<Project, ProjectApiError> {
    if !valid_project_id(id) {
        return Err(ProjectApiError::bad("invalid project id"));
    }
    match store.load(id) {
        Ok(Some(project)) => Ok(project),
        Ok(None) => Err(ProjectApiError::NotFound("unknown project")),
        Err(error) => Err(ProjectApiError::internal("load project", error)),
    }
}

fn load_chapter(project: &Project, index: usize) -> Result<&Chapter, ProjectApiError> {
    project
        .chapter(index)
        .ok_or(ProjectApiError::NotFound("unknown chapter"))
}

fn save_project(store: &ProjectStore, project: &mut Project) -> Result<(), ProjectApiError> {
    project
        .validate()
        .map_err(|error| ProjectApiError::bad(error.to_string()))?;
    store
        .save(project)
        .map_err(|error| ProjectApiError::internal("save project", error))
}

/// Перевірити, що проєктні regex-правила компілюються.
fn validate_pronunciation(state: &AppState, project: &Project) -> Result<(), ProjectApiError> {
    state
        .pronunciation
        .merged_pronouncer(&project.pronunciation)
        .map(|_| ())
        .map_err(|error| ProjectApiError::bad(error.to_string()))
}

fn map_import_error(error: ImportError) -> ProjectApiError {
    match error {
        ImportError::UnsupportedFormat => ProjectApiError::bad("unsupported book format"),
        ImportError::Invalid(message) => ProjectApiError::bad(message),
        ImportError::LimitExceeded(message) => ProjectApiError::PayloadTooLarge(message),
        ImportError::Zip(message) => {
            ProjectApiError::bad(format!("invalid ZIP archive: {message}"))
        }
        ImportError::Xml(message) => ProjectApiError::bad(format!("invalid XML: {message}")),
    }
}

#[derive(Debug, Deserialize)]
struct ChapterInput {
    #[serde(default)]
    title: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateProjectRequest {
    title: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    chapters: Vec<ChapterInput>,
    #[serde(default)]
    settings: Option<ProjectSettings>,
    #[serde(default)]
    pronunciation: Vec<PronunciationRule>,
    #[serde(default)]
    roles: Vec<Role>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProjectRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    settings: Option<ProjectSettings>,
    #[serde(default)]
    pronunciation: Option<Vec<PronunciationRule>>,
    #[serde(default)]
    roles: Option<Vec<Role>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateChapterRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct GenerateRequest {
    /// Індекси глав; відсутнє — усі глави.
    #[serde(default)]
    chapters: Option<Vec<usize>>,
    /// Перегенерувати навіть завершені глави.
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Deserialize)]
pub struct ListProjectsQuery {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SubtitleQuery {
    #[serde(default)]
    format: Option<String>,
}

/// `GET /v1/projects`: список проєктів (найновіші першими).
pub async fn list_projects(
    State(state): State<AppState>,
    Query(query): Query<ListProjectsQuery>,
) -> Result<Json<serde_json::Value>, ProjectApiError> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let mut projects = state
        .projects
        .list()
        .map_err(|error| ProjectApiError::internal("list projects", error))?;
    projects.truncate(limit);
    Ok(Json(json!({ "projects": projects })))
}

/// `POST /v1/projects`: створити проєкт із JSON.
pub async fn create_project(
    State(state): State<AppState>,
    Json(request): Json<CreateProjectRequest>,
) -> Result<Response, ProjectApiError> {
    if request.title.trim().is_empty() {
        return Err(ProjectApiError::bad("project title must not be empty"));
    }
    let mut project = Project::new(request.title.trim());
    project.author = request.author;
    project.language = request
        .language
        .filter(|language| !language.trim().is_empty())
        .unwrap_or_else(|| "uk".to_string());
    project.settings = request
        .settings
        .unwrap_or_else(|| default_settings(&state.default_loudness));
    project.pronunciation = request.pronunciation;
    project.roles = request.roles;
    project.chapters = request
        .chapters
        .into_iter()
        .enumerate()
        .map(|(index, chapter)| {
            let title = chapter
                .title
                .filter(|title| !title.trim().is_empty())
                .unwrap_or_else(|| format!("Розділ {}", index + 1));
            Chapter::new(index, title, chapter.text)
        })
        .collect();
    project.normalize();
    project
        .validate()
        .map_err(|error| ProjectApiError::bad(error.to_string()))?;
    validate_pronunciation(&state, &project)?;
    state
        .projects
        .create(&project)
        .map_err(|error| ProjectApiError::internal("create project", error))?;
    Ok((StatusCode::CREATED, Json(project)).into_response())
}

/// `POST /v1/projects/import`: multipart-імпорт книги або `.speechproj`.
pub async fn import_project(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ProjectApiError> {
    let mut file: Option<(String, axum::body::Bytes)> = None;
    let mut title: Option<String> = None;
    let mut author: Option<String> = None;
    let mut language: Option<String> = None;
    let mut settings: Option<ProjectSettings> = None;
    let mut pronunciation: Option<Vec<PronunciationRule>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ProjectApiError::bad(format!("invalid multipart body: {error}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" | "book" => {
                let filename = field
                    .file_name()
                    .filter(|name| !name.trim().is_empty())
                    .unwrap_or("book")
                    .to_string();
                let data = field.bytes().await.map_err(|error| {
                    ProjectApiError::bad(format!("cannot read upload: {error}"))
                })?;
                file = Some((filename, data));
            }
            "title" => title = Some(read_text_field(field).await?),
            "author" => author = Some(read_text_field(field).await?),
            "language" => language = Some(read_text_field(field).await?),
            "settings" => {
                let raw = read_text_field(field).await?;
                if !raw.trim().is_empty() {
                    settings = Some(serde_json::from_str(&raw).map_err(|error| {
                        ProjectApiError::bad(format!("invalid settings JSON: {error}"))
                    })?);
                }
            }
            "pronunciation" => {
                let raw = read_text_field(field).await?;
                if !raw.trim().is_empty() {
                    let payload: PronunciationPayload =
                        serde_json::from_str(&raw).map_err(|error| {
                            ProjectApiError::bad(format!("invalid pronunciation JSON: {error}"))
                        })?;
                    pronunciation = Some(payload.into_rules());
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some((filename, bytes)) = file else {
        return Err(ProjectApiError::bad("multipart field `file` is required"));
    };
    if bytes.len() > speech_text::import::MAX_UPLOAD_BYTES {
        return Err(ProjectApiError::PayloadTooLarge(format!(
            "file is too large: {} bytes (max {})",
            bytes.len(),
            speech_text::import::MAX_UPLOAD_BYTES
        )));
    }

    let extension = StdPath::new(&filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let speechproj = extension.as_deref() == Some("speechproj") || looks_like_speechproj(&bytes);
    let mut project = if speechproj {
        let mut project = ProjectStore::parse_speechproj(&bytes)
            .map_err(|error| ProjectApiError::bad(error.to_string()))?;
        project.reset_for_import();
        project
    } else {
        let book = import_book(&filename, &bytes).map_err(map_import_error)?;
        let format = book.format.as_str().to_string();
        let mut project = project_from_book(book, &state.default_loudness);
        project.source.format = Some(format);
        project
    };

    if let Some(title) = title {
        if title.trim().is_empty() {
            return Err(ProjectApiError::bad("project title must not be empty"));
        }
        project.title = title;
    }
    if let Some(author) = author {
        project.author = author;
    }
    if let Some(language) = language.filter(|language| !language.trim().is_empty()) {
        project.language = language;
    }
    if let Some(settings) = settings {
        project.settings = settings;
    }
    if let Some(pronunciation) = pronunciation {
        project.pronunciation = pronunciation;
    }
    project.source.filename = Some(filename);
    project.source.imported_at = Some(now_rfc3339());
    project
        .validate()
        .map_err(|error| ProjectApiError::bad(error.to_string()))?;
    validate_pronunciation(&state, &project)?;
    state
        .projects
        .create(&project)
        .map_err(|error| ProjectApiError::internal("create imported project", error))?;
    Ok((StatusCode::CREATED, Json(project)).into_response())
}

async fn read_text_field(
    field: axum::extract::multipart::Field<'_>,
) -> Result<String, ProjectApiError> {
    field
        .text()
        .await
        .map_err(|error| ProjectApiError::bad(format!("cannot read multipart field: {error}")))
}

fn looks_like_speechproj(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(8_192)];
    let Ok(text) = std::str::from_utf8(head) else {
        return false;
    };
    let text = text.trim_start();
    text.starts_with('{') && text.contains("\"chapters\"") && text.contains("\"title\"")
}

fn project_from_book(book: ImportedBook, loudness: &LoudnessSettings) -> Project {
    let chapters = book
        .chapters
        .into_iter()
        .enumerate()
        .map(|(index, chapter)| Chapter::new(index, chapter.title, chapter.text))
        .collect();
    let mut project = Project::from_chapters(book.title, chapters);
    project.author = book.author;
    project.language = book.language;
    project.settings = default_settings(loudness);
    project
}

/// `GET /v1/projects/{id}`: повний проєкт.
pub async fn get_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Project>, ProjectApiError> {
    Ok(Json(load_project(&state.projects, &id)?))
}

/// `PUT /v1/projects/{id}`: метадані, налаштування та словник вимови.
pub async fn update_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateProjectRequest>,
) -> Result<Json<Project>, ProjectApiError> {
    let _guard = state.project_lock.lock().await;
    let mut project = load_project(&state.projects, &id)?;
    let mut stale = false;

    if let Some(title) = request.title {
        if title.trim().is_empty() {
            return Err(ProjectApiError::bad("project title must not be empty"));
        }
        project.title = title;
    }
    if let Some(author) = request.author {
        project.author = author;
    }
    if let Some(language) = request.language {
        if !language.trim().is_empty() {
            project.language = language;
        }
    }
    if let Some(settings) = request.settings {
        settings
            .validate()
            .map_err(|error| ProjectApiError::bad(error.to_string()))?;
        if settings != project.settings {
            project.settings = settings;
            stale = true;
        }
    }
    if let Some(pronunciation) = request.pronunciation {
        state
            .pronunciation
            .merged_pronouncer(&pronunciation)
            .map_err(|error| ProjectApiError::bad(error.to_string()))?;
        if pronunciation != project.pronunciation {
            project.pronunciation = pronunciation;
            stale = true;
        }
    }
    if let Some(mut roles) = request.roles {
        for role in &mut roles {
            role.normalize();
        }
        validate_roles(&roles).map_err(|error| ProjectApiError::bad(error.to_string()))?;
        if roles != project.roles {
            project.roles = roles;
            stale = true;
        }
    }

    if stale {
        project.mark_all_stale();
    } else {
        project.touch();
    }
    save_project(&state.projects, &mut project)?;
    Ok(Json(project))
}

/// `DELETE /v1/projects/{id}`: скасувати активні job-и та видалити каталог.
pub async fn delete_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ProjectApiError> {
    if !valid_project_id(&id) {
        return Err(ProjectApiError::bad("invalid project id"));
    }
    let _guard = state.project_lock.lock().await;
    if !state.projects.exists(&id) {
        return Err(ProjectApiError::NotFound("unknown project"));
    }
    let cancelled = state.jobs.cancel_project_jobs(&id);
    state.exports.cancel_project(&id);
    state
        .projects
        .delete(&id)
        .map_err(|error| ProjectApiError::internal("delete project", error))?;
    tracing::info!(project_id = id, cancelled, "project deleted");
    Ok(StatusCode::NO_CONTENT)
}

/// `PUT /v1/projects/{id}/pronunciation`: замінити словник проєкту.
pub async fn put_project_pronunciation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<PronunciationPayload>,
) -> Result<Json<Project>, ProjectApiError> {
    let rules = payload.into_rules();
    let _guard = state.project_lock.lock().await;
    let mut project = load_project(&state.projects, &id)?;
    state
        .pronunciation
        .merged_pronouncer(&rules)
        .map_err(|error| ProjectApiError::bad(error.to_string()))?;
    if project.pronunciation != rules {
        project.pronunciation = rules;
        project.mark_all_stale();
        save_project(&state.projects, &mut project)?;
    }
    Ok(Json(project))
}

/// `POST /v1/projects/{id}/generate`: поставити глави в чергу.
pub async fn generate_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<GenerateRequest>,
) -> Result<Response, ProjectApiError> {
    let mut skipped = Vec::new();
    let mut errors = Vec::new();
    let mut pending: Vec<(usize, String, JobRequest)> = Vec::new();
    let settings;
    let pronunciation;
    let roles;
    let project_id;

    // Спершу під lock фіксуємо `queued`-статуси та зберігаємо проєкт.
    // Job-и ставимо в чергу вже після відпускання lock: інакше воркер може
    // заблокуватися на оновленні глави, поки ми чекаємо на переповнену чергу.
    {
        let _guard = state.project_lock.lock().await;
        let mut project = load_project(&state.projects, &id)?;
        if project.chapters.is_empty() {
            return Err(ProjectApiError::bad("project has no chapters"));
        }

        let indices = match request.chapters {
            Some(chapters) => {
                if chapters.is_empty() {
                    return Err(ProjectApiError::bad("chapters must not be empty"));
                }
                let mut unique = chapters;
                unique.sort_unstable();
                unique.dedup();
                if unique.iter().any(|index| project.chapter(*index).is_none()) {
                    return Err(ProjectApiError::NotFound("unknown chapter"));
                }
                unique
            }
            None => (0..project.chapters.len()).collect(),
        };

        settings = project.settings.clone();
        pronunciation = project.pronunciation.clone();
        roles = project.roles.clone();
        project_id = project.id.clone();

        for index in indices {
            let chapter = project
                .chapter(index)
                .expect("chapter index validated above");
            if chapter.text.trim().is_empty() {
                errors.push(json!({
                    "chapter_index": index,
                    "error": "chapter text is empty",
                }));
                continue;
            }
            if chapter.is_active() {
                skipped.push(json!({
                    "chapter_index": index,
                    "reason": "chapter is already queued or running",
                }));
                continue;
            }
            if chapter.status == ChapterStatus::Completed && !request.force {
                skipped.push(json!({
                    "chapter_index": index,
                    "reason": "chapter is already completed (use force to regenerate)",
                }));
                continue;
            }

            let job_request = JobRequest {
                text: chapter.text.clone(),
                voice: settings.voice.clone(),
                speed: settings.speed,
                auto_verbalize: settings.auto_verbalize.clone(),
                raw_stress: settings.raw_stress,
                webhook_url: None,
                max_chars: Some(settings.max_chars),
                crossfade_ms: Some(settings.crossfade_ms),
            };
            let job_id = uuid::Uuid::new_v4().to_string();
            project
                .chapter_mut(index)
                .expect("chapter index validated above")
                .mark_queued(&job_id);
            pending.push((index, job_id, job_request));
        }

        if !pending.is_empty() {
            save_project(&state.projects, &mut project)?;
        }
    }

    let mut jobs = Vec::new();
    for (index, job_id, job_request) in pending {
        if !state.projects.exists(&project_id) {
            errors.push(json!({
                "chapter_index": index,
                "error": "project was deleted before the job was queued",
            }));
            continue;
        }
        match state
            .jobs
            .submit_chapter_with_id(
                job_id,
                &project_id,
                index,
                job_request,
                Some(settings.clone()),
                pronunciation.clone(),
                roles.clone(),
            )
            .await
        {
            Ok(job) => jobs.push(json!({
                "job_id": job.id,
                "chapter_index": index,
            })),
            Err(error) => {
                mark_chapter_failed(&state, &project_id, index, &error).await;
                errors.push(json!({
                    "chapter_index": index,
                    "error": error,
                }));
            }
        }
    }

    let status = if jobs.is_empty() && !errors.is_empty() {
        StatusCode::BAD_REQUEST
    } else if jobs.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    Ok((
        status,
        Json(json!({
            "project_id": project_id,
            "jobs": jobs,
            "skipped": skipped,
            "errors": errors,
        })),
    )
        .into_response())
}

async fn mark_chapter_failed(state: &AppState, project_id: &str, index: usize, error: &str) {
    let _guard = state.project_lock.lock().await;
    let Ok(Some(mut project)) = state.projects.load(project_id) else {
        return;
    };
    if let Some(chapter) = project.chapter_mut(index) {
        chapter.mark_failed(error.to_string());
    }
    if let Err(save_error) = state.projects.save(&mut project) {
        tracing::error!(
            project_id,
            chapter_index = index,
            error = %save_error,
            "cannot mark chapter as failed"
        );
    }
}

/// `GET /v1/projects/{id}/chapters/{index}`: одна глава.
pub async fn get_chapter(
    State(state): State<AppState>,
    Path((id, index)): Path<(String, usize)>,
) -> Result<Json<Chapter>, ProjectApiError> {
    let project = load_project(&state.projects, &id)?;
    Ok(Json(load_chapter(&project, index)?.clone()))
}

/// `PUT /v1/projects/{id}/chapters/{index}`: змінити назву/текст глави.
pub async fn update_chapter(
    State(state): State<AppState>,
    Path((id, index)): Path<(String, usize)>,
    Json(request): Json<UpdateChapterRequest>,
) -> Result<Json<Chapter>, ProjectApiError> {
    let _guard = state.project_lock.lock().await;
    let mut project = load_project(&state.projects, &id)?;
    let chapter = project
        .chapter_mut(index)
        .ok_or(ProjectApiError::NotFound("unknown chapter"))?;
    if let Some(title) = request.title {
        if title.chars().count() > MAX_TITLE_CHARS {
            return Err(ProjectApiError::bad("chapter title is too long"));
        }
        chapter.title = title;
    }
    if let Some(text) = request.text {
        if text != chapter.text {
            chapter.text = text;
            chapter.mark_stale();
        }
    }
    save_project(&state.projects, &mut project)?;
    Ok(Json(
        project
            .chapter(index)
            .expect("chapter index validated above")
            .clone(),
    ))
}

/// `GET /v1/projects/{id}/chapters/{index}/audio`: готовий WAV глави.
pub async fn chapter_audio(
    State(state): State<AppState>,
    Path((id, index)): Path<(String, usize)>,
) -> Result<Response, ProjectApiError> {
    let project = load_project(&state.projects, &id)?;
    let chapter = load_chapter(&project, index)?;
    if chapter.status != ChapterStatus::Completed {
        return Err(ProjectApiError::conflict(format!(
            "chapter is not completed (status: {})",
            chapter.status.as_str()
        )));
    }
    let bytes = state
        .projects
        .load_chapter_audio(&id, index)
        .map_err(|error| ProjectApiError::internal("read chapter audio", error))?;
    let Some(bytes) = bytes else {
        return Err(ProjectApiError::NotFound("chapter audio is missing"));
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"project-{id}-chapter-{index}.wav\""
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    Ok((headers, bytes).into_response())
}

/// `GET /v1/projects/{id}/chapters/{index}/subtitles?format=srt|vtt`.
pub async fn chapter_subtitles(
    State(state): State<AppState>,
    Path((id, index)): Path<(String, usize)>,
    Query(query): Query<SubtitleQuery>,
) -> Result<Response, ProjectApiError> {
    let project = load_project(&state.projects, &id)?;
    let chapter = load_chapter(&project, index)?;
    if chapter.status != ChapterStatus::Completed || chapter.chunk_timings.is_empty() {
        return Err(ProjectApiError::conflict(
            "subtitles are available only for completed chapters",
        ));
    }
    let format = query.format.unwrap_or_else(|| "srt".to_string());
    let format = format.trim().to_ascii_lowercase();
    let (body, content_type, extension) = match format.as_str() {
        "srt" => (
            render_srt(&chapter.chunk_timings),
            "application/x-subrip; charset=utf-8",
            "srt",
        ),
        "vtt" | "webvtt" => (
            render_vtt(&chapter.chunk_timings),
            "text/vtt; charset=utf-8",
            "vtt",
        ),
        other => {
            return Err(ProjectApiError::bad(format!(
                "unsupported subtitle format: {other}"
            )))
        }
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"project-{id}-chapter-{index}.{extension}\""
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    Ok((headers, body).into_response())
}

/// `GET /v1/projects/{id}/manifest`: JSON із посиланнями на аудіо/субтитри.
pub async fn project_manifest(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ProjectApiError> {
    let project = load_project(&state.projects, &id)?;
    let mut value = serde_json::to_value(&project)
        .map_err(|error| ProjectApiError::internal("serialize manifest", error))?;
    let base = state.public_url.trim_end_matches('/');
    if let Some(chapters) = value
        .get_mut("chapters")
        .and_then(|value| value.as_array_mut())
    {
        for (index, chapter) in chapters.iter_mut().enumerate() {
            let Some(object) = chapter.as_object_mut() else {
                continue;
            };
            let completed = project
                .chapter(index)
                .map(|chapter| chapter.status == ChapterStatus::Completed)
                .unwrap_or(false)
                && state.projects.chapter_audio_exists(&id, index);
            if completed {
                object.insert(
                    "audio_url".to_string(),
                    json!(format!("{base}/v1/projects/{id}/chapters/{index}/audio")),
                );
                object.insert(
                    "subtitles_srt_url".to_string(),
                    json!(format!(
                        "{base}/v1/projects/{id}/chapters/{index}/subtitles?format=srt"
                    )),
                );
                object.insert(
                    "subtitles_vtt_url".to_string(),
                    json!(format!(
                        "{base}/v1/projects/{id}/chapters/{index}/subtitles?format=vtt"
                    )),
                );
            }
        }
    }
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "speechproj_url".to_string(),
            json!(format!("{base}/v1/projects/{id}/speechproj")),
        );
    }
    Ok(Json(value))
}

/// `GET /v1/projects/{id}/speechproj`: скачати JSON-манифест проєкту.
pub async fn project_speechproj(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ProjectApiError> {
    let project = load_project(&state.projects, &id)?;
    let bytes = ProjectStore::speechproj_bytes(&project)
        .map_err(|error| ProjectApiError::internal("serialize .speechproj", error))?;
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"project-{id}.speechproj\""))
            .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    Ok((headers, bytes).into_response())
}

/// Після рестарту активні глави не мають живих job-ів, а завершені глави
/// без файлу аудіо не можна віддавати. Позначаємо їх відповідно.
pub fn reconcile_projects(store: &ProjectStore) -> anyhow::Result<usize> {
    let mut changed = 0;
    for entry in std::fs::read_dir(store.dir())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_project_id(&id) {
            continue;
        }
        let mut project = match store.load(&id) {
            Ok(Some(project)) => project,
            Ok(None) => continue,
            Err(error) => {
                tracing::warn!(project = %id, error = %error, "skipping malformed project");
                continue;
            }
        };
        let mut project_changed = false;
        for chapter in &mut project.chapters {
            if chapter.status.is_active() {
                chapter.mark_failed("generation interrupted by server restart");
                project_changed = true;
            } else if chapter.status == ChapterStatus::Completed
                && !store.chapter_audio_exists(&project.id, chapter.index)
            {
                chapter.mark_stale();
                chapter.error = Some("audio file is missing".to_string());
                project_changed = true;
            }
        }
        if project_changed {
            store.save(&mut project)?;
            changed += 1;
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use speech_text::import::{BookFormat, ImportedChapter};

    fn temp_store() -> ProjectStore {
        let dir =
            std::env::temp_dir().join(format!("promin-api-projects-{}", uuid::Uuid::new_v4()));
        ProjectStore::new(dir).unwrap()
    }

    fn imported_book() -> ImportedBook {
        ImportedBook {
            title: "Книга".to_string(),
            author: "Автор".to_string(),
            language: "uk".to_string(),
            format: BookFormat::Epub,
            chapters: vec![
                ImportedChapter {
                    title: "Розділ 1".to_string(),
                    text: "Перший текст.".to_string(),
                },
                ImportedChapter {
                    title: "Розділ 2".to_string(),
                    text: "Другий текст.".to_string(),
                },
            ],
        }
    }

    #[test]
    fn book_conversion_keeps_metadata_and_default_loudness() {
        let loudness = LoudnessSettings {
            enabled: true,
            target_lufs: -23.0,
            true_peak_db: -2.0,
        };
        let project = project_from_book(imported_book(), &loudness);
        assert_eq!(project.title, "Книга");
        assert_eq!(project.author, "Автор");
        assert_eq!(project.language, "uk");
        assert_eq!(project.chapters.len(), 2);
        assert_eq!(project.chapters[0].title, "Розділ 1");
        assert_eq!(project.settings.loudness, loudness);
    }

    #[test]
    fn speechproj_magic_detection() {
        assert!(looks_like_speechproj(br#"{"title":"Book","chapters":[]}"#));
        assert!(!looks_like_speechproj(b"<FictionBook></FictionBook>"));
        assert!(!looks_like_speechproj(b"\xff\xfe"));
    }

    #[test]
    fn reconcile_marks_interrupted_and_missing_audio() {
        let store = temp_store();
        let mut project = Project::from_chapters(
            "Книга",
            vec![
                Chapter::new(0, "Розділ 1", "Перший."),
                Chapter::new(1, "Розділ 2", "Другий."),
                Chapter::new(2, "Розділ 3", "Третій."),
            ],
        );
        project.chapters[0].mark_queued("job-1");
        project.chapters[1].mark_completed(10, 1_000, None, Vec::new());
        project.chapters[2].mark_completed(10, 1_000, None, Vec::new());
        store.create(&project).unwrap();
        store.save_chapter_audio(&project.id, 2, b"wav").unwrap();

        assert_eq!(reconcile_projects(&store).unwrap(), 1);
        let reloaded = store.load(&project.id).unwrap().unwrap();
        assert_eq!(reloaded.chapters[0].status, ChapterStatus::Failed);
        assert!(reloaded.chapters[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("restart"));
        assert_eq!(reloaded.chapters[1].status, ChapterStatus::Stale);
        assert!(reloaded.chapters[1]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("missing"));
        assert_eq!(reloaded.chapters[2].status, ChapterStatus::Completed);

        std::fs::remove_dir_all(store.dir()).unwrap();
    }
}
