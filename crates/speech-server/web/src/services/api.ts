import type {
  Project,
  ProjectSummary,
  CreateProjectRequest,
  UpdateProjectRequest,
  UpdateChapterRequest,
  AddChapterRequest,
  Chapter,
  ChapterTakes,
  Take,
  RegenerateTakeRequest,
  JobStatus,
  ModelInfo,
  QCJob,
} from '../types';

let baseUrl: string = '';
let apiKey: string | null = null;

export function setBaseUrl(url: string) {
  baseUrl = url.replace(/\/$/, '');
  if (baseUrl) {
    localStorage.setItem('promin_base_url', baseUrl);
  } else {
    localStorage.removeItem('promin_base_url');
  }
}

export function getBaseUrl(): string {
  if (!baseUrl && typeof localStorage !== 'undefined') {
    baseUrl = localStorage.getItem('promin_base_url') ?? '';
  }
  return baseUrl;
}

export function setApiKey(key: string | null) {
  apiKey = key;
  if (key) {
    localStorage.setItem('promin_api_key', key);
  } else {
    localStorage.removeItem('promin_api_key');
  }
}

export function loadApiKey(): string | null {
  if (apiKey) return apiKey;
  if (typeof localStorage !== 'undefined') {
    apiKey = localStorage.getItem('promin_api_key');
  }
  return apiKey;
}

function headers(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  const key = loadApiKey();
  if (key) {
    h['Authorization'] = `Bearer ${key}`;
    h['X-API-Key'] = key;
  }
  return h;
}

function authHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  const key = loadApiKey();
  if (key) {
    h['Authorization'] = `Bearer ${key}`;
    h['X-API-Key'] = key;
  }
  return h;
}

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  // FormData потребує Content-Type із boundary, який виставляє браузер.
  const isFormData = opts?.body instanceof FormData;
  const baseHeaders = isFormData ? authHeaders() : headers();
  const res = await fetch(url, {
    ...opts,
    headers: { ...baseHeaders, ...(opts?.headers as Record<string, string> | undefined) },
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    let message = errText;
    try {
      const parsed = JSON.parse(errText) as { error?: string };
      if (parsed.error) message = parsed.error;
    } catch {
      /* тіло не JSON — лишаємо як є */
    }
    throw new Error(
      message || `${res.status} ${res.statusText}`
    );
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

/* ---------- Projects ---------- */
export async function listProjects(): Promise<ProjectSummary[]> {
  const data = await request<{ projects: ProjectSummary[] }>('/v1/projects');
  return data.projects ?? [];
}

export function createProject(body: CreateProjectRequest) {
  return request<Project>('/v1/projects', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function getProject(id: string) {
  return request<Project>(`/v1/projects/${id}`);
}

/**
 * Короткий зразок голосу для прослуховування перед застосуванням.
 *
 * Використовує той самий `/api/synthesize`, що й основний синтез, але
 * повертає WAV як Blob, щоб UI міг відтворити його без збереження файлу.
 */
export async function synthesizeVoiceSample(
  text: string,
  voice?: string,
  speed = 1
): Promise<Blob> {
  const res = await fetch(`${getBaseUrl()}/api/synthesize`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({
      text,
      voice: voice || undefined,
      speed,
    }),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    let message = errText;
    try {
      const parsed = JSON.parse(errText) as { error?: string };
      if (parsed.error) message = parsed.error;
    } catch {
      /* тіло не JSON — лишаємо як є */
    }
    throw new Error(message || `${res.status} ${res.statusText}`);
  }
  return res.blob();
}

export function updateProject(id: string, body: UpdateProjectRequest) {
  return request<Project>(`/v1/projects/${id}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export function deleteProject(id: string) {
  return request<void>(`/v1/projects/${id}`, { method: 'DELETE' });
}

export function importProject(formData: FormData) {
  // Do NOT set Content-Type for FormData — browser sets the boundary.
  return request<Project>('/v1/projects/import', {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });
}

/* ---------- Chapters ---------- */
export function addChapter(id: string, body: AddChapterRequest) {
  return request<Chapter>(`/v1/projects/${id}/chapters`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function updateChapter(id: string, index: number, body: UpdateChapterRequest) {
  return request<Chapter>(`/v1/projects/${id}/chapters/${index}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export function chapterAudioUrl(id: string, index: number) {
  return `${getBaseUrl()}/v1/projects/${id}/chapters/${index}/audio`;
}

export function chapterSubtitlesUrl(id: string, index: number, format: string = 'srt') {
  return `${getBaseUrl()}/v1/projects/${id}/chapters/${index}/subtitles?format=${format}`;
}

export function unitTakeAudioUrl(
  id: string,
  index: number,
  unit: number,
  take: number
) {
  return `${getBaseUrl()}/v1/projects/${id}/chapters/${index}/takes/units/${unit}/takes/${take}/audio`;
}

/* ---------- Generation ---------- */
export interface GenerateResult {
  project_id: string;
  jobs: { job_id: string; chapter_index: number }[];
  skipped: { chapter_index: number; reason: string }[];
  errors: { chapter_index: number; error: string }[];
}

export function generateChapters(
  id: string,
  chapters?: number[],
  force: boolean = false
) {
  return request<GenerateResult>(`/v1/projects/${id}/generate`, {
    method: 'POST',
    body: JSON.stringify({ chapters, force }),
  });
}

/* ---------- Takes ---------- */
export function chapterTakes(id: string, index: number) {
  return request<ChapterTakes>(`/v1/projects/${id}/chapters/${index}/takes`);
}

export function activateTake(
  id: string,
  index: number,
  unit: number,
  take: number
) {
  return request<ChapterTakes>(
    `/v1/projects/${id}/chapters/${index}/takes/units/${unit}/takes/${take}/activate`,
    { method: 'POST' }
  );
}

export function deleteTake(
  id: string,
  index: number,
  unit: number,
  take: number
) {
  return request<void>(
    `/v1/projects/${id}/chapters/${index}/takes/units/${unit}/takes/${take}`,
    { method: 'DELETE' }
  );
}

export function regenerateChapterUnit(
  id: string,
  index: number,
  unit: number,
  body: RegenerateTakeRequest
) {
  return request<Take>(
    `/v1/projects/${id}/chapters/${index}/takes/units/${unit}`,
    { method: 'POST', body: JSON.stringify(body) }
  );
}

export function rebuildChapter(id: string, index: number) {
  return request<ChapterTakes>(
    `/v1/projects/${id}/chapters/${index}/takes/rebuild`,
    { method: 'POST' }
  );
}

export function invalidateTakes(id: string, index: number) {
  return request<void>(`/v1/projects/${id}/chapters/${index}/takes`, {
    method: 'POST',
  });
}

/* ---------- Jobs ---------- */
export async function listJobs(): Promise<JobStatus[]> {
  const data = await request<{ jobs: JobStatus[] }>('/v1/jobs');
  return data.jobs ?? [];
}

export function getJob(id: string) {
  return request<JobStatus>(`/v1/jobs/${id}`);
}

export function cancelJob(id: string) {
  return request<void>(`/v1/jobs/${id}`, { method: 'DELETE' });
}

export function jobEvents(
  id: string,
  onEvent: (event: MessageEvent) => void
): EventSource {
  const es = new EventSource(`${getBaseUrl()}/v1/jobs/${id}/events`);
  es.addEventListener('job', (event) => onEvent(event as MessageEvent));
  return es;
}

/* ---------- Exports ---------- */
export interface CreateExportRequest {
  format: string;
  bitrate_kbps?: number;
  cover_base64?: string;
  title?: string;
  author?: string;
  album?: string;
  narrator?: string;
  year?: string;
}

export interface ExportRecord {
  id: string;
  project_id: string;
  format: string;
  status: string;
  progress: number;
  chapters_total: number;
  chapters_done: number;
  bitrate_kbps: number;
  duration_ms?: number | null;
  audio_bytes?: number | null;
  error?: string | null;
  created_at: string;
  finished_at?: string | null;
  stale?: boolean;
  audio_url?: string | null;
  subtitles_srt_url?: string | null;
  subtitles_vtt_url?: string | null;
  subtitles_ass_url?: string | null;
}

export async function listExports(id: string): Promise<ExportRecord[]> {
  const data = await request<{ exports: ExportRecord[] }>(
    `/v1/projects/${id}/exports`
  );
  return data.exports ?? [];
}

export function createExport(id: string, body: CreateExportRequest) {
  return request<ExportRecord>(`/v1/projects/${id}/exports`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function exportAudioUrl(id: string, exportId: string) {
  return `${getBaseUrl()}/v1/projects/${id}/exports/${exportId}/audio`;
}

export function exportSubtitlesUrl(
  id: string,
  exportId: string,
  format: string = 'srt'
) {
  return `${getBaseUrl()}/v1/projects/${id}/exports/${exportId}/subtitles?format=${format}`;
}

/**
 * Завантажує файл із заголовком авторизації та зберігає його через blob.
 * Звичайний `<a href>` не передає API-ключ, тому для захищених маршрутів
 * використовуємо fetch.
 */
export async function downloadFile(url: string, filename: string) {
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? ': ' + text : ''}`);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

/* ---------- Per-chapter exports ---------- */
export interface CreatePerChapterExportRequest {
  format: string;
  bitrate_kbps?: number;
  /** Індекси глав; порожньо — усі завершені. */
  chapters?: number[];
  include_subtitles?: boolean;
}

export interface PerChapterExportJob {
  id: string;
  project_id: string;
  format: string;
  bitrate_kbps: number;
  chapters: number[];
  include_subtitles: boolean;
  status: string;
  progress: number;
  chapters_total: number;
  chapters_done: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  results?: {
    chapter_index: number;
    status: string;
    audio_file?: string | null;
    subtitles_srt_file?: string | null;
    subtitles_vtt_file?: string | null;
    subtitles_ass_file?: string | null;
    duration_ms?: number | null;
    bytes?: number | null;
    error?: string | null;
  }[];
  error?: string | null;
}

export function createPerChapterExport(
  id: string,
  body: CreatePerChapterExportRequest
) {
  return request<PerChapterExportJob>(
    `/v1/projects/${id}/exports/per-chapter`,
    { method: 'POST', body: JSON.stringify(body) }
  );
}

export function listPerChapterExports(id: string) {
  return request<PerChapterExportJob[]>(
    `/v1/projects/${id}/exports/per-chapter`
  );
}

export function perChapterAudioUrl(id: string, jobId: string, index: number) {
  return `${getBaseUrl()}/v1/projects/${id}/exports/per-chapter/${jobId}/chapter/${index}/audio`;
}

export function perChapterSubtitlesUrl(
  id: string,
  jobId: string,
  index: number,
  format: string = 'srt'
) {
  return `${getBaseUrl()}/v1/projects/${id}/exports/per-chapter/${jobId}/chapter/${index}/subtitles?format=${format}`;
}

/* ---------- Voices / Models ---------- */
export interface VoicesResponse {
  voices: string[];
  custom: string[];
  default: string;
  sample_rate: number;
}

export function listVoices() {
  return request<VoicesResponse>('/api/voices');
}

export function listModels() {
  return request<{ object: string; data: ModelInfo[] }>('/v1/models');
}

/* ---------- QC ---------- */
export async function listQC(id: string): Promise<QCJob[]> {
  const data = await request<{ project_id: string; jobs: QCJob[] }>(
    `/v1/projects/${id}/qc`
  );
  return data.jobs ?? [];
}

export function getQC(id: string, jobId: string) {
  return request<QCJob>(`/v1/projects/${id}/qc/${jobId}`);
}

export function startQC(id: string, index: number, force: boolean = false) {
  return request<QCJob>(
    `/v1/projects/${id}/chapters/${index}/verify?force=${force}`,
    { method: 'POST' }
  );
}

/* ---------- Health ---------- */
export function healthz() {
  return request<{ status: string }>('/healthz');
}
