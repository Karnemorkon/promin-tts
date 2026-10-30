export type ChapterStatus =
  | 'empty'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'stale';

export interface Role {
  name: string;
  voice?: string;
}

export interface LoudnessSettings {
  enabled: boolean;
  target_lufs?: number;
  true_peak?: number;
}

export interface ProjectSettings {
  voice?: string;
  speed: number;
  max_chars: number;
  crossfade_ms: number;
  auto_verbalize?: unknown;
  raw_stress: boolean;
  loudness: LoudnessSettings;
}

export interface ChunkTiming {
  start_ms: number;
  end_ms: number;
  text: string;
  role?: string | null;
}

export interface Chapter {
  index: number;
  title: string;
  text: string;
  status: ChapterStatus;
  job_id?: string | null;
  chunk_timings?: ChunkTiming[];
  audio_bytes?: number;
  duration_ms?: number;
  error?: string | null;
  updated_at?: string | null;
}

export interface ProjectSummary {
  id: string;
  title: string;
  author: string;
  language: string;
  chapters: number;
  completed_chapters: number;
  total_chars: number;
  total_duration_ms: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Project {
  id: string;
  version: number;
  title: string;
  author: string;
  language: string;
  settings: ProjectSettings;
  roles: Role[];
  chapters: Chapter[];
  pronunciation: unknown[];
  created_at: string;
  updated_at: string;
}

export interface CreateProjectRequest {
  title: string;
  author?: string;
  language?: string;
  settings?: ProjectSettings;
  pronunciation?: unknown[];
  roles?: Role[];
  chapters?: { title?: string; text: string }[];
}

export interface UpdateProjectRequest {
  title?: string;
  author?: string;
  language?: string;
  settings?: ProjectSettings;
  pronunciation?: unknown[];
  roles?: Role[];
}

export interface UpdateChapterRequest {
  title?: string;
  text?: string;
}

export interface JobStatus {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  voice?: string | null;
  speed: number;
  text_chars: number;
  error?: string | null;
  audio_url?: string | null;
  audio_bytes?: number | null;
  duration_ms?: number | null;
  chunks?: number | null;
  verbalizer?: string | null;
  project_id?: string | null;
  chapter_index?: number | null;
}

export interface VoicesResponse {
  voices: string[];
  custom: string[];
  default: string;
  sample_rate: number;
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface QcDiff {
  reference_words: number;
  hypothesis_words: number;
  correct: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  wer: number;
  cer: number;
  reference_chars: number;
  hypothesis_chars: number;
  char_errors: number;
}

export interface QcIssue {
  code: string;
  severity: string;
  message: string;
}

export interface QCJob {
  id: string;
  project_id: string;
  chapter_index: number;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  stage: string;
  progress: number;
  model: string;
  audio_sha256: string;
  audio_bytes: number;
  duration_ms: number;
  segments_total: number;
  segments_done: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_ms?: number | null;
  error?: string | null;
  report?: {
    reference_source: string;
    model: string;
    duration_ms: number;
    segments_total: number;
    segments_with_errors: number;
    diff: QcDiff;
    segments: {
      index: number;
      start_ms: number;
      end_ms: number;
      script: string;
      reference: string;
      recognized: string;
      diff: QcDiff;
    }[];
    issues: QcIssue[];
    elapsed_ms: number;
    asr_ms: number;
    rtf: number;
  } | null;
}

export interface ConnectionState {
  mode: 'local' | 'remote';
  baseUrl: string;
  apiKey: string | null;
  connected: boolean;
}
