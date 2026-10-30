import { useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Download,
  Trash2,
  Save,
  BookOpen,
  FileText,
  X,
  RefreshCw,
  Users,
  BarChart3,
  Mic,
  AlertTriangle,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import {
  getProject,
  updateProject,
  deleteProject,
  updateChapter,
  listJobs,
  cancelJob,
  generateChapters,
  chapterAudioUrl,
  listVoices,
  listExports,
  createExport,
  exportAudioUrl,
  exportSubtitlesUrl,
  listQC,
  getQC,
  startQC,
  listPerChapterExports,
  createPerChapterExport,
  perChapterAudioUrl,
  perChapterSubtitlesUrl,
  downloadFile,
} from '../services/api';
import type { PerChapterExportJob } from '../services/api';
import type {
  Chapter,
  Project,
  QCJob,
  UpdateChapterRequest,
  VoicesResponse,
  JobStatus,
} from '../types';
import type { ExportRecord } from '../services/api';
import { useAudioUrl } from '../hooks/useAudioUrl';

type Tab = 'editor' | 'roles' | 'export' | 'qc';

const STATUS_LABELS: Record<string, string> = {
  empty: 'Порожня',
  queued: 'У черзі',
  running: 'Генерується',
  completed: 'Готова',
  failed: 'Помилка',
  cancelled: 'Скасовано',
  stale: 'Застаріла',
};

const STATUS_STYLES: Record<string, string> = {
  empty: 'bg-gray-100 text-gray-600',
  queued: 'bg-amber-100 text-amber-700',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
  cancelled: 'bg-gray-100 text-gray-500',
  stale: 'bg-orange-100 text-orange-700',
};

export default function ProjectEditor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [project, setProject] = useState<Project | null>(null);
  const [tab, setTab] = useState<Tab>('editor');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [editingChapter, setEditingChapter] = useState<number | null>(null);
  const [chapterText, setChapterText] = useState('');
  const [chapterTitle, setChapterTitle] = useState('');

  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [voices, setVoices] = useState<VoicesResponse | null>(null);
  const [exports, setExports] = useState<ExportRecord[]>([]);
  const [qcJobs, setQcJobs] = useState<QCJob[]>([]);
  const [perChapterJobs, setPerChapterJobs] = useState<PerChapterExportJob[]>([]);

  const [exportFormat, setExportFormat] = useState('mp3');
  const [exportBitrate, setExportBitrate] = useState(128);
  const [perChapterFormat, setPerChapterFormat] = useState('mp3');
  const [perChapterBitrate, setPerChapterBitrate] = useState(128);
  const [perChapterSubtitles, setPerChapterSubtitles] = useState(true);
  const [qcChapter, setQcChapter] = useState<number | ''>('');

  useEffect(() => {
    void loadProject();
    void loadJobs();
    void loadVoices();
    void loadExports();
    void loadPerChapterExports();
    void loadQC();
  }, [id]);

  // Періодичне оновлення job-ів, поки є активні.
  useEffect(() => {
    const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
    if (!active) return;
    const timer = window.setInterval(() => {
      void loadJobs();
      void loadProject();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [jobs]);

  // Періодичне оновлення експортів і QC, поки є активні.
  useEffect(() => {
    const exportsActive = exports.some(
      (e) => e.status === 'queued' || e.status === 'running'
    );
    const perChapterActive = perChapterJobs.some(
      (j) => j.status === 'queued' || j.status === 'running'
    );
    const qcActive = qcJobs.some(
      (j) => j.status === 'queued' || j.status === 'running'
    );
    if (!exportsActive && !perChapterActive && !qcActive) return;
    const timer = window.setInterval(() => {
      if (exportsActive) void loadExports();
      if (perChapterActive) void loadPerChapterExports();
      if (qcActive) void loadQC();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [exports, perChapterJobs, qcJobs]);

  async function loadProject() {
    if (!id) return;
    try {
      const p = await getProject(id);
      setProject(p);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося завантажити проєкт');
    }
  }

  async function loadJobs() {
    try {
      setJobs(await listJobs());
    } catch {
      /* ignore */
    }
  }

  async function loadVoices() {
    try {
      setVoices(await listVoices());
    } catch {
      /* ignore */
    }
  }

  async function loadExports() {
    if (!id) return;
    try {
      setExports(await listExports(id));
    } catch {
      /* ignore */
    }
  }

  async function loadPerChapterExports() {
    if (!id) return;
    try {
      setPerChapterJobs(await listPerChapterExports(id));
    } catch {
      /* ignore */
    }
  }

  async function loadQC() {
    if (!id) return;
    try {
      const jobs = await listQC(id);
      // Список не містить report — дозавантажуємо деталі завершених job-ів.
      const detailed = await Promise.all(
        jobs.map(async (job) => {
          if (job.status === 'completed' && !job.report) {
            try {
              return await getQC(id, job.id);
            } catch {
              return job;
            }
          }
          return job;
        })
      );
      setQcJobs(detailed);
    } catch {
      /* ignore */
    }
  }

  const projectJobs = useMemo(
    () => jobs.filter((j) => j.project_id === id),
    [jobs, id]
  );

  async function handleSaveProject() {
    if (!project || !id) return;
    setLoading(true);
    setError(null);
    try {
      const updated = await updateProject(id, {
        title: project.title,
        author: project.author,
        language: project.language,
        settings: project.settings,
        roles: project.roles,
      });
      setProject(updated);
      setNotice('Проєкт збережено');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося зберегти проєкт');
    } finally {
      setLoading(false);
    }
  }

  function startEdit(chapter: Chapter) {
    setEditingChapter(chapter.index);
    setChapterTitle(chapter.title);
    setChapterText(chapter.text);
    setNotice(null);
  }

  async function handleSaveChapter(index: number) {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const body: UpdateChapterRequest = { title: chapterTitle, text: chapterText };
      await updateChapter(id, index, body);
      setEditingChapter(null);
      await loadProject();
      setNotice('Главу збережено');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося зберегти главу');
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerate(chapterIndex?: number) {
    if (!id) return;
    setError(null);
    setNotice(null);
    try {
      const result = await generateChapters(
        id,
        chapterIndex === undefined ? undefined : [chapterIndex],
        false
      );
      const queued = result.jobs.length;
      if (queued > 0) {
        setNotice(`Заплановано глав: ${queued}`);
      } else if (result.errors.length > 0) {
        setError(result.errors.map((e) => e.error).join('; '));
      } else if (result.skipped.length > 0) {
        setNotice('Усі вибрані глави вже готові');
      }
      await loadJobs();
      await loadProject();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося запустити синтез');
    }
  }

  async function handleDelete() {
    if (!id) return;
    if (!window.confirm('Видалити проєкт?')) return;
    try {
      await deleteProject(id);
      navigate('/projects');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося видалити проєкт');
    }
  }

  async function handleExport() {
    if (!id) return;
    setError(null);
    setNotice(null);
    try {
      const record = await createExport(id, {
        format: exportFormat,
        bitrate_kbps: exportBitrate,
      });
      setNotice(`Експорт запущено: ${record.id}`);
      await loadExports();
    } catch (err: unknown) {
      setError(
        humanizeError(
          err instanceof Error ? err.message : 'Не вдалося запустити експорт'
        )
      );
    }
  }

  async function handlePerChapterExport() {
    if (!id) return;
    setError(null);
    setNotice(null);
    try {
      const job = await createPerChapterExport(id, {
        format: perChapterFormat,
        bitrate_kbps: perChapterBitrate,
        include_subtitles: perChapterSubtitles,
      });
      setNotice(`Експорт по главах запущено: ${job.id}`);
      await loadPerChapterExports();
    } catch (err: unknown) {
      setError(
        humanizeError(
          err instanceof Error
            ? err.message
            : 'Не вдалося запустити експорт по главах'
        )
      );
    }
  }

  async function handleStartQC() {
    if (!id || qcChapter === '') return;
    setError(null);
    setNotice(null);
    try {
      await startQC(id, qcChapter);
      setNotice(`QC для глави ${qcChapter + 1} запущено`);
      await loadQC();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося запустити QC');
    }
  }

  async function handleDownload(url: string, filename: string) {
    setError(null);
    try {
      await downloadFile(url, filename);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося завантажити файл');
    }
  }

  function updateRoleVoice(index: number, voice: string) {
    if (!project) return;
    const roles = project.roles.map((role, i) =>
      i === index ? { ...role, voice: voice || undefined } : role
    );
    setProject({ ...project, roles });
  }

  if (!project) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-500">{error || 'Завантаження проєкту...'}</p>
          <button
            onClick={() => navigate('/projects')}
            className="mt-4 text-blue-600 hover:text-blue-700"
          >
            Назад до проєктів
          </button>
        </div>
      </div>
    );
  }

  const currentChapter =
    editingChapter !== null
      ? project.chapters.find((c) => c.index === editingChapter) ?? null
      : null;

  return (
    <div className="flex-1 overflow-hidden bg-gray-50 flex flex-col">
      {/* Заголовок */}
      <div className="border-b border-gray-200 bg-white px-4 py-3 flex items-center justify-between">
        <button
          onClick={() => navigate('/projects')}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Проєкти
        </button>
        <div className="flex-1 text-center min-w-0 px-4">
          <input
            value={project.title}
            onChange={(e) => setProject({ ...project, title: e.target.value })}
            className="text-xl font-bold text-gray-900 text-center bg-transparent border-b border-transparent hover:border-gray-300 focus:border-blue-500 outline-none w-full max-w-xl"
          />
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <input
              value={project.author}
              onChange={(e) => setProject({ ...project, author: e.target.value })}
              placeholder="Автор"
              className="bg-transparent text-center border-b border-transparent hover:border-gray-300 focus:border-blue-500 outline-none w-40"
            />
            <span>•</span>
            <input
              value={project.language}
              onChange={(e) => setProject({ ...project, language: e.target.value })}
              className="bg-transparent text-center border-b border-transparent hover:border-gray-300 focus:border-blue-500 outline-none w-12"
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleSaveProject}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            Зберегти
          </button>
          <button onClick={handleDelete} className="p-2 text-red-500 hover:text-red-700">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Вкладки */}
      <div className="bg-white border-b border-gray-200 px-4 flex gap-1">
        {(
          [
            ['editor', 'Редактор', BookOpen],
            ['roles', 'Ролі', Users],
            ['export', 'Експорт', Download],
            ['qc', 'QC', BarChart3],
          ] as [Tab, string, typeof BookOpen][]
        ).map(([key, label, Icon]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === key
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-gray-500 hover:text-gray-800'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {(error || notice) && (
        <div className="px-4 pt-3">
          {error && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-100 text-sm text-red-700 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}
          {notice && (
            <div className="p-3 rounded-lg bg-green-50 border border-green-100 text-sm text-green-700 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
              {notice}
            </div>
          )}
        </div>
      )}

      {tab === 'editor' && (
        <div className="flex-1 overflow-hidden flex">
          {/* Список глав */}
          <div className="w-80 border-r border-gray-200 bg-white flex flex-col">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">
                Глави <span className="text-gray-400 font-normal">({project.chapters.length})</span>
              </h2>
              <button
                onClick={() => handleGenerate()}
                className="flex items-center gap-1 text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
              >
                <Play className="w-3 h-3" />
                Синтез усіх
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {project.chapters.map((ch) => (
                <button
                  key={ch.index}
                  onClick={() => startEdit(ch)}
                  className={`w-full text-left p-3 border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                    editingChapter === ch.index ? 'bg-blue-50' : ''
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-gray-900 truncate">{ch.title}</span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full whitespace-nowrap ${
                        STATUS_STYLES[ch.status] ?? STATUS_STYLES.empty
                      }`}
                    >
                      {STATUS_LABELS[ch.status] ?? ch.status}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1 truncate">
                    {ch.text.slice(0, 90)}
                  </p>
                  <div className="flex items-center gap-3 mt-1 text-[10px] text-gray-400">
                    <span>{ch.text.length} симв.</span>
                    {ch.duration_ms ? <span>{formatDuration(ch.duration_ms)}</span> : null}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Редактор глави */}
          <div className="flex-1 overflow-hidden flex flex-col">
            {currentChapter ? (
              <>
                <div className="p-3 border-b border-gray-200 bg-white flex items-center justify-between">
                  <div>
                    <h2 className="font-semibold text-gray-900">
                      Глава {currentChapter.index + 1}
                    </h2>
                    <p className="text-xs text-gray-500">
                      {currentChapter.text.length} символів
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleGenerate(currentChapter.index)}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                      <Mic className="w-3.5 h-3.5" />
                      Синтезувати
                    </button>
                    <button
                      onClick={() => handleSaveChapter(currentChapter.index)}
                      disabled={loading}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50"
                    >
                      <Save className="w-3.5 h-3.5" />
                      Зберегти
                    </button>
                    <button
                      onClick={() => setEditingChapter(null)}
                      className="p-1.5 text-gray-500 hover:text-gray-800"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {currentChapter.status === 'completed' && (
                  <div className="p-3 bg-white border-b border-gray-200">
                    <ChapterAudio id={id!} index={currentChapter.index} />
                  </div>
                )}

                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Назва глави
                    </label>
                    <input
                      value={chapterTitle}
                      onChange={(e) => setChapterTitle(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Текст
                    </label>
                    <textarea
                      value={chapterText}
                      onChange={(e) => setChapterText(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 font-mono text-sm leading-relaxed"
                      rows={24}
                    />
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4 text-gray-400">
                  <BookOpen className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">
                  Оберіть главу
                </h3>
                <p className="text-gray-500 text-sm">
                  Натисніть на главу у списку, щоб редагувати текст
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'roles' && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-2xl mx-auto bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-1">Ролі діалогу</h2>
            <p className="text-sm text-gray-500 mb-4">
              Призначте голос кожному персонажу. Нарація використовує голос проєкту.
            </p>

            <div className="grid grid-cols-2 gap-3 mb-5 p-3 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Голос нарації
                </label>
                <select
                  value={project.settings.voice ?? ''}
                  onChange={(e) =>
                    setProject({
                      ...project,
                      settings: {
                        ...project.settings,
                        voice: e.target.value || undefined,
                      },
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
                >
                  <option value="">
                    За замовчуванням{voices?.default ? ` (${voices.default})` : ''}
                  </option>
                  {voices?.voices.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                  {voices?.custom.map((v) => (
                    <option key={v} value={v}>
                      {v} (клон)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Швидкість
                </label>
                <input
                  type="number"
                  step="0.05"
                  min={0.5}
                  max={2}
                  value={project.settings.speed}
                  onChange={(e) =>
                    setProject({
                      ...project,
                      settings: {
                        ...project.settings,
                        speed: Number(e.target.value),
                      },
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
                />
              </div>
            </div>

            {project.roles.length === 0 ? (
              <p className="text-sm text-gray-500">
                Ролей ще немає. Вони з'являться після імпорту книги з діалогами.
              </p>
            ) : (
              <div className="space-y-2">
                {project.roles.map((role, i) => (
                  <div
                    key={`${role.name}-${i}`}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg"
                  >
                    <span className="flex-1 font-medium text-gray-900">{role.name}</span>
                    <select
                      value={role.voice ?? ''}
                      onChange={(e) => updateRoleVoice(i, e.target.value)}
                      className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white"
                    >
                      <option value="">Голос проєкту</option>
                      {voices?.voices.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                      {voices?.custom.map((v) => (
                        <option key={v} value={v}>
                          {v} (клон)
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            )}
            <div className="mt-4 flex justify-end">
              <button
                onClick={handleSaveProject}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                <Save className="w-4 h-4" />
                Зберегти ролі
              </button>
            </div>
          </div>
        </div>
      )}

      {tab === 'export' && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-3xl mx-auto space-y-4">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Експорт книги
              </h2>
              <div className="flex items-end gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    Формат
                  </label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg bg-white"
                  >
                    <option value="m4b">M4B (аудіокнига)</option>
                    <option value="mp3">MP3</option>
                    <option value="ogg">OGG</option>
                    <option value="flac">FLAC</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    Бітрейт, kbps
                  </label>
                  <input
                    type="number"
                    value={exportBitrate}
                    min={32}
                    max={320}
                    onChange={(e) => setExportBitrate(Number(e.target.value))}
                    className="w-28 px-3 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <button
                  onClick={handleExport}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  <Download className="w-4 h-4" />
                  Експортувати
                </button>
                <button
                  onClick={loadExports}
                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                  title="Оновити"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Історія експорту
              </h2>
              {exports.length === 0 ? (
                <p className="text-sm text-gray-500">Експортів ще не було.</p>
              ) : (
                <div className="space-y-2">
                  {exports.map((record) => (
                    <div
                      key={record.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-gray-900 uppercase">
                            {record.format}
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600">
                            {record.status}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {record.chapters_done}/{record.chapters_total} глав
                          {record.duration_ms
                            ? ` • ${formatDuration(record.duration_ms)}`
                            : ''}
                          {record.error ? ` • ${record.error}` : ''}
                        </p>
                      </div>
                      {record.status === 'completed' && (
                        <div className="flex items-center gap-3">
                          <button
                            onClick={() =>
                              handleDownload(
                                exportAudioUrl(id!, record.id),
                                `${project.title || 'book'}.${record.format}`
                              )
                            }
                            className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700"
                          >
                            <Download className="w-4 h-4" />
                            Аудіо
                          </button>
                          <button
                            onClick={() =>
                              handleDownload(
                                exportSubtitlesUrl(id!, record.id, 'srt'),
                                `${project.title || 'book'}.srt`
                              )
                            }
                            className="text-sm text-gray-600 hover:text-gray-800"
                          >
                            SRT
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Експорт окремими файлами по главах (v0.13) */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-lg font-semibold text-gray-900">
                  Окремі файли по главах
                </h2>
                <button
                  onClick={loadPerChapterExports}
                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                  title="Оновити"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
              <p className="text-sm text-gray-500 mb-4">
                Кожна готова глава — окремий файл із власними субтитрами.
              </p>
              <div className="flex flex-wrap items-end gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    Формат
                  </label>
                  <select
                    value={perChapterFormat}
                    onChange={(e) => setPerChapterFormat(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg bg-white"
                  >
                    <option value="m4b">M4B</option>
                    <option value="mp3">MP3</option>
                    <option value="ogg">OGG</option>
                    <option value="flac">FLAC</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    Бітрейт, kbps
                  </label>
                  <input
                    type="number"
                    value={perChapterBitrate}
                    min={32}
                    max={320}
                    onChange={(e) => setPerChapterBitrate(Number(e.target.value))}
                    className="w-28 px-3 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <label className="flex items-center gap-2 py-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={perChapterSubtitles}
                    onChange={(e) => setPerChapterSubtitles(e.target.checked)}
                    className="rounded border-gray-300"
                  />
                  Субтитри
                </label>
                <button
                  onClick={handlePerChapterExport}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  <Download className="w-4 h-4" />
                  Експортувати глави
                </button>
              </div>

              <div className="mt-4 space-y-2">
                {perChapterJobs.length === 0 ? (
                  <p className="text-sm text-gray-500">Експортів по главах ще не було.</p>
                ) : (
                  perChapterJobs.map((job) => (
                    <div key={job.id} className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-gray-900 uppercase">
                            {job.format}
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600">
                            {job.status}
                          </span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {job.chapters_done}/{job.chapters_total} глав
                        </span>
                      </div>
                      {job.error ? (
                        <p className="text-xs text-red-600 mt-1">{job.error}</p>
                      ) : null}
                      {job.results && job.results.length > 0 ? (
                        <div className="mt-2 space-y-1">
                          {job.results
                            .filter((r) => r.status === 'completed')
                            .map((r) => (
                              <div
                                key={r.chapter_index}
                                className="flex items-center justify-between text-xs"
                              >
                                <span className="text-gray-700">
                                  Глава {r.chapter_index + 1}
                                  {r.duration_ms
                                    ? ` • ${formatDuration(r.duration_ms)}`
                                    : ''}
                                </span>
                                <div className="flex items-center gap-3">
                                  <button
                                    onClick={() =>
                                      handleDownload(
                                        perChapterAudioUrl(
                                          id!,
                                          job.id,
                                          r.chapter_index
                                        ),
                                        `${project.title || 'book'}-${r.chapter_index + 1}.${job.format}`
                                      )
                                    }
                                    className="text-blue-600 hover:text-blue-700"
                                  >
                                    Аудіо
                                  </button>
                                  {job.include_subtitles ? (
                                    <button
                                      onClick={() =>
                                        handleDownload(
                                          perChapterSubtitlesUrl(
                                            id!,
                                            job.id,
                                            r.chapter_index,
                                            'srt'
                                          ),
                                          `${project.title || 'book'}-${r.chapter_index + 1}.srt`
                                        )
                                      }
                                      className="text-gray-600 hover:text-gray-800"
                                    >
                                      SRT
                                    </button>
                                  ) : null}
                                </div>
                              </div>
                            ))}
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'qc' && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-3xl mx-auto bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Контроль якості</h2>
              <button
                onClick={loadQC}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                title="Оновити"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
            <div className="flex flex-wrap items-end gap-3 mb-5 p-3 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Глава для перевірки
                </label>
                <select
                  value={qcChapter}
                  onChange={(e) =>
                    setQcChapter(e.target.value === '' ? '' : Number(e.target.value))
                  }
                  className="px-3 py-2 border border-gray-300 rounded-lg bg-white min-w-48"
                >
                  <option value="">Оберіть главу</option>
                  {project.chapters
                    .filter((ch) => ch.status === 'completed')
                    .map((ch) => (
                      <option key={ch.index} value={ch.index}>
                        {ch.title}
                      </option>
                    ))}
                </select>
              </div>
              <button
                onClick={handleStartQC}
                disabled={qcChapter === ''}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                <BarChart3 className="w-4 h-4" />
                Запустити QC
              </button>
              {project.chapters.every((ch) => ch.status !== 'completed') ? (
                <p className="text-xs text-gray-500">
                  Спочатку синтезуйте хоча б одну главу.
                </p>
              ) : null}
            </div>
            {qcJobs.length === 0 ? (
              <p className="text-sm text-gray-500">
                Перевірок ще не було. Запустіть QC для готової глави.
              </p>
            ) : (
              <div className="space-y-2">
                {qcJobs.map((job) => (
                  <div key={job.id} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-900">
                        Глава {job.chapter_index + 1}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600">
                        {job.status}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1 flex gap-3">
                      {job.report ? (
                        <>
                          <span>WER: {(job.report.diff.wer * 100).toFixed(1)}%</span>
                          <span>CER: {(job.report.diff.cer * 100).toFixed(1)}%</span>
                          <span>
                            Помилок: {job.report.segments_with_errors}/
                            {job.report.segments_total}
                          </span>
                        </>
                      ) : (
                        <span>Прогрес: {job.progress}%</span>
                      )}
                      {job.error ? <span className="text-red-600">{job.error}</span> : null}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Job-и */}
      <div className="border-t border-gray-200 bg-white px-4 py-3 max-h-40 overflow-y-auto">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase">
            Завдання проєкту
          </h3>
          <button
            onClick={loadJobs}
            className="text-xs text-gray-500 hover:text-gray-800"
          >
            Оновити
          </button>
        </div>
        {projectJobs.length === 0 ? (
          <p className="text-xs text-gray-400">Немає завдань</p>
        ) : (
          <div className="space-y-1.5">
            {projectJobs.slice(0, 8).map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between px-2 py-1.5 bg-gray-50 rounded text-xs"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <Clock className="w-3 h-3 text-gray-400 shrink-0" />
                  <span className="text-gray-700 truncate">
                    Глава {(job.chapter_index ?? 0) + 1}
                  </span>
                  <span className="text-gray-400">{job.status}</span>
                  {(job.status === 'running' || job.status === 'queued') && (
                    <span className="text-gray-400">{job.progress}%</span>
                  )}
                  {job.error ? (
                    <span className="text-red-600 truncate">{job.error}</span>
                  ) : null}
                </div>
                {(job.status === 'running' || job.status === 'queued') && (
                  <button
                    onClick={() => cancelJob(job.id)}
                    className="p-1 text-red-500 hover:text-red-700"
                    title="Скасувати"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ChapterAudio({ id, index }: { id: string; index: number }) {
  const url = chapterAudioUrl(id, index);
  const { src, loading, error } = useAudioUrl(url);

  if (loading) return <p className="text-xs text-gray-500">Завантаження аудіо...</p>;
  if (error) return <p className="text-xs text-red-600">Аудіо: {error}</p>;
  if (!src) return null;

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-medium text-gray-500 uppercase">Прослухати</span>
      <audio src={src} controls className="flex-1 h-9" />
      <a
        href={src}
        download={`chapter-${index + 1}.wav`}
        className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700"
      >
        <Download className="w-4 h-4" />
        WAV
      </a>
    </div>
  );
}

/** Перекладає типові помилки API для показу в UI. */
function humanizeError(message: string): string {
  if (message.includes('all chapters must be completed')) {
    return 'Спочатку завершіть синтез усіх глав — експорт книги потребує готового аудіо для кожної глави.';
  }
  if (message.includes('no completed chapters')) {
    return 'Немає готових глав для експорту. Синтезуйте хоча б одну главу.';
  }
  return message;
}

function formatDuration(ms: number): string {
  if (!ms) return '—';
  const totalSeconds = Math.round(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours} год ${minutes} хв`;
  if (minutes > 0) return `${minutes} хв ${seconds} с`;
  return `${seconds} с`;
}
