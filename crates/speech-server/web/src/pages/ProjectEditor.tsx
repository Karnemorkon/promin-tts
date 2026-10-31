import { useEffect, useMemo, useRef, useState } from 'react';
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
  Plus,
  Layers,
} from 'lucide-react';
import {
  getProject,
  updateProject,
  deleteProject,
  addChapter,
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
  chapterTakes,
  activateTake,
  deleteTake,
  regenerateChapterUnit,
  rebuildChapter,
  invalidateTakes,
  unitTakeAudioUrl,
} from '../services/api';
import type { PerChapterExportJob } from '../services/api';
import type {
  Chapter,
  ChapterTakes,
  Project,
  QCJob,
  RegenerateTakeRequest,
  Take,
  UnitTakes,
  UpdateChapterRequest,
  VoicesResponse,
  JobStatus,
} from '../types';
import type { ExportRecord } from '../services/api';
import { useAudioUrl } from '../hooks/useAudioUrl';

type Tab = 'editor' | 'roles' | 'takes' | 'export' | 'qc';

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

  const [takesChapter, setTakesChapter] = useState<number | ''>('');
  const [takes, setTakes] = useState<ChapterTakes | null>(null);
  const [takesLoading, setTakesLoading] = useState(false);
  const [takesBusy, setTakesBusy] = useState(false);
  const [takesError, setTakesError] = useState<string | null>(null);
  const [regenUnit, setRegenUnit] = useState<number | null>(null);
  const [regenText, setRegenText] = useState('');
  const [regenVoice, setRegenVoice] = useState('');
  const [regenSpeed, setRegenSpeed] = useState('1');

  const [newRoleName, setNewRoleName] = useState('');
  const [newRoleVoice, setNewRoleVoice] = useState('');
  const [assignRole, setAssignRole] = useState('');
  const chapterTextRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setTakesChapter('');
    setTakes(null);
    setTakesError(null);
    setRegenUnit(null);
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

  // Автоматично обираємо першу готову главу для вкладки тейків.
  useEffect(() => {
    if (tab !== 'takes' || takesChapter !== '' || !project || project.id !== id)
      return;
    const first = project.chapters.find((ch) => ch.status === 'completed');
    if (first) setTakesChapter(first.index);
  }, [tab, takesChapter, project, id]);

  // Завантажуємо маніфест тейків для вибраної глави.
  useEffect(() => {
    if (tab !== 'takes' || takesChapter === '') return;
    void loadTakes(takesChapter);
  }, [tab, takesChapter, id]);

  async function loadProject() {
    if (!id) return;
    try {
      const p = await getProject(id);
      // Старі проєкти (до v0.8) не містять поля roles у JSON.
      setProject({ ...p, roles: p.roles ?? [] });
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

  async function loadTakes(index: number) {
    if (!id) return;
    setTakesLoading(true);
    setTakesError(null);
    try {
      setTakes(await chapterTakes(id, index));
    } catch (err: unknown) {
      setTakes(null);
      setTakesError(
        err instanceof Error ? err.message : 'Не вдалося завантажити тейки'
      );
    } finally {
      setTakesLoading(false);
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

  async function handleAddChapter() {
    if (!id) return;
    const suggested = `Розділ ${(project?.chapters.length ?? 0) + 1}`;
    const title = window.prompt('Назва нової глави:', suggested);
    if (title === null) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const chapter = await addChapter(id, {
        title: title.trim() || undefined,
        text: '',
      });
      await loadProject();
      startEdit(chapter);
      setNotice('Главу додано');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося додати главу');
    } finally {
      setLoading(false);
    }
  }

  async function handleActivateTake(unit: number, take: number) {
    if (!id || takesChapter === '') return;
    setTakesBusy(true);
    setTakesError(null);
    try {
      setTakes(await activateTake(id, takesChapter, unit, take));
      await loadProject();
      setNotice(`Активовано версію ${take}`);
    } catch (err: unknown) {
      setTakesError(
        err instanceof Error ? humanizeError(err.message) : 'Не вдалося активувати тейк'
      );
    } finally {
      setTakesBusy(false);
    }
  }

  async function handleDeleteTake(unit: number, take: number) {
    if (!id || takesChapter === '') return;
    if (!window.confirm(`Видалити версію ${take}?`)) return;
    setTakesBusy(true);
    setTakesError(null);
    try {
      await deleteTake(id, takesChapter, unit, take);
      await loadTakes(takesChapter);
      await loadProject();
      setNotice(`Версію ${take} видалено`);
    } catch (err: unknown) {
      setTakesError(
        err instanceof Error ? humanizeError(err.message) : 'Не вдалося видалити тейк'
      );
    } finally {
      setTakesBusy(false);
    }
  }

  function openRegenerate(unit: UnitTakes) {
    const active = unit.takes.find((take) => take.id === unit.active_take);
    setRegenUnit(unit.index);
    setRegenText(unit.text);
    setRegenVoice(active?.voice ?? '');
    setRegenSpeed(String(active?.speed ?? project?.settings.speed ?? 1));
    setTakesError(null);
  }

  async function handleRegenerateSubmit(unit: UnitTakes) {
    if (!id || takesChapter === '' || !project) return;
    if (!regenText.trim()) {
      setTakesError('Текст не може бути порожнім');
      return;
    }
    const speed = Number(regenSpeed);
    if (!Number.isFinite(speed) || speed < 0.5 || speed > 2) {
      setTakesError('Темп має бути в межах 0.5–2.0');
      return;
    }
    const body: RegenerateTakeRequest = {
      text: regenText.trim() === unit.text.trim() ? undefined : regenText,
      voice: regenVoice || undefined,
      speed,
    };
    setTakesBusy(true);
    setTakesError(null);
    try {
      const take = await regenerateChapterUnit(
        id,
        takesChapter,
        unit.index,
        body
      );
      setRegenUnit(null);
      await loadTakes(takesChapter);
      setNotice(`Створено версію ${take.id}`);
    } catch (err: unknown) {
      setTakesError(
        err instanceof Error ? humanizeError(err.message) : 'Не вдалося створити новий тейк'
      );
    } finally {
      setTakesBusy(false);
    }
  }

  async function handleRebuildChapter() {
    if (!id || takesChapter === '') return;
    setTakesBusy(true);
    setTakesError(null);
    try {
      setTakes(await rebuildChapter(id, takesChapter));
      await loadProject();
      setNotice('Главу перезібрано з активних тейків');
    } catch (err: unknown) {
      setTakesError(
        err instanceof Error ? humanizeError(err.message) : 'Не вдалося перезібрати главу'
      );
    } finally {
      setTakesBusy(false);
    }
  }

  async function handleInvalidateTakes() {
    if (!id || takesChapter === '') return;
    if (
      !window.confirm(
        'Скинути кеш тейків глави? Поточні версії речень буде видалено; для нових тейків главу доведеться синтезувати повторно.'
      )
    ) {
      return;
    }
    setTakesBusy(true);
    setTakesError(null);
    try {
      await invalidateTakes(id, takesChapter);
      setTakes(null);
      setNotice('Кеш тейків скинуто');
    } catch (err: unknown) {
      setTakesError(
        err instanceof Error ? humanizeError(err.message) : 'Не вдалося скинути кеш тейків'
      );
    } finally {
      setTakesBusy(false);
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

  function updateRoleName(index: number, name: string) {
    if (!project) return;
    const roles = project.roles.map((role, i) =>
      i === index ? { ...role, name } : role
    );
    setProject({ ...project, roles });
  }

  function handleAddRole() {
    if (!project) return;
    const name = newRoleName.trim();
    if (!name) {
      setError('Введіть ім\'я ролі');
      return;
    }
    if (name.includes(':') || name.includes('[') || name.includes(']')) {
      setError('Ім\'я ролі не може містити двокрапку або дужки');
      return;
    }
    if (
      project.roles.some(
        (role) => role.name.trim().toLowerCase() === name.toLowerCase()
      )
    ) {
      setError('Роль з таким ім\'ям уже існує');
      return;
    }
    setProject({
      ...project,
      roles: [
        ...project.roles,
        { name, voice: newRoleVoice || undefined },
      ],
    });
    setNewRoleName('');
    setNewRoleVoice('');
    setError(null);
    setNotice(`Роль «${name}» додано. Натисніть «Зберегти ролі».`);
  }

  function handleDeleteRole(index: number) {
    if (!project) return;
    const role = project.roles[index];
    if (!window.confirm(`Видалити роль «${role.name}»?`)) return;
    setProject({
      ...project,
      roles: project.roles.filter((_, i) => i !== index),
    });
    setNotice(`Роль «${role.name}» видалено. Натисніть «Зберегти ролі».`);
  }

  /**
   * Позначити виділений у редакторі глави текст як репліку ролі.
   *
   * Парсер діалогів розпізнає роль лише на початку рядка, тому перед
   * виділенням додаємо `Ім'я: `; якщо виділення починається всередині
   * рядка — спершу переносимо його на новий рядок.
   */
  function handleAssignRoleToSelection() {
    const textarea = chapterTextRef.current;
    if (!textarea) return;
    if (!assignRole) {
      setError('Оберіть роль для виділеного тексту');
      return;
    }
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    if (start === end) {
      setError('Спочатку виділіть текст у редакторі глави');
      return;
    }
    const selected = chapterText.slice(start, end);
    const atLineStart = start === 0 || chapterText[start - 1] === '\n';
    const prefix = atLineStart ? `${assignRole}: ` : `\n${assignRole}: `;
    const next = chapterText.slice(0, start) + prefix + chapterText.slice(start);
    setChapterText(next);
    setError(null);
    setNotice(
      `Текст позначено як репліка ролі «${assignRole}». Не забудьте зберегти главу.`
    );
    window.requestAnimationFrame(() => {
      const caret = start + prefix.length + selected.length;
      textarea.focus();
      textarea.setSelectionRange(caret, caret);
    });
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
            ['takes', 'Тейки', Layers],
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
              <div className="flex items-center gap-1">
                <button
                  onClick={() => handleAddChapter()}
                  disabled={loading}
                  className="flex items-center gap-1 text-xs bg-white border border-gray-300 text-gray-700 px-2 py-1 rounded hover:bg-gray-50 disabled:opacity-50"
                  title="Додати главу"
                >
                  <Plus className="w-3 h-3" />
                  Додати
                </button>
                <button
                  onClick={() => handleGenerate()}
                  className="flex items-center gap-1 text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                >
                  <Play className="w-3 h-3" />
                  Синтез усіх
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {project.chapters.length === 0 && (
                <div className="p-4 text-center text-sm text-gray-500">
                  <p className="mb-3">У проєкті ще немає глав.</p>
                  <button
                    onClick={() => handleAddChapter()}
                    disabled={loading}
                    className="inline-flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    <Plus className="w-4 h-4" />
                    Додати першу главу
                  </button>
                </div>
              )}
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
                      ref={chapterTextRef}
                      value={chapterText}
                      onChange={(e) => setChapterText(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 font-mono text-sm leading-relaxed"
                      rows={24}
                    />
                  </div>
                  <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg">
                    <p className="text-xs font-medium text-blue-900 mb-2">
                      Призначити виділений текст ролі
                    </p>
                    {project.roles.length === 0 ? (
                      <p className="text-xs text-blue-800">
                        Спершу додайте роль на вкладці «Ролі».
                      </p>
                    ) : (
                      <div className="flex flex-wrap items-center gap-2">
                        <select
                          value={assignRole}
                          onChange={(e) => setAssignRole(e.target.value)}
                          className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white min-w-48"
                        >
                          <option value="">Оберіть роль</option>
                          {project.roles.map((role) => (
                            <option key={role.name} value={role.name}>
                              {role.name}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={handleAssignRoleToSelection}
                          className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
                        >
                          <Users className="w-4 h-4" />
                          Позначити репліку
                        </button>
                        <p className="text-xs text-blue-800">
                          Виділіть фрагмент у полі «Текст» вище, оберіть роль і
                          натисніть кнопку. Перед текстом з'явиться{' '}
                          <code className="font-mono">Ім'я:</code>.
                        </p>
                      </div>
                    )}
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

            <div className="mb-5 p-3 bg-blue-50 border border-blue-100 rounded-lg">
              <p className="text-xs font-medium text-blue-900 mb-1">
                Як позначити репліку ролі в тексті
              </p>
              <p className="text-xs text-blue-800">
                На початку рядка напишіть <code className="font-mono">Ім'я: текст</code>{' '}
                або <code className="font-mono">[Ім'я] текст</code>. Усе інше —
                нарація голосом проєкту. У редакторі глави можна виділити
                фрагмент і призначити його ролі кнопкою.
              </p>
            </div>

            <div className="mb-5 p-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 mb-2">Додати роль</p>
              <div className="flex flex-wrap items-end gap-2">
                <div className="flex-1 min-w-40">
                  <label className="block text-xs text-gray-500 mb-1">Ім'я</label>
                  <input
                    value={newRoleName}
                    onChange={(e) => setNewRoleName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleAddRole();
                      }
                    }}
                    placeholder="Наприклад, Анна"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
                  />
                </div>
                <div className="flex-1 min-w-40">
                  <label className="block text-xs text-gray-500 mb-1">Голос</label>
                  <select
                    value={newRoleVoice}
                    onChange={(e) => setNewRoleVoice(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
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
                <button
                  onClick={handleAddRole}
                  className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
                >
                  <Plus className="w-4 h-4" />
                  Додати роль
                </button>
              </div>
            </div>

            {project.roles.length === 0 ? (
              <p className="text-sm text-gray-500">
                Ролей ще немає. Додайте роль вище або імпортуйте книгу з
                діалогами.
              </p>
            ) : (
              <div className="space-y-2">
                {project.roles.map((role, i) => (
                  <div
                    key={i}
                    className="flex flex-wrap items-center gap-2 p-3 bg-gray-50 rounded-lg"
                  >
                    <input
                      value={role.name}
                      onChange={(e) => updateRoleName(i, e.target.value)}
                      className="flex-1 min-w-40 px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white font-medium text-gray-900"
                    />
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
                    <button
                      onClick={() => handleDeleteRole(i)}
                      title="Видалити роль"
                      className="p-1.5 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
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

      {tab === 'takes' && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Версії речень
                </h2>
                <p className="text-sm text-gray-500">
                  Прослухайте альтернативні тейки, активуйте найкращий або
                  перегенеруйте окреме речення.
                </p>
              </div>
              <button
                onClick={() =>
                  takesChapter !== '' && void loadTakes(takesChapter)
                }
                disabled={takesChapter === '' || takesLoading}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg disabled:opacity-50"
                title="Оновити"
              >
                <RefreshCw
                  className={`w-4 h-4 ${takesLoading ? 'animate-spin' : ''}`}
                />
              </button>
            </div>

            <div className="flex flex-wrap items-end gap-3 mb-5 p-3 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Глава
                </label>
                <select
                  value={takesChapter}
                  onChange={(e) =>
                    setTakesChapter(
                      e.target.value === '' ? '' : Number(e.target.value)
                    )
                  }
                  disabled={takesBusy}
                  className="px-3 py-2 border border-gray-300 rounded-lg bg-white min-w-48 disabled:opacity-50"
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
                onClick={handleRebuildChapter}
                disabled={
                  takesChapter === '' ||
                  takesBusy ||
                  !takes ||
                  takes.units.length === 0
                }
                className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                <RefreshCw className="w-4 h-4" />
                Перезібрати главу
              </button>
              <button
                onClick={handleInvalidateTakes}
                disabled={
                  takesChapter === '' ||
                  takesBusy ||
                  !takes ||
                  takes.units.length === 0
                }
                className="flex items-center gap-2 px-3 py-2 bg-white border border-red-200 text-red-600 rounded-lg hover:bg-red-50 disabled:opacity-50"
              >
                <Trash2 className="w-4 h-4" />
                Скинути кеш
              </button>
              {project.chapters.every((ch) => ch.status !== 'completed') ? (
                <p className="text-xs text-gray-500">
                  Спочатку синтезуйте хоча б одну главу.
                </p>
              ) : null}
            </div>

            {takesError && (
              <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-100 text-sm text-red-700">
                {takesError}
              </div>
            )}

            {takesLoading ? (
              <p className="text-sm text-gray-500">Завантаження тейків...</p>
            ) : takesChapter === '' ? (
              <p className="text-sm text-gray-500">Оберіть готову главу.</p>
            ) : !takes || takes.units.length === 0 ? (
              <p className="text-sm text-gray-500">
                Маніфест тейків порожній. Синтезуйте главу повторно, щоб
                створити версії речень.
              </p>
            ) : (
              <div className="space-y-3">
                {takes.units.map((unit) => (
                  <div
                    key={unit.index}
                    className="border border-gray-200 rounded-lg p-3"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>Речення {unit.index + 1}</span>
                          {unit.role ? <span>• {unit.role}</span> : null}
                          <span>• {unit.takes.length} верс.</span>
                        </div>
                        <p className="text-sm text-gray-800 mt-1">
                          {unit.text}
                        </p>
                      </div>
                      <button
                        onClick={() => openRegenerate(unit)}
                        disabled={takesBusy || regenUnit === unit.index}
                        className="flex items-center gap-1 text-xs bg-white border border-gray-300 text-gray-700 px-2 py-1 rounded hover:bg-gray-50 disabled:opacity-50 shrink-0"
                      >
                        <Mic className="w-3 h-3" />
                        Перегенерувати
                      </button>
                    </div>

                    {regenUnit === unit.index && (
                      <div className="mt-3 p-3 bg-blue-50 border border-blue-100 rounded-lg space-y-2">
                        <div>
                          <label className="block text-xs font-medium text-gray-500 mb-1">
                            Текст для синтезу
                          </label>
                          <textarea
                            value={regenText}
                            onChange={(e) => setRegenText(e.target.value)}
                            rows={2}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 text-sm"
                          />
                        </div>
                        <div className="flex flex-wrap items-end gap-2">
                          <div className="flex-1 min-w-48">
                            <label className="block text-xs font-medium text-gray-500 mb-1">
                              Голос
                            </label>
                            <select
                              value={regenVoice}
                              onChange={(e) => setRegenVoice(e.target.value)}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white"
                            >
                              <option value="">
                                Як у проєкті / активному тейку
                              </option>
                              {[
                                ...(voices?.voices ?? []),
                                ...(voices?.custom ?? []),
                              ].map((voice) => (
                                <option key={voice} value={voice}>
                                  {voice}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">
                              Темп
                            </label>
                            <input
                              type="number"
                              min={0.5}
                              max={2}
                              step={0.05}
                              value={regenSpeed}
                              onChange={(e) => setRegenSpeed(e.target.value)}
                              className="w-24 px-3 py-2 border border-gray-300 rounded-lg"
                            />
                          </div>
                          <button
                            onClick={() => void handleRegenerateSubmit(unit)}
                            disabled={takesBusy}
                            className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                          >
                            <Mic className="w-4 h-4" />
                            Створити тейк
                          </button>
                          <button
                            onClick={() => setRegenUnit(null)}
                            disabled={takesBusy}
                            className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded-lg disabled:opacity-50"
                          >
                            Скасувати
                          </button>
                        </div>
                        <p className="text-xs text-gray-500">
                          Щоб отримати інший тейк, змініть текст, голос або
                          темп.
                        </p>
                      </div>
                    )}

                    <div className="mt-2 space-y-1">
                      {unit.takes.map((take) => (
                        <TakeRow
                          key={take.id}
                          id={id!}
                          chapter={takesChapter as number}
                          unit={unit.index}
                          take={take}
                          active={take.id === unit.active_take}
                          canDelete={unit.takes.length > 1}
                          busy={takesBusy}
                          onActivate={() =>
                            void handleActivateTake(unit.index, take.id)
                          }
                          onDelete={() =>
                            void handleDeleteTake(unit.index, take.id)
                          }
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
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

function TakeRow({
  id,
  chapter,
  unit,
  take,
  active,
  canDelete,
  busy,
  onActivate,
  onDelete,
}: {
  id: string;
  chapter: number;
  unit: number;
  take: Take;
  active: boolean;
  canDelete: boolean;
  busy: boolean;
  onActivate: () => void;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const { src, loading, error } = useAudioUrl(
    expanded ? unitTakeAudioUrl(id, chapter, unit, take.id) : null
  );

  return (
    <div
      className={`rounded-lg border px-3 py-2 ${
        active ? 'border-green-200 bg-green-50' : 'border-gray-200 bg-gray-50'
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {active ? (
            <CheckCircle2 className="w-4 h-4 text-green-600 shrink-0" />
          ) : (
            <span className="w-4 h-4 shrink-0" />
          )}
          <span className="text-xs font-medium text-gray-800">
            Версія {take.id}
          </span>
          <span className="text-xs text-gray-500 truncate">
            {take.voice ?? 'голос проєкту'} • {take.speed.toFixed(2)}× •{' '}
            {formatDuration(take.duration_ms)}
          </span>
          {active && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-green-100 text-green-700">
              активна
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-600 hover:text-blue-700"
          >
            {expanded ? 'Закрити' : 'Прослухати'}
          </button>
          {!active && (
            <button
              onClick={onActivate}
              disabled={busy}
              className="text-xs text-gray-700 hover:text-gray-900 disabled:opacity-50"
            >
              Активувати
            </button>
          )}
          <button
            onClick={onDelete}
            disabled={busy || !canDelete}
            title={
              canDelete
                ? 'Видалити версію'
                : 'Не можна видалити єдину версію'
            }
            className="text-red-500 hover:text-red-700 disabled:opacity-30"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      {expanded && (
        <div className="mt-2">
          {loading ? (
            <p className="text-xs text-gray-500">Завантаження аудіо...</p>
          ) : null}
          {error ? (
            <p className="text-xs text-red-600">Аудіо: {error}</p>
          ) : null}
          {src ? (
            <audio src={src} controls autoPlay className="w-full h-9" />
          ) : null}
          <p className="text-[10px] text-gray-400 mt-1">
            Синтезовано: {take.spoken}
          </p>
        </div>
      )}
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
  if (message.includes('identical take already exists')) {
    return 'Такий самий тейк уже існує. Змініть текст, голос або темп.';
  }
  if (message.includes('takes cache is invalid or missing')) {
    return 'Кеш тейків застарілий або відсутній. Синтезуйте главу повторно.';
  }
  if (message.includes('cannot delete the last take')) {
    return 'Не можна видалити єдину версію речення.';
  }
  if (message.includes('already has') && message.includes('takes')) {
    return 'Досягнуто ліміт версій для речення. Видаліть старішу версію.';
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
