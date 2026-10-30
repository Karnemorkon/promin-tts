import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Plus, Trash2, RefreshCw, FileText, Upload } from 'lucide-react';
import {
  listProjects,
  createProject,
  deleteProject,
  importProject,
} from '../services/api';
import type { ProjectSummary } from '../types';

export default function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listProjects();
      setProjects(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося завантажити проєкти');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    const title = window.prompt('Назва нового проєкту:');
    if (!title?.trim()) return;
    try {
      const project = await createProject({ title: title.trim() });
      navigate(`/projects/${project.id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося створити проєкт');
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Видалити проєкт?')) return;
    try {
      await deleteProject(id);
      loadProjects();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося видалити проєкт');
    }
  };

  const handleImport = async (file: File) => {
    setImporting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const project = await importProject(form);
      navigate(`/projects/${project.id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Не вдалося імпортувати файл');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-gray-50 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Проєкти</h1>
          <p className="text-gray-600">Керуйте вашими проєктами озвучення</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
            <Upload className="w-4 h-4" />
            {importing ? 'Імпорт...' : 'Імпорт книги'}
            <input
              type="file"
              accept=".epub,.fb2,.pdf,.docx,.txt,.md,.speechproj"
              className="hidden"
              disabled={importing}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImport(file);
                e.target.value = '';
              }}
            />
          </label>
          <button
            onClick={loadProjects}
            className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            title="Оновити"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleCreate}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Новий проєкт
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-100 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {loading ? (
          <div className="text-center py-10 text-gray-500">Завантаження...</div>
        ) : projects.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-xl border border-gray-200">
            <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3 text-gray-400">
              <FileText className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-medium text-gray-900">Проєктів не знайдено</h3>
            <p className="text-gray-500 text-sm">
              Створіть перший проєкт або імпортуйте книгу
            </p>
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full text-left">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase">Назва</th>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase">Автор</th>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase">Мова</th>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase">Глави</th>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase">Тривалість</th>
                  <th className="px-6 py-3 text-xs font-semibold text-gray-500 uppercase text-right">Дії</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {projects.map((p) => (
                  <tr key={p.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4">
                      <Link
                        to={`/projects/${p.id}`}
                        className="font-medium text-gray-900 hover:text-blue-600"
                      >
                        {p.title}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-gray-600">{p.author || '—'}</td>
                    <td className="px-6 py-4 text-gray-600">{p.language || 'uk'}</td>
                    <td className="px-6 py-4 text-gray-600">
                      {p.completed_chapters}/{p.chapters}
                    </td>
                    <td className="px-6 py-4 text-gray-600 text-sm">
                      {formatDuration(p.total_duration_ms)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => handleDelete(p.id)}
                        className="text-red-500 hover:text-red-700"
                        title="Видалити"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
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
