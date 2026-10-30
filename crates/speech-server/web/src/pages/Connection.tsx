import { useState } from 'react';
import {
  Server,
  Wifi,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  KeyRound,
} from 'lucide-react';
import { healthz, loadApiKey, setApiKey, setBaseUrl } from '../services/api';
import { useConnection } from '../stores/connection';

interface ConnectionProps {
  onConnect: (
    next: Partial<{
      mode: 'local' | 'remote';
      baseUrl: string;
      apiKey: string | null;
      connected: boolean;
    }>
  ) => void;
}

export default function Connection({ onConnect }: ConnectionProps) {
  const { state } = useConnection();
  const [mode, setMode] = useState<'local' | 'remote'>(state.mode);
  const [baseUrl, setBaseUrlState] = useState(
    state.baseUrl || 'http://localhost:8010'
  );
  const [apiKey, setApiKeyState] = useState(loadApiKey() ?? '');
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setConnecting(true);
    setError(null);

    const url = mode === 'local' ? '' : baseUrl.trim();
    setBaseUrl(url);
    setApiKey(apiKey.trim() || null);

    try {
      await healthz();
      onConnect({
        mode,
        baseUrl: url,
        apiKey: apiKey.trim() || null,
        connected: true,
      });
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? `Не вдалося підключитися: ${err.message}`
          : 'Не вдалося підключитися до сервера'
      );
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-gray-100 p-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Server className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Promin TTS</h1>
          <p className="text-gray-600 text-sm">Audio Studio — веб-інтерфейс</p>
        </div>

        <form onSubmit={handleConnect} className="space-y-5">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Режим роботи</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setMode('local')}
                className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  mode === 'local'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Server className="w-4 h-4" />
                Локальний
              </button>
              <button
                type="button"
                onClick={() => setMode('remote')}
                className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  mode === 'remote'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Wifi className="w-4 h-4" />
                Віддалений
              </button>
            </div>
            <p className="text-xs text-gray-500">
              {mode === 'local'
                ? 'Веб-інтерфейс відкрито з того самого сервера, що й API.'
                : 'Вкажіть адресу віддаленого speech-server.'}
            </p>
          </div>

          {mode === 'remote' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 block">
                URL сервера
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrlState(e.target.value)}
                placeholder="http://192.168.22.146:8010"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              />
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 flex items-center gap-1">
              <KeyRound className="w-3.5 h-3.5" />
              API-ключ (опційно)
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKeyState(e.target.value)}
              placeholder="PROMIN_API_KEYS"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
            />
            <p className="text-xs text-gray-500">
              Потрібен, якщо сервер запущено з <code>PROMIN_API_KEYS</code>.
            </p>
          </div>

          {error && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-red-50 border border-red-100">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 shrink-0" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={connecting}
            className="w-full py-3 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
          >
            {connecting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Підключення...
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4" />
                Підключитися
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
