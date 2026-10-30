import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
  useCallback,
} from 'react';
import { loadApiKey } from '../services/api';

export interface ConnectionState {
  mode: 'local' | 'remote';
  baseUrl: string;
  apiKey: string | null;
  connected: boolean;
}

const STORAGE_KEY = 'promin_connection';

const initial: ConnectionState = {
  mode: 'local',
  baseUrl: '',
  apiKey: null,
  connected: false,
};

/** Відновлює режим/URL із localStorage (API-ключ зберігається окремо). */
function loadInitial(): ConnectionState {
  if (typeof localStorage === 'undefined') return initial;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return initial;
    const parsed = JSON.parse(raw) as Partial<ConnectionState>;
    return {
      mode: parsed.mode === 'remote' ? 'remote' : 'local',
      baseUrl: typeof parsed.baseUrl === 'string' ? parsed.baseUrl : '',
      apiKey: loadApiKey(),
      connected: parsed.connected === true,
    };
  } catch {
    return initial;
  }
}

const ConnectionContext = createContext<{
  state: ConnectionState;
  setState: (next: Partial<ConnectionState>) => void;
} | null>(null);

export function ConnectionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ConnectionState>(loadInitial);

  const updateState = useCallback((next: Partial<ConnectionState>) => {
    setState((s) => ({ ...s, ...next }));
  }, []);

  // Зберігаємо режим, URL і прапорець підключення між перезавантаженнями.
  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          mode: state.mode,
          baseUrl: state.baseUrl,
          connected: state.connected,
        })
      );
    } catch {
      /* localStorage може бути недоступним (приватний режим) */
    }
  }, [state.mode, state.baseUrl, state.connected]);

  return (
    <ConnectionContext.Provider value={{ state, setState: updateState }}>
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  const ctx = useContext(ConnectionContext);
  if (!ctx) {
    throw new Error('useConnection must be used within ConnectionProvider');
  }
  return ctx;
}
