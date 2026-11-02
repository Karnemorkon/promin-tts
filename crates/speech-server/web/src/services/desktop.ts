/** Інтеграція з desktop-оболонкою Tauri (`desktop/src-tauri`).
 *
 * У браузері всі функції — no-op: там «local» означає той самий origin,
 * який уже обслуговує завантажений UI. У Tauri вебв'ю має власний origin
 * (`tauri://…`), тому local-режим запускає окремий процес `speech-server`
 * на `127.0.0.1` і чекає його `/healthz`.
 */
import { getBaseUrl, healthz } from './api';

type InvokeFn = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

function tauriInvoke(): InvokeFn | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    __TAURI__?: { core?: { invoke?: InvokeFn } };
  };
  const invoke = w.__TAURI__?.core?.invoke;
  return typeof invoke === 'function' ? invoke : null;
}

/** Чи запущено під desktop-оболонкою Tauri. */
export function isDesktop(): boolean {
  return tauriInvoke() !== null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Скільки чекати локальний сервер (перший запуск вантажить моделі). */
const LOCAL_SERVER_TIMEOUT_MS = 60_000;

/**
 * Local-режим: у Tauri запустити локальний `speech-server` (якщо він ще
 * не працює) і дочекатися його `/healthz`; у браузері — no-op.
 *
 * Помилка запуску не є фатальною сама по собі: якщо сервер уже працює
 * (наприклад, його запустив Docker), здоров'я перевіриться `/healthz`.
 */
export async function ensureLocalBackend(): Promise<void> {
  const invoke = tauriInvoke();
  if (!invoke) return;

  let spawnError: string | null = null;
  try {
    await invoke('start_local_server');
  } catch (err) {
    spawnError = err instanceof Error ? err.message : String(err);
  }

  const deadline = Date.now() + LOCAL_SERVER_TIMEOUT_MS;
  let lastError: unknown = spawnError;
  while (Date.now() < deadline) {
    try {
      await healthz();
      return;
    } catch (err) {
      lastError = err;
      await sleep(500);
    }
  }
  const detail =
    lastError instanceof Error ? lastError.message : String(lastError ?? 'невідома помилка');
  throw new Error(`Локальний сервер не відповів (${getBaseUrl()}): ${detail}`);
}
