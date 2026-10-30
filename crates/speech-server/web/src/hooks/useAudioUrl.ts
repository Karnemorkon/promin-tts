import { useEffect, useState } from 'react';
import { loadApiKey } from '../services/api';

/**
 * Завантажує аудіо з Authorization-заголовком і повертає blob URL.
 * `<audio src>` не вміє додавати заголовки, тому для API-ключів
 * використовуємо fetch + object URL.
 */
export function useAudioUrl(url: string | null): {
  src: string | null;
  loading: boolean;
  error: string | null;
} {
  const [src, setSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!url) {
      setSrc(null);
      return;
    }
    let revoked: string | null = null;
    let cancelled = false;
    setLoading(true);
    setError(null);

    const key = loadApiKey();
    const headers: Record<string, string> = {};
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
      headers['X-API-Key'] = key;
    }

    fetch(url, { headers })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}`);
        }
        const blob = await res.blob();
        if (cancelled) return;
        const objectUrl = URL.createObjectURL(blob);
        revoked = objectUrl;
        setSrc(objectUrl);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Не вдалося завантажити аудіо');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [url]);

  return { src, loading, error };
}
