import { useEffect, useRef, useState } from 'react';
import { AlertCircle, Loader2, Play, Square } from 'lucide-react';
import { synthesizeVoiceSample } from '../services/api';

/** Коротка фраза для прослуховування голосу перед застосуванням. */
const SAMPLE_TEXT = 'Привіт! Це короткий зразок голосу.';

/** Blob URL кешуємо на сесію, щоб повторне прослуховування не синтезувало звук. */
const sampleCache = new Map<string, string>();
let activeAudio: HTMLAudioElement | null = null;

interface VoicePreviewButtonProps {
  voice?: string;
  speed?: number;
  className?: string;
  title?: string;
}

export default function VoicePreviewButton({
  voice,
  speed = 1,
  className = '',
  title = 'Прослухати зразок голосу',
}: VoicePreviewButtonProps) {
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Зупиняємо звук цього компонента при розмонтуванні або зміні голосу.
  useEffect(() => {
    return () => {
      const audio = audioRef.current;
      if (audio) {
        audio.pause();
        if (activeAudio === audio) activeAudio = null;
      }
    };
  }, [voice, speed]);

  useEffect(() => {
    setError(null);
  }, [voice, speed]);

  function stop() {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      if (activeAudio === audio) activeAudio = null;
    }
    audioRef.current = null;
    setPlaying(false);
  }

  async function play() {
    setError(null);
    const key = `${voice ?? ''}|${speed}|${SAMPLE_TEXT}`;
    let url = sampleCache.get(key);
    try {
      if (!url) {
        setLoading(true);
        const blob = await synthesizeVoiceSample(SAMPLE_TEXT, voice, speed);
        url = URL.createObjectURL(blob);
        sampleCache.set(key, url);
      }
      // Одночасно грає лише один зразок.
      activeAudio?.pause();
      const audio = new Audio(url);
      audioRef.current = audio;
      activeAudio = audio;
      audio.onended = () => {
        setPlaying(false);
        if (audioRef.current === audio) audioRef.current = null;
        if (activeAudio === audio) activeAudio = null;
      };
      audio.onerror = () => {
        setPlaying(false);
        setError('Не вдалося відтворити зразок');
      };
      setPlaying(true);
      await audio.play();
    } catch (err: unknown) {
      setPlaying(false);
      setError(
        err instanceof Error ? err.message : 'Не вдалося синтезувати зразок'
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <button
      type="button"
      onClick={() => (playing ? stop() : void play())}
      disabled={loading}
      title={error ?? title}
      aria-label={title}
      className={`inline-flex shrink-0 items-center justify-center rounded-lg border p-2 transition-colors disabled:opacity-50 ${
        error
          ? 'border-red-200 bg-red-50 text-red-600'
          : 'border-gray-300 bg-white text-gray-600 hover:bg-gray-50'
      } ${className}`}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : playing ? (
        <Square className="h-4 w-4" />
      ) : error ? (
        <AlertCircle className="h-4 w-4" />
      ) : (
        <Play className="h-4 w-4" />
      )}
    </button>
  );
}
