import { useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import type { PronunciationRule } from '../types';

/** Стільки ж, скільки MAX_PRONUNCIATION_RULES на сервері. */
export const MAX_PRONUNCIATION_RULES = 1000;

interface PronunciationRulesEditorProps {
  rules: PronunciationRule[];
  onChange: (rules: PronunciationRule[]) => void;
  disabled?: boolean;
  emptyHint?: string;
}

/** Редактор списку правил вимови (шаблон → заміна). */
export default function PronunciationRulesEditor({
  rules,
  onChange,
  disabled = false,
  emptyHint = 'Правил ще немає.',
}: PronunciationRulesEditorProps) {
  const [pattern, setPattern] = useState('');
  const [replacement, setReplacement] = useState('');
  const [regex, setRegex] = useState(false);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function addRule() {
    if (!pattern.trim()) {
      setError('Шаблон не може бути порожнім');
      return;
    }
    if (rules.length >= MAX_PRONUNCIATION_RULES) {
      setError(`Максимум ${MAX_PRONUNCIATION_RULES} правил`);
      return;
    }
    onChange([
      ...rules,
      {
        pattern: pattern.trim(),
        replacement,
        regex,
        case_sensitive: caseSensitive,
      },
    ]);
    setPattern('');
    setReplacement('');
    setRegex(false);
    setCaseSensitive(false);
    setError(null);
  }

  function updateRule(index: number, patch: Partial<PronunciationRule>) {
    onChange(rules.map((rule, i) => (i === index ? { ...rule, ...patch } : rule)));
  }

  function removeRule(index: number) {
    onChange(rules.filter((_, i) => i !== index));
  }

  return (
    <div>
      {rules.length === 0 ? (
        <p className="text-xs text-gray-500 mb-3">{emptyHint}</p>
      ) : (
        <ul className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-white mb-3">
          {rules.map((rule, index) => (
            <li key={index} className="px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <input
                  value={rule.pattern}
                  onChange={(e) => updateRule(index, { pattern: e.target.value })}
                  disabled={disabled}
                  placeholder="Шаблон"
                  className="flex-1 min-w-32 px-2 py-1 border border-gray-300 rounded text-sm bg-white disabled:opacity-60"
                />
                <span className="text-gray-400 text-sm">→</span>
                <input
                  value={rule.replacement}
                  onChange={(e) =>
                    updateRule(index, { replacement: e.target.value })
                  }
                  disabled={disabled}
                  placeholder="Заміна"
                  className="flex-1 min-w-32 px-2 py-1 border border-gray-300 rounded text-sm bg-white disabled:opacity-60"
                />
                <label className="flex items-center gap-1 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={!!rule.regex}
                    onChange={(e) =>
                      updateRule(index, { regex: e.target.checked })
                    }
                    disabled={disabled}
                  />
                  regex
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={!!rule.case_sensitive}
                    onChange={(e) =>
                      updateRule(index, {
                        case_sensitive: e.target.checked,
                      })
                    }
                    disabled={disabled || !!rule.regex}
                  />
                  регістр
                </label>
                <button
                  type="button"
                  onClick={() => removeRule(index)}
                  disabled={disabled}
                  title="Видалити правило"
                  className="p-1.5 text-gray-400 hover:text-red-600 rounded hover:bg-red-50 disabled:opacity-50"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <div className="flex-1 min-w-32">
          <label className="block text-xs text-gray-500 mb-1">Шаблон</label>
          <input
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addRule();
              }
            }}
            disabled={disabled}
            placeholder="Наприклад, ІТ"
            className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white disabled:opacity-60"
          />
        </div>
        <div className="flex-1 min-w-32">
          <label className="block text-xs text-gray-500 mb-1">Заміна</label>
          <input
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addRule();
              }
            }}
            disabled={disabled}
            placeholder="ай-ті"
            className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white disabled:opacity-60"
          />
        </div>
        <label className="flex items-center gap-1 text-xs text-gray-600 pb-2">
          <input
            type="checkbox"
            checked={regex}
            onChange={(e) => setRegex(e.target.checked)}
            disabled={disabled}
          />
          regex
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-600 pb-2">
          <input
            type="checkbox"
            checked={caseSensitive}
            onChange={(e) => setCaseSensitive(e.target.checked)}
            disabled={disabled || regex}
          />
          регістр
        </label>
        <button
          type="button"
          onClick={addRule}
          disabled={disabled || !pattern.trim()}
          className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Plus className="w-4 h-4" />
          Додати
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      <p className="mt-2 text-xs text-gray-400">
        {rules.length} / {MAX_PRONUNCIATION_RULES} правил
      </p>
    </div>
  );
}
