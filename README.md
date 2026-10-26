# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required).

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

**v0.3 «Accentor»** — повний ланцюжок `текст → вербалізація → наголос → фонеми → ONNX → WAV`.
Працюють обидва вербалізатори: rule (числа, дати, час) і Gemma-3-270M INT8
(decoder-only, KV-cache); без Gemma-моделі автоматично вмикається rule fallback.
Слова поза словником отримують neural-наголос від accentor-lite.

Реалізовано:

- `speech-text` — порт C++ препроцесу: нормалізація, словниковий наголос (marisa-trie,
  2.9 млн форм), українська IPA-транскрипція. Фонеми звірені 1:1 з `holos-tts-core`.
- `speech-inference::accentor` — accentor-lite (ONNX LSTM, 1.2 MB): наголос для
  OOV-слів; словник має пріоритет, accentor підключається лише коли словник не дав
  U+0301. Golden-тест 54/54 проти C++ `holos::Accentor`.
- `speech-text::verbalize` — rule-вербалізатор: кардинальні числа до 999 999 999 999,
  дати `ДД.ММ.РРРР`, час `ГГ:ХХ`. Golden-тест 68/68 проти C++ `holos::verbalize`.
- `speech-text::gemma_prep` — поцифрове читання номерів і групування розрядів
  (порт `gemma_prep`), 33/33 кейси C++.
- `speech-inference::gemma` — Gemma-3-270M INT8 з KV-кешем: промпт Gemma-3 chat,
  бан цифрових токенів, fallback на оригінал речення. Golden-тест 46/46;
  token IDs звірені 1:1 із системним SentencePiece 0.2.0.
- `speech-inference` — ONNX Runtime 1.17.3 (остання версія без вимоги AVX2),
  автоматичний біндінг входів `tokens` / `voice` / `speed`.
- `speech-audio` — PCM16 з піковою нормалізацією + WAV 24 кГц mono.
- `speech-server` — Axum: `GET /healthz`, `GET /api/voices`,
  `POST /api/verbalize`, `POST /api/synthesize`.

## Архітектура

```
crates/
├── speech-core/        # Бізнес-логіка, проєкти, черга генерації (каркас)
├── speech-text/        # Препроцесинг: нормалізація, вербалізація, наголоси, IPA
├── speech-inference/   # ONNX Runtime: TTS-генератор, Gemma-вербалізатор, accentor
├── speech-audio/       # PCM16, WAV-експорт
└── speech-server/      # Axum: REST API
deploy/                 # Dockerfile, docker-compose (cpu-legacy / cpu-modern / gpu)
```

## Моделі

Очікувана розкладка (підтримується і плоска, без підкаталогів):

```
models/
├── tts/
│   ├── holos_cpu_int8.onnx
│   ├── vocab.json
│   └── voices.json
├── stress/
│   └── stress.trie
├── accentor/
│   └── accentor-lite.onnx    # опційно: neural-наголос для OOV-слів
└── verbalizer/gemma-int8/   # опційно: без нього працює rule fallback
    ├── model.onnx
    └── tokenizer.model
```

Шлях задається через `PROMIN_MODELS_DIR`.

## Швидкий старт (Docker)

```bash
PROMIN_MODELS_HOST=/path/to/models \
  docker compose -f deploy/docker-compose.cpu-legacy.yml up -d --build

curl http://localhost:8010/healthz
curl http://localhost:8010/api/voices

curl -X POST http://localhost:8010/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привіт, світе!","voice":"Speaker_0","speed":1.0}' \
  -o out.wav

# Вербалізація чисел/дат без синтезу
curl -X POST http://localhost:8010/api/verbalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ціна 1250 грн, зустріч о 14:30.","mode":"rule"}'
```

Основна цільова платформа — сервери з процесорами, що мають лише **AVX** (без AVX2):
Intel Xeon E5-2650 тощо. Саме тому образ фіксує ONNX Runtime **1.17.3**.

## Локальний запуск

Потрібен ONNX Runtime 1.17.3 (наприклад, у `/opt/ort/current`), а також `cmake`
і `g++` — `sentencepiece-sys` збирає ту саму C++ бібліотеку, що використовує
`holos-tts-core` (гарантує 1:1 token IDs).

```bash
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/ort/current/lib
export LD_LIBRARY_PATH=/opt/ort/current/lib
export PROMIN_MODELS_DIR=/path/to/models

cargo run --release -p speech-server -- --http-addr 127.0.0.1:8010
```

### API

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/healthz` | Стан сервера, кількість голосів, активний вербалізатор та accentor |
| `GET` | `/api/voices` | Список голосів + голос за замовчуванням |
| `POST` | `/api/verbalize` | `{text, mode?}` → `{text}`; `mode`: `neural` (типово), `rule`, `off` |
| `POST` | `/api/synthesize` | `{text, voice?, speed?, auto_verbalize?, raw_stress?}` → `audio/wav` |

`POST /v1/tts/synthesize` — синонім `/api/synthesize`.

- `auto_verbalize`: `true`/`false` або `"neural"`/`"rule"`/`"off"`; відсутнє ⇒ `neural`
  (як у `holos-tts-core`). Якщо Gemma-моделі немає — автоматичний fallback на `rule`.
- `raw_stress`: не додавати автоматичні наголоси (очікуються `+`/U+0301 у тексті).
- Наголос: спершу словник; якщо форма відсутня або неоднозначна — accentor-lite.
  Без `accentor-lite.onnx` OOV-слова лишаються без наголосу (як у v0.2).
- Заголовки відповіді: `X-Promin-Verbalizer` (`off`/`rule`/`neural`) і `X-Promin-Vb-Ms`.
- Обмеження: до 5000 символів на запит, `speed` у діапазоні `0.5..=2.0`.

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
