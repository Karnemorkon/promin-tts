# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required). v0.5 adds async jobs, SSE, streaming WAV, OpenAI-compatible API, API keys, webhooks, model auto-download, Wyoming (Home Assistant) and MCP.

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

**v0.5 «Server & Integrations»** — повний ланцюжок `текст → вербалізація → наголос → фонеми → ONNX → WAV`
плюс клонування голосу, асинхронна черга, потоковий синтез та інтеграції.

Реалізовано:

- `speech-text` — порт C++ препроцесу: нормалізація, словниковий наголос (marisa-trie,
  2.9 млн форм), українська IPA-транскрипція. Фонеми звірені 1:1 з `holos-tts-core`.
- `speech-text::segment` — нарізка довгого тексту на речення/чанки з урахуванням
  скорочень, ініціалів, дробів і лапок.
- `speech-inference::accentor` — accentor-lite (ONNX LSTM, 1.2 MB): наголос для
  OOV-слів; словник має пріоритет. Golden-тест 54/54 проти C++ `holos::Accentor`.
- `speech-inference::voice_encoder` — ONNX voice encoder: wave float32 `[1,N]` @24 кГц
  → ембеддінг `[1,256]`. ORT-сесія створюється ліниво; golden-тест cos ≥ 0.995.
- `speech-core::custom_voices` — сховище `<dir>/<name>.npy` (float32[256],
  numpy-сумісне): атомарний запис, рескан каталогу при кожному зверненні.
- `speech-core::jobs` — файлове сховище завдань `<jobs>/<id>.json` + `<id>.wav`,
  атомарний запис, відновлення перерваних завдань після рестарту.
- `speech-audio` — PCM16 з піковою нормалізацією + WAV 24 кГц mono; декодування
  WAV (PCM 8/16/24/32, float32, багатоканальне → моно), ресемплінг через `rubato`,
  рівнопотужний кросфейд між чанками.
- `speech-text::verbalize` — rule-вербалізатор: числа, дати, час. Golden 68/68.
- `speech-inference::gemma` — Gemma-3-270M INT8 з KV-кешем; golden 46/46,
  token IDs звірені 1:1 із системним SentencePiece 0.2.0.
- `speech-inference` — ONNX Runtime 1.31.0 (офіційний x64 build; перевірено на
  AVX-only Xeon E5-2650), автоматичний біндінг входів `tokens` / `voice` / `speed`.
- `speech-server` — Axum: REST, streaming WAV, асинхронні job-и + SSE, webhook-и,
  OpenAI-сумісний API, API-ключі, автозавантаження моделей, Wyoming TCP, MCP.

## Архітектура

```
crates/
├── speech-core/        # Бізнес-логіка: кастомні голоси (.npy), черга та сховище job-ів
├── speech-text/        # Препроцесинг: нормалізація, вербалізація, наголоси, IPA, чанки
├── speech-inference/   # ONNX Runtime: TTS-генератор, Gemma, accentor, voice encoder
├── speech-audio/       # PCM16, WAV, декодування, ресемплінг, кросфейд
└── speech-server/      # Axum + Wyoming + MCP (HTTP і stdio-проксі promin-mcp)
deploy/                 # Dockerfile, docker-compose (cpu-legacy / cpu-modern / gpu)
```

## Моделі

Очікувана розкладка (підтримується і плоска, без підкаталогів):

```
models/
├── tts/
│   ├── holos_cpu_int8.onnx
│   ├── vocab.json
│   ├── voices.json
│   └── voice_encoder.onnx    # опційно: клонування голосу
├── stress/
│   └── stress.trie
├── accentor/
│   └── accentor-lite.onnx    # опційно: neural-наголос для OOV-слів
├── verbalizer/gemma-int8/   # опційно: без нього працює rule fallback
│   ├── model.onnx
│   ├── tokenizer.model
│   └── config.json
├── speakers/                 # клоновані голоси: <name>.npy (float32[256])
└── jobs/                     # асинхронні завдання: <id>.json + <id>.wav
```

Шлях до моделей задається через `PROMIN_MODELS_DIR`; каталог клонованих голосів —
`PROMIN_SPEAKERS_DIR` (типово `<models>/speakers`), job-ів — `PROMIN_JOBS_DIR`
(типово `<models>/jobs`). Без `voice_encoder.onnx` клонування вимкнено
(`/healthz` → `"voice_encoder":"off"`, енролмент → HTTP 503).

### Автозавантаження моделей

Якщо обов'язкових файлів немає, сервер завантажує їх із
[`Morkon/holos-native-models`](https://huggingface.co/Morkon/holos-native-models)
у `PROMIN_MODELS_DIR` (потоково, з `.part` і докачкою через HTTP Range).
Опційні моделі (accentor, voice encoder, Gemma) завантажуються за наявності.

| Змінна | Типово | Опис |
|---|---|---|
| `PROMIN_AUTO_DOWNLOAD` | `auto` | `auto`/`1` — завантажувати відсутнє; `0`/`off` — вимкнено |
| `PROMIN_DOWNLOAD_OPTIONAL` | `1` | `0` — не завантажувати опційні моделі |
| `PROMIN_HF_REPO` | `Morkon/holos-native-models` | Hugging Face репозиторій |
| `PROMIN_HF_REVISION` | `main` | Гілка/тег/коміт |
| `PROMIN_HF_TOKEN` | — | Токен для приватних репозиторіїв |
| `PROMIN_HF_BASE_URL` | `https://huggingface.co` | Дзеркало |
| `PROMIN_DOWNLOAD_TIMEOUT` | `3600` | Таймаут завантаження, секунди |

Якщо каталог моделей read-only і чогось не вистачає, сервер завершиться з
зрозумілою помилкою; для автозавантаження потрібен writable volume.

## Швидкий старт (Docker)

```bash
mkdir -p /srv/promin/{models,speakers,jobs}

PROMIN_MODELS_HOST=/srv/promin/models \
PROMIN_SPEAKERS_HOST=/srv/promin/speakers \
PROMIN_JOBS_HOST=/srv/promin/jobs \
PROMIN_API_KEYS=change-me \
PROMIN_PUBLIC_URL=http://192.168.1.10:8010 \
  docker compose -f deploy/docker-compose.cpu-legacy.yml up -d --build

curl http://localhost:8010/healthz
curl -H 'Authorization: Bearer change-me' http://localhost:8010/api/voices
```

Порти: `8010` — HTTP/REST/OpenAI/MCP, `10200` — Wyoming (Home Assistant).
Для immutable-моделей додайте `:ro` до mount-у `/data/models`, якщо всі файли вже є.

```bash
# Короткий синтез
curl -X POST http://localhost:8010/api/synthesize \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"text":"Привіт, світе!","voice":"Speaker_0","speed":1.0}' \
  -o out.wav

# Вербалізація чисел/дат без синтезу
curl -X POST http://localhost:8010/api/verbalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ціна 1250 грн, зустріч о 14:30.","mode":"rule"}'

# Клонування голосу: WAV (0.25–120 с) -> base64 -> .npy
B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8010/api/voices \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"Мій голос\",\"audio_base64\":\"$B64\"}"

# Синтез збереженим клоном або ad-hoc (без збереження)
curl -X POST http://localhost:8010/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Це мій клонований голос.","voice":"Мій голос"}' \
  -o clone.wav
curl -X POST http://localhost:8010/api/synthesize \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Це ad-hoc клон.\",\"audio_base64\":\"$B64\"}" \
  -o adhoc.wav
```

Основна цільова платформа — сервери з процесорами, що мають лише **AVX** (без AVX2):
Intel Xeon E5-2650 тощо. Образ фіксує ONNX Runtime **1.31.0** (офіційний x64 build).

## Локальний запуск

Потрібен ONNX Runtime 1.31.0 (наприклад, у `/opt/ort-1.31.0`), а також `cmake`
і `g++` — `sentencepiece-sys` збирає ту саму C++ бібліотеку, що використовує
`holos-tts-core` (гарантує 1:1 token IDs).

```bash
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/ort-1.31.0/lib
export LD_LIBRARY_PATH=/opt/ort-1.31.0/lib
export PROMIN_MODELS_DIR=/path/to/models
export PROMIN_SPEAKERS_DIR=/path/to/models/speakers
export PROMIN_JOBS_DIR=/path/to/models/jobs
export PROMIN_WYOMING_ADDR=127.0.0.1:10200

cargo run --release -p speech-server -- --http-addr 127.0.0.1:8010
```

## API

Усі маршрути, крім `GET /healthz`, захищені, якщо задано `PROMIN_API_KEYS`
(`Authorization: Bearer <key>` або `X-API-Key: <key>`).

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/healthz` | Стан сервера, версія, голоси, вербалізатор, accentor, voice encoder |
| `GET` | `/api/voices` | Пресети + `custom` (клоновані) + голос за замовчуванням |
| `POST` | `/api/voices` | `{name, audio_base64}` → `{name, dim}`; 400 — некоректні дані, 503 — енкодер вимкнено |
| `DELETE` | `/api/voices/{name}` | Видалити клонований голос: 204 або 404 |
| `POST` | `/api/verbalize` | `{text, mode?}` → `{text}`; `mode`: `neural` (типово), `rule`, `off` |
| `POST` | `/api/synthesize` | `{text, voice?, audio_base64?, speed?, auto_verbalize?, raw_stress?}` → `audio/wav` |
| `POST` | `/api/synthesize/stream` | Те саме, але потоковий WAV (chunked) |
| `POST` | `/v1/tts/synthesize` | Синонім `/api/synthesize` |
| `POST` | `/v1/tts/stream` | Синонім `/api/synthesize/stream` |
| `POST` | `/v1/projects/generate-async` | Довгий текст → `202` + job record |
| `GET` | `/v1/jobs?limit=50` | Останні завдання |
| `GET` | `/v1/jobs/{id}` | Статус, прогрес, метадані |
| `DELETE` | `/v1/jobs/{id}` | Скасувати: 204, 404 або 409 (уже завершено) |
| `GET` | `/v1/jobs/{id}/audio` | Готовий WAV; 409, якщо завдання не завершене |
| `GET` | `/v1/jobs/{id}/events` | SSE-потік подій `job` |
| `POST` | `/v1/audio/speech` | OpenAI-сумісний синтез (`wav`/`pcm`) |
| `GET` | `/v1/audio/voices` | Голоси у форматі OpenAI-подібного списку |
| `GET` | `/v1/models` | OpenAI-сумісний список моделей |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 (Streamable HTTP, stateless) |

### Синтез

- `audio_base64`: WAV-референс для клонування (PCM 8/16/24/32 або float32,
  будь-яка частота/канали, 0.25–120 с; ресемплиться до 24 кГц). Має **пріоритет**
  над `voice`; потребує `voice_encoder.onnx`, інакше HTTP 503.
- `voice`: пресет із `voices.json` або клонований `<name>.npy`. Кастомні голоси
  шукаються після пресетів; типово — `Speaker_0`.
- `auto_verbalize`: `true`/`false` або `"neural"`/`"rule"`/`"off"`; відсутнє ⇒ `neural`
  (як у `holos-tts-core`). Якщо Gemma-моделі немає — автоматичний fallback на `rule`.
- `raw_stress`: не додавати автоматичні наголоси (очікуються `+`/U+0301 у тексті).
- Короткий синтез: до 5000 символів; заголовки `X-Promin-Verbalizer`,
  `X-Promin-Vb-Ms`, `X-Promin-Chunks`, `X-Promin-Audio-Ms`.
- Потоковий WAV: PCM16 24 кГц mono, кожен чанк нормалізується окремо;
  заголовок WAV містить `u32::MAX` замість невідомих розмірів.

### Довгі тексти та job-и

`POST /v1/projects/generate-async` приймає до **200 000** символів. Текст
вербалізується, ріжеться на чанки (`max_chars`, типово 1200, 50–5000) і
склеюється з рівнопотужним кросфейдом (`crossfade_ms`, типово 40, 0–200).
Пікова нормалізація виконується один раз на весь файл.

```bash
curl -X POST http://localhost:8010/v1/projects/generate-async \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Дуже довгий текст...",
    "voice": "Speaker_0",
    "max_chars": 1200,
    "crossfade_ms": 40,
    "webhook_url": "https://example.com/promin-hook"
  }'
# -> 202 {"id":"...","status":"queued",...}

curl http://localhost:8010/v1/jobs/<id>
curl -N http://localhost:8010/v1/jobs/<id>/events
curl http://localhost:8010/v1/jobs/<id>/audio -o long.wav
```

Статуси: `queued`, `running`, `completed`, `failed`, `cancelled`; `progress` 0–100.
SSE надсилає початковий стан одразу, далі події `job` з JSON. Webhook отримує
`job_id`, `status`, `progress`, `audio_url`, `audio_bytes`, `duration_ms`, `chunks`,
`verbalizer`, `error`, `finished_at`. `PROMIN_PUBLIC_URL` використовується для
`audio_url` і має бути доступним ззовні.

Job-и зберігаються у `PROMIN_JOBS_DIR`; після рестарту незавершені позначаються
`failed` («server restarted before completion»), завершені лишаються доступними.
`PROMIN_JOBS_CONCURRENCY` (типово 1) обмежує одночасний синтез.

### OpenAI-сумісний API

```bash
curl -X POST http://localhost:8010/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"promin-tts","input":"Привіт!","voice":"alloy","response_format":"wav"}' \
  -o speech.wav
```

`response_format`: `wav` (типово) або `pcm` (raw PCM16 24 кГц mono). Стандартні
OpenAI-назви голосів (`alloy`, `nova`, …) мапляться на голос за замовчуванням;
пресети та клони можна передавати напряму. MP3/OGG поки не підтримуються.

### Wyoming (Home Assistant)

Сервер слухає Wyoming TCP на `PROMIN_WYOMING_ADDR` (типово `0.0.0.0:10200`;
`off` вимикає). Підтримуються `describe`, `synthesize`, `ping`; відповіді —
`info`, `audio-start`, `audio-chunk`, `audio-stop`, `pong`, `error`. Аудіо —
PCM16 24 кГц mono. У Home Assistant: **Settings → Devices & Services → Add
Integration → Wyoming Protocol**, host = IP сервера, port = `10200`.

```bash
python3 - <<'PY'
import json, socket
s = socket.create_connection(("127.0.0.1", 10200))
s.sendall((json.dumps({"type": "describe"}) + "\n").encode())
print(s.makefile().readline().strip())
PY
```

### MCP

MCP доступний двома способами:

1. **HTTP JSON-RPC**: `POST http://localhost:8010/mcp` (під тим самим API-ключем).
2. **stdio-проксі** `promin-mcp` для Claude Desktop / Cursor:

```json
{
  "mcpServers": {
    "promin-tts": {
      "command": "/usr/local/bin/promin-mcp",
      "env": {
        "PROMIN_MCP_URL": "http://127.0.0.1:8010/mcp",
        "PROMIN_API_KEY": "change-me"
      }
    }
  }
}
```

Інструменти: `synthesize`, `synthesize_async`, `get_job`, `list_jobs`,
`list_voices`, `clone_voice`, `delete_voice`, `verbalize`. `synthesize` повертає
MCP audio content (`audio/wav` base64) і текст із метаданими.

## Змінні середовища

| Змінна | Типово | Опис |
|---|---|---|
| `PROMIN_HTTP_ADDR` | `0.0.0.0:8010` | HTTP listen address |
| `PROMIN_WYOMING_ADDR` | `0.0.0.0:10200` | Wyoming TCP; `off` вимикає |
| `PROMIN_MODELS_DIR` | `models` | Каталог моделей |
| `PROMIN_SPEAKERS_DIR` | `<models>/speakers` | Клоновані голоси |
| `PROMIN_JOBS_DIR` | `<models>/jobs` | Job-и та аудіо |
| `PROMIN_JOBS_CONCURRENCY` | `1` | Одночасних job-ів |
| `PROMIN_PUBLIC_URL` | `http://127.0.0.1:<port>` | База для `audio_url`/webhook |
| `PROMIN_API_KEYS` | — | Ключі через кому; порожньо = без авторизації |
| `PROMIN_THREADS` | `4` | ONNX intra-op threads |
| `PROMIN_AUTO_DOWNLOAD` | `auto` | Автозавантаження моделей |
| `PROMIN_HF_*` | — | Репозиторій/ревізія/токен/дзеркало |
| `RUST_LOG` | `info` | Рівень логування |

## Тести

```bash
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/ort-1.31.0/lib
export LD_LIBRARY_PATH=/opt/ort-1.31.0/lib
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Golden-тести accentor/Gemma/voice encoder потребують моделей і запускаються
окремо (див. `crates/*/tests`).

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
