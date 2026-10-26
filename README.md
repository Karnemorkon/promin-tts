# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required).

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

**v0.4 «Voice Encoder»** — повний ланцюжок `текст → вербалізація → наголос → фонеми → ONNX → WAV`
плюс клонування голосу з референсного аудіо.
Працюють обидва вербалізатори: rule (числа, дати, час) і Gemma-3-270M INT8
(decoder-only, KV-cache); без Gemma-моделі автоматично вмикається rule fallback.
Слова поза словником отримують neural-наголос від accentor-lite.

Реалізовано:

- `speech-text` — порт C++ препроцесу: нормалізація, словниковий наголос (marisa-trie,
  2.9 млн форм), українська IPA-транскрипція. Фонеми звірені 1:1 з `holos-tts-core`.
- `speech-inference::accentor` — accentor-lite (ONNX LSTM, 1.2 MB): наголос для
  OOV-слів; словник має пріоритет, accentor підключається лише коли словник не дав
  U+0301. Golden-тест 54/54 проти C++ `holos::Accentor`.
- `speech-inference::voice_encoder` — ONNX voice encoder: wave float32 `[1,N]` @24 кГц
  → ембеддінг `[1,256]`. ORT-сесія створюється ліниво при першому енролменті;
  golden-тест проти C++: cos ≥ 0.995, детермінізм 1.0.
- `speech-core::custom_voices` — сховище `<dir>/<name>.npy` (float32[256],
  numpy-сумісне): атомарний запис, рескан каталогу при кожному зверненні,
  валідація назв `^[\w \-]{1,40}$` (Unicode).
- `speech-audio` — PCM16 з піковою нормалізацією + WAV 24 кГц mono; декодування
  WAV (PCM 8/16/24/32, float32, багатоканальне → моно) і ресемплінг до 24 кГц
  через `rubato` (FFT).
- `speech-text::verbalize` — rule-вербалізатор: кардинальні числа до 999 999 999 999,
  дати `ДД.ММ.РРРР`, час `ГГ:ХХ`. Golden-тест 68/68 проти C++ `holos::verbalize`.
- `speech-text::gemma_prep` — поцифрове читання номерів і групування розрядів
  (порт `gemma_prep`), 33/33 кейси C++.
- `speech-inference::gemma` — Gemma-3-270M INT8 з KV-кешем: промпт Gemma-3 chat,
  бан цифрових токенів, fallback на оригінал речення. Golden-тест 46/46;
  token IDs звірені 1:1 із системним SentencePiece 0.2.0.
- `speech-inference` — ONNX Runtime 1.31.0 (офіційний x64 build; перевірено на
  AVX-only Xeon E5-2650), автоматичний біндінг входів `tokens` / `voice` / `speed`.
- `speech-server` — Axum: `GET /healthz`, `GET/POST /api/voices`,
  `DELETE /api/voices/{name}`, `POST /api/verbalize`, `POST /api/synthesize`.

## Архітектура

```
crates/
├── speech-core/        # Бізнес-логіка: кастомні голоси (.npy), проєкти, черга
├── speech-text/        # Препроцесинг: нормалізація, вербалізація, наголоси, IPA
├── speech-inference/   # ONNX Runtime: TTS-генератор, Gemma, accentor, voice encoder
├── speech-audio/       # PCM16, WAV-експорт, декодування та ресемплінг
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
│   ├── voices.json
│   └── voice_encoder.onnx    # опційно: клонування голосу
├── stress/
│   └── stress.trie
├── accentor/
│   └── accentor-lite.onnx    # опційно: neural-наголос для OOV-слів
├── verbalizer/gemma-int8/   # опційно: без нього працює rule fallback
│   ├── model.onnx
│   └── tokenizer.model
└── speakers/                 # клоновані голоси: <name>.npy (float32[256])
```

Шлях до моделей задається через `PROMIN_MODELS_DIR`; каталог клонованих голосів —
`PROMIN_SPEAKERS_DIR` (типово `<models>/speakers`). Без `voice_encoder.onnx`
клонування вимкнено (`/healthz` → `"voice_encoder":"off"`, енролмент → HTTP 503),
решта синтезу працює як раніше.

## Швидкий старт (Docker)

```bash
PROMIN_MODELS_HOST=/path/to/models \
PROMIN_SPEAKERS_HOST=/path/to/speakers \
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
Intel Xeon E5-2650 тощо. Образ фіксує ONNX Runtime **1.31.0** (офіційний x64 build);
він підтримує IR version 10 (`voice_encoder.onnx`) і перевірений на AVX-only CPU.

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

cargo run --release -p speech-server -- --http-addr 127.0.0.1:8010
```

### API

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/healthz` | Стан сервера, кількість голосів, вербалізатор, accentor, voice encoder |
| `GET` | `/api/voices` | Пресети + `custom` (клоновані) + голос за замовчуванням |
| `POST` | `/api/voices` | `{name, audio_base64}` → `{name, dim}`; 400 — некоректні дані, 503 — енкодер вимкнено |
| `DELETE` | `/api/voices/{name}` | Видалити клонований голос: 204 або 404 |
| `POST` | `/api/verbalize` | `{text, mode?}` → `{text}`; `mode`: `neural` (типово), `rule`, `off` |
| `POST` | `/api/synthesize` | `{text, voice?, audio_base64?, speed?, auto_verbalize?, raw_stress?}` → `audio/wav` |

`POST /v1/tts/synthesize` — синонім `/api/synthesize`.

- `audio_base64`: WAV-референс для клонування (PCM 8/16/24/32 або float32,
  будь-яка частота/канали, 0.25–120 с; ресемплиться до 24 кГц). Має **пріоритет**
  над `voice`; потребує `voice_encoder.onnx`, інакше HTTP 503.
- `voice`: пресет із `voices.json` або клонований `<name>.npy`. Кастомні голоси
  шукаються після пресетів; типово — `Speaker_0`.
- Клоновані голоси: `<PROMIN_SPEAKERS_DIR>/<name>.npy`, float32[256],
  numpy-сумісні (`np.load`/`np.save`). Каталог ресканується при кожному запиті,
  тож `.npy`, покладені у volume, зʼявляються без рестарту. Назва:
  `^[\w \-]{1,40}$` (Unicode: літери, цифри, `_`, пробіл, дефіс).
- `auto_verbalize`: `true`/`false` або `"neural"`/`"rule"`/`"off"`; відсутнє ⇒ `neural`
  (як у `holos-tts-core`). Якщо Gemma-моделі немає — автоматичний fallback на `rule`.
- `raw_stress`: не додавати автоматичні наголоси (очікуються `+`/U+0301 у тексті).
- Наголос: спершу словник; якщо форма відсутня або неоднозначна — accentor-lite.
  Без `accentor-lite.onnx` OOV-слова лишаються без наголосу (як у v0.2).
- Заголовки відповіді: `X-Promin-Verbalizer` (`off`/`rule`/`neural`) і `X-Promin-Vb-Ms`.
- Обмеження: до 5000 символів на запит, `speed` у діапазоні `0.5..=2.0`,
  тіло запиту до 64 МБ (для `audio_base64`).

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
