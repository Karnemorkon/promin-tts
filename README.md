# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required).

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

**v0.1 «Core Parity»** — повний ланцюжок `текст → фонеми → ONNX → WAV` працює через REST.
Повне технічне завдання: [tz-rust-tts-service-v5.md](tz-rust-tts-service-v5.md).

Реалізовано:

- `speech-text` — порт C++ препроцесу: нормалізація, словниковий наголос (marisa-trie,
  2.9 млн форм), українська IPA-транскрипція. Фонеми звірені 1:1 з `holos-tts-core`.
- `speech-inference` — ONNX Runtime 1.17.3 (остання версія без вимоги AVX2),
  автоматичний біндінг входів `tokens` / `voice` / `speed`.
- `speech-audio` — PCM16 з піковою нормалізацією + WAV 24 кГц mono.
- `speech-server` — Axum: `GET /healthz`, `GET /api/voices`, `POST /api/synthesize`.

## Архітектура

```
crates/
├── speech-core/        # Бізнес-логіка, проєкти, черга генерації (каркас)
├── speech-text/        # Препроцесинг: нормалізація, наголоси, IPA
├── speech-inference/   # ONNX Runtime: TTS-генератор
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
└── stress/
    └── stress.trie
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
```

Основна цільова платформа — сервери з процесорами, що мають лише **AVX** (без AVX2):
Intel Xeon E5-2650 тощо. Саме тому образ фіксує ONNX Runtime **1.17.3**.

## Локальний запуск

Потрібен ONNX Runtime 1.17.3 (наприклад, у `/opt/ort/current`):

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
| `GET` | `/healthz` | Стан сервера, кількість голосів |
| `GET` | `/api/voices` | Список голосів + голос за замовчуванням |
| `POST` | `/api/synthesize` | `{text, voice?, speed?}` → `audio/wav` |

`POST /v1/tts/synthesize` — синонім `/api/synthesize`. Обмеження v0.1:
до 5000 символів на запит, `speed` у діапазоні `0.5..=2.0`.

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
