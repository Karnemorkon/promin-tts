# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required).

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

Активна розробка. Етап 0 — каркас monorepo; повне технічне завдання: [tz-rust-tts-service-v5.md](tz-rust-tts-service-v5.md).

## Архітектура

```
crates/
├── speech-core/        # Бізнес-логіка, проєкти, черга генерації
├── speech-text/        # Препроцесинг: імпорт, вербалізація, наголоси
├── speech-inference/   # ONNX Runtime: генератор, вербалізатор, акцентор, voice encoder
├── speech-audio/       # DSP, кросфейд, LUFS, експорт
└── speech-server/      # Axum: REST, WebSocket, OpenAI API, Wyoming, MCP
desktop/                # Tauri 2 + React (тонкий клієнт / локальний GUI)
deploy/                 # Dockerfile, docker-compose (cpu-legacy / cpu-modern / gpu)
```

## Швидкий старт (Docker)

```bash
docker compose -f deploy/docker-compose.cpu-legacy.yml up -d --build
curl http://localhost:8010/healthz
```

Основна цільова платформа — сервери з процесорами, що мають лише **AVX** (без AVX2): Intel Xeon E5-2650 тощо.

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
