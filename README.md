# Promin TTS

> **English summary:** native Ukrainian text-to-speech studio service written in **Rust** (Axum + ONNX Runtime, no Python at runtime). Hybrid deployment: local all-in-one desktop mode or thin client + remote server. Target hardware includes **CPU-only servers with baseline AVX** (no AVX2 required). v0.10 adds the multi-stage **HolosTTS** ONNX bundle (encoder → prosody → decoder) with windowed streaming that emits audio before a sentence finishes, plus model-identity tracking that invalidates cached takes when the model changes. v0.11 extends audiobook export with lossless **FLAC**, **ASS** karaoke subtitles with per-word timing, and per-chapter **audio diagnostics** (clipping, long pauses, DC offset, level). v0.12 adds **Back-ASR quality control**: a compact Ukrainian W2V-BERT CTC model transcribes finished chapters, aligns the transcript with the spoken script and reports **WER/CER** with per-word errors, segment windows and audio diagnostics. v0.13 adds **per-chapter export**: any completed chapters can be rendered into separate M4B/MP3/OGG/FLAC files with per-file SRT/VTT/ASS subtitles, as restart-safe asynchronous jobs. v0.14 adds an **embedded web UI**: a React + TypeScript SPA served by the same binary, covering projects, chapter editing, synthesis, roles, export and Back-ASR QC. v0.15 adds a **takes editor** to that UI: per-sentence A/B playback, regeneration with another voice/speed/text, activation and deletion of versions. v0.16 adds **role management** with inline dialogue markup. v0.17 adds **voice previews** before applying a voice, **local voice aliases** for friendlier names, a **collapsible sidebar**, and an inline chapter-creation form that does not depend on browser prompts. It builds on v0.9 per-sentence takes, v0.8 dialogue roles and v0.7 PDF import, with export to **M4B**, **MP3**, **OGG/Opus** and **FLAC** plus combined SRT/VTT/ASS subtitles.

Нативний (Rust) сервіс синтезу мовлення та озвучення довгого контенту (аудіокниги, підкасти) з гібридною архітектурою «клієнт — сервер», Docker-розгортанням та інтеграціями (Home Assistant / Wyoming, n8n, MCP).

## Статус

**v0.17 «Прослуховування голосів»** — повний ланцюжок
`текст → вербалізація → наголос → фонеми → ONNX → WAV`
плюс клонування голосу, асинхронна черга, потоковий синтез, інтеграції,
озвучення книг по главах з LUFS-нормалізацією і субтитрами, імпорт PDF,
експорт готової аудіокниги в M4B/MP3/OGG/FLAC, багатоголосий синтез діалогів
із ролями персонажів та кеш аудіо по реченнях із кількома тейками,
A/B-вибором і перегенерацією окремого речення. ASS-субтитри містять
послівну караоке-розмітку, а діагностика глав знаходить кліпінг, довгі
паузи, постійну складову та проблеми рівня. Вбудований веб-інтерфейс
(React SPA у тому самому бінарнику) керує проєктами, синтезом, ролями,
експортом і Back-ASR QC.

v0.13 додає експорт окремих глав: можна вибрати будь-які завершені глави
(або всі) і отримати для кожної власний аудіофайл M4B/MP3/OGG/FLAC разом
із SRT/VTT/ASS. Job-и асинхронні, зберігаються в
`<project>/exports/per-chapter/`, переживають рестарт і скасовуються
кооперативно. Імена файлів містять id job-а, тому кілька експортів не
перезаписують файли один одного, а невдала глава не блокує решту.

v0.14 додає вбудований **веб-інтерфейс**: React + TypeScript SPA
збирається у `crates/speech-server/web`, вбудовується в бінарник і
відкривається за `http://<host>:8010/`. UI працює з локальним або
віддаленим сервером, дозволяє керувати проєктами, редагувати глави,
запускати синтез, слухати WAV, призначати голоси ролям, експортувати
книгу та переглядати Back-ASR QC. Статичний UI доступний без API-ключа,
а всі маршрути `/v1/*`, `/api/*` і `/mcp` лишаються захищеними.

v0.15 додає у веб-інтерфейс редактор тейків: для готової глави можна
переглянути речення, прослухати альтернативні версії, перегенерувати
окреме речення з іншим голосом, темпом або текстом, активувати найкращий
тейк (глава перезбирається) та видалити зайві версії.

v0.16 додає керування ролями у веб-інтерфейсі: ролі можна створювати
вручну (ім'я + голос), перейменовувати та видаляти, а в редакторі глави —
виділити фрагмент тексту й позначити його реплікою обраної ролі. Заодно
виправлено білий екран на вкладці «Ролі» для старих проєктів, де поле
`roles` було відсутнє в JSON.

v0.17 додає прослуховування короткого зразка голосу перед застосуванням
(кнопка біля кожного селекта голосу та в списку назв), локальні назви
голосів (`voice_aliases` у налаштуваннях проєкту), згортуваний лівий
сайдбар зі збереженням стану та вбудовану форму створення глави замість
`window.prompt`, який блокується в деяких браузерах.

v0.12 додає зворотне розпізнавання готових глав: компактна українська
W2V-BERT CTC-модель (ONNX + SentencePiece) транскрибує аудіо вікнами ~25 с,
текст зіставляється з тим, що реально пішло в синтез (тейк або скрипт),
а звіт містить WER/CER, помилки на рівні слів (заміна/пропуск/вставка),
сегменти з таймінгами та аудіо-діагностику. Перевірки асинхронні,
зберігаються в `<project>/qc/`, переживають рестарт і скасовуються
кооперативно. Модель не входить до Docker-образу: вона завантажується
ліниво при першому QC-запиті (або монтується вручну).

Нова багатоетапна архітектура HolosTTS: замість одного графа — бандл
`holos_onnx.json` + `encoder` / `prosody` / `decoder` (exact) або
`source` + `decoder_window` (стрімінг вікнами). Сервіс сам визначає формат,
а `PROMIN_MODEL_FORMAT` дозволяє вибрати його явно. Ідентичність моделі
(SHA-256 графів і вокабуляра) входить у відбиток глави й тейків, тому
після заміни вагів застарілий кеш не змішується з новим аудіо.

Реалізовано:

- `speech-text` — порт C++ препроцесу: нормалізація, словниковий наголос (marisa-trie,
  2.9 млн форм), українська IPA-транскрипція. Фонеми звірені 1:1 з `holos-tts-core`.
- `speech-text::segment` — нарізка довгого тексту на речення/чанки з урахуванням
  скорочень, ініціалів, дробів і лапок.
- `speech-text::dialogue` — розмітка діалогів `Ім'я: репліка` та `[Ім'я] репліка`
  для зареєстрованих ролей; невідомі префікси лишаються нарацією.
- `speech-inference::accentor` — accentor-lite (ONNX LSTM, 1.2 MB): наголос для
  OOV-слів; словник має пріоритет. Golden-тест 54/54 проти C++ `holos::Accentor`.
- `speech-inference::voice_encoder` — ONNX voice encoder: wave float32 `[1,N]` @24 кГц
  → ембеддінг `[1,256]`. ORT-сесія створюється ліниво; golden-тест cos ≥ 0.995.
- `speech-core::custom_voices` — сховище `<dir>/<name>.npy` (float32[256],
  numpy-сумісне): атомарний запис, рескан каталогу при кожному зверненні.
- `speech-core::jobs` — файлове сховище завдань `<jobs>/<id>.json` + `<id>.wav`,
  атомарний запис, відновлення перерваних завдань після рестарту.
- `speech-core::projects` — модель книги/глав, ролі діалогів (ім'я → голос),
  статуси, `.speechproj` (JSON), атомарне сховище
  `<projects>/<id>/project.json` + `chapters/<n>.wav`.
- `speech-core::takes` — маніфест тейків глави (`chapters/<n>/takes.json`),
  float32 WAV на кожен тейк, SHA-256, відбиток плану глави для інвалідації
  кешу після зміни тексту, налаштувань, ролей або словника вимови.
- `speech-text::import` — EPUB 2/3, FB2 (UTF-8/CP1251), DOCX, PDF (текстовий
  шар), TXT, Markdown; захист ZIP, ліміти розміру/глав, розбиття великих XHTML
  за заголовками. PDF: закладки → евристика заголовків → групування сторінок;
  скановані та зашифровані PDF відхиляються.
- `speech-text::pronunciation` — глобальні та проєктні правила вимови
  (підрядок/regex, регістронезалежність), проєктні перекривають глобальні.
- `speech-asr` — українське розпізнавання для Back-ASR: W2V-BERT frontend
  (80 mel × stride 2, ресемплінг у 16 кГц), greedy CTC + SentencePiece,
  нормалізація тексту, WER/CER і построкове вирівнювання (заміни, пропуски,
  вставки). Модель — `speech-uk/w2v-bert-v3-onnx` (Apache-2.0).
- `speech-audio` — PCM16 з піковою нормалізацією + WAV 24 кГц mono; декодування
  WAV (PCM 8/16/24/32, float32, багатоканальне → моно), ресемплінг через `rubato`,
  рівнопотужний кросфейд між чанками.
- `speech-audio::loudness` — EBU R128 / ACX: вимірювання LUFS + true peak,
  статичний gain, look-ahead лімітер, фінальна TP-корекція (типово −19 LUFS / −3 dBTP).
- `speech-audio::diagnostics` — пік/RMS/crest factor, DC offset, кліпінг,
  провідна/хвостова тиша, найдовша пауза та перелік знахідок для UI.
- `speech-text::verbalize` — rule-вербалізатор: числа, дати, час. Golden 68/68.
- `speech-inference::gemma` — Gemma-3-270M INT8 з KV-кешем; golden 46/46,
  token IDs звірені 1:1 із системним SentencePiece 0.2.0.
- `speech-inference::bundle` — читач бандла HolosTTS: `holos_onnx.json`
  (формат 2, `decoder_mode=exact`, `alignment_format=gather`), `voices.npz`
  (NPY `<f4`/`<U*`), безпечні відносні шляхи етапів.
- `speech-inference::pipeline` — багатоетапний рушій: encoder → prosody →
  exact decoder, а також віконний стрімінг `source` + `decoder_window`
  з обрізанням стартової тиші до 10 мс і фейд-іном 5 мс (паритет із
  Python-еталоном).
- `speech-inference` — ONNX Runtime 1.31.0 (офіційний x64 build; перевірено на
  AVX-only Xeon E5-2650), автоматичний біндінг входів `tokens` / `voice` / `speed`.
- `speech-export` — експорт книги в **M4B** (AAC, глави, обкладинка, метадані),
  **MP3** (ID3v2.4 CHAP/CTOC/APIC), **OGG/Opus** (OpusTags, глави, обкладинка)
  та **FLAC** (lossless, Vorbis comments + нативний Picture-блок); об'єднані
  SRT/VTT/ASS. Pure-Rust енкодери, без ffmpeg у рантаймі.
- `speech-server` — Axum: REST, streaming WAV, асинхронні job-и + SSE, webhook-и,
  OpenAI-сумісний API, API-ключі, автозавантаження моделей, Wyoming TCP, MCP,
  API проєктів книг, імпорт, генерація глав, багатоголосий синтез діалогів,
  SRT/VTT/ASS, `.speechproj`, тейки й перегенерація речень, асинхронні
  export-job-и аудіокниг, діагностика готових глав, Back-ASR перевірки
  з WER/CER і збереженням історії, експорт окремих глав із субтитрами.

## Архітектура

```
crates/
├── speech-core/        # Голоси (.npy), job-и, проєкти книг, `.speechproj`
├── speech-text/        # Препроцесинг, вербалізація, наголоси, IPA, чанки, імпорт, вимова
├── speech-inference/   # ONNX Runtime: TTS-генератор, Gemma, accentor, voice encoder
├── speech-asr/         # Back-ASR: W2V-BERT CTC, SentencePiece, WER/CER, diff
├── speech-audio/       # PCM16, WAV, декодування, ресемплінг, кросфейд, LUFS/TP, діагностика
├── speech-export/      # M4B (AAC), MP3, OGG/Opus, FLAC, глави, обкладинка, SRT/VTT/ASS
└── speech-server/      # Axum + Wyoming + MCP (HTTP і stdio-проксі promin-mcp)
    └── web/            # React + TypeScript + Vite, вбудовується у бінарник
deploy/                 # Dockerfile, docker-compose (cpu-legacy / cpu-modern / gpu)
```

## Моделі

Очікувана розкладка (підтримується і плоска, без підкаталогів):

```
models/
├── holos-onnx/              # багатоетапний бандл HolosTTS (пріоритет у auto)
│   ├── holos_onnx.json      # маніфест: вокабуляр, частота, етапи
│   ├── encoder.onnx         # токени → acoustic/prosodic + durations
│   ├── prosody.onnx         # frame_to_token → f0/energy
│   ├── decoder.onnx         # exact-декодер на всю репліку
│   ├── source.onnx          # опційно: збудження для віконного декодера
│   ├── decoder_window.onnx  # опційно: декодер вікна
│   └── voices.npz           # пресети голосів (names/values)
├── tts/                     # старий однографовий формат (fallback)
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
├── asr/w2v-bert-v3/         # опційно: Back-ASR, ліниве завантаження
│   ├── model.onnx           # граф CTC
│   ├── model.onnx_data      # зовнішні ваги (~2.3 ГБ)
│   └── tokenizer.model      # SentencePiece
├── speakers/                 # клоновані голоси: <name>.npy (float32[256])
├── jobs/                     # асинхронні завдання: <id>.json + <id>.wav
└── projects/                 # книги: <id>/project.json + chapters/<n>.wav
                              #        + chapters/<n>/takes/ (тейки)
                              #        + exports/ (книга, per-chapter/)
```

Бандл шукається в корені `models/`, а також у `tts/`, `onnx/` і `holos-onnx/`.
У режимі `auto` він має пріоритет над старим графом. Якщо локальних моделей
немає зовсім, сервіс перевіряє наявність `holos_onnx.json` у HF: знайдений
бандл завантажується, інакше використовується legacy-граф. Уже розгорнутий
legacy-інстанс автоматично не перемикається. Формат можна зафіксувати:

| Значення `PROMIN_MODEL_FORMAT` | Опис |
|---|---|
| `auto` (типово) | бандл, якщо є `holos_onnx.json`, інакше legacy |
| `bundle` / `holos-onnx` / `onnx` | тільки багатоетапний бандл |
| `legacy` / `holos-cpu` / `int8` | тільки старий однографовий формат |

Частота дискретизації бандла мусить збігатися з серверною (24 кГц).
Розмірність голосу береться з `voices.npz`; наразі очікується 256.

Шлях до моделей задається через `PROMIN_MODELS_DIR`; каталог клонованих голосів —
`PROMIN_SPEAKERS_DIR` (типово `<models>/speakers`), job-ів — `PROMIN_JOBS_DIR`
(типово `<models>/jobs`), проєктів — `PROMIN_PROJECTS_DIR` (типово
`<models>/projects`). Глобальний словник вимови — `<data>/pronunciation.json`
(override: `PROMIN_PRONUNCIATION_FILE`). Без `voice_encoder.onnx` клонування вимкнено
(`/healthz` → `"voice_encoder":"off"`, енролмент → HTTP 503).

Back-ASR модель лежить окремо від TTS: `PROMIN_ASR_MODEL_DIR` (типово
`<models>/asr/w2v-bert-v3`). Вона не потрібна для синтезу й завантажується
ліниво — при першому `POST .../verify`, якщо `PROMIN_ASR_AUTO_DOWNLOAD`
не вимкнено. `/healthz` показує `asr.loaded`, `asr.files_present` і
`asr.missing`, тому стан можна перевірити без запуску перевірки.

### Автозавантаження моделей

Якщо обов'язкових файлів немає, сервер завантажує їх із
[`Morkon/holos-native-models`](https://huggingface.co/Morkon/holos-native-models)
у `PROMIN_MODELS_DIR` (потоково, з `.part` і докачкою через HTTP Range).
У режимі бандла спершу завантажується `holos_onnx.json`, а далі — усі етапи,
які він перелічує; словник наголосів потрібен обом режимам.
Опційні моделі (accentor, voice encoder, Gemma) завантажуються за наявності.

| Змінна | Типово | Опис |
|---|---|---|
| `PROMIN_AUTO_DOWNLOAD` | `auto` | `auto`/`1` — завантажувати відсутнє; `0`/`off` — вимкнено |
| `PROMIN_DOWNLOAD_OPTIONAL` | `1` | `0` — не завантажувати опційні моделі |
| `PROMIN_HF_REPO` | `Morkon/holos-native-models` | Hugging Face репозиторій |
| `PROMIN_HF_REVISION` | `main` | Гілка/тег/коміт |
| `PROMIN_HF_TOKEN` | — | Токен для приватних репозиторіїв |
| `PROMIN_HF_BASE_URL` | `https://huggingface.co` | Дзеркало |
| `PROMIN_HF_BUNDLE_PREFIX` | `holos-onnx` | Каталог бандла в репозиторії HF |
| `PROMIN_DOWNLOAD_TIMEOUT` | `3600` | Таймаут завантаження, секунди |

Якщо каталог моделей read-only і чогось не вистачає, сервер завершиться з
зрозумілою помилкою; для автозавантаження потрібен writable volume.

Back-ASR модель завантажується з `PROMIN_ASR_REPO` (типово
`speech-uk/w2v-bert-v3-onnx`) у `PROMIN_ASR_MODEL_DIR`. Потрібні три файли:
`model.onnx`, `model.onnx_data`, `tokenizer.model`. Якщо
`PROMIN_ASR_AUTO_DOWNLOAD=0`, а файлів немає, `POST .../verify` поверне
HTTP 503 із переліком відсутніх файлів; синтез і експорт при цьому працюють.
Для повторного завантаження після обриву використовується той самий
`.part`-механізм із HTTP Range.

## Швидкий старт (Docker)

```bash
mkdir -p /srv/promin/{models,speakers,jobs,projects}

PROMIN_MODELS_HOST=/srv/promin/models \
PROMIN_SPEAKERS_HOST=/srv/promin/speakers \
PROMIN_JOBS_HOST=/srv/promin/jobs \
PROMIN_PROJECTS_HOST=/srv/promin/projects \
PROMIN_API_KEYS=change-me \
PROMIN_PUBLIC_URL=http://192.168.1.10:8010 \
  docker compose -f deploy/docker-compose.cpu-legacy.yml up -d --build

curl http://localhost:8010/healthz
curl -H 'Authorization: Bearer change-me' http://localhost:8010/api/voices
```

Веб-інтерфейс: **http://localhost:8010/** — відкривається в браузері й працює
з тим самим API. Статичні файли вбудовані в бінарник, окремий Node.js у
рантаймі не потрібен.

Порти: `8010` — HTTP/REST/OpenAI/MCP + веб-UI, `10200` — Wyoming (Home Assistant).
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
export PROMIN_PROJECTS_DIR=/path/to/models/projects
export PROMIN_WYOMING_ADDR=127.0.0.1:10200

cargo run --release -p speech-server -- --http-addr 127.0.0.1:8010
```

## Веб-інтерфейс

Веб-UI — основний інтерфейс для роботи з проєктами. Він збирається з
`crates/speech-server/web` (React 18 + TypeScript + Vite + Tailwind) і
вбудовується у `speech-server` через `rust-embed`.

Можливості MVP:

- підключення до локального або віддаленого сервера (URL + API-ключ);
- список, створення та видалення проєктів; імпорт `EPUB`, `FB2`, `PDF`,
  `DOCX`, `TXT`, `Markdown` і `.speechproj`;
- редактор глав (назва, текст) зі статусами `empty`/`queued`/`running`/
  `completed`/`failed`/`stale`; додавання глав до наявного проєкту;
- синтез глави або всієї книги, прогрес job-ів і скасування;
- прослуховування готового WAV глави;
- тейки речень: A/B-прослуховування, перегенерація з іншим голосом,
  темпом або текстом, активація та видалення версій;
- ролі діалогу, голос нарації та швидкість;
- експорт книги й окремих глав у `M4B`/`MP3`/`OGG`/`FLAC` із субтитрами;
- запуск і перегляд Back-ASR QC (WER/CER, проблемні сегменти).

Статуси job-ів оновлюються опитуванням кожні 3 секунди: SSE-ендпоінт
`/v1/jobs/{id}/events` існує, але браузерний `EventSource` не передає
заголовок `Authorization`, тому UI поки не використовує його.

Збірка фронтенду (потрібен Node.js 20+):

```bash
cd crates/speech-server/web
npm ci
npm run build
```

`cargo build` без зібраного фронтенду створює placeholder-сторінку через
`build.rs`, тому збірка не падає. Docker і CI збирають справжній UI.

Для розробки з гарячим перезавантаженням:

```bash
cd crates/speech-server/web
npm run dev   # http://localhost:5173, проксі на 127.0.0.1:8010
```

## API

Усі маршрути API (`/v1/*`, `/api/*`, `/mcp`) захищені, якщо задано
`PROMIN_API_KEYS` (`Authorization: Bearer <key>` або `X-API-Key: <key>`).
Статичний веб-UI (`/`, `/assets/*`) і `GET /healthz` доступні без ключа,
інакше неможливо було б відкрити сторінку для введення ключа.

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/healthz` | Стан сервера, версія, голоси, вербалізатор, accentor, voice encoder |
| `GET` | `/v1/model`, `/api/model` | Ідентичність моделі: SHA-256 графів, вокабуляра, формат, стрімінг |
| `GET` | `/api/voices` | Пресети + `custom` (клоновані) + голос за замовчуванням |
| `POST` | `/api/voices` | `{name, audio_base64}` → `{name, dim}`; 400 — некоректні дані, 503 — енкодер вимкнено |
| `DELETE` | `/api/voices/{name}` | Видалити клонований голос: 204 або 404 |
| `POST` | `/api/verbalize` | `{text, mode?}` → `{text}`; `mode`: `neural` (типово), `rule`, `off` |
| `POST` | `/api/synthesize` | `{text, voice?, audio_base64?, speed?, auto_verbalize?, raw_stress?}` → `audio/wav` |
| `POST` | `/api/synthesize/stream` | Те саме, але потоковий WAV (chunked) |
| `POST` | `/v1/tts/synthesize` | Синонім `/api/synthesize` |
| `POST` | `/v1/tts/stream` | Синонім `/api/synthesize/stream` |
| `POST` | `/v1/projects/generate-async` | Довгий текст → `202` + job record (legacy) |
| `GET` | `/v1/projects` | Список книг |
| `POST` | `/v1/projects` | Створити проєкт із JSON |
| `POST` | `/v1/projects/import` | Multipart-імпорт EPUB/FB2/DOCX/PDF/TXT/MD або `.speechproj` |
| `GET` | `/v1/projects/{id}` | Повний проєкт із главами |
| `PUT` | `/v1/projects/{id}` | Метадані, налаштування синтезу, словник вимови |
| `DELETE` | `/v1/projects/{id}` | Скасувати job-и глави та видалити проєкт |
| `PUT` | `/v1/projects/{id}/pronunciation` | Замінити проєктний словник вимови |
| `POST` | `/v1/projects/{id}/generate` | Поставити глави в чергу (`chapters?`, `force?`) |
| `POST` | `/v1/projects/{id}/chapters` | Додати главу в кінець проєкту (`title?`, `text?`) |
| `GET`/`PUT` | `/v1/projects/{id}/chapters/{i}` | Отримати/змінити назву й текст глави |
| `GET` | `/v1/projects/{id}/chapters/{i}/audio` | Готовий WAV глави |
| `GET` | `/v1/projects/{id}/chapters/{i}/subtitles?format=srt\|vtt\|ass` | Субтитри глави (ASS — з послівним караоке) |
| `GET` | `/v1/projects/{id}/chapters/{i}/diagnostics` | Аудіо-діагностика глави: кліпінг, паузи, DC offset, рівень |
| `GET` | `/v1/projects/{id}/chapters/{i}/takes` | Маніфест тейків глави (усі речення й версії) |
| `POST` | `/v1/projects/{id}/chapters/{i}/takes` | Скинути кеш тейків глави (204) |
| `POST` | `/v1/projects/{id}/chapters/{i}/takes/units/{u}` | Перегенерувати речення `{voice?, speed?, text?}` як новий тейк |
| `POST` | `/v1/projects/{id}/chapters/{i}/takes/units/{u}/takes/{t}/activate` | Зробити тейк активним і перезібрати главу |
| `DELETE` | `/v1/projects/{id}/chapters/{i}/takes/units/{u}/takes/{t}` | Видалити тейк (204) |
| `GET` | `/v1/projects/{id}/chapters/{i}/takes/units/{u}/takes/{t}/audio` | Float32 WAV окремого тейку |
| `POST` | `/v1/projects/{id}/chapters/{i}/takes/rebuild` | Перезібрати главу з активних тейків без синтезу |
| `GET` | `/v1/projects/{id}/manifest` | JSON-манифест із посиланнями на аудіо/субтитри |
| `GET` | `/v1/projects/{id}/speechproj` | Скачати `.speechproj` |
| `GET` | `/v1/projects/{id}/exports` | Список export-job-ів проєкту |
| `POST` | `/v1/projects/{id}/exports` | Створити експорт `{format, bitrate_kbps?, cover_base64?, ...}`; 409, якщо не всі глави `completed` |
| `GET` | `/v1/projects/{id}/exports/{export_id}` | Статус, прогрес, тривалість, розмір, помилка |
| `DELETE` | `/v1/projects/{id}/exports/{export_id}` | Видалити експорт і його файли: 204/404 |
| `GET` | `/v1/projects/{id}/exports/{export_id}/audio` | Готовий M4B/MP3/OGG/FLAC; 409, якщо експорт не завершений |
| `GET` | `/v1/projects/{id}/exports/{export_id}/subtitles?format=srt\|vtt\|ass` | Об'єднані субтитри книги |
| `GET` | `/v1/projects/{id}/exports/per-chapter` | Історія job-ів експорту окремих глав |
| `POST` | `/v1/projects/{id}/exports/per-chapter` | Експорт вибраних/усіх завершених глав `{format, chapters?, include_subtitles?, bitrate_kbps?}` → 202; 409 для незавершених |
| `GET` | `/v1/projects/{id}/exports/per-chapter/{job_id}` | Статус, прогрес і результат по кожній главі |
| `DELETE` | `/v1/projects/{id}/exports/per-chapter/{job_id}` | Скасувати/видалити job і його файли: 204/404 |
| `GET` | `/v1/projects/{id}/exports/per-chapter/{job_id}/chapter/{i}/audio` | Аудіо окремої глави; 409, якщо глава ще не готова |
| `GET` | `/v1/projects/{id}/exports/per-chapter/{job_id}/chapter/{i}/subtitles?format=srt\|vtt\|ass` | Субтитри окремої глави |
| `GET` | `/v1/jobs?limit=50` | Останні завдання |
| `GET` | `/v1/jobs/{id}` | Статус, прогрес, метадані |
| `DELETE` | `/v1/jobs/{id}` | Скасувати: 204, 404 або 409 (уже завершено) |
| `GET` | `/v1/jobs/{id}/audio` | Готовий WAV; 409, якщо завдання не завершене |
| `GET` | `/v1/jobs/{id}/events` | SSE-потік подій `job` |
| `POST` | `/v1/audio/speech` | OpenAI-сумісний синтез (`wav`/`pcm`) |
| `GET` | `/v1/audio/voices` | Голоси у форматі OpenAI-подібного списку |
| `GET` | `/v1/models` | OpenAI-сумісний список моделей |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 (Streamable HTTP, stateless) |

Потоковий синтез (`/api/synthesize/stream`, `/v1/tts/stream`, Wyoming) ділить
текст на **окремі речення**. З legacy-моделлю перший аудіочанк віддається після
синтезу першого речення, а наступні генеруються послідовно, поки клієнт
відтворює попередні. З бандлом HolosTTS (`source` + `decoder_window`) аудіо
віддається ще дрібніше — по одному декодованому вікну, тому перший звук
з'являється, не чекаючи кінця речення. Стартова тиша зрізається до 10 мс,
після чого перші 5 мс отримують фейд-ін, щоб уникнути клацання.
Сусідні речення не об'єднуються; `max_chars` (50–5000) обмежує лише занадто
довге речення, яке додатково ріжеться за комами/тире/пробілами.

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

### Книги та проєкти (v0.6+)

Проєкт — це книга з главами, налаштуваннями синтезу та власним словником вимови.
Аудіо кожної глави генерується окремим job-ом і зберігається в
`<projects>/<id>/chapters/<index>.wav`; `project.json` — атомарний манифест.

Підтримка імпорту: **EPUB 2/3** (NCX/nav), **FB2** (UTF-8/CP1251), **DOCX**,
**PDF** (лише текстовий шар), **TXT**, **Markdown**, а також `.speechproj`.
Скановані PDF без текстового шару та зашифровані PDF відхиляються (потрібен OCR
або зняття пароля). Ліміти: 64 МБ на файл, 2000 глав, 200 000 символів на главу,
2 000 000 на книгу.

```bash
# Імпорт книги (multipart, поле `file`)
curl -X POST http://localhost:8010/v1/projects/import \
  -H 'Authorization: Bearer change-me' \
  -F 'file=@frankenstein.epub'
# -> 201 {"id":"...","title":"...","chapters":[...]}

# Або створити проєкт вручну
curl -X POST http://localhost:8010/v1/projects \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{
    "title": "Моя книга",
    "author": "Автор",
    "chapters": [
      {"title": "Розділ 1", "text": "Текст першого розділу."},
      {"title": "Розділ 2", "text": "Текст другого розділу."}
    ]
  }'

# Поставити всі незавершені глави в чергу
curl -X POST http://localhost:8010/v1/projects/<id>/generate \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"force": false}'
# -> 202 {"jobs":[{"job_id":"...","chapter_index":0}],"skipped":[],"errors":[]}

# Прогрес — у проєкті або в job-і
curl -H 'Authorization: Bearer change-me' http://localhost:8010/v1/projects/<id>
curl -H 'Authorization: Bearer change-me' http://localhost:8010/v1/jobs/<job_id>

# Готове аудіо та субтитри
curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/chapters/0/audio -o chapter-0.wav
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/chapters/0/subtitles?format=srt' -o chapter-0.srt
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/chapters/0/subtitles?format=vtt' -o chapter-0.vtt
```

Статуси глави: `empty`, `queued`, `running`, `completed`, `failed`, `cancelled`,
`stale`. Якщо текст, налаштування, ролі або словник вимови змінилися під час
генерації, готова глава стає `stale` і потребує перегенерації (`force: true`).
Після рестарту активні глави позначаються `failed`, а `completed` без файлу
аудіо — `stale`.

Нормалізація гучності типово ACX: **−19 LUFS** та **−3 dBTP** (можна змінити
`PROMIN_LOUDNESS_TARGET`/`PROMIN_TRUE_PEAK` або `settings.loudness` проєкту).
Ланцюжок: вимірювання → статичний gain → look-ahead лімітер → перевірка true peak.
Субтитри будуються з пост-кросфейдних меж чанків і діляться на речення.

```bash
# Змінити налаштування проєкту (готові глави стануть stale)
curl -X PUT http://localhost:8010/v1/projects/<id> \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"settings":{"voice":"Speaker_0","speed":1.0,"max_chars":1200,"crossfade_ms":40,"loudness":{"enabled":true,"target_lufs":-19,"true_peak_db":-3}}}'

# Проєктний словник вимови (перекриває глобальний за однаковим pattern)
curl -X PUT http://localhost:8010/v1/projects/<id>/pronunciation \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"rules":[{"pattern":"ІТ","replacement":"ай-ті","case_sensitive":true}]}'

# Експорт/імпорт .speechproj (JSON без аудіо)
curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/speechproj -o book.speechproj
curl -X POST http://localhost:8010/v1/projects/import \
  -H 'Authorization: Bearer change-me' \
  -F 'file=@book.speechproj'
```

### Діалоги та ролі (v0.8)

Ролі задаються на рівні проєкту: ім'я в тексті → голос. У главі репліка
позначається на початку рядка як `Іван: текст` або `[Іван] текст`; усе інше —
нарація голосом проєкту. Порівняння імен регістронезалежне, невідомі префікси
не змінюють текст. Порожній рядок розділяє абзаци.

```bash
# Додати/замінити ролі (готові глави стануть stale)
curl -X PUT http://localhost:8010/v1/projects/<id> \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"roles":[
    {"name":"Іван","voice":"Speaker_0"},
    {"name":"Марія","voice":"Speaker_1"}
  ]}'
```

Текст глави може виглядати так:

```text
Був теплий вечір.
Іван: Привіт, Маріє!
Марія: Привіт, Іване.
```

Кожен голос резолвиться в ембеддінг один раз на главу; чанки різних ролей
склеюються тим самим кросфейдом, а `chunk_timings[].role` зберігає автора
репліки для субтитрів і майбутнього редактора. Якщо `voice` ролі не задано,
використовується голос проєкту. Ролі входять у `.speechproj` і у знімок job-а:
зміна ролей під час генерації позначає главу `stale`.

### Тейки та перегенерація речень (v0.9)

Після генерації глави кожен чанк (речення) зберігається окремо як float32 WAV
у `<projects>/<id>/chapters/<n>/takes/`, а `takes.json` описує версії. Глави
нарізаються через `sentence_chunks`, тому один юніт — це одне речення; занадто
довге речення ділиться за `max_chars`:

```text
chapters/0/
├── takes.json          # маніфест: fingerprint, юніти, активні тейки
├── u0-t1.wav           # речення 0, тейк 1
├── u0-t2.wav           # речення 0, тейк 2 (перегенерований)
└── u1-t1.wav
```

Маніфест містить SHA-256 кожного тейку, голос, темп і текст, що пішов у синтез.
`fingerprint` — SHA-256 від тексту глави, `ProjectSettings`, ролей і словника
вимови; якщо він не збігається з поточним проєктом, кеш вважається застарілим і
видаляється (глави стають `stale`). Дублікат аудіо (той самий SHA-256) не
створюється — `POST` повертає `409`.

```bash
# Усі тейки глави
curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/chapters/0/takes

# Перегенерувати речення 3 іншим голосом і темпом
curl -X POST http://localhost:8010/v1/projects/<id>/chapters/0/takes/units/3 \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"voice":"Speaker_3","speed":1.05}'

# Або з альтернативним текстом лише для озвучення (субтитри не змінюються)
curl -X POST http://localhost:8010/v1/projects/<id>/chapters/0/takes/units/3 \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"text":"Текст із іншим наголосом або вимовою."}'

# A/B: зробити тейк 2 активним (глава перезбирається з кешу)
curl -X POST \
  http://localhost:8010/v1/projects/<id>/chapters/0/takes/units/3/takes/2/activate \
  -H 'Authorization: Bearer change-me'

# Окремий тейк для прослуховування
curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/chapters/0/takes/units/3/takes/2/audio -o take.wav

# Видалити тейк
curl -X DELETE \
  http://localhost:8010/v1/projects/<id>/chapters/0/takes/units/3/takes/2 \
  -H 'Authorization: Bearer change-me'
```

Перезбирання глави використовує той самий кросфейд і LUFS-нормалізацію, що й
повний синтез, тому результат побайтово збігається з оригінальним WAV
(за умови, що активні тейки — це тейки першого проходу). Перегенерація
оновлює лише вибране речення; решта аудіо береться з кешу.

### Експорт аудіокниг (v0.7)

Експорт — асинхронний job, як і генерація глав. Вимагається, щоб **усі** глави
проєкту мали статус `completed` і файл аудіо; інакше `POST` повертає `409` зі
списком `pending_chapters`. Експорт прив'язаний до `project.updated_at`: якщо
проєкт змінився після старту, запис позначається `stale` (аудіо лишається
доступним, але його слід перегенерувати).

Формати та типові бітрейти:

| Формат | Кодек | Типово | Діапазон | Особливості |
|---|---|---|---|---|
| `m4b` | AAC-LC | 96 кбіт/с | 32–256 | глави `chpl`, обкладинка `covr`, `stik=9`, metadata |
| `mp3` | MPEG-1/2 Layer III | 64 кбіт/с | 32–160 | ID3v2.4 `CHAP`/`CTOC`/`APIC` |
| `ogg` | Opus | 48 кбіт/с | 16–128 | OpusTags, глави, обкладинка |
| `flac` | FLAC | — | — | lossless, Vorbis comments, нативний Picture-блок; `bitrate_kbps` ігнорується |

```bash
# Створити експорт (metadata опційна; cover_base64 — data URI або base64)
curl -X POST http://localhost:8010/v1/projects/<id>/exports \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{
    "format": "m4b",
    "bitrate_kbps": 96,
    "title": "Моя книга",
    "author": "Автор",
    "album": "Моя книга",
    "narrator": "Диктор",
    "year": "2026",
    "description": "Опис"
  }'
# -> 202 {"id":"...","status":"queued","format":"m4b","bitrate_kbps":96,...}

curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/exports/<export_id>
# статуси: queued, running, completed, failed, cancelled; progress 0-100

curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/exports/<export_id>/audio -o book.m4b
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/exports/<export_id>/subtitles?format=srt' -o book.srt
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/exports/<export_id>/subtitles?format=vtt' -o book.vtt
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/exports/<export_id>/subtitles?format=ass' -o book.ass
```

ASS містить заголовок, стиль `Default` і `Dialogue`-події з `{\k<cs>}` для
кожного слова. Оскільки модель поки не повертає точних послівних таймінгів,
тривалість слова оцінюється пропорційно його довжині та розділовим знакам.

Файли лежать у `<projects>/<id>/exports/<export_id>.{m4b,mp3,ogg,flac,srt,vtt,ass,json}`;
під час роботи поруч є `.part`. Після рестарту незавершені експорти позначаються
`failed`, завершені лишаються доступними. `PROMIN_EXPORT_CONCURRENCY` (типово 1)
обмежує одночасне кодування. Без `cover_base64` вбудовується стандартна
обкладинка 600×600 PNG.

### Експорт окремими файлами по главах (v0.13)

`POST /v1/projects/{id}/exports/per-chapter` створює асинхронний job, який
кодує кожну вибрану главу в **окремий** файл. `chapters` — індекси глав;
якщо поле пропущене або порожнє, беруться всі глави зі статусом `completed`.
Дублікати індексів відкидаються, а незавершена або відсутня глава дає `409`
(без часткового запуску). `include_subtitles=true` додає SRT/VTT/ASS для
кожної глави.

```bash
# Глави 0 і 1 окремими MP3 + субтитри
curl -X POST http://localhost:8010/v1/projects/<id>/exports/per-chapter \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"format":"mp3","chapters":[0,1],"include_subtitles":true}'
# -> 202 {"id":"<job_id>","status":"queued","chapters":[0,1],...}

curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/exports/per-chapter/<job_id>
# statuses: queued, running, completed, failed, cancelled
# results[]: chapter_index, status, audio_file, subtitles_*_file,
#            duration_ms, bytes, error

curl -H 'Authorization: Bearer change-me' \
  http://localhost:8010/v1/projects/<id>/exports/per-chapter/<job_id>/chapter/0/audio \
  -o chapter-0.mp3
curl -H 'Authorization: Bearer change-me' \
  'http://localhost:8010/v1/projects/<id>/exports/per-chapter/<job_id>/chapter/0/subtitles?format=ass' \
  -o chapter-0.ass
```

Файли лежать у
`<projects>/<id>/exports/per-chapter/<job_id>-chapter-<n>.{m4b,mp3,ogg,flac,srt,vtt,ass}`;
записи job-ів — `<job_id>.json`, під час кодування поруч є `.part`. Загальний
статус job-а — `completed`, лише якщо всі глави успішні; інакше `failed` зі
списком глав у `error`, але аудіо успішних глав лишається доступним. Після
рестарту незавершені job-и позначаються `failed`. `DELETE` скасовує активний
job і видаляє всі його файли; `PROMIN_EXPORT_CONCURRENCY` спільний із повним
експортом книги.

### Аудіо-діагностика

`GET /v1/projects/{id}/chapters/{i}/diagnostics` повертає JSON для готової
глави: `peak_dbfs`, `rms_dbfs`, `crest_factor_db`, `dc_offset`,
`clipped_samples`, `clipping_ratio`, `silence_ratio`, `leading_silence_ms`,
`trailing_silence_ms`, `longest_silence` і `issues`. Коди знахідок:
`clipping`, `near_full_scale`, `low_level`, `dc_offset`, `leading_silence`,
`trailing_silence`, `long_pause`, `silent_audio`, `short_audio`, `empty_audio`.
Пороги орієнтовані на аудіокниги: тиша — нижче −50 dBFS, довга пауза — понад
2 с, кліпінг — семпли ≥ 0.999.

### Back-ASR контроль якості (v0.12)

Перевірка виконується для готової глави (`status=completed`). Аудіо ріжеться
на вікна до `PROMIN_QC_MAX_SEGMENT_MS` (типово 25 с) за `chunk_timings`,
кожне вікно розпізнається українською W2V-BERT CTC-моделлю, після чого
транскрипт порівнюється з еталоном:

- `reference_source="takes"` — `spoken` активного тейку з `takes.json`
  (те, що реально пішло в синтез, з урахуванням вербалізації);
- `reference_source="script"` — `verbalize(chunk.text)`, якщо маніфесту тейків
  немає або він не збігається з таймінгами.

```bash
# Запустити перевірку: 202 + job; повторний запит повертає наявний job (200)
curl -X POST 'http://localhost:8010/v1/projects/<id>/chapters/0/verify'

# Примусово перезапустити навіть для того самого аудіо
curl -X POST 'http://localhost:8010/v1/projects/<id>/chapters/0/verify?force=true'

# Останній звіт глави, історія проєкту, конкретний job
curl 'http://localhost:8010/v1/projects/<id>/chapters/0/verify'
curl 'http://localhost:8010/v1/projects/<id>/qc'
curl 'http://localhost:8010/v1/projects/<id>/qc/<job_id>'

# Скасувати активну перевірку (кооперативно)
curl -X DELETE 'http://localhost:8010/v1/projects/<id>/qc/<job_id>'
```

Статуси job-а: `queued`, `running`, `completed`, `failed`, `cancelled`.
Записи лежать у `<project>/qc/<job_id>.json`; після рестарту активні job-и
позначаються `failed` (`quality check interrupted by server restart`),
завершені лишаються доступними. `?force=true` для активного job-а ставить
його в чергу на скасування і створює новий.

Звіт (`report`) містить `diff` з `wer`, `cer`, `reference_words`,
`substitutions`/`deletions`/`insertions` і масивом `errors` (кожна помилка —
`kind`, індекси та слова з обох боків), `segments` із таймінгами,
`script`/`reference`/`recognized` та власним `diff`, `diagnostics` (ті самі
перевірки, що й у `/diagnostics`), `issues` і застосовані `thresholds`.
Пороги: WER warn/fail — `0.10`/`0.25`, CER — `0.05`/`0.15`; перевизначаються
через `PROMIN_QC_WER_WARN`, `PROMIN_QC_WER_FAIL`, `PROMIN_QC_CER_WARN`,
`PROMIN_QC_CER_FAIL`. `elapsed_ms` включає завантаження моделі, `asr_ms` —
лише інференс, `rtf = elapsed_ms / duration_ms`.

`/healthz` додає `asr` (`model`, `model_dir`, `loaded`, `files_present`,
`missing`, `auto_download`, `threads`) і `qc_active`. Видалення проєкту
скасовує його активні перевірки.

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
| `PROMIN_PROJECTS_DIR` | `<models>/projects` | Проєкти книг (`project.json` + WAV + тейки + exports) |
| `PROMIN_EXPORT_CONCURRENCY` | `1` | Одночасних експортів аудіокниг (повних і по главах) |
| `PROMIN_ASR_MODEL_DIR` | `<models>/asr/w2v-bert-v3` | Каталог Back-ASR моделі |
| `PROMIN_ASR_MODEL_ID` | `speech-uk/w2v-bert-v3-onnx` | Id моделі у звітах |
| `PROMIN_ASR_REPO` | `speech-uk/w2v-bert-v3-onnx` | HF-репозиторій ASR |
| `PROMIN_ASR_REVISION` | `main` | Гілка/тег/коміт ASR |
| `PROMIN_ASR_THREADS` | `4` | ONNX intra-op threads для ASR |
| `PROMIN_ASR_AUTO_DOWNLOAD` | як `PROMIN_AUTO_DOWNLOAD` | Автозавантаження ASR |
| `PROMIN_QC_CONCURRENCY` | `1` | Одночасних Back-ASR перевірок |
| `PROMIN_QC_MAX_SEGMENT_MS` | `25000` | Максимальне вікно розпізнавання |
| `PROMIN_QC_WER_WARN` | `0.10` | WER warning |
| `PROMIN_QC_WER_FAIL` | `0.25` | WER failure |
| `PROMIN_QC_CER_WARN` | `0.05` | CER warning |
| `PROMIN_QC_CER_FAIL` | `0.15` | CER failure |
| `PROMIN_PRONUNCIATION_FILE` | `<data>/pronunciation.json` | Глобальний словник вимови |
| `PROMIN_LOUDNESS_TARGET` | `-19` | Ціль LUFS для нових проєктів |
| `PROMIN_TRUE_PEAK` | `-3` | Ціль dBTP для нових проєктів |
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
cargo test -p speech-text --features import
cargo clippy --workspace --all-targets -- -D warnings
```

Golden-тести accentor/Gemma/voice encoder потребують моделей і запускаються
окремо (див. `crates/*/tests`). Імпорт книг перевіряється синтетичними
EPUB2/EPUB3/FB2/DOCX/PDF/ZIP-фікстурами, LUFS — еталонним синусом і сплесками,
експорт — синтетичними WAV-главами з перевіркою контейнерів і субтитрів.
Експорт по главах додатково перевіряє round-trip job-а на диск, валідацію
id/імен файлів і статуси.

Back-ASR E2E-тест (реальна модель + WAV) запускається окремо:

```bash
PROMIN_ASR_TEST_MODEL_DIR=/models/asr/w2v-bert-v3 \
PROMIN_ASR_TEST_WAV=/path/to/chapter.wav \
cargo test -p speech-asr --release -- --ignored --nocapture
```

## Ліцензія та застереження

- Код: [MIT](LICENSE).
- Back-ASR: модель [`speech-uk/w2v-bert-v3-onnx`](https://huggingface.co/speech-uk/w2v-bert-v3-onnx) — Apache-2.0; крейт `asr-features` — MIT.
- Ваги нейромереж належать їхнім авторам (див. [Morkon/holos-native-models](https://huggingface.co/Morkon/holos-native-models), [patriotyk/HolosTTS](https://github.com/patriotyk/HolosTTS)); проєкт не афілійований з авторами моделей і надається «як є».
