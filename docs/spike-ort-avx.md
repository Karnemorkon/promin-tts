# Spike A: ONNX Runtime без AVX2 (цільовий CPU: Intel Xeon E5-2650)

**Дата:** 2026-10-23
**Платформа:** 192.168.22.146 — Debian 13, 2× Xeon E5-2650 (Sandy Bridge-EP, 16C/32T), 24 GB RAM, **лише AVX** (без AVX2 / FMA / BMI2), без GPU.
**Мета:** обрати стратегію інтеграції `ort` crate з ONNX Runtime так, щоб синтез працював на процесорах без AVX2 без `SIGILL`.

## Результати

| Конфігурація | Результат |
|---|---|
| `ort = { git = "https://github.com/pykeio/ort", tag = "v1.16.3" }` (ORT 1.16.0, pyke prebuilt CPU) | **PASS** — інференс MNIST onnx виконано, exit 0, без SIGILL |
| `ort = "=2.0.0-rc.15"` (pyke prebuilt binaries) | **FAIL** — `Illegal instruction` (exit 132) + вбудоване попередження: *"This CPU does not support AVX2 … use a custom build of ONNX Runtime to fix"* |

Висновок прямо підтверджує документацію ONNX Runtime: **офіційні prebuilt-бінарники ORT ≥1.18 вимагають AVX2**; остання лінійка, що працює на AVX-only — ORT ≤1.17.

## Рішення (v0.1) — фінальне

- Залежність: `ort = { git = "https://github.com/pykeio/ort", tag = "v1.16.3" }`.
  - Усі версії `ort` 1.x зі *crates.io* **yanked** — тому пін саме через git-тег (він не зникне після yanked).
  - `ort` 1.16.x API відрізняється від 2.0-rc: `Environment` + `SessionBuilder::new(&env)` + `Value::from_array(allocator, &cow_array)`; `ndarray` **0.15** (не 0.16).
- **Вбудовані бінарники `download-binaries` (ORT 1.16.0) не годяться**: модель `holos_cpu_int8.onnx` використовує **ONNX opset 20**, який офіційно підтримується лише ORT ≥1.17.
- **Робоча конфігурація (перевірено на E5-2650, модель 321 MB завантажилась, тест `load_holos_model` — PASS)**:

```bash
# один раз: офіційний ORT 1.17.3 (остання AVX-безпечна версія, підтримка opset 20)
curl -fL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar xf ort.tgz && mv onnxruntime-linux-x64-1.17.3 /opt/ort/current

# збірка та запуск
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/ort/current/lib
export LD_LIBRARY_PATH=/opt/ort/current/lib
export PROMIN_MODELS_DIR=/root/models
cargo test -p speech-inference -- --ignored
```

- `ORT_STRATEGY=system` обов'язковий: з увімкненим дефолтним `download-binaries` змінна `ORT_LIB_LOCATION` ігнорується (див. build.rs ort v1.16.3).
- У продакшен-Docker копіюємо `libonnxruntime.so.1.17.3` в образ і виставляємо `LD_LIBRARY_PATH`.

## Довгострокова стратегія (міграція на ort 2.x)

`ort` 2.x за замовчуванням не підтримує AVX-only. Шляхи:

1. **Своя збірка ONNX Runtime** (реліз-артефакт) з базовим AVX без AVX2 + `ort::init_from(path)` / `load-dynamic` — рекомендований шлях (саме так радить і сам `ort`).
2. Триматись 1.16.x до появи потреби в EP, недоступних ORT 1.16 (CUDA EP у 1.16 частково підтримується; перевірити окремо).

## Відтворення

Спайк-крейти залишені на сервері: `/root/spike/ort-avx` (PASS) та `/root/spike/ort2-avx` (FAIL SIGILL).
Тестова модель: `/root/spike/mnist-12.onnx` (26 KB, MNIST-12 з onnx/models).

```bash
cd /root/spike/ort-avx && LD_LIBRARY_PATH=target/release ./target/release/ort-avx
# SPIKE-A-PASS inference OK on AVX-only CPU (E5-2650)
```

Завантаження реальної моделі TTS (перевірка opset 20 + AVX):

```bash
cd /root/promin-tts
ORT_STRATEGY=system ORT_LIB_LOCATION=/opt/ort/current/lib LD_LIBRARY_PATH=/opt/ort/current/lib \
  PROMIN_MODELS_DIR=/root/models cargo test -p speech-inference -- --ignored
# test engine::tests::load_holos_model ... ok
```
