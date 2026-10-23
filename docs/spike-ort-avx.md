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

## Рішення (v0.1)

- Залежність: `ort = { git = "https://github.com/pykeio/ort", tag = "v1.16.3" }`.
  - Усі версії `ort` 1.x зі *crates.io* **yanked** — тому пін саме через git-тег (він не зникне після yanked).
  - `ort` 1.16.x API відрізняється від 2.0-rc: `Environment` + `SessionBuilder::new(&env)` + `Value::from_array(allocator, &cow_array)`; `ndarray` **0.15** (не 0.16).
- Динамічна бібліотека `libonnxruntime.so` завантажується під час `cargo build` у `target/.../ort-sys-*/out/`; **без `LD_LIBRARY_PATH` бінарник не стартує**. У продакшен-Docker копіюємо `.so` в образ і виставляємо `LD_LIBRARY_PATH` або rpath.

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
