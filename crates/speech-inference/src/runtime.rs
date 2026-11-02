//! Вибір execution provider (EP) ONNX Runtime під час запуску.
//!
//! ТЗ (Етап 2): «динамічне автовиявлення наборів інструкцій CPU
//! (SIMD dispatching) … CUDA → AVX-512 → AVX2 → AVX fallback».
//!
//! Як це реалізовано чесно:
//! * **CUDA** — реєструється лише у збірці з ознакою `gpu`
//!   (`ort/cuda` + `ort/load-dynamic`) і лише якщо ONNX Runtime
//!   повідомляє про наявність `CUDAExecutionProvider`
//!   (`GetAvailableProviders`, тобто GPU-бібліотека завантажена).
//! * **CPU (AVX-512 → AVX2 → AVX → SSE4.2 → baseline)** — це внутрішня
//!   диспетчеризація кернелів ONNX Runtime (MLAS): вона виконується при
//!   створенні сесії на підставі тих самих runtime-CPUID, без `SIGILL`.
//!   Наші крейти не містять власного SIMD-коду (див. [`crate::cpu`]).
//!
//! Керування змінною середовища `PROMIN_ORT_EP`:
//! * `auto` (типово) — CUDA, якщо доступна, інакше CPU;
//! * `cuda` — вимагати CUDA (попередження + CPU, якщо недоступна);
//! * `cpu` — завжди CPU.
//!
//! У збірці без ознаки `gpu` будь-яке значення, крім `cpu`, вирішується
//! в `cpu` (без CUDA-коду у бінарнику).
//!
//! ### Чесне попередження
//! За документацією `ort`, неправильно сконфігурований CUDA (немає
//! драйвера/на неправильній версії чи відсутній cuDNN) може завершити
//! **весь процес** під час першого `Run`, а не впасти з помилкою.
//! Тому: типове значення — `auto`; CPU-образи ніколи не реєструють CUDA;
//! GPU-образ слід запускати лише на хості з NVIDIA Container Toolkit
//! (`deploy/docker-compose.gpu.yml`).

use std::sync::OnceLock;

use anyhow::{Context, Result};
use ort::environment::{EnvBuilder, Environment};

#[cfg(feature = "gpu")]
use ort::execution_providers::{CUDAExecutionProviderOptions, ExecutionProvider};

/// Фінальний вибір: `"cpu"` або `"cuda"`. Обчислюється один раз при першому
/// створенні середовища ORT (перший запуск, до обслуговування запитів).
static SELECTED: OnceLock<&'static str> = OnceLock::new();

/// Запит із `PROMIN_ORT_EP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preference {
    /// `cpu` — завжди CPU.
    Cpu,
    /// `cuda` — вимагати CUDA.
    Cuda,
    /// `auto` (типово) — CUDA за наявності.
    Auto,
}

fn preference_from_env() -> Preference {
    match std::env::var("PROMIN_ORT_EP") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Preference::Cpu,
            "cuda" | "gpu" => Preference::Cuda,
            "auto" | "" => Preference::Auto,
            other => {
                tracing::warn!(
                    value = %other,
                    "unknown PROMIN_ORT_EP (expected cpu|cuda|auto); falling back to `auto`"
                );
                Preference::Auto
            }
        },
        Err(_) => Preference::Auto,
    }
}

/// CUDA доступна у цій збірці та у цій ONNX Runtime бібліотеці.
#[cfg(feature = "gpu")]
fn cuda_available() -> bool {
    ExecutionProvider::CUDA(CUDAExecutionProviderOptions::default()).is_available()
}

/// CPU-збірка: CUDA-коду немає, доступної CUDA теж.
#[cfg(not(feature = "gpu"))]
fn cuda_available() -> bool {
    false
}

fn select() -> &'static str {
    let preference = preference_from_env();
    let required = matches!(preference, Preference::Cuda);
    match preference {
        Preference::Cpu => "cpu",
        Preference::Cuda | Preference::Auto => {
            if !cfg!(feature = "gpu") {
                if required {
                    tracing::warn!(
                        "PROMIN_ORT_EP=cuda requested but this binary is built without the \
                         `gpu` feature; using CPU execution provider"
                    );
                }
                "cpu"
            } else if cuda_available() {
                "cuda"
            } else {
                if required {
                    tracing::warn!(
                        "PROMIN_ORT_EP=cuda requested but CUDAExecutionProvider is not available \
                         in this ONNX Runtime build; using CPU execution provider"
                    );
                }
                "cpu"
            }
        }
    }
}

/// Обраний виконавець: `"cpu"` або `"cuda"` (для `/healthz` та логів).
///
/// Якщо реєстрація CUDA з якихось причин не вдасться, `ort` сам друкує
/// попередження і застосує CPU; це значення показує **запитаний** вибір.
pub fn execution_provider() -> &'static str {
    SELECTED.get_or_init(select)
}

/// Створити середовище ONNX Runtime із налаштованим виконавцем.
///
/// Усі сесії (генератор, вербалізатор, акцентор, енкодер голосу) створюються
/// з середовища, тому EP реєструється рівно один раз на середовище.
pub fn build_environment(name: &str) -> Result<Environment> {
    with_selected_provider(Environment::builder().with_name(name))
        .build()
        .with_context(|| format!("create ONNX Runtime environment `{name}`"))
}

#[cfg(feature = "gpu")]
fn with_selected_provider(builder: EnvBuilder) -> EnvBuilder {
    if execution_provider() == "cuda" {
        // CPU EP в ORT завжди доступний як fallback, явно додаємо лише CUDA.
        builder.with_execution_providers([ExecutionProvider::CUDA(
            CUDAExecutionProviderOptions::default(),
        )])
    } else {
        builder
    }
}

#[cfg(not(feature = "gpu"))]
fn with_selected_provider(builder: EnvBuilder) -> EnvBuilder {
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Типове значення без `PROMIN_ORT_EP` — `auto`; у CPU-збірці (як і в
    /// тестах CI без GPU) воно має вирішитися в `cpu`.
    #[test]
    fn selection_is_stable_and_cpu_without_gpu_build() {
        let selected = execution_provider();
        assert!(
            selected == "cpu" || selected == "cuda",
            "невідомий EP: {selected}"
        );
        assert_eq!(execution_provider(), selected);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn cpu_build_never_selects_cuda() {
        assert_eq!(execution_provider(), "cpu");
    }
}
