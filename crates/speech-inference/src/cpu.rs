//! Детекція CPU-можливостей (SIMD) під час запуску.
//!
//! ТЗ (Етап 2, п. «динамічне автовиявлення наборів інструкцій CPU»):
//! визначення рівня SIMD виконується **лише** через runtime CPUID
//! (`is_x86_feature_detected!`), який лише читає біти CPUID і ніколи не
//! виконує незнайомі інструкції — тому `SIGILL` на AVX-без AVX2 процесорах
//! неможливий.
//!
//! Власного SIMD-коду в проєкті немає: кернели обирає сам ONNX Runtime
//! (MLAS) при створенні сесії — AVX-512 → AVX2 → AVX → SSE4.2, зокрема
//! для INT8-бандла. Збірка ведеться без `-C target-cpu=native`
//! (базовий x86-64), тож бінарник запускається на будь-якому x86-64 CPU,
//! а швидкі кернели вмикаються динамічно.

use serde::Serialize;

/// Прапорці CPU, виявлені під час запуску (для логів та `/healthz`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CpuFeatures {
    /// Цільова архітектура збірки (`std::env::consts::ARCH`).
    pub arch: &'static str,
    /// Найвищий доступний рівень SIMD:
    /// `avx512` | `avx2` | `avx` | `sse4.2` | `neon` | `baseline`.
    pub simd: &'static str,
    /// SSE4.2 (присутній на x86-64 від Nehalem+).
    pub sse42: bool,
    /// AVX (без нього — виключно базові SSE-кернели).
    pub avx: bool,
    /// AVX2 + FMA: типовий рівень серверів останніх років.
    pub avx2: bool,
    /// FMA (вимагається разом із AVX2 для FMA-кернелів).
    pub fma: bool,
    /// AVX-512 F (перший мікроархітектурний рівень AVX-512).
    pub avx512f: bool,
    /// AVX-512 BW (байтові/словні операції: INT8 GEMM).
    pub avx512bw: bool,
    /// AVX-512 VNNI — прискорює int8 GEMM (INT8-бандл Holos).
    pub avx512vnni: bool,
}

/// Виявити можливості CPU. Безпечна на будь-якому процесорі: виконується
/// лише CPUID, заборонених інструкцій код не містить і не виконує.
pub fn detect() -> CpuFeatures {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let flags = (
        is_x86_feature_detected!("sse4.2"),
        is_x86_feature_detected!("avx"),
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512bw"),
        is_x86_feature_detected!("avx512vnni"),
    );
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    let flags = (false, false, false, false, false, false, false);

    let (sse42, avx, avx2, fma, avx512f, avx512bw, avx512vnni) = flags;

    // Порядок відповідає ТЗ: AVX-512 → AVX2 → AVX → SSE4.2 → базовий.
    let simd = if avx512f && avx512bw {
        "avx512"
    } else if avx2 && fma {
        "avx2"
    } else if avx {
        "avx"
    } else if sse42 {
        "sse4.2"
    } else if cfg!(target_arch = "aarch64") {
        // NEON обов'язковий для aarch64; прапорці вище стосуються лише x86.
        "neon"
    } else {
        "baseline"
    };

    CpuFeatures {
        arch: std::env::consts::ARCH,
        simd,
        sse42,
        avx,
        avx2,
        fma,
        avx512f,
        avx512bw,
        avx512vnni,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детекція має виконатися без `SIGILL` і лишити прапорці узгодженими
    /// з оголошеним рівнем SIMD.
    #[test]
    fn detection_is_consistent_and_safe() {
        let features = detect();
        assert_eq!(features.arch, std::env::consts::ARCH);

        match features.simd {
            "avx512" => assert!(
                features.avx512f && features.avx512bw,
                "avx512 рівень вимагає avx512f+avx512bw"
            ),
            "avx2" => {
                assert!(
                    features.avx2 && features.fma,
                    "avx2 рівень вимагає avx2+fma"
                );
                assert!(
                    !(features.avx512f && features.avx512bw),
                    "avx2 не має обиратися за наявності AVX-512"
                );
            }
            "avx" => {
                assert!(features.avx && !features.avx2, "avx рівень без AVX2");
                assert!(!(features.avx512f && features.avx512bw));
            }
            "sse4.2" => {
                assert!(features.sse42 && !features.avx, "sse4.2 рівень без AVX");
            }
            "neon" => assert_eq!(features.arch, "aarch64"),
            "baseline" => assert!(!features.sse42 && !features.avx, "baseline без SSE4.2/AVX"),
            other => panic!("невідомий рівень SIMD: {other}"),
        }
    }

    /// Повторні виклики стабільні (без мінливості під час роботи процесу).
    #[test]
    fn detection_is_stable() {
        assert_eq!(detect(), detect());
    }
}
