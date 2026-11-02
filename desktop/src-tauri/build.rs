use std::path::Path;

fn main() {
    ensure_sidecar_files();
    tauri_build::build()
}

/// Sidecar-файли з `bundle.externalBin` мають існувати вже на етапі
/// компіляції: `tauri-build` завершується помилкою `ResourcePathNotFound`,
/// якщо поруч із крейтом немає файлу `bin/<name>-<target triple>`.
///
/// У CI перед `tauri build` туди кладуться справжні `speech-server` і
/// `libonnxruntime.so.1` (див. джоб `package`); локально для
/// `cargo check/clippy/test` створюється порожня заглушка — десктоп-крейт
/// компілюється без збірки сервера. Пакування (`tauri build`) без справжніх
/// файлів дасть битий пакет, тому про заглушку друкується попередження.
fn ensure_sidecar_files() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let extension = if target.contains("windows") {
        ".exe"
    } else {
        ""
    };
    for name in ["speech-server", "libonnxruntime.so.1"] {
        let path = Path::new("bin").join(format!("{name}-{target}{extension}"));
        if path.exists() {
            continue;
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(err) = std::fs::write(&path, b"") {
            println!(
                "cargo:warning=не вдалося створити заглушку sidecar {}: {err}",
                path.display()
            );
            continue;
        }
        println!(
            "cargo:warning=sidecar {} відсутній — створено порожню заглушку \
             (для пакування потрібні справжні бінарники)",
            path.display()
        );
    }
}
