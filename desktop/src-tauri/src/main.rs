//! Promin TTS — desktop-оболонка Tauri 2 (ТЗ: `desktop/`, «Local Rust
//! Backend» або «Remote Axum Server»).
//!
//! Оболонка відкриває той самий SPA (`crates/speech-server/web`), що й
//! веб-сервер, і реалізує вибір режиму рендерингу:
//!
//! * **Local** — команда `start_local_server` запускає локальний процес
//!   `speech-server` (бінарник поруч із застосунком, `PROMIN_SERVER_BIN`
//!   або PATH) на `127.0.0.1:<port>`; UI чекає його `/healthz`. Дочірній
//!   процес вбивається при виході із застосунку.
//! * **Remote** — користувач вводить URL віддаленого сервера на сторінці
//!   Connection; оболонка участі не бере (`services/api.ts` зберігає
//!   base URL у localStorage).
//!
//! Default baseUrl для UI під Tauri (`http://127.0.0.1:8010`) задано у
//! `crates/speech-server/web/src/services/api.ts`.
//!
//! ## Побудова
//! Crate залежить від системних GTK/WebKit-бібліотек, тому навмисно
//! винесений за кореневий workspace: `cargo build/clippy/test --workspace`
//! його не бачать. Локальна збірка потребує `libwebkit2gtk-4.1-dev` +
//! `libgtk-3-dev`; у CI це джоб `desktop` (web-SPA збирається першим,
//! бо `generate_context!` вбудовує `web/dist` у бінарник).

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, State};

/// Типовий порт локального сервера: збігається з `PROMIN_HTTP_ADDR` за
/// замовчуванням і з дефолтом UI (`DEFAULT_LOCAL_URL` в `api.ts`).
const DEFAULT_LOCAL_PORT: u16 = 8010;

/// Локальний дочірній процес `speech-server` (порожньо — не запущено).
struct LocalServer {
    child: Mutex<Option<Child>>,
}

#[derive(Debug, serde::Serialize)]
struct ServerInfo {
    running: bool,
    port: u16,
    base_url: String,
    pid: Option<u32>,
    /// `"spawned"` — запущено цим застосунком.
    source: String,
}

/// Де шукати бінарник `speech-server` для local-режиму.
///
/// Пріоритет: `PROMIN_SERVER_BIN` → файл поруч із застосунком →
/// (`"speech-server"`) пошук у `PATH` при `spawn`.
fn resolve_server_program() -> String {
    if let Ok(explicit) = std::env::var("PROMIN_SERVER_BIN") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            let name = if cfg!(windows) {
                "speech-server.exe"
            } else {
                "speech-server"
            };
            let sibling = dir.join(name);
            if sibling.is_file() {
                return sibling.to_string_lossy().into_owned();
            }
        }
    }
    "speech-server".to_string()
}

/// Каталог моделей локального сервера: `PROMIN_MODELS_DIR` застосунку
/// (якщо задано) або `<app data dir>/models`.
fn local_models_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("PROMIN_MODELS_DIR") {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|err| format!("app data dir: {err}"))?;
    Ok(dir.join("models"))
}

/// Запустити (або повернути вже запущений) локальний speech-server.
///
/// Дочірній процес успадковує середовище застосунку; явно виставляються
/// лише `PROMIN_HTTP_ADDR` (loopback + порт), `PROMIN_WYOMING_ADDR=off`
/// (десктопу Wyoming не потрібен і він може конфліктувати зі
/// systemd-сервісом) і `PROMIN_MODELS_DIR`.
#[tauri::command]
fn start_local_server(
    app: tauri::AppHandle,
    state: State<'_, LocalServer>,
    port: Option<u16>,
) -> Result<ServerInfo, String> {
    let port = port.unwrap_or(DEFAULT_LOCAL_PORT);
    let base_url = format!("http://127.0.0.1:{port}");
    let mut slot = state.child.lock().map_err(|err| err.to_string())?;

    // Уже запущено цим застосунком — не перевантажуємо.
    if let Some(child) = slot.as_mut() {
        match child.try_wait() {
            Ok(None) => {
                return Ok(ServerInfo {
                    running: true,
                    port,
                    base_url,
                    pid: Some(child.id()),
                    source: "spawned".to_string(),
                });
            }
            // Помер — прибираємо і запускаємо заново.
            Ok(Some(_)) | Err(_) => *slot = None,
        }
    }

    let program = resolve_server_program();
    let models_dir = local_models_dir(&app)?;
    // Не фатально: сервер теж створює каталоги при автозавантаженні.
    let _ = std::fs::create_dir_all(&models_dir);

    let child = Command::new(&program)
        .env("PROMIN_HTTP_ADDR", format!("127.0.0.1:{port}"))
        .env("PROMIN_WYOMING_ADDR", "off")
        .env("PROMIN_MODELS_DIR", &models_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            format!(
                "не вдалося запустити локальний сервер `{program}`: {err}. \
                 Покладіть `speech-server` поруч із застосунком, задайте \
                 PROMIN_SERVER_BIN, або оберіть Remote-режим із URL вже \
                 працюючого сервера"
            )
        })?;
    let pid = child.id();
    *slot = Some(child);

    Ok(ServerInfo {
        running: true,
        port,
        base_url,
        pid: Some(pid),
        source: "spawned".to_string(),
    })
}

fn main() {
    tauri::Builder::default()
        .manage(LocalServer {
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![start_local_server])
        .build(tauri::generate_context!())
        .expect("error while building Promin TTS desktop")
        .run(|app, event| {
            // Дочірній speech-server не має пережити застосунок.
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app.try_state::<LocalServer>() {
                    if let Ok(mut slot) = state.child.lock() {
                        if let Some(mut child) = slot.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Порт десктопного запуску має збігатися з дефолтом UI.
    #[test]
    fn default_port_matches_ui_local_url() {
        assert_eq!(DEFAULT_LOCAL_PORT, 8010);
    }

    /// Пошук бінарника ніколи не повертає порожнє значення
    /// (джереро: `PROMIN_SERVER_BIN`, файл поруч із бінарником, PATH).
    #[test]
    fn resolve_server_program_is_never_empty() {
        let program = resolve_server_program();
        assert!(!program.trim().is_empty());
    }
}
