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
//! ## Системний трей
//! На Linux лівий клік по іконці трею відкриває контекстне меню
//! (`show_menu_on_left_click(true)` за замовчуванням);містяться там дії
//! «Розгорнути вікно», «Приховати вікно», «Вийти». При закритті вікна
//! хрестиком вікно приховується в трей (замість повного виходу), щоб
//! synthesis не переривався у тлі.
//!
//! ## Динамічний заголовок
//! frontend викликає команду `set_synth_active(active)`, коли починається
//! або закінчується хоча б одне завдання синтезу; заголовок вікна змінюється
//! на «Promin TTS — синтезується…» / «Promin TTS».

//!
//! ## Побудова
//! Crate залежить від системних GTK/WebKit-бібліотек, тому навмисно
//! винесений за кореневий workspace: `cargo build/clippy/test --workspace`
//! його не бачить. Локальна збірка потребує `libwebkit2gtk-4.1-dev` +
//! `libgtk-3-dev`; у CI це джоб `desktop` (web-SPA збирається першим,
//! бо `generate_context!` вбудовує `web/dist` у бінарник).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Manager, RunEvent, State, WindowEvent};

use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder, PredefinedMenuItem};
use tauri::tray::{TrayIconBuilder, TrayIconEvent};
use tauri::WebviewWindow;

/// Типовий порт локального сервера: збігається з `PROMIN_HTTP_ADDR` за
/// замовчуванням і з дефолтом UI (`DEFAULT_LOCAL_URL` в `api.ts`).
const DEFAULT_LOCAL_PORT: u16 = 8010;

/// Локальний дочірній процес `speech-server` (порожньо — не запущено).
struct LocalServer {
    child: Mutex<Option<Child>>,
}

/// Чи йде зараз хоча б одне завдання синтезу (для заголовка вікна).
struct SynthState {
    active: Mutex<bool>,
}

/// Чи застосунок зараз виходить (щоб не приховати вікно в трей замість виходу).
struct Exiting {
    flag: Mutex<bool>,
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

/// Додати каталог бінарника сервера в `LD_LIBRARY_PATH`.
fn set_sidecar_library_path(program: &str, command: &mut Command) -> Result<(), String> {
    let Some(dir) = Path::new(program)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    let mut entries = vec![dir.to_path_buf()];
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        if !existing.is_empty() {
            entries.extend(std::env::split_paths(&existing));
        }
    }
    let value = std::env::join_paths(entries).map_err(|err| format!("LD_LIBRARY_PATH: {err}"))?;
    command.env("LD_LIBRARY_PATH", value);
    Ok(())
}

/// Запустити (або повернути вже запущений) локальний speech-server.
#[tauri::command]
fn start_local_server(
    app: tauri::AppHandle,
    state: State<'_, LocalServer>,
    port: Option<u16>,
) -> Result<ServerInfo, String> {
    let port = port.unwrap_or(DEFAULT_LOCAL_PORT);
    let base_url = format!("http://127.0.0.1:{port}");
    let mut slot = state.child.lock().map_err(|err| err.to_string())?;

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
            Ok(Some(_)) | Err(_) => *slot = None,
        }
    }

    let program = resolve_server_program();
    let models_dir = local_models_dir(&app)?;
    let _ = std::fs::create_dir_all(&models_dir);

    let mut command = Command::new(&program);
    set_sidecar_library_path(&program, &mut command)?;
    let child = command
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

/// Установити чи йде зараз синтез (викликається з frontend).
#[tauri::command]
fn set_synth_active(app: AppHandle, state: State<'_, SynthState>, active: bool) {
    let mut slot = state.active.lock().unwrap();
    *slot = active;
    let _ = set_window_title(&app, active);
}

/// Перемикає видимість головного вікна (з меню трею).
#[tauri::command]
fn toggle_main_window(app: AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let is_visible = win.is_visible().unwrap_or(true);
        if is_visible {
            let _ = win.hide();
        } else {
            let _ = win.show();
            let _ = win.set_focus();
        }
    }
}

/// Оновити заголовок головного вікна залежно від стану синтезу.
fn set_window_title<R: tauri::Runtime>(app: &AppHandle<R>, active: bool) -> Result<(), String> {
    let title = if active {
        "Promin TTS — синтезується…"
    } else {
        "Promin TTS"
    };
    if let Some(win) = app.get_webview_window("main") {
        win.set_title(title).map_err(|e| e.to_string())
    } else {
        Ok(())
    }
}

fn show_main_window<R: tauri::Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.unminimize();
        let _ = win.show();
        let _ = win.set_focus();
    }
    Ok(())
}

fn build_tray_menu<R: tauri::Runtime>(app: &AppHandle<R>) -> Result<Menu<R>, String> {
    let show_i = MenuItemBuilder::new("Розгорнути вікно")
        .id("show")
        .build(app)
        .map_err(|e| e.to_string())?;
    let hide_i = MenuItemBuilder::new("Приховати вікно")
        .id("hide")
        .build(app)
        .map_err(|e| e.to_string())?;
    let sep = tauri::menu::PredefinedMenuItem::separator(app).map_err(|e| e.to_string())?;
    let quit_i = MenuItemBuilder::new("Вийти")
        .id("quit")
        .build(app)
        .map_err(|e| e.to_string())?;
    MenuBuilder::new(app)
        .items(&[&show_i, &hide_i, &sep, &quit_i])
        .build()
        .map_err(|e| e.to_string())
}

fn build_tray<R: tauri::Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let menu = build_tray_menu(app)?;
    let app_clone = app.clone();
    let builder = TrayIconBuilder::<R>::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "show" => {
                let _ = show_main_window(app);
            }
            "hide" => {
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.hide();
                }
            }
            "quit" => {
                let menu_id = event.id();
                {
                    let mut ex = app_clone.state::<Exiting>().flag.lock().unwrap();
                    *ex = true;
                }
                match menu_id.as_ref() {
                    "quit" => app.exit(0),
                    _ => {}
                }
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: tauri::tray::MouseButton::Left,
                button_state: tauri::tray::MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.unminimize();
                    let _ = win.show();
                    let _ = win.set_focus();
                }
            }
        });
    builder.build(app).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(LocalServer {
            child: Mutex::new(None),
        })
        .manage(SynthState {
            active: Mutex::new(false),
        })
        .manage(Exiting {
            flag: Mutex::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            start_local_server,
            set_synth_active,
            toggle_main_window,
        ])
        .setup(|app| {
            let app_handle = app.handle().clone();
            // Системний трей.
            let _ = build_tray(&app_handle);

            // Закриття хрестиком — приховати в трей, а не вийти.
            if let Some(window) = app.get_webview_window("main") {
                let app_handle_clone = app_handle.clone();
                window.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        let is_exiting = *app_handle_clone.state::<Exiting>().flag.lock().unwrap();
                        if !is_exiting {
                            api.prevent_close();
                            let _ = app_handle_clone.get_webview_window("main").unwrap().hide();
                        }
                    }
                });
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Promin TTS desktop")
        .run(|app, event| {
            if let RunEvent::Exit = event {
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
    /// (джерело: `PROMIN_SERVER_BIN`, файл поруч із бінарником, PATH).
    #[test]
    fn resolve_server_program_is_never_empty() {
        let program = resolve_server_program();
        assert!(!program.trim().is_empty());
    }

    /// Каталог із `libonnxruntime.so.1` (поруч із сервером) має бути
    /// першим у `LD_LIBRARY_PATH`, решта наявних шляхів — збережені.
    #[test]
    fn sidecar_library_path_prepends_server_dir() {
        let mut command = Command::new("x");
        set_sidecar_library_path("/usr/bin/speech-server", &mut command).unwrap();
        let value = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("LD_LIBRARY_PATH"))
            .and_then(|(_, value)| value)
            .expect("LD_LIBRARY_PATH має бути виставлений");
        assert!(
            value.to_string_lossy().starts_with("/usr/bin"),
            "каталог сервера має йти першим: {}",
            value.to_string_lossy()
        );
    }

    /// Сервер, знайдений у `PATH` (каталог невідомий) — змінні не чіпаємо.
    #[test]
    fn sidecar_library_path_skips_path_lookup() {
        let mut command = Command::new("x");
        set_sidecar_library_path("speech-server", &mut command).unwrap();
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != std::ffi::OsStr::new("LD_LIBRARY_PATH")),
            "без відомого каталогу LD_LIBRARY_PATH не виставляється"
        );
    }
}
