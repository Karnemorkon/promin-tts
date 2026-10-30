//! Гарантує, що `web/dist` існує до компіляції `rust-embed`.
//!
//! Якщо фронтенд ще не зібрано (`npm run build`), створюємо placeholder,
//! щоб `cargo build` не падав на свіжому клоні. Docker і CI збирають
//! справжній фронтенд перед `cargo build`.

use std::path::Path;

const PLACEHOLDER: &str = r#"<!doctype html>
<html lang="uk">
<head><meta charset="utf-8"><title>Promin TTS</title></head>
<body>
<h1>Promin TTS</h1>
<p>Веб-інтерфейс не зібрано. Виконайте <code>npm install &amp;&amp; npm run build</code> у <code>crates/speech-server/web</code>.</p>
</body>
</html>
"#;

fn main() {
    let dist = Path::new("web/dist");
    let index = dist.join("index.html");

    if !index.exists() {
        let _ = std::fs::create_dir_all(dist);
        let _ = std::fs::write(&index, PLACEHOLDER);
    }

    println!("cargo:rerun-if-changed=web/dist");
    println!("cargo:rerun-if-changed=build.rs");
}
