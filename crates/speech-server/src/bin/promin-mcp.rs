//! stdio-проксі MCP: новий рядок stdin -> `POST $PROMIN_MCP_URL` -> stdout.
//!
//! Призначений для клієнтів (Claude Desktop, Cursor), які запускають MCP
//! сервер як дочірній процес. Сам синтез виконує `speech-server`.

use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url =
        std::env::var("PROMIN_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8010/mcp".to_string());
    let api_key = std::env::var("PROMIN_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    let timeout = std::env::var("PROMIN_MCP_TIMEOUT")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(600);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .build()?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let id = serde_json::from_str::<Value>(trimmed)
            .ok()
            .and_then(|value| value.get("id").cloned())
            .filter(|value| !value.is_null());

        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .body(trimmed.to_string());
        if let Some(key) = &api_key {
            request = request.bearer_auth(key);
        }

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if !body.trim().is_empty() {
                    stdout.write_all(body.trim().as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                } else if !status.is_success() {
                    if let Some(id) = id {
                        write_error(&mut stdout, id, &format!("HTTP {status}")).await?;
                    }
                }
            }
            Err(error) => {
                if let Some(id) = id {
                    write_error(&mut stdout, id, &format!("proxy error: {error}")).await?;
                } else {
                    eprintln!("promin-mcp: {error}");
                }
            }
        }
    }

    Ok(())
}

async fn write_error<W>(stdout: &mut W, id: Value, message: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32603, "message": message },
    });
    stdout.write_all(payload.to_string().as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}
