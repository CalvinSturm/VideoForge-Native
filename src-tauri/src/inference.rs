use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{ChildStdin, ChildStdout, Command},
    sync::watch,
    time::{timeout, Duration},
};

/* ───────────────────────────────────────────── */

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub command: String,
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Response {
    pub id: String,
    pub status: String,
    pub result: Option<Value>,
    pub error: Option<Value>,
    pub event: Option<String>,
}

/* ───────────────────────────────────────────── */

pub struct InferenceBridge {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl InferenceBridge {
    /// Spawn the Python RealESRGAN worker and wait for READY event
    pub async fn spawn(python_path: &str, script_path: &str) -> Result<Self> {
        let mut child = Command::new(python_path)
            .arg(script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .context("Failed to spawn Python worker")?;

        let stdin = child.stdin.take().context("stdin missing")?;
        let stdout = child.stdout.take().context("stdout missing")?;
        let mut reader = BufReader::new(stdout);

        // Wait for "READY" line from worker
        let mut line = String::new();
        timeout(Duration::from_secs(10), async {
            loop {
                line.clear();
                reader.read_line(&mut line).await?;
                if line.contains("READY") {
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("Python worker READY timeout")??;

        Ok(Self { stdin, stdout: reader })
    }

    /// Send a request to the Python worker, optionally handle events, and support cancellation
    pub async fn send<F>(
        &mut self,
        req: Request,
        mut on_event: Option<F>,
        mut cancel_rx: watch::Receiver<bool>,
    ) -> Result<Response>
    where
        F: FnMut(Response) + Send,
    {
        // Serialize and send request
        let msg = serde_json::to_string(&req)?;
        self.stdin.write_all(msg.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        let mut line = String::new();

        loop {
            tokio::select! {
                _ = cancel_rx.changed() => {
                    if *cancel_rx.borrow() {
                        anyhow::bail!("Cancelled");
                    }
                }
                read_res = self.stdout.read_line(&mut line) => {
                    let bytes_read = read_res?;
                    if bytes_read == 0 {
                        anyhow::bail!("Python worker exited unexpectedly");
                    }

                    let resp: Response = match serde_json::from_str(line.trim()) {
                        Ok(r) => r,
                        Err(_) => {
                            line.clear();
                            continue;
                        }
                    };

                    // Invoke callback for any events
                    if let Some(cb) = &mut on_event {
                        cb(resp.clone());
                    }

                    // Return if this is the final response for this request
                    if resp.id == req.id && (resp.status == "ok" || resp.status == "error") {
                        return Ok(resp);
                    }

                    line.clear();
                }
            }
        }
    }
}