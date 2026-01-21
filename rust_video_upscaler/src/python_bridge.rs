use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine};
use serde_json::json;

pub struct PythonBridge {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
}

impl PythonBridge {
    pub fn spawn(python: &str, script: &str) -> Result<Self> {
        let mut child = Command::new(python)
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().ok_or(anyhow!("stdin missing"))?;
        let stdout = child.stdout.take().ok_or(anyhow!("stdout missing"))?;

        Ok(Self {
            child,
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
        })
    }

    pub fn upscale_frame(&self, job_id: &str, png_bytes: &[u8]) -> Result<Vec<u8>> {
        let b64 = general_purpose::STANDARD.encode(png_bytes);

        let req = json!({
            "command": "upscale_frame",
            "id": job_id,
            "params": { "image_base64": b64 }
        });

        {
            let mut stdin = self.stdin.lock().unwrap();
            stdin.write_all(req.to_string().as_bytes())?;
            stdin.write_all(b"\n")?;
            stdin.flush()?;
        }

        let mut line = String::new();
        let mut stdout = self.stdout.lock().unwrap();
        stdout.read_line(&mut line)?;

        let msg: serde_json::Value = serde_json::from_str(&line)?;
        let out_b64 = msg["result"]["image_base64"]
            .as_str()
            .ok_or(anyhow!("missing image_base64"))?;

        let decoded = general_purpose::STANDARD.decode(out_b64)?;
        Ok(decoded)
    }

    pub fn cancel(&self, job_id: &str) -> Result<()> {
        let req = json!({
            "command": "cancel",
            "params": { "id": job_id }
        });

        let mut stdin = self.stdin.lock().unwrap();
        stdin.write_all(req.to_string().as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }
}

impl Drop for PythonBridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
