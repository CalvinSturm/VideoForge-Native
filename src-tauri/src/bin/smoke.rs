//! VideoForge smoke-test binary.
//!
//! Validates prerequisites and the Python IPC handshake end-to-end,
//! without requiring the Tauri UI.
//!
//! # Usage
//!
//! ```text
//! cargo run --manifest-path src-tauri/Cargo.toml --bin smoke -- [OPTIONS]
//! ```
//!
//! # Options
//!
//! ```text
//! --model <name>       Model name to request (e.g. RCAN_x4).
//!                      If omitted, the model-load check is skipped.
//! --precision <mode>   fp32 | fp16 | deterministic  (default: fp32)
//! --timeout <secs>     Zenoh handshake timeout in seconds  (default: 60)
//! ```
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | All checks passed |
//! | 1    | One or more checks failed |

use std::process::{self, Stdio};
use std::time::Duration;

use app_lib::ipc::{self, protocol::RequestEnvelope};
use app_lib::python_env::{get_free_port, resolve_python_environment, ProcessGuard, PYTHON_PIDS};
use serde_json::json;
use tokio::time::timeout;
use zenoh::Config as ZenohConfig;

// ─── check helpers ───────────────────────────────────────────────────────────

fn check(label: &str, passed: bool, detail: &str) -> bool {
    let icon = if passed { "PASS" } else { "FAIL" };
    if passed {
        println!("[{icon}] {label}");
    } else {
        eprintln!("[{icon}] {label}: {detail}");
    }
    passed
}

fn check_ffmpeg() -> bool {
    let out = std::process::Command::new("ffmpeg")
        .args(["-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    check(
        "FFmpeg in PATH",
        out.map(|s| s.success()).unwrap_or(false),
        "ffmpeg not found — install FFmpeg and ensure it is in PATH",
    )
}

fn check_ffprobe() -> bool {
    let out = std::process::Command::new("ffprobe")
        .args(["-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    check(
        "FFprobe in PATH",
        out.map(|s| s.success()).unwrap_or(false),
        "ffprobe not found — install FFmpeg (includes ffprobe) and ensure it is in PATH",
    )
}

fn check_python_env() -> Option<(String, String)> {
    match resolve_python_environment() {
        Ok((bin, script)) => {
            check(
                "Python environment",
                true,
                "",
            );
            println!("       python = {bin}");
            println!("       script = {script}");
            Some((bin, script))
        }
        Err(e) => {
            check("Python environment", false, &e.to_string());
            None
        }
    }
}

// ─── Zenoh handshake + optional model load ───────────────────────────────────

async fn check_python_ipc(
    python_bin: &str,
    script_path: &str,
    model: Option<&str>,
    precision: &str,
    handshake_timeout_secs: u64,
) -> bool {
    let port = get_free_port();
    let ipc_endpoint = format!("tcp/127.0.0.1:{}", port);
    let zenoh_prefix = format!("videoforge/ipc/{}", port);

    // ── Open Zenoh listener ──────────────────────────────────────────────────
    let mut config = ZenohConfig::default();
    if config
        .insert_json5("listen/endpoints", &format!("[\"{}\"]", ipc_endpoint))
        .is_err()
    {
        return check("Zenoh listener", false, "failed to configure endpoint");
    }
    let session = match zenoh::open(config).await {
        Ok(s) => s,
        Err(e) => return check("Zenoh listener", false, &e.to_string()),
    };
    let publisher = match session
        .declare_publisher(format!("{}/req", zenoh_prefix))
        .await
    {
        Ok(p) => p,
        Err(e) => return check("Zenoh publisher", false, &e.to_string()),
    };
    let subscriber = match session
        .declare_subscriber(format!("{}/res", zenoh_prefix))
        .await
    {
        Ok(s) => s,
        Err(e) => return check("Zenoh subscriber", false, &e.to_string()),
    };
    check("Zenoh listener", true, "");

    // ── Spawn Python ─────────────────────────────────────────────────────────
    let mut cmd = tokio::process::Command::new(python_bin);
    cmd.arg(script_path);
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--parent-pid").arg(std::process::id().to_string());
    cmd.arg("--precision").arg(precision);
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let python_child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return check("Python spawn", false, &e.to_string()),
    };
    if let Some(pid) = python_child.id() {
        PYTHON_PIDS.lock().unwrap().insert(pid);
    }
    let mut guard = ProcessGuard::new(python_child);
    check("Python spawn", true, "");

    // ── Handshake — wait for first message (MODEL_LOADED or error from default load) ──
    let handshake_ok = timeout(
        Duration::from_secs(handshake_timeout_secs),
        subscriber.recv_async(),
    )
    .await
    .is_ok();
    if !check(
        "Python Zenoh handshake",
        handshake_ok,
        &format!("no message received within {}s", handshake_timeout_secs),
    ) {
        shutdown_python(&publisher, &mut guard).await;
        return false;
    }

    // ── Optional model load ───────────────────────────────────────────────────
    if let Some(model_name) = model {
        let job_id = app_lib::ipc::protocol::next_request_id();
        if let Err(e) = ipc::put_request(
            &publisher,
            RequestEnvelope::new("load_model", &job_id, json!({"model_name": model_name})),
        )
        .await
        {
            check("Model load send", false, &e.to_string());
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }

        // Read response (up to 30 s)
        let load_result = timeout(Duration::from_secs(30), subscriber.recv_async()).await;
        let passed = match load_result {
            Err(_) => false,
            Ok(Err(e)) => {
                check("Model load response", false, &e.to_string());
                shutdown_python(&publisher, &mut guard).await;
                return false;
            }
            Ok(Ok(msg)) => {
                let payload = String::from_utf8_lossy(&msg.payload().to_bytes()).to_string();
                payload.contains("MODEL_LOADED")
            }
        };
        if !check(
            &format!("Model load ({model_name})"),
            passed,
            "response did not contain MODEL_LOADED — check model name and weights path",
        ) {
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    shutdown_python(&publisher, &mut guard).await;
    true
}

async fn shutdown_python(
    publisher: &zenoh::pubsub::Publisher<'_>,
    guard: &mut ProcessGuard,
) {
    let job_id = app_lib::ipc::protocol::next_request_id();
    let _ = ipc::put_request(
        publisher,
        RequestEnvelope::new("shutdown", &job_id, json!({})),
    )
    .await;
    if let Some(mut child) = guard.disarm() {
        if let Some(pid) = child.id() {
            PYTHON_PIDS.lock().unwrap().remove(&pid);
        }
        let _ = timeout(Duration::from_secs(5), child.wait()).await;
    }
}

// ─── Argument parsing ────────────────────────────────────────────────────────

struct Args {
    model: Option<String>,
    precision: String,
    timeout_secs: u64,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1).peekable();
    let mut model = None;
    let mut precision = "fp32".to_string();
    let mut timeout_secs = 60u64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = args.next();
            }
            "--precision" => {
                if let Some(p) = args.next() {
                    precision = p;
                }
            }
            "--timeout" => {
                if let Some(t) = args.next() {
                    timeout_secs = t.parse().unwrap_or(60);
                }
            }
            _ => {}
        }
    }
    Args { model, precision, timeout_secs }
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("error")
        .try_init();

    let args = parse_args();

    println!();
    println!("=== VideoForge Smoke Test ===");
    println!();

    let mut all_passed = true;

    // 1. Prerequisites
    println!("── Prerequisites ─────────────────────────────────────────────");
    all_passed &= check_ffmpeg();
    all_passed &= check_ffprobe();

    // 2. Python environment
    println!();
    println!("── Python Environment ────────────────────────────────────────");
    let python_env = check_python_env();
    all_passed &= python_env.is_some();

    // 3. Python IPC (only if env resolved)
    if let Some((python_bin, script_path)) = python_env {
        println!();
        println!("── Python IPC Handshake ──────────────────────────────────────");
        let ipc_ok = check_python_ipc(
            &python_bin,
            &script_path,
            args.model.as_deref(),
            &args.precision,
            args.timeout_secs,
        )
        .await;
        all_passed &= ipc_ok;
    }

    // 4. Native engine status
    println!();
    println!("── Native Engine ─────────────────────────────────────────────");
    #[cfg(feature = "native_engine")]
    println!("[INFO] native_engine feature: ENABLED");
    #[cfg(not(feature = "native_engine"))]
    println!(
        "[SKIP] native_engine feature: BLOCKED (ort ^2.0 not on crates.io as stable release). \
         See docs/NATIVE_ENGINE_MVP.md for resolution steps."
    );

    // 5. Summary
    println!();
    println!("─────────────────────────────────────────────────────────────");
    if all_passed {
        println!("Result: ALL CHECKS PASSED");
        println!();
        process::exit(0);
    } else {
        eprintln!("Result: ONE OR MORE CHECKS FAILED");
        eprintln!();
        process::exit(1);
    }
}
