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
//! --shm-roundtrip      Run the SHM roundtrip check
//! --e2e-python         Run the full FFmpeg E2E pipeline check
//! --input <path>       Input file for E2E check
//! --output <path>      Output file for E2E check (optional, auto-generated)
//! --e2e-model <name>   Model name for E2E check (default: RCAN_x4)
//! --e2e-scale <n>      Scale factor for E2E check (default: 1)
//! --timeout-ms <N>     E2E job timeout in milliseconds (default: 600000)
//! --keep-temp          Keep E2E output file instead of deleting it
//! --native-direct      Force `VIDEOFORGE_NATIVE_ENGINE_DIRECT=1` for native E2E
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
use std::{env, path::PathBuf};

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

fn configure_repo_runtime_path() {
    let ffmpeg_exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    let ffprobe_exe = if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" };

    let root = match PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
    {
        Ok(p) => p,
        Err(_) => return,
    };

    let mut additions: Vec<PathBuf> = Vec::new();

    for dir in [
        root.join("third_party").join("ffmpeg").join("bin"),
        root.join("third_party").join("ffmpeg"),
    ] {
        if dir.join(ffmpeg_exe).exists() && dir.join(ffprobe_exe).exists() {
            additions.push(dir);
            break;
        }
    }

    let trt = root.join("third_party").join("tensorrt");
    if trt.join("nvinfer_10.dll").exists() {
        additions.push(trt);
    }

    if additions.is_empty() {
        return;
    }

    let current = env::var_os("PATH").unwrap_or_default();
    let mut paths: Vec<PathBuf> = env::split_paths(&current).collect();
    for dir in additions {
        if dir.is_dir() && !paths.iter().any(|p| p == &dir) {
            paths.insert(0, dir);
        }
    }
    if let Ok(joined) = env::join_paths(paths) {
        // SAFETY: process-local PATH setup for smoke prerequisites/runtime.
        unsafe { env::set_var("PATH", joined) };
    }
}

fn check_python_env() -> Option<(String, String)> {
    match resolve_python_environment() {
        Ok((bin, script)) => {
            check("Python environment", true, "");
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

fn runtime_truth_lines(
    runtime_snapshot: &app_lib::runtime_truth::RuntimeConfigSnapshot,
    observed_metrics: Option<&app_lib::runtime_truth::RunObservedMetrics>,
) -> Vec<String> {
    let mut lines = Vec::new();
    if let Ok(snapshot_json) = serde_json::to_string(runtime_snapshot) {
        lines.push(format!("  → runtime_snapshot: {snapshot_json}"));
    }
    if let Some(metrics) = observed_metrics {
        if let Ok(metrics_json) = serde_json::to_string(metrics) {
            lines.push(format!("  → observed_metrics: {metrics_json}"));
        }
    }
    lines
}

fn print_runtime_truth(
    runtime_snapshot: &app_lib::runtime_truth::RuntimeConfigSnapshot,
    observed_metrics: Option<&app_lib::runtime_truth::RunObservedMetrics>,
) {
    for line in runtime_truth_lines(runtime_snapshot, observed_metrics) {
        println!("{line}");
    }
}

#[derive(Debug, Clone)]
struct PythonSharedArgs {
    model: Option<String>,
    precision: String,
    timeout_secs: u64,
}

#[derive(Debug, Clone)]
struct ShmRoundtripMode {
    width: u32,
    height: u32,
    scale: u32,
    roundtrip_timeout_ms: u64,
}

#[derive(Debug, Clone)]
struct PythonE2eMode {
    input: Option<String>,
    output: Option<String>,
    model: String,
    scale: u32,
    timeout_ms: u64,
    keep_temp: bool,
}

#[derive(Debug, Clone)]
struct PythonSmokeMode {
    shared: PythonSharedArgs,
    shm_roundtrip: Option<ShmRoundtripMode>,
    e2e: Option<PythonE2eMode>,
}

impl PythonSmokeMode {
    fn enabled(&self) -> bool {
        self.shared.model.is_some() || self.shm_roundtrip.is_some() || self.e2e.is_some()
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeE2eMode {
    input: Option<String>,
    output: Option<String>,
    model_path: Option<String>,
    scale: u32,
    precision: String,
    native_direct: bool,
    keep_temp: bool,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeSmokeMode {
    e2e: Option<NativeE2eMode>,
}

#[derive(Debug, Clone)]
struct SmokeArgs {
    python: PythonSmokeMode,
    #[cfg(feature = "native_engine")]
    native: NativeSmokeMode,
    #[cfg(not(feature = "native_engine"))]
    native_e2e_requested: bool,
}

// ─── ffprobe helpers ─────────────────────────────────────────────────────────

/// Returns (width, height) of first video stream or None on failure.
fn ffprobe_dims(path: &str) -> Option<(usize, usize)> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let mut it = s.lines();
    Some((
        it.next()?.trim().parse().ok()?,
        it.next()?.trim().parse().ok()?,
    ))
}

/// Returns duration in seconds of the file or None on failure.
fn ffprobe_duration(path: &str) -> Option<f64> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
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

async fn shutdown_python(publisher: &zenoh::pubsub::Publisher<'_>, guard: &mut ProcessGuard) {
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

// ─── SHM Roundtrip ───────────────────────────────────────────────────────────

async fn check_shm_roundtrip(
    python_bin: &str,
    script_path: &str,
    mode: &ShmRoundtripMode,
    shared: &PythonSharedArgs,
) -> bool {
    use app_lib::shm::{VideoShm, RING_SIZE, SLOT_READY_FOR_AI};

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

    // ── Spawn Python (stderr visible so exceptions surface) ──────────────────
    let mut cmd = tokio::process::Command::new(python_bin);
    cmd.arg(script_path);
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--parent-pid").arg(std::process::id().to_string());
    cmd.arg("--precision").arg(&shared.precision);
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::inherit());
    #[cfg(target_os = "windows")]
    {
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

    // ── Handshake ────────────────────────────────────────────────────────────
    let handshake_ok = timeout(
        Duration::from_secs(shared.timeout_secs),
        subscriber.recv_async(),
    )
    .await
    .is_ok();
    if !check(
        "Python Zenoh handshake",
        handshake_ok,
        &format!("no message received within {}s", shared.timeout_secs),
    ) {
        shutdown_python(&publisher, &mut guard).await;
        return false;
    }

    // ── create_shm ───────────────────────────────────────────────────────────
    let job_id = app_lib::ipc::protocol::next_request_id();
    if let Err(e) = ipc::put_request(
        &publisher,
        RequestEnvelope::new(
            "create_shm",
            &job_id,
            json!({
                "width": mode.width,
                "height": mode.height,
                "scale": mode.scale,
                "ring_size": RING_SIZE
            }),
        ),
    )
    .await
    {
        check("SHM create send", false, &e.to_string());
        shutdown_python(&publisher, &mut guard).await;
        return false;
    }

    // Wait for SHM_CREATED
    let shm_path = match timeout(Duration::from_secs(10), subscriber.recv_async()).await {
        Err(_) => {
            check(
                "SHM created",
                false,
                "SHM_CREATE_TIMEOUT: no response within 10s",
            );
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
        Ok(Err(e)) => {
            check("SHM created", false, &e.to_string());
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
        Ok(Ok(msg)) => {
            let raw = String::from_utf8_lossy(&msg.payload().to_bytes()).to_string();
            let v: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    check("SHM created", false, &format!("JSON parse error: {}", e));
                    shutdown_python(&publisher, &mut guard).await;
                    return false;
                }
            };
            if v["status"].as_str() != Some("SHM_CREATED") {
                check(
                    "SHM created",
                    false,
                    &format!(
                        "SHM_CREATE_FAILED: status={}",
                        v["status"].as_str().unwrap_or("<missing>")
                    ),
                );
                shutdown_python(&publisher, &mut guard).await;
                return false;
            }
            match v["shm_path"].as_str() {
                Some(p) => p.to_string(),
                None => {
                    check(
                        "SHM created",
                        false,
                        "SHM_CREATE_FAILED: missing shm_path field",
                    );
                    shutdown_python(&publisher, &mut guard).await;
                    return false;
                }
            }
        }
    };
    check("SHM created", true, &format!("path: {}", shm_path));

    // ── Open SHM ─────────────────────────────────────────────────────────────
    let mut shm = match VideoShm::open(
        &shm_path,
        mode.width as usize,
        mode.height as usize,
        mode.scale as usize,
    ) {
        Ok(s) => s,
        Err(e) => {
            check(
                "SHM header validated",
                false,
                &format!("SHM_OPEN_FAILED: {}", e),
            );
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
    };
    check("SHM header validated", true, "");

    // ── Write synthetic frame ─────────────────────────────────────────────────
    shm.reset_all_slots();
    {
        let input = shm.input_slot_mut(0).expect("slot 0 in bounds");
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
    }
    shm.set_slot_frame_bytes(0, mode.width * mode.height * 3);
    shm.set_slot_write_index(0, 1);
    shm.set_slot_state(0, SLOT_READY_FOR_AI);
    check(
        "Synthetic frame written \u{2192} SLOT_READY_FOR_AI",
        true,
        "",
    );

    // ── process_one_frame ─────────────────────────────────────────────────────
    let job_id2 = app_lib::ipc::protocol::next_request_id();
    if let Err(e) = ipc::put_request(
        &publisher,
        RequestEnvelope::new("process_one_frame", &job_id2, json!({})),
    )
    .await
    {
        check("process_one_frame sent", false, &e.to_string());
        shutdown_python(&publisher, &mut guard).await;
        return false;
    }
    check("process_one_frame sent", true, "");

    // Wait for FRAME_DONE
    let frame_ok = match timeout(
        Duration::from_millis(mode.roundtrip_timeout_ms),
        subscriber.recv_async(),
    )
    .await
    {
        Err(_) => {
            check(
                "FRAME_DONE received",
                false,
                &format!(
                    "FRAME_DONE_TIMEOUT: no response within {}ms",
                    mode.roundtrip_timeout_ms
                ),
            );
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
        Ok(Err(e)) => {
            check("FRAME_DONE received", false, &e.to_string());
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
        Ok(Ok(msg)) => {
            let raw = String::from_utf8_lossy(&msg.payload().to_bytes()).to_string();
            let v: serde_json::Value =
                serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
            v["status"].as_str() == Some("FRAME_DONE")
        }
    };
    if !check(
        "FRAME_DONE received",
        frame_ok,
        "unexpected status in response",
    ) {
        shutdown_python(&publisher, &mut guard).await;
        return false;
    }

    // ── Validate output ───────────────────────────────────────────────────────
    let expected_len = (mode.width * mode.scale * mode.height * mode.scale * 3) as usize;
    let output_ok = match shm.output_slot(0) {
        Err(e) => {
            check(
                "Output validated",
                false,
                &format!("SHM_OPEN_FAILED: {}", e),
            );
            shutdown_python(&publisher, &mut guard).await;
            return false;
        }
        Ok(output) => {
            if output.len() != expected_len {
                check(
                    "Output validated",
                    false,
                    &format!(
                        "OUTPUT_SIZE_MISMATCH: expected {} bytes, got {}",
                        expected_len,
                        output.len()
                    ),
                );
                shutdown_python(&publisher, &mut guard).await;
                return false;
            }
            if output.iter().all(|&b| b == 0) {
                check(
                    "Output validated",
                    false,
                    "OUTPUT_ALL_ZEROS: output slot is all zeros",
                );
                shutdown_python(&publisher, &mut guard).await;
                return false;
            }
            true
        }
    };
    check(
        &format!("Output validated ({} bytes, non-zero)", expected_len),
        output_ok,
        "",
    );

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    shutdown_python(&publisher, &mut guard).await;
    true
}

// ─── Python E2E (FFmpeg path) ─────────────────────────────────────────────────

async fn check_e2e_python(
    python_bin: &str,
    script_path: &str,
    mode: &PythonE2eMode,
    shared: &PythonSharedArgs,
) -> bool {
    use app_lib::commands::upscale::{
        run_upscale_job, JobProgress, JobProgressFn, UpscaleJobConfig,
    };
    use app_lib::control::ResearchConfig;
    use app_lib::edit_config::EditConfig;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // A) Input prereq checks
    let input_path = mode.input.as_deref().unwrap_or("");
    if !std::path::Path::new(input_path).exists() {
        return check(
            "Input file exists",
            false,
            &format!("E2E_INPUT_NOT_FOUND: {}", input_path),
        );
    }
    check("Input file exists", true, "");

    let (in_w, in_h) = match ffprobe_dims(input_path) {
        Some(d) => {
            check(&format!("Probe input ({}×{})", d.0, d.1), true, "");
            d
        }
        None => return check("Probe input", false, "E2E_FFPROBE_PROBE: ffprobe failed"),
    };

    // B) Build config
    let research_config = Arc::new(Mutex::new(ResearchConfig::default()));
    let out = mode.output.clone().unwrap_or_default();
    let config = UpscaleJobConfig {
        python_bin: python_bin.to_string(),
        script_path: script_path.to_string(),
        input_path: input_path.to_string(),
        output_path: out,
        model: mode.model.clone(),
        scale: mode.scale,
        precision: shared.precision.clone(),
        edit_config: EditConfig::default(),
        research_config,
        zenoh_timeout_secs: (mode.timeout_ms / 1000).max(60),
        enable_run_artifacts: false,
        use_shm_proto_v2: false,
        shm_ring_size_override: None,
    };

    // C) Progress callback
    let last_pct = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let last_pct_clone = last_pct.clone();
    let progress: JobProgressFn = Arc::new(move |p: JobProgress| {
        let prev = last_pct_clone.swap(p.pct, std::sync::atomic::Ordering::Relaxed);
        if p.pct >= 100 || p.pct / 10 > prev / 10 {
            println!("  [progress] {}% frame={} {}", p.pct, p.frame, p.message);
        }
        if let Some(op) = &p.output_path {
            println!("  → output: {}", op);
        }
    });

    // D) Run job with timeout
    let result = tokio::time::timeout(
        Duration::from_millis(mode.timeout_ms),
        run_upscale_job(config, progress),
    )
    .await;

    let report = match result {
        Err(_) => {
            return check(
                "Job completed",
                false,
                &format!("E2E_TIMEOUT: exceeded {}ms", mode.timeout_ms),
            );
        }
        Ok(Err(e)) => return check("Job completed", false, &e),
        Ok(Ok(r)) => r,
    };
    check("Job completed", true, "");
    print_runtime_truth(&report.runtime_snapshot, report.observed_metrics.as_ref());
    let actual_out = &report.output_path;

    // E) Validate output
    let size = std::fs::metadata(actual_out).map(|m| m.len()).unwrap_or(0);
    if size < 4096 {
        return check(
            "Output file size",
            false,
            &format!("E2E_OUTPUT_MISSING: {} bytes at {}", size, actual_out),
        );
    }
    check(
        "Output file size",
        true,
        &format!("> 4 KB ({} bytes)", size),
    );

    let (out_w, out_h) = match ffprobe_dims(actual_out) {
        Some(d) => d,
        None => return check("Output dimensions", false, "E2E_FFPROBE_VALIDATE"),
    };
    let exp_w = in_w * mode.scale as usize;
    let exp_h = in_h * mode.scale as usize;
    if out_w != exp_w || out_h != exp_h {
        return check(
            "Output dimensions",
            false,
            &format!(
                "E2E_DIM_MISMATCH: expected {}×{}, got {}×{}",
                exp_w, exp_h, out_w, out_h
            ),
        );
    }
    check(
        &format!("Output dimensions ({}×{})", out_w, out_h),
        true,
        "",
    );

    let dur = ffprobe_duration(actual_out).unwrap_or(0.0);
    if dur <= 0.0 {
        return check("Output duration", false, "E2E_ZERO_DURATION");
    }
    check(&format!("Output duration ({:.2}s > 0)", dur), true, "");

    // F) Cleanup
    if !mode.keep_temp {
        let _ = std::fs::remove_file(actual_out);
    } else {
        println!("  → kept: {}", actual_out);
    }

    true
}

// ─── Native Engine E2E ────────────────────────────────────────────────────────

#[cfg(feature = "native_engine")]
async fn check_e2e_native(
    mode: &NativeE2eMode,
) -> bool {
    use app_lib::commands::native_engine::{
        native_smoke_success_lines, native_tool_run_banner, run_native_tool_request,
        NativeToolRunRequest,
    };

    let input_path = mode.input.as_deref().unwrap_or("");
    let model_path = mode.model_path.as_deref().unwrap_or("");

    // A) Input prereq checks
    if !std::path::Path::new(input_path).exists() {
        return check(
            "Input file exists",
            false,
            &format!("E2E_INPUT_NOT_FOUND: {}", input_path),
        );
    }
    check("Input file exists", true, "");

    let (in_w, in_h) = match ffprobe_dims(input_path) {
        Some(d) => {
            check(&format!("Probe input ({}×{})", d.0, d.1), true, "");
            d
        }
        None => return check("Probe input", false, "E2E_FFPROBE_PROBE: ffprobe failed"),
    };

    if !std::path::Path::new(model_path).exists() {
        return check(
            "Model file exists",
            false,
            &format!("E2E_MODEL_NOT_FOUND: {}", model_path),
        );
    }
    check("Model file exists", true, model_path);

    let request = NativeToolRunRequest::new(
        input_path.to_string(),
        model_path.to_string(),
        mode.scale,
        mode.precision.clone(),
    )
    .with_optional_output_path(mode.output.clone())
    .with_max_batch(Some(1))
    .with_native_direct(mode.native_direct);
    println!("{}", native_tool_run_banner(&request));
    let result = run_native_tool_request(request).await;

    let report = match result {
        Ok(r) => r,
        Err(e) => {
            // Error is a JSON string
            return check("Native pipeline completed", false, &e);
        }
    };

    let summary_lines = native_smoke_success_lines(&report);
    check(
        "Native pipeline completed",
        true,
        summary_lines.first().map(String::as_str).unwrap_or("native result"),
    );
    for line in summary_lines.iter().skip(1) {
        println!("  → {line}");
    }
    if let Some(snapshot) = &report.runtime_snapshot {
        print_runtime_truth(snapshot, report.observed_metrics.as_ref());
    }
    let actual_out = &report.output_path;

    // C) Validate output
    let size = std::fs::metadata(actual_out).map(|m| m.len()).unwrap_or(0);
    if size < 4096 {
        return check(
            "Output file size",
            false,
            &format!("E2E_OUTPUT_MISSING: {} bytes at {}", size, actual_out),
        );
    }
    check(
        "Output file size",
        true,
        &format!("> 4 KB ({} bytes)", size),
    );

    let (out_w, out_h) = match ffprobe_dims(actual_out) {
        Some(d) => d,
        None => return check("Output dimensions", false, "E2E_FFPROBE_VALIDATE"),
    };
    let exp_w = in_w * mode.scale as usize;
    let exp_h = in_h * mode.scale as usize;
    if out_w != exp_w || out_h != exp_h {
        return check(
            "Output dimensions",
            false,
            &format!(
                "E2E_DIM_MISMATCH: expected {}×{}, got {}×{}",
                exp_w, exp_h, out_w, out_h
            ),
        );
    }
    check(
        &format!("Output dimensions ({}×{})", out_w, out_h),
        true,
        "",
    );

    let dur = ffprobe_duration(actual_out).unwrap_or(0.0);
    if dur <= 0.0 {
        return check("Output duration", false, "E2E_ZERO_DURATION");
    }
    check(&format!("Output duration ({:.2}s > 0)", dur), true, "");

    // D) Cleanup
    if !mode.keep_temp {
        let _ = std::fs::remove_file(actual_out);
    } else {
        println!("  → kept: {}", actual_out);
    }

    true
}

// ─── Argument parsing ────────────────────────────────────────────────────────

fn parse_args_from<I>(args: I) -> SmokeArgs
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter().peekable();
    let mut model = None;
    let mut precision = "fp32".to_string();
    let mut timeout_secs = 60u64;
    let mut shm_roundtrip = false;
    let mut shm_width = 8u32;
    let mut shm_height = 8u32;
    let mut shm_scale = 1u32;
    let mut roundtrip_timeout_ms = 5000u64;
    let mut e2e_python = false;
    let mut e2e_input: Option<String> = None;
    let mut e2e_output: Option<String> = None;
    let mut e2e_model = "RCAN_x4".to_string();
    let mut e2e_scale = 1u32;
    let mut e2e_timeout_ms = 600_000u64;
    let mut keep_temp = false;
    let mut native_e2e_enabled = false;
    #[cfg(feature = "native_engine")]
    let mut native_direct = false;
    #[cfg(feature = "native_engine")]
    let mut e2e_onnx: Option<String> = None;

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
            "--shm-roundtrip" => {
                shm_roundtrip = true;
            }
            "--width" => {
                if let Some(w) = args.next() {
                    shm_width = w.parse().unwrap_or(8);
                }
            }
            "--height" => {
                if let Some(h) = args.next() {
                    shm_height = h.parse().unwrap_or(8);
                }
            }
            "--scale" => {
                if let Some(s) = args.next() {
                    shm_scale = s.parse().unwrap_or(1);
                }
            }
            "--roundtrip-timeout-ms" => {
                if let Some(t) = args.next() {
                    roundtrip_timeout_ms = t.parse().unwrap_or(5000);
                }
            }
            "--e2e-python" => {
                e2e_python = true;
            }
            "--input" => {
                e2e_input = args.next();
            }
            "--output" => {
                e2e_output = args.next();
            }
            "--e2e-model" => {
                if let Some(m) = args.next() {
                    e2e_model = m;
                }
            }
            "--e2e-scale" => {
                if let Some(s) = args.next() {
                    e2e_scale = s.parse().unwrap_or(1);
                }
            }
            "--timeout-ms" => {
                if let Some(t) = args.next() {
                    e2e_timeout_ms = t.parse().unwrap_or(600_000);
                }
            }
            "--keep-temp" => {
                keep_temp = true;
            }
            "--e2e-native" => {
                native_e2e_enabled = true;
            }
            "--native-direct" => {
                #[cfg(feature = "native_engine")]
                {
                    native_direct = true;
                }
            }
            "--e2e-onnx" => {
                #[cfg(feature = "native_engine")]
                {
                    e2e_onnx = args.next();
                }
            }
            _ => {}
        }
    }

    let python = PythonSmokeMode {
        shared: PythonSharedArgs {
            model,
            precision: precision.clone(),
            timeout_secs,
        },
        shm_roundtrip: shm_roundtrip.then_some(ShmRoundtripMode {
            width: shm_width,
            height: shm_height,
            scale: shm_scale,
            roundtrip_timeout_ms,
        }),
        e2e: e2e_python.then_some(PythonE2eMode {
            input: e2e_input.clone(),
            output: e2e_output.clone(),
            model: e2e_model,
            scale: e2e_scale,
            timeout_ms: e2e_timeout_ms,
            keep_temp,
        }),
    };

    SmokeArgs {
        python,
        #[cfg(feature = "native_engine")]
        native: NativeSmokeMode {
            e2e: native_e2e_enabled.then_some(NativeE2eMode {
                input: e2e_input,
                output: e2e_output,
                model_path: e2e_onnx,
                scale: e2e_scale,
                precision,
                native_direct,
                keep_temp,
            }),
        },
        #[cfg(not(feature = "native_engine"))]
        native_e2e_requested: native_e2e_enabled,
    }
}

fn parse_args() -> SmokeArgs {
    parse_args_from(std::env::args().skip(1))
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
        )
        .try_init();

    let args = parse_args();
    configure_repo_runtime_path();

    println!();
    println!("=== VideoForge Smoke Test ===");
    println!();

    let mut all_passed = true;

    // 1. Prerequisites
    println!("── Prerequisites ─────────────────────────────────────────────");
    all_passed &= check_ffmpeg();
    all_passed &= check_ffprobe();

    // 2. Python environment
    // Only verify Python if we're running Python tests or NOT running native-only
    let run_python_tests = args.python.enabled();
    #[cfg(feature = "native_engine")]
    let native_only = args.native.e2e.is_some() && !run_python_tests;
    #[cfg(not(feature = "native_engine"))]
    let native_only = args.native_e2e_requested && !run_python_tests;

    let mut python_env = None;
    if !native_only {
        println!();
        println!("── Python Environment ────────────────────────────────────────");
        python_env = check_python_env();
        all_passed &= python_env.is_some();
    }

    // 3. Python IPC
    if let Some((python_bin, script_path)) = python_env {
        if run_python_tests {
            println!();
            println!("── Python IPC Handshake ──────────────────────────────────────");
            let ipc_ok = check_python_ipc(
                &python_bin,
                &script_path,
                args.python.shared.model.as_deref(),
                &args.python.shared.precision,
                args.python.shared.timeout_secs,
            )
            .await;
            all_passed &= ipc_ok;

            if let Some(shm_mode) = &args.python.shm_roundtrip {
                println!();
                println!("── SHM Roundtrip ─────────────────────────────────────────────");
                let ok = check_shm_roundtrip(
                    &python_bin,
                    &script_path,
                    shm_mode,
                    &args.python.shared,
                )
                .await;
                all_passed &= ok;
            }

            if let Some(python_e2e) = &args.python.e2e {
                println!();
                println!("── Python E2E (FFmpeg path) ──────────────────────────────────");
                let ok = check_e2e_python(
                    &python_bin,
                    &script_path,
                    python_e2e,
                    &args.python.shared,
                )
                .await;
                all_passed &= ok;
            }
        }
    }

    // 4. Native engine status
    println!();
    println!("── Native Engine ─────────────────────────────────────────────");
    #[cfg(feature = "native_engine")]
    {
        println!("[INFO] native_engine feature: ENABLED");
        if let Some(native_e2e) = &args.native.e2e {
            println!();
            println!("── Native E2E (engine-v2) ────────────────────────────────────");
            let input = native_e2e.input.as_deref().unwrap_or("");
            let model = native_e2e.model_path.as_deref().unwrap_or("");
            if input.is_empty() || model.is_empty() {
                eprintln!("[FAIL] --e2e-native requires --input and --e2e-onnx");
                all_passed = false;
            } else {
                let ok = check_e2e_native(native_e2e).await;
                all_passed &= ok;
            }
        }
    }
    #[cfg(not(feature = "native_engine"))]
    {
        println!("[SKIP] native_engine feature: BLOCKED / DISABLED.");
        if args.native_e2e_requested {
            eprintln!("[FAIL] --e2e-native requested but feature is disabled.");
            all_passed = false;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::{parse_args_from, runtime_truth_lines};
    use app_lib::runtime_truth::{
        RunObservedMetrics, RunStatus, RuntimeConfigSnapshot, RuntimeEngineFamily,
    };

    #[test]
    fn runtime_truth_lines_embed_runtime_owned_snapshot_and_metrics() {
        let snapshot = RuntimeConfigSnapshot::new(
            "run-smoke-1",
            "python_sidecar",
            RuntimeEngineFamily::Python,
            "in.mp4",
            "out.mp4",
        );
        let metrics = RunObservedMetrics::new(
            "run-smoke-1",
            "python_sidecar",
            RunStatus::Succeeded,
        );

        let lines = runtime_truth_lines(&snapshot, Some(&metrics));

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"run_id\":\"run-smoke-1\""));
        assert!(lines[0].contains("\"route_id\":\"python_sidecar\""));
        assert!(lines[1].contains("\"status\":\"succeeded\""));
    }

    #[test]
    fn parse_args_routes_python_modes_to_nested_structs() {
        let args = parse_args_from([
            "--model".to_string(),
            "RCAN_x4".to_string(),
            "--precision".to_string(),
            "fp16".to_string(),
            "--timeout".to_string(),
            "42".to_string(),
            "--shm-roundtrip".to_string(),
            "--width".to_string(),
            "16".to_string(),
            "--height".to_string(),
            "12".to_string(),
            "--scale".to_string(),
            "2".to_string(),
            "--roundtrip-timeout-ms".to_string(),
            "1234".to_string(),
            "--e2e-python".to_string(),
            "--input".to_string(),
            "input.mp4".to_string(),
            "--output".to_string(),
            "output.mp4".to_string(),
            "--e2e-model".to_string(),
            "SPAN_x2".to_string(),
            "--e2e-scale".to_string(),
            "3".to_string(),
            "--timeout-ms".to_string(),
            "9000".to_string(),
            "--keep-temp".to_string(),
        ]);

        assert_eq!(args.python.shared.model.as_deref(), Some("RCAN_x4"));
        assert_eq!(args.python.shared.precision, "fp16");
        assert_eq!(args.python.shared.timeout_secs, 42);

        let shm = args.python.shm_roundtrip.expect("shm roundtrip mode");
        assert_eq!(shm.width, 16);
        assert_eq!(shm.height, 12);
        assert_eq!(shm.scale, 2);
        assert_eq!(shm.roundtrip_timeout_ms, 1234);

        let e2e = args.python.e2e.expect("python e2e mode");
        assert_eq!(e2e.input.as_deref(), Some("input.mp4"));
        assert_eq!(e2e.output.as_deref(), Some("output.mp4"));
        assert_eq!(e2e.model, "SPAN_x2");
        assert_eq!(e2e.scale, 3);
        assert_eq!(e2e.timeout_ms, 9000);
        assert!(e2e.keep_temp);
    }

    #[cfg(feature = "native_engine")]
    #[test]
    fn parse_args_routes_native_modes_to_nested_structs() {
        let args = parse_args_from([
            "--e2e-native".to_string(),
            "--input".to_string(),
            "clip.mp4".to_string(),
            "--output".to_string(),
            "native-out.mp4".to_string(),
            "--e2e-scale".to_string(),
            "4".to_string(),
            "--precision".to_string(),
            "deterministic".to_string(),
            "--native-direct".to_string(),
            "--e2e-onnx".to_string(),
            "model.onnx".to_string(),
            "--keep-temp".to_string(),
        ]);

        let native = args.native.e2e.expect("native e2e mode");
        assert_eq!(native.input.as_deref(), Some("clip.mp4"));
        assert_eq!(native.output.as_deref(), Some("native-out.mp4"));
        assert_eq!(native.model_path.as_deref(), Some("model.onnx"));
        assert_eq!(native.scale, 4);
        assert_eq!(native.precision, "deterministic");
        assert!(native.native_direct);
        assert!(native.keep_temp);
    }
}
