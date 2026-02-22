//! AI upscaling command — orchestrates Python sidecar, SHM ring buffer,
//! FFmpeg decode/encode, and Zenoh IPC.

use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;
use serde_json::json;
use tauri::{AppHandle, Emitter};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration, Instant};
use zenoh::Config;

use crate::control::ResearchConfig;
use crate::edit_config::{build_ffmpeg_filters, calculate_output_dimensions, EditConfig};
use crate::ipc::{self, protocol::RequestEnvelope};
use crate::python_env::{
    build_worker_argv, get_free_port, resolve_python_environment, BaseWorkerArgs, ProcessGuard,
    WorkerCaps, PYTHON_PIDS,
};
use crate::run_manifest::{maybe_write_run_manifest, RunManifestInputs};
use crate::video_pipeline;
use crate::{
    commands::export::{get_smart_output_path, is_image_file},
    shm,
};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration for a single upscale job (no Tauri types).
pub struct UpscaleJobConfig {
    pub python_bin: String,
    pub script_path: String,
    pub input_path: String,
    /// Empty string → auto-generate via get_smart_output_path.
    pub output_path: String,
    pub model: String,
    pub scale: u32,
    pub precision: String,
    pub edit_config: EditConfig,
    /// Shared research params — polled live every 500 ms by the poll task.
    pub research_config: Arc<Mutex<ResearchConfig>>,
    /// Seconds to wait for Python handshake (default 60).
    pub zenoh_timeout_secs: u64,
    /// Internal opt-in only. Default false keeps behavior/output unchanged.
    pub enable_run_artifacts: bool,
}

pub struct UpscaleJobReport {
    pub output_path: String,
    pub frames_encoded: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTimingsMs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

/// Single progress tick emitted during the job.
#[derive(Debug, Clone, Serialize)]
pub struct JobProgress {
    pub pct: u32,
    pub frame: u64,
    pub message: String,
    pub output_path: Option<String>,
    pub eta_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_decoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_encoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_ms: Option<StageTimingsMs>,
}

pub type JobProgressFn = Arc<dyn Fn(JobProgress) + Send + Sync + 'static>;

fn progress_to_event_payload(p: &JobProgress) -> serde_json::Value {
    let mut j = json!({
        "jobId": "active",
        "progress": p.pct,
        "message": p.message,
        "eta": p.eta_secs
    });
    if let Some(op) = &p.output_path {
        j["outputPath"] = json!(op);
    }
    if let Some(v) = p.frames_decoded {
        j["frames_decoded"] = json!(v);
    }
    if let Some(v) = p.frames_processed {
        j["frames_processed"] = json!(v);
    }
    if let Some(v) = p.frames_encoded {
        j["frames_encoded"] = json!(v);
    }
    if let Some(stage) = &p.stage_ms {
        j["stage_ms"] = serde_json::to_value(stage).unwrap_or_default();
    }
    j
}

fn extract_handshake_protocol_version(raw: &str) -> Option<u32> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    if let Some(v) = value
        .get("protocol_version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
    {
        return Some(v);
    }
    value
        .get("payload")
        .and_then(|p| p.get("protocol_version"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
}

fn validate_worker_protocol_version(found: Option<u32>, typed_ipc: bool) -> Result<(), String> {
    let expected = crate::ipc::PROTOCOL_VERSION;
    match found {
        None => {
            tracing::warn!("Worker protocol_version missing (older worker?)");
            Ok(())
        }
        Some(v) if v == expected => Ok(()),
        Some(v) => {
            if typed_ipc {
                Err(format!(
                    "IPC_PROTOCOL_VERSION_MISMATCH: expected={}, found={}. Update or reinstall the Python worker to match host protocol.",
                    expected, v
                ))
            } else {
                tracing::warn!(
                    expected,
                    found = v,
                    "Worker protocol version mismatch; continuing because typed IPC is disabled"
                );
                Ok(())
            }
        }
    }
}

// ─── run_upscale_job ─────────────────────────────────────────────────────────

pub async fn run_upscale_job(
    config: UpscaleJobConfig,
    progress: JobProgressFn,
) -> Result<UpscaleJobReport, String> {
    let precision = match config.precision.as_str() {
        "fp32" | "fp16" | "deterministic" => config.precision.clone(),
        _ => {
            return Err(format!(
                "Invalid precision mode '{}'. Use fp32, fp16, or deterministic.",
                config.precision
            ))
        }
    };

    let is_img = is_image_file(&config.input_path);
    let output_path = if config.output_path.trim().is_empty() {
        get_smart_output_path(&config.input_path, !is_img)
    } else {
        config.output_path.clone()
    };
    let worker_caps = WorkerCaps::default();

    if let Some(manifest_path) = maybe_write_run_manifest(
        config.enable_run_artifacts,
        &RunManifestInputs {
            input_path: &config.input_path,
            output_path: &output_path,
            scale: config.scale,
            precision: &precision,
            model_key: Some(&config.model),
            worker_caps: &worker_caps,
            ipc_protocol_version: Some(crate::ipc::PROTOCOL_VERSION),
            shm_protocol_version: None,
            app_version: Some(env!("CARGO_PKG_VERSION")),
        },
    )
    .map_err(|e| format!("Failed to write run manifest: {e}"))?
    {
        tracing::info!(path = %manifest_path.display(), "Run manifest written");
    }

    // Generate a session-scoped job ID for IPC correlation.
    let job_id = ipc::protocol::next_request_id();

    tracing::info!(
        job_id = %job_id,
        input = %config.input_path,
        output = %output_path,
        model = %config.model,
        precision = %precision,
        is_image = is_img,
        "Upscale request started"
    );

    // ── Zenoh setup ──────────────────────────────────────────────────────────
    let port = get_free_port();
    let ipc_endpoint = format!("tcp/127.0.0.1:{}", port);
    let zenoh_prefix = format!("videoforge/ipc/{}", port);

    let mut zenoh_cfg = Config::default();
    zenoh_cfg
        .insert_json5("listen/endpoints", &format!("[\"{}\"]", ipc_endpoint))
        .map_err(|e| e.to_string())?;
    let session = zenoh::open(zenoh_cfg)
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    let publisher = session
        .declare_publisher(format!("{}/req", zenoh_prefix))
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;
    let subscriber = session
        .declare_subscriber(format!("{}/res", zenoh_prefix))
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

    // ── Spawn Python sidecar ─────────────────────────────────────────────────
    tracing::info!(
        job_id = %job_id,
        python = %config.python_bin,
        script = %config.script_path,
        "Spawning Python worker"
    );

    let mut cmd = tokio::process::Command::new(&config.python_bin);
    let worker_argv = build_worker_argv(
        &BaseWorkerArgs {
            script_path: &config.script_path,
            port,
            parent_pid: std::process::id(),
            precision: &precision,
        },
        &worker_caps,
    );
    cmd.args(&worker_argv);
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x08000000);
    }

    let python_child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn Python worker: {}", e))?;

    if let Some(pid) = python_child.id() {
        PYTHON_PIDS.lock().unwrap().insert(pid);
        tracing::info!(job_id = %job_id, pid, "Python worker spawned");
    }

    let mut python_guard = ProcessGuard::new(python_child);

    // Wait for Python startup handshake (first message = ready signal).
    let handshake_msg = timeout(
        Duration::from_secs(config.zenoh_timeout_secs),
        subscriber.recv_async(),
    )
    .await
    .map_err(|_| {
        format!(
            "Python worker handshake timeout ({}s)",
            config.zenoh_timeout_secs
        )
    })?
    .map_err(|e: zenoh::Error| e.to_string())?;

    let handshake_data = String::from_utf8(handshake_msg.payload().to_bytes().to_vec())
        .map_err(|e| e.to_string())?;
    validate_worker_protocol_version(
        extract_handshake_protocol_version(&handshake_data),
        worker_caps.use_typed_ipc,
    )?;

    tracing::info!(job_id = %job_id, "Python worker handshake received");

    // ── Load model ───────────────────────────────────────────────────────────
    ipc::put_request(
        &publisher,
        RequestEnvelope::new("load_model", &job_id, json!({"model_name": config.model})),
    )
    .await
    .map_err(|e: zenoh::Error| e.to_string())?;

    let load_msg = timeout(Duration::from_secs(30), subscriber.recv_async())
        .await
        .map_err(|_| "Model load timeout (30s)")?
        .map_err(|e: zenoh::Error| e.to_string())?;
    let load_data =
        String::from_utf8(load_msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;

    if !load_data.contains("MODEL_LOADED") {
        tracing::error!(job_id = %job_id, response = %load_data, "Model load failed");
        return Err(format!("Failed to load model: {}", load_data));
    }
    tracing::info!(job_id = %job_id, model = %config.model, "Model loaded");

    // ── Push initial research config ─────────────────────────────────────────
    {
        let rc = config.research_config.lock().await;
        let params_req = RequestEnvelope::new(
            "update_research_params",
            &job_id,
            json!({"params": serde_json::to_value(&*rc).unwrap_or_default()}),
        );
        let _ = ipc::put_request(&publisher, params_req).await;
        let _ = timeout(Duration::from_secs(2), subscriber.recv_async()).await;
    }

    // ── Image pipeline ───────────────────────────────────────────────────────
    if is_img {
        let research_params = {
            let guard = config.research_config.lock().await;
            serde_json::to_value(&*guard).unwrap_or_default()
        };

        ipc::put_request(
            &publisher,
            RequestEnvelope::new(
                "upscale_image_file",
                &job_id,
                json!({
                    "id": "single_shot",
                    "params": {
                        "input_path": config.input_path,
                        "output_path": output_path,
                        "config": config.edit_config
                    },
                    "research_params": research_params
                }),
            ),
        )
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;

        loop {
            let msg = timeout(Duration::from_secs(300), subscriber.recv_async())
                .await
                .map_err(|_| "Image upscale timeout (5 min)")?
                .map_err(|e: zenoh::Error| e.to_string())?;

            let payload_str =
                String::from_utf8(msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;
            let resp: serde_json::Value =
                serde_json::from_str(&payload_str).map_err(|e| e.to_string())?;

            if resp["status"] == "progress" {
                let current = resp["current"].as_u64().unwrap_or(0);
                let total = resp["total"].as_u64().unwrap_or(1);
                let pct = (current as f64 / total as f64) * 100.0;

                tracing::debug!(
                    job_id = %job_id,
                    tile = current,
                    total_tiles = total,
                    "Image upscale progress"
                );

                progress(JobProgress {
                    pct: pct as u32,
                    frame: current,
                    message: format!("Processing Tile {}/{}", current, total),
                    output_path: None,
                    eta_secs: 0,
                    frames_decoded: None,
                    frames_processed: Some(current),
                    frames_encoded: None,
                    stage_ms: None,
                });
            } else if resp["status"] == "ok" {
                tracing::info!(job_id = %job_id, output = %output_path, "Image upscale complete");
                break;
            } else if resp["status"] == "error" {
                let msg = resp["error"]["message"]
                    .as_str()
                    .or_else(|| resp["message"].as_str())
                    .unwrap_or("(no message)");
                tracing::error!(job_id = %job_id, error = %msg, "Python error during image upscale");
                return Err(format!("Python error: {}", msg));
            }
        }

        let _ = ipc::put_request(
            &publisher,
            RequestEnvelope::new("shutdown", &job_id, json!({})),
        )
        .await;

        if let Some(mut child) = python_guard.disarm() {
            if let Some(pid) = child.id() {
                PYTHON_PIDS.lock().unwrap().remove(&pid);
            }
            if timeout(Duration::from_secs(10), child.wait())
                .await
                .is_err()
            {
                tracing::warn!(job_id = %job_id, "Image worker did not exit in time, killing");
                let _ = child.start_kill();
            }
        }

        return Ok(UpscaleJobReport {
            output_path,
            frames_encoded: 0,
        });
    }

    // ── Video pipeline ───────────────────────────────────────────────────────
    let probe_res = video_pipeline::probe_video(&config.input_path).map_err(|e| e.to_string())?;
    let (input_w, input_h, duration, fps, _total_frames) = probe_res;

    let (process_w, process_h) = calculate_output_dimensions(&config.edit_config, input_w, input_h);
    let start_time = config.edit_config.trim_start;
    let end_time = if config.edit_config.trim_end > 0.0 {
        config.edit_config.trim_end
    } else {
        duration
    };
    let process_duration = (end_time - start_time).max(0.1);
    let process_frames = (process_duration * fps).round() as u64;
    let scale_factor = config.scale as usize;

    tracing::info!(
        job_id = %job_id,
        width = process_w,
        height = process_h,
        fps,
        frames = process_frames,
        scale = scale_factor,
        "Video pipeline starting"
    );

    // Request Python to create the SHM ring buffer.
    ipc::put_request(
        &publisher,
        RequestEnvelope::new(
            "create_shm",
            &job_id,
            json!({
                "width": process_w,
                "height": process_h,
                "scale": scale_factor,
                "ring_size": shm::RING_SIZE
            }),
        ),
    )
    .await
    .map_err(|e: zenoh::Error| e.to_string())?;

    let shm_msg = timeout(Duration::from_secs(10), subscriber.recv_async())
        .await
        .map_err(|_| "SHM creation timeout (10s)")?
        .map_err(|e: zenoh::Error| e.to_string())?;
    let shm_data =
        String::from_utf8(shm_msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;
    let shm_resp: serde_json::Value = serde_json::from_str(&shm_data).map_err(|e| e.to_string())?;

    if shm_resp["status"] != "SHM_CREATED" {
        let msg = shm_resp["error"]["message"]
            .as_str()
            .or_else(|| shm_resp["message"].as_str())
            .unwrap_or("(no message)");
        return Err(format!("SHM init failed: {}", msg));
    }
    let shm_path = shm_resp
        .get("shm_path")
        .and_then(|s| s.as_str())
        .unwrap()
        .to_string();

    let shm = shm::VideoShm::open(&shm_path, process_w, process_h, scale_factor)
        .map_err(|e| e.to_string())?;
    shm.reset_all_slots();

    tracing::info!(
        job_id = %job_id,
        shm_path = %shm_path,
        "SHM ring buffer opened and reset"
    );

    let shared_shm = Mutex::new(shm);
    let frames_decoded_ctr = Arc::new(AtomicU64::new(0));
    let frames_processed_ctr = Arc::new(AtomicU64::new(0));
    let frames_encoded_ctr = Arc::new(AtomicU64::new(0));

    // Flow-control channels (Rust-only, not cross-process).
    let (free_tx, mut free_rx) = mpsc::channel::<usize>(shm::RING_SIZE);
    let (pending_tx, mut pending_rx) = mpsc::channel::<usize>(shm::RING_SIZE);
    let (enc_tx, mut enc_rx) = mpsc::channel::<usize>(shm::RING_SIZE);

    for i in 0..shm::RING_SIZE {
        free_tx.send(i).await.map_err(|e| e.to_string())?;
    }

    // Start Python frame polling loop.
    let research_config_ref = config.research_config.clone();
    {
        let guard = research_config_ref.lock().await;
        ipc::put_request(
            &publisher,
            RequestEnvelope::new(
                "start_frame_loop",
                &job_id,
                json!({"research_params": serde_json::to_value(&*guard).unwrap_or_default()}),
            ),
        )
        .await
        .map_err(|e: zenoh::Error| e.to_string())?;
    }

    let loop_msg = timeout(Duration::from_secs(10), subscriber.recv_async())
        .await
        .map_err(|_| "Frame loop start timeout (10s)")?
        .map_err(|e: zenoh::Error| e.to_string())?;
    let loop_data =
        String::from_utf8(loop_msg.payload().to_bytes().to_vec()).map_err(|e| e.to_string())?;
    if !loop_data.contains("FRAME_LOOP_STARTED") {
        return Err(format!("Frame loop start failed: {}", loop_data));
    }
    tracing::info!(job_id = %job_id, "Python frame loop started (SHM atomic polling)");

    let filters = build_ffmpeg_filters(&config.edit_config, input_w, input_h);

    // ── Decoder task ─────────────────────────────────────────────────────────
    let decoder_task = async {
        let frames_decoded_ctr = Arc::clone(&frames_decoded_ctr);
        let use_nvdec = video_pipeline::probe_nvdec();
        let mut decoder = video_pipeline::VideoDecoder::new(
            &config.input_path,
            start_time,
            process_duration,
            &filters,
            use_nvdec,
        )
        .await
        .map_err(|e| e.to_string())?;

        if decoder.using_hwaccel {
            tracing::info!(job_id = %job_id, "Decoder using NVDEC hardware acceleration");
        }

        let mut frame_id: u32 = 0;
        while let Some(slot_idx) = free_rx.recv().await {
            let mut shm_guard = shared_shm.lock().await;
            shm_guard.set_slot_state(slot_idx, shm::SLOT_RUST_WRITING);

            let frame_size = {
                let input_slot = match shm_guard.input_slot_mut(slot_idx) {
                    Ok(slot) => slot,
                    Err(e) => {
                        tracing::error!(job_id = %job_id, slot = slot_idx, error = %e, "SHM input slot error");
                        break;
                    }
                };
                let size = input_slot.len() as u32;
                let got_frame = decoder
                    .read_raw_frame_into(input_slot)
                    .await
                    .unwrap_or(false);
                if got_frame {
                    Some(size)
                } else {
                    None
                }
            };

            if let Some(size) = frame_size {
                frame_id += 1;
                frames_decoded_ctr.fetch_add(1, Ordering::Relaxed);
                shm_guard.set_slot_write_index(slot_idx, frame_id);
                shm_guard.set_slot_frame_bytes(slot_idx, size);
                shm_guard.set_slot_state(slot_idx, shm::SLOT_READY_FOR_AI);
                drop(shm_guard);

                tracing::debug!(
                    job_id = %job_id,
                    frame_id,
                    slot_index = slot_idx,
                    bytes = size,
                    "Frame written to SHM → READY_FOR_AI"
                );

                if pending_tx.send(slot_idx).await.is_err() {
                    break;
                }
            } else {
                shm_guard.set_slot_state(slot_idx, shm::SLOT_EMPTY);
                tracing::info!(job_id = %job_id, total_frames = frame_id, "Decoder EOS");
                break;
            }
        }
        Ok::<(), String>(())
    };

    // ── Poll task ────────────────────────────────────────────────────────────
    let poll_task = async {
        let frames_processed_ctr = Arc::clone(&frames_processed_ctr);
        let mut last_params_push = Instant::now();

        while let Some(slot_idx) = pending_rx.recv().await {
            // Periodically push research params via Zenoh control plane.
            if last_params_push.elapsed() > Duration::from_millis(500) {
                let mut guard = research_config_ref.lock().await;
                let has_reset = guard.reset_temporal;
                let val = serde_json::to_value(&*guard).unwrap_or_default();
                if has_reset {
                    guard.reset_temporal = false;
                }
                drop(guard);

                let _ = ipc::put_request(
                    &publisher,
                    RequestEnvelope::new("update_research_params", &job_id, json!({"params": val})),
                )
                .await;
                last_params_push = Instant::now();
            }

            // Poll SHM slot state until Python signals READY_FOR_ENCODE.
            let poll_start = Instant::now();
            loop {
                {
                    let shm_guard = shared_shm.lock().await;
                    if shm_guard.slot_state(slot_idx) == shm::SLOT_READY_FOR_ENCODE {
                        frames_processed_ctr.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
                if poll_start.elapsed() > Duration::from_secs(30) {
                    let shm_guard = shared_shm.lock().await;
                    shm_guard.set_slot_state(slot_idx, shm::SLOT_EMPTY);
                    tracing::error!(
                        job_id = %job_id,
                        slot_index = slot_idx,
                        "AI inference timeout (30s) — aborting"
                    );
                    return Err("AI processing timeout (30s)".to_string());
                }
                tokio::time::sleep(Duration::from_micros(200)).await;
            }

            tracing::debug!(
                job_id = %job_id,
                slot_index = slot_idx,
                "Slot READY_FOR_ENCODE — forwarding to encoder"
            );

            if enc_tx.send(slot_idx).await.is_err() {
                break;
            }
        }
        Ok(())
    };

    // ── Encoder task ─────────────────────────────────────────────────────────
    let target_fps = config.edit_config.fps;
    let encoder_task = async {
        let frames_decoded_ctr = Arc::clone(&frames_decoded_ctr);
        let frames_processed_ctr = Arc::clone(&frames_processed_ctr);
        let frames_encoded_ctr = Arc::clone(&frames_encoded_ctr);
        let mut encoder = video_pipeline::VideoEncoder::new_with_audio(
            &output_path,
            fps as u32,
            target_fps,
            process_w * scale_factor,
            process_h * scale_factor,
            Some(&config.input_path),
            start_time,
            process_duration,
            precision == "deterministic",
        )
        .await
        .map_err(|e| e.to_string())?;

        let mut processed_count = 0u64;
        let eta_start = Instant::now();
        let mut last_emit = Instant::now();

        while let Some(slot_idx) = enc_rx.recv().await {
            let shm_guard = shared_shm.lock().await;
            shm_guard.set_slot_state(slot_idx, shm::SLOT_ENCODING);

            let output_slot = match shm_guard.output_slot(slot_idx) {
                Ok(slot) => slot,
                Err(e) => {
                    tracing::error!(job_id = %job_id, slot = slot_idx, error = %e, "SHM output slot error");
                    break;
                }
            };

            if let Err(e) = encoder.write_raw_frame(output_slot).await {
                tracing::error!(job_id = %job_id, slot = slot_idx, error = %e, "Encoder write error");
                break;
            }

            shm_guard.set_slot_state(slot_idx, shm::SLOT_EMPTY);
            drop(shm_guard);
            let _ = free_tx.send(slot_idx).await;

            processed_count += 1;
            frames_encoded_ctr.store(processed_count, Ordering::Relaxed);
            let is_last_frame = processed_count >= process_frames;

            if last_emit.elapsed() > Duration::from_millis(100) || is_last_frame {
                let pct =
                    ((processed_count as f64 / process_frames as f64) * 100.0).min(100.0) as u32;
                let elapsed = eta_start.elapsed().as_secs_f64();
                let fps_proc = processed_count as f64 / elapsed;
                let eta = if fps_proc > 0.0 {
                    (process_frames.saturating_sub(processed_count) as f64 / fps_proc) as u64
                } else {
                    0
                };

                tracing::debug!(
                    job_id = %job_id,
                    frame = processed_count,
                    total = process_frames,
                    pct,
                    "Encode progress"
                );

                progress(JobProgress {
                    pct,
                    frame: processed_count,
                    message: format!("Processing Frame {}/{}", processed_count, process_frames),
                    output_path: None,
                    eta_secs: eta,
                    frames_decoded: Some(frames_decoded_ctr.load(Ordering::Relaxed)),
                    frames_processed: Some(frames_processed_ctr.load(Ordering::Relaxed)),
                    frames_encoded: Some(frames_encoded_ctr.load(Ordering::Relaxed)),
                    stage_ms: Some(StageTimingsMs {
                        decode: None,
                        ai: None,
                        encode: None,
                        total: Some(eta_start.elapsed().as_millis() as u64),
                    }),
                });
                last_emit = Instant::now();
            }

            if is_last_frame {
                break;
            }
        }

        encoder.finish().await.map_err(|e| e.to_string())?;
        tracing::info!(
            job_id = %job_id,
            frames_encoded = processed_count,
            output = %output_path,
            "Encode complete"
        );
        progress(JobProgress {
            pct: 100,
            frame: processed_count,
            message: "Finalizing...".to_string(),
            output_path: Some(output_path.clone()),
            eta_secs: 0,
            frames_decoded: Some(frames_decoded_ctr.load(Ordering::Relaxed)),
            frames_processed: Some(frames_processed_ctr.load(Ordering::Relaxed)),
            frames_encoded: Some(frames_encoded_ctr.load(Ordering::Relaxed)),
            stage_ms: Some(StageTimingsMs {
                decode: None,
                ai: None,
                encode: None,
                total: Some(eta_start.elapsed().as_millis() as u64),
            }),
        });
        Ok::<u64, String>(processed_count)
    };

    let (_, _, frames_encoded) =
        tokio::try_join!(decoder_task, poll_task, encoder_task).map_err(|e| {
            tracing::error!(job_id = %job_id, error = %e, "Pipeline task failed");
            e
        })?;

    // ── Graceful shutdown ────────────────────────────────────────────────────
    let _ = ipc::put_request(
        &publisher,
        RequestEnvelope::new("stop_frame_loop", &job_id, json!({})),
    )
    .await;
    let _ = ipc::put_request(
        &publisher,
        RequestEnvelope::new("shutdown", &job_id, json!({})),
    )
    .await;

    if let Some(mut child) = python_guard.disarm() {
        if let Some(pid) = child.id() {
            PYTHON_PIDS.lock().unwrap().remove(&pid);
        }
        if timeout(Duration::from_secs(3), child.wait()).await.is_err() {
            tracing::warn!(job_id = %job_id, "Worker did not exit gracefully, killing");
            let _ = child.start_kill();

            #[cfg(target_os = "windows")]
            if let Some(pid) = child.id() {
                use std::os::windows::process::CommandExt;
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000)
                    .output();
            }
        }
    }

    tracing::info!(job_id = %job_id, output = %output_path, "Upscale request complete");
    Ok(UpscaleJobReport {
        output_path,
        frames_encoded,
    })
}

#[cfg(test)]
mod tests {
    use super::{validate_worker_protocol_version, JobProgress, StageTimingsMs};

    #[test]
    fn test_handshake_version_match_ok() {
        assert!(
            validate_worker_protocol_version(Some(crate::ipc::PROTOCOL_VERSION), false).is_ok()
        );
        assert!(validate_worker_protocol_version(Some(crate::ipc::PROTOCOL_VERSION), true).is_ok());
    }

    #[test]
    fn test_handshake_missing_version_warn_only_ok() {
        assert!(validate_worker_protocol_version(None, false).is_ok());
        assert!(validate_worker_protocol_version(None, true).is_ok());
    }

    #[test]
    fn test_handshake_mismatch_warn_only_ok_when_typed_ipc_off() {
        assert!(
            validate_worker_protocol_version(Some(crate::ipc::PROTOCOL_VERSION + 1), false).is_ok()
        );
    }

    #[test]
    fn test_handshake_mismatch_fails_when_typed_ipc_on() {
        let err = validate_worker_protocol_version(Some(crate::ipc::PROTOCOL_VERSION + 1), true)
            .unwrap_err();
        assert!(err.contains("IPC_PROTOCOL_VERSION_MISMATCH"));
        assert!(err.contains("expected="));
        assert!(err.contains("found="));
    }

    #[test]
    fn test_job_progress_serialization_includes_perf_fields_when_present() {
        let p = JobProgress {
            pct: 50,
            frame: 10,
            message: "x".to_string(),
            output_path: None,
            eta_secs: 1,
            frames_decoded: Some(10),
            frames_processed: Some(9),
            frames_encoded: Some(8),
            stage_ms: Some(StageTimingsMs {
                decode: Some(1),
                ai: Some(2),
                encode: Some(3),
                total: Some(6),
            }),
        };
        let v = serde_json::to_value(&p).expect("serialize job progress");
        assert_eq!(v["frames_decoded"], 10);
        assert_eq!(v["frames_processed"], 9);
        assert_eq!(v["frames_encoded"], 8);
        assert_eq!(v["stage_ms"]["decode"], 1);
        assert_eq!(v["stage_ms"]["ai"], 2);
        assert_eq!(v["stage_ms"]["encode"], 3);
        assert_eq!(v["stage_ms"]["total"], 6);
    }

    #[test]
    fn test_job_progress_serialization_omits_perf_fields_when_absent() {
        let p = JobProgress {
            pct: 50,
            frame: 10,
            message: "x".to_string(),
            output_path: None,
            eta_secs: 1,
            frames_decoded: None,
            frames_processed: None,
            frames_encoded: None,
            stage_ms: None,
        };
        let v = serde_json::to_value(&p).expect("serialize job progress");
        assert!(v.get("frames_decoded").is_none());
        assert!(v.get("frames_processed").is_none());
        assert!(v.get("frames_encoded").is_none());
        assert!(v.get("stage_ms").is_none());
    }
}

// ─── upscale_request (thin Tauri wrapper) ────────────────────────────────────

#[tauri::command]
#[allow(clippy::too_many_arguments)] // TODO(clippy): Tauri IPC command signature is intentionally flat.
pub async fn upscale_request(
    app: AppHandle,
    research_state: tauri::State<'_, Arc<Mutex<ResearchConfig>>>,
    input_path: String,
    output_path: String,
    model: String,
    edit_config: EditConfig,
    scale: u32,
    #[allow(unused_variables)] precision: Option<String>,
) -> Result<String, String> {
    if !Path::new(&input_path).exists() {
        return Err(format!("Input file not found: {}", input_path));
    }

    let (python_bin, script_path) = resolve_python_environment().map_err(|e| e.to_string())?;

    let app_clone = app.clone();
    let progress: JobProgressFn = Arc::new(move |p: JobProgress| {
        let j = progress_to_event_payload(&p);
        let _ = app_clone.emit("upscale-progress", j);
    });

    let job_config = UpscaleJobConfig {
        python_bin,
        script_path,
        input_path,
        output_path,
        model,
        scale,
        precision: precision.unwrap_or_else(|| "fp32".to_string()),
        edit_config,
        research_config: research_state.inner().clone(),
        zenoh_timeout_secs: 60,
        enable_run_artifacts: false,
    };

    let report = run_upscale_job(job_config, progress).await?;
    Ok(report.output_path)
}
