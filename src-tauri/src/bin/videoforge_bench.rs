use std::env;
use std::path::Path;
use std::process::{self, Stdio};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use app_lib::commands::upscale::{run_upscale_job, JobProgress, JobProgressFn, UpscaleJobConfig};
#[cfg(feature = "native_engine")]
use app_lib::commands::native_engine::{
    native_result_summary_json, run_native_tool_request, NativeToolRunRequest,
};
use app_lib::control::ResearchConfig;
use app_lib::edit_config::EditConfig;
use app_lib::models;
use app_lib::python_env::resolve_python_environment;
use serde_json::json;
use tokio::sync::Mutex as TokioMutex;

#[derive(Debug, Clone)]
struct BenchArgs {
    input: String,
    output: String,
    model: String,
    scale: u32,
    precision: String,
    deterministic: bool,
    edit_config: EditConfig,
    dry_run: bool,
    #[allow(dead_code)]
    native: bool,
    #[allow(dead_code)]
    native_direct: bool,
    #[allow(dead_code)]
    onnx_model: Option<String>,
    #[allow(dead_code)]
    max_batch: Option<u32>,
    #[allow(dead_code)]
    preserve_audio: bool,
    #[allow(dead_code)]
    trt_cache: bool,
    #[allow(dead_code)]
    warmup_runs: u32,
}

#[derive(Default)]
struct ProgressState {
    started_at: Option<Instant>,
}

#[tokio::main]
async fn main() {
    init_tracing();

    let args = match parse_args(env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(CliExit::Help) => {
            print_usage();
            return;
        }
        Err(CliExit::Error(message)) => emit_error_and_exit(&message),
    };

    if let Err(message) = args.edit_config.validate().map_err(|e| e.to_string()) {
        emit_error_and_exit(&format!("Invalid --edit-config: {message}"));
    }

    if args.dry_run {
        if let Err(message) = run_dry_run_checks(&args) {
            emit_error_and_exit(&message);
        }
        emit_json(json!({ "event": "dry_run_ok" }));
        return;
    }

    if !Path::new(&args.input).exists() {
        emit_error_and_exit(&format!("Input file not found: {}", args.input));
    }

    let effective_precision = if args.deterministic {
        // Existing pipeline already supports a "deterministic" precision mode string.
        "deterministic".to_string()
    } else {
        args.precision.clone()
    };

    emit_json(json!({
        "event": "start",
        "input": args.input,
        "output": args.output,
        "model": args.model,
        "scale": args.scale,
        "precision": effective_precision,
        "mode": if args.native { "native" } else { "python" },
    }));

    let started = Instant::now();
    if args.native {
        run_native_bench(&args, &effective_precision, started).await;
    } else {
        let (python_bin, script_path) = match resolve_python_environment() {
            Ok(v) => v,
            Err(e) => emit_error_and_exit(&e.to_string()),
        };

        let state = Arc::new(StdMutex::new(ProgressState {
            started_at: Some(Instant::now()),
        }));
        let progress: JobProgressFn = {
            let state = Arc::clone(&state);
            Arc::new(move |p: JobProgress| {
                let elapsed_secs = state
                    .lock()
                    .ok()
                    .and_then(|s| s.started_at)
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                let fps = if elapsed_secs > 0.0 {
                    Some(p.frame as f64 / elapsed_secs)
                } else {
                    None
                };
                emit_json(json!({
                    "event": "progress",
                    "frame": p.frame,
                    "total_frames": parse_total_from_message(&p.message),
                    "fps": fps,
                    "pct": p.pct,
                    "message": p.message,
                    "eta_secs": p.eta_secs,
                    "output": p.output_path,
                }));
            }) as JobProgressFn
        };

        let job = UpscaleJobConfig {
            python_bin,
            script_path,
            input_path: args.input.clone(),
            output_path: args.output.clone(),
            model: args.model.clone(),
            scale: args.scale,
            precision: effective_precision,
            edit_config: args.edit_config.clone(),
            research_config: Arc::new(TokioMutex::new(ResearchConfig::default())),
            zenoh_timeout_secs: 60,
            enable_run_artifacts: false,
            use_shm_proto_v2: false,
            shm_ring_size_override: None,
        };

        match run_upscale_job(job, progress).await {
            Ok(report) => emit_json(json!({
                "event": "done",
                "output": report.output_path,
                "elapsed_ms": started.elapsed().as_millis(),
                "frames_encoded": report.frames_encoded,
                "mode": "python",
            })),
            Err(message) => emit_error_and_exit(&message),
        }
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("videoforge=info")),
        )
        .try_init();
}

fn run_dry_run_checks(args: &BenchArgs) -> Result<(), String> {
    check_command("ffmpeg")?;
    check_command("ffprobe")?;

    if args.native {
        #[cfg(not(feature = "native_engine"))]
        {
            return Err(
                "Native benchmarking requires the native_engine feature. Rebuild with `cargo build --features native_engine`.".to_string(),
            );
        }

        #[cfg(feature = "native_engine")]
        {
            let model_path = args
                .onnx_model
                .as_deref()
                .ok_or_else(|| "Native benchmarking requires --onnx-model <path>.".to_string())?;
            if !Path::new(model_path).exists() {
                return Err(format!("ONNX model not found: {model_path}"));
            }
            return Ok(());
        }
    }

    let _ = resolve_python_environment().map_err(|e| e.to_string())?;

    let available = models::list_models();
    if available.is_empty() {
        return Err("No model weights discovered. Install engine weights or place weights in a supported path.".to_string());
    }
    if !available.iter().any(|m| m.id == args.model) {
        let sample_keys = available
            .iter()
            .take(10)
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Model key '{}' not found. Available keys (sample): {}",
            args.model, sample_keys
        ));
    }

    Ok(())
}

#[cfg(feature = "native_engine")]
async fn run_native_bench(args: &BenchArgs, precision: &str, started: Instant) {
    let onnx_model = args
        .onnx_model
        .as_ref()
        .unwrap_or_else(|| emit_error_and_exit("Native benchmarking requires --onnx-model <path>."));

    let base_request = NativeToolRunRequest::new(
        args.input.clone(),
        onnx_model.to_string(),
        args.scale,
        precision.to_string(),
    )
    .with_preserve_audio(args.preserve_audio)
    .with_max_batch(args.max_batch)
    .with_native_direct(args.native_direct)
    .with_default_benchmark_trt_cache(args.trt_cache)
    .with_output_path(args.output.clone());

    let cache_dir = base_request.trt_cache_dir.clone();
    if let Some(cache_dir) = &cache_dir {
        if let Err(err) = std::fs::create_dir_all(cache_dir) {
            emit_error_and_exit(&format!(
                "Failed to create TensorRT cache directory {}: {err}",
                cache_dir.display()
            ));
        }
    }
    for warmup_idx in 0..args.warmup_runs {
        let warmup_output = base_request.warmup_output_path(warmup_idx + 1);
        emit_json(json!({
            "event": "warmup_start",
            "index": warmup_idx + 1,
            "output": warmup_output,
            "native_direct": args.native_direct,
            "trt_cache_enabled": args.trt_cache,
        }));
        let warmup_started = Instant::now();
        match run_native_tool_request(
            base_request.clone().with_output_path(warmup_output.clone()),
        )
        .await {
            Ok(report) => {
                let mut payload = native_result_summary_json(&report);
                payload.insert("event".to_string(), json!("warmup_done"));
                payload.insert("index".to_string(), json!(warmup_idx + 1));
                payload.insert(
                    "elapsed_ms".to_string(),
                    json!(warmup_started.elapsed().as_millis()),
                );
                emit_json(serde_json::Value::Object(payload));
                let _ = std::fs::remove_file(&warmup_output);
            }
            Err(message) => emit_error_and_exit(&message),
        }
    }

    match run_native_tool_request(base_request).await {
        Ok(report) => {
            let mut payload = native_result_summary_json(&report);
            payload.insert("event".to_string(), json!("done"));
            payload.insert("elapsed_ms".to_string(), json!(started.elapsed().as_millis()));
            payload.insert("mode".to_string(), json!("native"));
            payload.insert("native_direct".to_string(), json!(args.native_direct));
            payload.insert("requested_max_batch".to_string(), json!(args.max_batch));
            payload.insert("warmup_runs".to_string(), json!(args.warmup_runs));
            emit_json(serde_json::Value::Object(payload));
        }
        Err(message) => emit_error_and_exit(&message),
    }
}

#[cfg(not(feature = "native_engine"))]
async fn run_native_bench(_args: &BenchArgs, _precision: &str, _started: Instant) {
    emit_error_and_exit(
        "Native benchmarking requires the native_engine feature. Rebuild with `cargo build --features native_engine`.",
    );
}

fn check_command(cmd: &str) -> Result<(), String> {
    let status = std::process::Command::new(cmd)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to invoke {cmd}: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{cmd} returned non-zero status when running '-version'"
        ))
    }
}

fn parse_total_from_message(message: &str) -> Option<u64> {
    let token = message.split_whitespace().last()?;
    let (_, total) = token.split_once('/')?;
    total.parse::<u64>().ok()
}

fn emit_json(value: serde_json::Value) {
    println!("{value}");
}

fn emit_error_and_exit(message: &str) -> ! {
    emit_json(json!({
        "event": "error",
        "message": message,
    }));
    process::exit(1);
}

enum CliExit {
    Help,
    Error(String),
}

fn parse_args(args: Vec<String>) -> Result<BenchArgs, CliExit> {
    if args.is_empty() {
        return Err(CliExit::Help);
    }

    let mut input = None;
    let mut output = None;
    let mut model = None;
    let mut scale = None;
    let mut precision = None;
    let mut deterministic = false;
    let mut edit_config_json: Option<String> = None;
    let mut dry_run = false;
    let mut native = false;
    let mut native_direct = false;
    let mut onnx_model = None;
    let mut max_batch = None;
    let mut preserve_audio = false;
    let mut trt_cache = false;
    let mut warmup_runs = 0u32;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Err(CliExit::Help),
            "--deterministic" => {
                deterministic = true;
                i += 1;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            "--native" => {
                native = true;
                i += 1;
            }
            "--native-direct" => {
                native_direct = true;
                i += 1;
            }
            "--preserve-audio" => {
                preserve_audio = true;
                i += 1;
            }
            "--trt-cache" => {
                trt_cache = true;
                i += 1;
            }
            "--input" => {
                input = Some(next_value(&args, &mut i, "--input")?);
            }
            "--output" => {
                output = Some(next_value(&args, &mut i, "--output")?);
            }
            "--onnx-model" => {
                onnx_model = Some(next_value(&args, &mut i, "--onnx-model")?);
            }
            "--model" => {
                model = Some(next_value(&args, &mut i, "--model")?);
            }
            "--max-batch" => {
                let raw = next_value(&args, &mut i, "--max-batch")?;
                let parsed = raw.parse::<u32>().map_err(|_| {
                    CliExit::Error(format!("Invalid --max-batch '{}': expected u32", raw))
                })?;
                if parsed == 0 {
                    return Err(CliExit::Error("--max-batch must be >= 1".to_string()));
                }
                max_batch = Some(parsed);
            }
            "--warmup-runs" => {
                let raw = next_value(&args, &mut i, "--warmup-runs")?;
                warmup_runs = raw.parse::<u32>().map_err(|_| {
                    CliExit::Error(format!("Invalid --warmup-runs '{}': expected u32", raw))
                })?;
            }
            "--scale" => {
                let raw = next_value(&args, &mut i, "--scale")?;
                let parsed = raw.parse::<u32>().map_err(|_| {
                    CliExit::Error(format!("Invalid --scale '{}': expected u32", raw))
                })?;
                if parsed == 0 {
                    return Err(CliExit::Error("--scale must be >= 1".to_string()));
                }
                scale = Some(parsed);
            }
            "--precision" => {
                let raw = next_value(&args, &mut i, "--precision")?;
                match raw.as_str() {
                    "fp16" | "fp32" => precision = Some(raw),
                    _ => {
                        return Err(CliExit::Error(format!(
                            "Invalid --precision '{}'. Use fp16 or fp32.",
                            raw
                        )));
                    }
                }
            }
            "--edit-config" => {
                edit_config_json = Some(next_value(&args, &mut i, "--edit-config")?);
            }
            other => {
                return Err(CliExit::Error(format!("Unknown argument: {other}")));
            }
        }
    }

    let edit_config = match edit_config_json {
        Some(raw) => serde_json::from_str::<EditConfig>(&raw)
            .map_err(|e| CliExit::Error(format!("Failed to parse --edit-config JSON: {e}")))?,
        None => EditConfig::default(),
    };

    let input = required_arg(input, "--input")?;
    let output = required_arg(output, "--output")?;
    let scale = required_arg(scale, "--scale")?;
    let precision = required_arg(precision, "--precision")?;

    let model = if native {
        onnx_model
            .clone()
            .unwrap_or_else(|| "native".to_string())
    } else {
        required_arg(model, "--model")?
    };

    Ok(BenchArgs {
        input,
        output,
        model,
        scale,
        precision,
        deterministic,
        edit_config,
        dry_run,
        native,
        native_direct,
        onnx_model,
        max_batch,
        preserve_audio,
        trt_cache,
        warmup_runs,
    })
}

fn next_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, CliExit> {
    let next = *i + 1;
    if next >= args.len() {
        return Err(CliExit::Error(format!("Missing value for {flag}")));
    }
    *i += 2;
    Ok(args[next].clone())
}

fn required_arg<T>(value: Option<T>, flag: &str) -> Result<T, CliExit> {
    value.ok_or_else(|| CliExit::Error(format!("Missing required argument: {flag}")))
}

fn print_usage() {
    eprintln!(
        "Usage:\n  Python mode:\n    cargo run --bin videoforge_bench -- --input <path> --output <path> --model <key> --scale <u32> --precision <fp16|fp32> [--deterministic] [--edit-config <json>] [--dry-run]\n  Native mode:\n    cargo run --features native_engine --bin videoforge_bench -- --native --input <path> --output <path> --onnx-model <path> --scale <u32> --precision <fp16|fp32> [--max-batch <u32>] [--native-direct] [--preserve-audio] [--trt-cache] [--warmup-runs <u32>] [--dry-run]"
    );
}
