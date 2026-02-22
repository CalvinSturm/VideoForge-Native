use std::env;
use std::path::Path;
use std::process::{self, Stdio};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use app_lib::commands::upscale::{run_upscale_job, JobProgress, JobProgressFn, UpscaleJobConfig};
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
    }));

    let started = Instant::now();
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
    };

    match run_upscale_job(job, progress).await {
        Ok(report) => emit_json(json!({
            "event": "done",
            "output": report.output_path,
            "elapsed_ms": started.elapsed().as_millis(),
            "frames_encoded": report.frames_encoded,
        })),
        Err(message) => emit_error_and_exit(&message),
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
            "--input" => {
                input = Some(next_value(&args, &mut i, "--input")?);
            }
            "--output" => {
                output = Some(next_value(&args, &mut i, "--output")?);
            }
            "--model" => {
                model = Some(next_value(&args, &mut i, "--model")?);
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

    Ok(BenchArgs {
        input: required_arg(input, "--input")?,
        output: required_arg(output, "--output")?,
        model: required_arg(model, "--model")?,
        scale: required_arg(scale, "--scale")?,
        precision: required_arg(precision, "--precision")?,
        deterministic,
        edit_config,
        dry_run,
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
        "Usage:\n  cargo run --bin videoforge_bench -- --input <path> --output <path> --model <key> --scale <u32> --precision <fp16|fp32> [--deterministic] [--edit-config <json>] [--dry-run]"
    );
}
