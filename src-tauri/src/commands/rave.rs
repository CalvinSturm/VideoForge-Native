use std::path::PathBuf;

use crate::commands::native_engine::{native_engine_runtime_enabled, workspace_root};
use crate::rave_cli::{
    run_benchmark, run_upscale, run_validate, RaveCliConfig, RaveCliError, RaveResult,
};

fn default_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "production_strict"
    }
}

fn env_profile_override() -> Result<Option<String>, String> {
    match std::env::var("VIDEOFORGE_RAVE_PROFILE") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "dev" | "production_strict" => Ok(Some(normalized)),
                _ => Err(format!(
                    "Invalid VIDEOFORGE_RAVE_PROFILE='{value}'. Use 'dev' or 'production_strict'."
                )),
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("Failed to read VIDEOFORGE_RAVE_PROFILE: {e}")),
    }
}

fn resolve_profile() -> Result<String, String> {
    Ok(env_profile_override()?.unwrap_or_else(|| default_profile().to_string()))
}

fn ensure_native_runtime_opt_in(ui_opt_in: bool) -> Result<(), String> {
    if ui_opt_in || native_engine_runtime_enabled() {
        return Ok(());
    }

    Err(encode_rave_error(
        "native_engine_disabled",
        "Native engine path is disabled by default for stability.",
        None,
        Some(
            "Set VIDEOFORGE_ENABLE_NATIVE_ENGINE=1 to opt in explicitly, or switch the header toggle to Python.",
        ),
    ))
}

fn ensure_profile_arg(mut args: Vec<String>, profile: &str) -> Vec<String> {
    let has_profile = args
        .windows(1)
        .any(|w| w.first().is_some_and(|v| v == "--profile"));
    if !has_profile {
        args.push("--profile".to_string());
        args.push(profile.to_string());
    }
    args
}

fn validate_max_batch_arg(args: &[String]) -> Result<(), String> {
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];

        if let Some(v) = arg.strip_prefix("--max-batch=") {
            let parsed = v.parse::<usize>().map_err(|_| {
                format!("Invalid --max-batch value '{v}'. Expected a positive integer.")
            })?;
            if parsed == 0 || parsed > 8 {
                return Err(format!(
                    "Invalid --max-batch value '{parsed}'. Must be in range 1–8."
                ));
            }
        } else if let Some(v) = arg.strip_prefix("--max_batch=") {
            let parsed = v.parse::<usize>().map_err(|_| {
                format!("Invalid --max_batch value '{v}'. Expected a positive integer.")
            })?;
            if parsed == 0 || parsed > 8 {
                return Err(format!(
                    "Invalid --max_batch value '{parsed}'. Must be in range 1–8."
                ));
            }
        } else if arg == "--max-batch" || arg == "--max_batch" {
            let next = args
                .get(idx + 1)
                .ok_or_else(|| format!("Missing value for {arg}. Expected a positive integer."))?;
            let parsed = next.parse::<usize>().map_err(|_| {
                format!("Invalid {arg} value '{next}'. Expected a positive integer.")
            })?;
            if parsed == 0 || parsed > 8 {
                return Err(format!(
                    "Invalid {arg} value '{parsed}'. Must be in range 1–8."
                ));
            }
            idx += 1;
        }

        idx += 1;
    }

    Ok(())
}

fn prepare_rave_cli_args(
    args: Vec<String>,
    ui_opt_in: bool,
    validate_max_batch: bool,
) -> Result<(RaveCliConfig, Vec<String>), String> {
    ensure_native_runtime_opt_in(ui_opt_in)?;
    if validate_max_batch {
        validate_max_batch_arg(&args)?;
    }

    let profile = resolve_profile()?;
    let root = workspace_root().ok_or_else(|| "Failed to resolve workspace root".to_string())?;
    let config = RaveCliConfig::from_workspace_root(root);
    let args = ensure_profile_arg(args, &profile);
    Ok((config, args))
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRaveCliCommand {
    config: RaveCliConfig,
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
}

fn prepare_rave_cli_command(
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
    ui_opt_in: bool,
    validate_max_batch: bool,
) -> Result<PreparedRaveCliCommand, String> {
    let (config, args) = prepare_rave_cli_args(args, ui_opt_in, validate_max_batch)?;
    Ok(PreparedRaveCliCommand {
        config,
        args,
        strict_audit,
        mock_run,
    })
}

pub(crate) fn prepare_rave_upscale_command(
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
    ui_opt_in: bool,
) -> Result<PreparedRaveCliCommand, String> {
    prepare_rave_cli_command(args, strict_audit, mock_run, ui_opt_in, true)
}

pub(crate) fn prepare_rave_benchmark_command(
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
    ui_opt_in: bool,
) -> Result<PreparedRaveCliCommand, String> {
    prepare_rave_cli_command(args, strict_audit, mock_run, ui_opt_in, true)
}

fn encode_rave_error(
    category: &str,
    message: &str,
    detail: Option<&str>,
    next_action: Option<&str>,
) -> String {
    let payload = if let Some(detail) = detail {
        serde_json::json!({
            "category": category,
            "message": message,
            "detail": detail,
            "next_action": next_action
        })
    } else {
        serde_json::json!({
            "category": category,
            "message": message,
            "next_action": next_action
        })
    };
    payload.to_string()
}

fn classify_exit_code(code: i32) -> Option<&'static str> {
    match code {
        100 | 101 => Some("runtime_dependency_missing"), // CUDA/NVRTC
        200..=202 => Some("inference_error"),            // Inference/ModelMetadata/NotInitialized
        300..=305 => Some("codec_error"),                // Decode/Encode/Demux/Mux/Probe
        400..=402 => Some("pipeline_error"),             // Pipeline/ChannelClosed/Shutdown
        500..=502 => Some("input_contract_error"), // FormatMismatch/DimensionMismatch/BufferTooSmall
        600 => Some("policy_violation"),           // InvariantViolation
        700..=703 => Some("runtime_error"), // PanicRecovered/VramLimit/Backpressure/DropOrder
        _ => None,
    }
}

fn classify_exit_stderr(stderr: &str) -> &'static str {
    let lower = stderr.to_ascii_lowercase();

    if lower.contains("max_batch") {
        return "input_contract_error";
    }

    if lower.contains("strict no-host-copies")
        || lower.contains("host copy audit")
        || lower.contains("production_strict")
    {
        return "policy_violation";
    }

    if lower.contains("inference") || lower.contains("model load") || lower.contains("ort error") {
        return "inference_error";
    }

    if (lower.contains("cuda")
        || lower.contains("tensorrt")
        || lower.contains("onnxruntime")
        || lower.contains("provider")
        || lower.contains("nvenc")
        || lower.contains("nvdec"))
        && (lower.contains("missing")
            || lower.contains("not found")
            || lower.contains("could not")
            || lower.contains("failed to load"))
    {
        return "runtime_dependency_missing";
    }

    if lower.contains("provider")
        || lower.contains("tensorrt")
        || lower.contains("onnxruntime")
        || lower.contains("driver")
        || lower.contains("loader")
    {
        return "provider_loader_error";
    }

    "runtime_error"
}

fn map_rave_error(error: RaveCliError) -> String {
    match error {
        RaveCliError::Spawn(msg) => encode_rave_error(
            "runtime_dependency_missing",
            "Failed to launch rave-cli process.",
            Some(&msg),
            Some("Build `rave-cli` release binary and verify runtime dependencies are installed."),
        ),
        RaveCliError::MissingPrebuiltBinary { searched_paths } => {
            let detail = searched_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("; ");
            let next_action = format!(
                "Use `upscale_request_native` (native engine) directly, or prebuild one of these binaries: `{}`.",
                detail
            );
            encode_rave_error(
                "runtime_dependency_missing",
                "rave-cli prebuilt binary not found; fallback compile is disabled.",
                Some(&detail),
                Some(&next_action),
            )
        }
        RaveCliError::Exit { status, stderr } => {
            // Prefer exit-code classification (well-defined in rave-core error codes),
            // fall back to stderr text heuristics for unknown exit codes.
            let category = classify_exit_code(status)
                .unwrap_or_else(|| classify_exit_stderr(&stderr));
            let next_action = match category {
                "policy_violation" => {
                    Some("Enable required strict-profile capabilities (for example audit-no-host-copies) or switch to dev profile.")
                }
                "provider_loader_error" => {
                    Some("Verify CUDA/driver/ORT/TensorRT provider libraries are installed and discoverable by the process.")
                }
                "runtime_dependency_missing" => {
                    Some("Install the missing runtime dependency and rerun the command.")
                }
                "input_contract_error" => {
                    Some("Fix input/CLI arguments to satisfy contract requirements (for example keep max_batch at 1).")
                }
                "inference_error" => {
                    Some("The model failed during inference. Check stderr for ORT/TensorRT details — the model may use unsupported ops or exceed VRAM. Try a smaller model or fp16 precision.")
                }
                "codec_error" => {
                    Some("A codec error occurred during decode/encode. Verify the input format is supported and FFmpeg/NVENC/NVDEC are available.")
                }
                "pipeline_error" => {
                    Some("The processing pipeline failed. Check stderr for details and retry with a simpler configuration.")
                }
                _ => Some("Inspect stderr details and rerun with corrected runtime/input configuration."),
            };
            encode_rave_error(
                category,
                &format!("rave-cli failed with exit status {status}."),
                Some(&stderr),
                next_action,
            )
        }
        RaveCliError::JsonContract(msg) => encode_rave_error(
            "input_contract_error",
            "rave-cli JSON output contract was violated.",
            Some(&msg),
            Some("Ensure --json mode writes exactly one final JSON object to stdout and logs to stderr."),
        ),
    }
}

#[tauri::command]
pub async fn rave_validate(
    fixture: Option<String>,
    profile: Option<String>,
    best_effort: Option<bool>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
) -> Result<serde_json::Value, String> {
    let root = workspace_root().ok_or_else(|| "Failed to resolve workspace root".to_string())?;
    let config = RaveCliConfig::from_workspace_root(root);
    let fixture_path = fixture
        .as_deref()
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from);
    let res = run_validate(
        &config,
        fixture_path.as_deref(),
        profile.as_deref().unwrap_or("production_strict"),
        best_effort.unwrap_or(false),
        strict_audit.unwrap_or(true),
        mock_run.unwrap_or(false),
    )
    .await
    .map_err(map_rave_error)?;

    Ok(res.json)
}

pub(crate) async fn run_prepared_rave_upscale(
    prepared: PreparedRaveCliCommand,
) -> Result<RaveResult, String> {
    run_upscale(
        &prepared.config,
        &prepared.args,
        prepared.strict_audit,
        prepared.mock_run,
    )
    .await
    .map_err(map_rave_error)
}

pub(crate) async fn run_prepared_rave_benchmark(
    prepared: PreparedRaveCliCommand,
) -> Result<RaveResult, String> {
    run_benchmark(
        &prepared.config,
        &prepared.args,
        prepared.strict_audit,
        prepared.mock_run,
    )
    .await
    .map_err(map_rave_error)
}

#[tauri::command]
pub async fn rave_upscale(
    args: Vec<String>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
    ui_opt_in: Option<bool>,
) -> Result<serde_json::Value, String> {
    Ok(run_rave_upscale_internal(
        args,
        strict_audit.unwrap_or(true),
        mock_run.unwrap_or(false),
        ui_opt_in.unwrap_or(false),
    )
    .await?
    .json)
}

pub async fn run_rave_upscale_internal(
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
    ui_opt_in: bool,
) -> Result<RaveResult, String> {
    run_prepared_rave_upscale(prepare_rave_upscale_command(
        args,
        strict_audit,
        mock_run,
        ui_opt_in,
    )?)
    .await
}

pub async fn run_rave_benchmark_internal(
    args: Vec<String>,
    strict_audit: bool,
    mock_run: bool,
    ui_opt_in: bool,
) -> Result<RaveResult, String> {
    run_prepared_rave_benchmark(prepare_rave_benchmark_command(
        args,
        strict_audit,
        mock_run,
        ui_opt_in,
    )?)
    .await
}

#[cfg(test)]
mod tests {
    use super::{
        classify_exit_stderr, map_rave_error, prepare_rave_upscale_command, validate_max_batch_arg,
    };
    use crate::rave_cli::RaveCliError;

    #[test]
    fn max_batch_allows_one_through_eight() {
        for n in 1usize..=8 {
            assert!(
                validate_max_batch_arg(&[format!("--max-batch={n}")]).is_ok(),
                "--max-batch={n} should be accepted"
            );
        }
        assert!(validate_max_batch_arg(&["--max-batch".to_string(), "4".to_string()]).is_ok());
        assert!(validate_max_batch_arg(&["--max_batch=4".to_string()]).is_ok());
    }

    #[test]
    fn max_batch_rejects_zero() {
        let err = validate_max_batch_arg(&["--max-batch=0".to_string()])
            .expect_err("max_batch=0 must fail");
        assert!(err.contains("1–8"));
    }

    #[test]
    fn max_batch_rejects_above_eight() {
        let err = validate_max_batch_arg(&["--max-batch=9".to_string()])
            .expect_err("max_batch=9 must fail");
        assert!(err.contains("1–8"));
    }

    #[test]
    fn max_batch_rejects_above_eight_positional() {
        let err = validate_max_batch_arg(&[
            "--foo".to_string(),
            "bar".to_string(),
            "--max_batch".to_string(),
            "9".to_string(),
        ])
        .expect_err("max_batch=9 must fail");
        assert!(err.contains("1–8"));
    }

    #[test]
    fn max_batch_requires_value() {
        let err = validate_max_batch_arg(&["--max-batch".to_string()])
            .expect_err("missing max-batch value must fail");
        assert!(err.contains("Missing value"));
    }

    #[test]
    fn classify_policy_violation_from_stderr() {
        let category = classify_exit_stderr("strict no-host-copies requires audit");
        assert_eq!(category, "policy_violation");
    }

    #[test]
    fn map_error_returns_json_payload() {
        let out = map_rave_error(RaveCliError::JsonContract(
            "stdout contained 2 JSON objects".to_string(),
        ));
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("mapped error should be JSON");
        assert_eq!(parsed["category"], "input_contract_error");
        assert!(parsed.get("next_action").is_some());
    }

    #[test]
    fn prepared_upscale_command_injects_profile_once() {
        let prepared = prepare_rave_upscale_command(
            vec!["-i".to_string(), "input.mp4".to_string()],
            true,
            false,
            true,
        )
        .expect("prepared command");

        let profile_positions = prepared
            .args
            .iter()
            .enumerate()
            .filter_map(|(idx, arg)| (arg == "--profile").then_some(idx))
            .collect::<Vec<_>>();
        assert_eq!(profile_positions.len(), 1);
        let profile_idx = profile_positions[0];
        assert!(prepared.args.get(profile_idx + 1).is_some());
    }
}

#[tauri::command]
pub async fn rave_benchmark(
    args: Vec<String>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
    ui_opt_in: Option<bool>,
) -> Result<serde_json::Value, String> {
    Ok(run_rave_benchmark_internal(
        args,
        strict_audit.unwrap_or(true),
        mock_run.unwrap_or(false),
        ui_opt_in.unwrap_or(false),
    )
    .await?
    .json)
}
