use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

use crate::commands::native_runtime::resolve_native_runtime_paths;

#[derive(Debug, Clone)]
pub struct RaveCliConfig {
    pub bin_path: PathBuf,
    pub candidate_paths: Vec<PathBuf>,
    pub workspace_root: PathBuf,
}

impl RaveCliConfig {
    pub fn from_workspace_root(workspace_root: impl AsRef<Path>) -> Self {
        let workspace_root = workspace_root.as_ref().to_path_buf();
        let bin_name = if cfg!(windows) { "rave.exe" } else { "rave" };
        let candidate_paths = vec![
            workspace_root
                .join("third_party")
                .join("rave")
                .join("target")
                .join("release")
                .join(bin_name),
            workspace_root
                .join("third_party")
                .join("rave")
                .join("rave-cli")
                .join("target")
                .join("release")
                .join(bin_name),
            workspace_root
                .join("third_party")
                .join("rave-main")
                .join("target")
                .join("release")
                .join(bin_name),
            workspace_root
                .join("third_party")
                .join("rave-main")
                .join("rave-cli")
                .join("target")
                .join("release")
                .join(bin_name),
        ];

        let resolved_bin = candidate_paths
            .iter()
            .find(|p| p.exists())
            .cloned()
            .unwrap_or_else(|| candidate_paths[0].clone());
        Self {
            bin_path: resolved_bin,
            candidate_paths,
            workspace_root,
        }
    }
}

fn prepend_runtime_path(cmd: &mut Command, workspace_root: &Path, bin_path: &Path) {
    let runtime = resolve_native_runtime_paths(Some(workspace_root), bin_path.parent());
    if runtime.path_additions.is_empty() {
        return;
    }

    let mut merged = String::new();
    for d in &runtime.path_additions {
        if !merged.is_empty() {
            merged.push(';');
        }
        merged.push_str(&d.to_string_lossy());
    }
    if let Ok(existing) = std::env::var("PATH") {
        if !existing.is_empty() {
            merged.push(';');
            merged.push_str(&existing);
        }
    }
    cmd.env("PATH", merged);
}

#[derive(Debug, Clone)]
pub struct RaveResult {
    pub json: Value,
    pub stderr: String,
    pub progress: Option<RaveProgressSummary>,
}

#[derive(Debug, Clone)]
pub struct RaveProgressSummary {
    pub elapsed_ms: u64,
    pub frames_decoded: u64,
    pub frames_inferred: u64,
    pub frames_encoded: u64,
    pub final_record_seen: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum RaveCliError {
    #[error("rave-cli invocation failed: {0}")]
    Spawn(String),
    #[error(
        "rave-cli prebuilt binary not found. Searched paths: {searched_paths:?}. Fallback compile is disabled; use upscale_request_native (native engine) or prebuild rave-cli."
    )]
    MissingPrebuiltBinary { searched_paths: Vec<PathBuf> },
    #[error("rave-cli exited with status {status}: {stderr}")]
    Exit { status: i32, stderr: String },
    #[error("rave-cli --json contract violated: {0}")]
    JsonContract(String),
}

#[cfg(test)]
mod runtime_path_tests {
    use super::*;
    use crate::commands::native_runtime::workspace_root;

    #[test]
    fn runtime_path_resolver_keeps_cli_bin_dir() {
        let root = workspace_root().expect("workspace root");
        let fake_bin = root
            .join("third_party")
            .join("rave")
            .join("target")
            .join("release");
        let runtime = resolve_native_runtime_paths(Some(&root), Some(&fake_bin));
        assert!(runtime.path_additions.iter().any(|p| p == &fake_bin));
    }
}

pub async fn run_validate(
    config: &RaveCliConfig,
    fixture: Option<&Path>,
    profile: &str,
    best_effort: bool,
    strict_audit: bool,
    mock_run: bool,
) -> Result<RaveResult, RaveCliError> {
    let mut args = vec![
        "validate".to_string(),
        "--json".to_string(),
        "--profile".to_string(),
        profile.to_string(),
    ];

    if let Some(fixture) = fixture {
        args.push("--fixture".to_string());
        args.push(fixture.to_string_lossy().to_string());
    }

    if best_effort {
        args.push("--best-effort".to_string());
    }

    run_cli(config, "validate", &args, strict_audit, mock_run).await
}

pub async fn run_upscale(
    config: &RaveCliConfig,
    args: &[String],
    strict_audit: bool,
    mock_run: bool,
) -> Result<RaveResult, RaveCliError> {
    let mut full = vec!["upscale".to_string(), "--json".to_string()];
    full.extend(args.iter().cloned());
    run_cli(config, "upscale", &full, strict_audit, mock_run).await
}

pub async fn run_benchmark(
    config: &RaveCliConfig,
    args: &[String],
    strict_audit: bool,
    mock_run: bool,
) -> Result<RaveResult, RaveCliError> {
    let mut full = vec!["benchmark".to_string(), "--json".to_string()];
    full.extend(args.iter().cloned());
    run_cli(config, "benchmark", &full, strict_audit, mock_run).await
}

async fn run_cli(
    config: &RaveCliConfig,
    command: &str,
    args: &[String],
    strict_audit: bool,
    mock_run: bool,
) -> Result<RaveResult, RaveCliError> {
    if !config.bin_path.exists() {
        return Err(RaveCliError::MissingPrebuiltBinary {
            searched_paths: config.candidate_paths.clone(),
        });
    }

    let mut cmd = Command::new(&config.bin_path);

    cmd.current_dir(&config.workspace_root)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    prepend_runtime_path(&mut cmd, &config.workspace_root, &config.bin_path);

    if strict_audit {
        cmd.env("RAVE_STRICT_AUDIT_REQUIRED", "1");
    }
    if mock_run {
        cmd.env("RAVE_MOCK_RUN", "1");
    }

    let out = cmd
        .output()
        .await
        .map_err(|e| RaveCliError::Spawn(e.to_string()))?;

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    if !out.status.success() {
        return Err(RaveCliError::Exit {
            status: out.status.code().unwrap_or(-1),
            stderr,
        });
    }

    let json = parse_json_stdout_contract(&stdout)?;
    assert_required_contract_fields(command, &json)?;
    let progress = parse_progress_summary(&stderr, command);
    Ok(RaveResult {
        json,
        stderr,
        progress,
    })
}

fn parse_progress_summary(stderr: &str, command: &str) -> Option<RaveProgressSummary> {
    let mut latest: Option<RaveProgressSummary> = None;

    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        if value.get("type").and_then(|v| v.as_str()) != Some("progress") {
            continue;
        }
        if value.get("command").and_then(|v| v.as_str()) != Some(command) {
            continue;
        }

        let Some(frames) = value.get("frames") else {
            continue;
        };
        let Some(elapsed_ms) = value.get("elapsed_ms").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(frames_decoded) = frames.get("decoded").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(frames_inferred) = frames.get("inferred").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(frames_encoded) = frames.get("encoded").and_then(|v| v.as_u64()) else {
            continue;
        };

        latest = Some(RaveProgressSummary {
            elapsed_ms,
            frames_decoded,
            frames_inferred,
            frames_encoded,
            final_record_seen: value
                .get("final")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        });
    }

    latest
}

fn parse_json_stdout_contract(stdout: &str) -> Result<Value, RaveCliError> {
    let mut json_lines = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Contract requires only one final JSON object on stdout in --json mode.
        let parsed: Value = serde_json::from_str(trimmed).map_err(|_| {
            RaveCliError::JsonContract(format!(
                "non-JSON content on stdout: {}",
                summarize_for_error(trimmed)
            ))
        })?;
        if !parsed.is_object() {
            return Err(RaveCliError::JsonContract(
                "stdout JSON payload must be an object".to_string(),
            ));
        }
        json_lines.push(parsed);
    }

    match json_lines.len() {
        1 => Ok(json_lines.remove(0)),
        0 => Err(RaveCliError::JsonContract(
            "stdout did not contain final JSON object".to_string(),
        )),
        n => Err(RaveCliError::JsonContract(format!(
            "stdout contained {n} JSON objects; expected exactly one"
        ))),
    }
}

fn summarize_for_error(value: &str) -> String {
    const MAX: usize = 120;
    if value.len() <= MAX {
        value.to_string()
    } else {
        format!("{}...", &value[..MAX])
    }
}

fn assert_required_contract_fields(command: &str, json: &Value) -> Result<(), RaveCliError> {
    if !matches!(command, "upscale" | "benchmark" | "validate") {
        return Ok(());
    }

    if !json.get("policy").is_some_and(|v| v.is_object()) {
        return Err(RaveCliError::JsonContract(format!(
            "{command} JSON missing required top-level policy object"
        )));
    }

    if command == "validate" {
        if json.get("host_copy_audit_enabled").is_none() {
            return Err(RaveCliError::JsonContract(
                "validate JSON missing host_copy_audit_enabled".to_string(),
            ));
        }
        if json.get("host_copy_audit_disable_reason").is_none() {
            return Err(RaveCliError::JsonContract(
                "validate JSON missing host_copy_audit_disable_reason".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        assert_required_contract_fields, parse_json_stdout_contract, parse_progress_summary,
    };

    #[test]
    fn parse_accepts_single_json_object() {
        let out = "{\"policy\":{\"mode\":\"production_strict\"}}\n";
        let v = parse_json_stdout_contract(out).expect("single object should parse");
        assert_eq!(v["policy"]["mode"], "production_strict");
    }

    #[test]
    fn parse_rejects_non_json_content() {
        let out = "hello\n{\"ok\":true}\n";
        let err = parse_json_stdout_contract(out).expect_err("must reject plain log lines");
        assert!(err.to_string().contains("contract violated"));
    }

    #[test]
    fn parse_rejects_multiple_json_objects() {
        let out = "{\"a\":1}\n{\"b\":2}\n";
        let err = parse_json_stdout_contract(out).expect_err("must reject two objects");
        assert!(err.to_string().contains("expected exactly one"));
    }

    #[test]
    fn parse_rejects_missing_json_object() {
        let out = "\n  \n";
        let err = parse_json_stdout_contract(out).expect_err("must reject empty output");
        assert!(err
            .to_string()
            .contains("did not contain final JSON object"));
    }

    #[test]
    fn contract_requires_policy_for_upscale() {
        let v: serde_json::Value = serde_json::json!({ "ok": true });
        let err = assert_required_contract_fields("upscale", &v).expect_err("must require policy");
        assert!(err.to_string().contains("policy object"));
    }

    #[test]
    fn contract_requires_validate_audit_fields() {
        let v: serde_json::Value = serde_json::json!({
            "policy": { "strict": true },
            "host_copy_audit_enabled": true
        });
        let err = assert_required_contract_fields("validate", &v)
            .expect_err("must require host_copy_audit_disable_reason");
        assert!(err.to_string().contains("host_copy_audit_disable_reason"));
    }

    #[test]
    fn contract_accepts_validate_policy_and_audit_fields() {
        let v: serde_json::Value = serde_json::json!({
            "policy": { "strict": true },
            "host_copy_audit_enabled": true,
            "host_copy_audit_disable_reason": null
        });
        assert!(assert_required_contract_fields("validate", &v).is_ok());
    }

    #[test]
    fn progress_summary_uses_latest_matching_record() {
        let stderr = r#"
noise
{"schema_version":1,"type":"progress","command":"upscale","elapsed_ms":100,"frames":{"decoded":5,"inferred":4,"encoded":3},"final":false}
{"schema_version":1,"type":"progress","command":"upscale","elapsed_ms":250,"frames":{"decoded":9,"inferred":8,"encoded":7},"final":true}
"#;
        let summary = parse_progress_summary(stderr, "upscale").expect("progress summary");
        assert_eq!(summary.elapsed_ms, 250);
        assert_eq!(summary.frames_decoded, 9);
        assert_eq!(summary.frames_inferred, 8);
        assert_eq!(summary.frames_encoded, 7);
        assert!(summary.final_record_seen);
    }

    #[test]
    fn progress_summary_ignores_non_matching_records() {
        let stderr = r#"
{"schema_version":1,"type":"progress","command":"benchmark","elapsed_ms":100,"frames":{"decoded":5,"inferred":4,"encoded":3},"final":true}
"#;
        assert!(parse_progress_summary(stderr, "upscale").is_none());
    }
}
