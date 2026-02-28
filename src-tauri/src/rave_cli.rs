use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct RaveCliConfig {
    pub bin_path: PathBuf,
    pub workspace_root: PathBuf,
}

impl RaveCliConfig {
    pub fn from_workspace_root(workspace_root: impl AsRef<Path>) -> Self {
        let workspace_root = workspace_root.as_ref().to_path_buf();
        let bin_name = if cfg!(windows) { "rave.exe" } else { "rave" };
        let workspace_target_bin = workspace_root
            .join("third_party")
            .join("rave")
            .join("target")
            .join("release")
            .join(bin_name);
        let legacy_crate_target_bin = workspace_root
            .join("third_party")
            .join("rave")
            .join("rave-cli")
            .join("target")
            .join("release")
            .join(bin_name);

        let resolved_bin = if workspace_target_bin.exists() {
            workspace_target_bin
        } else {
            // Compatibility fallback for non-workspace builds.
            legacy_crate_target_bin
        };
        Self {
            bin_path: resolved_bin,
            workspace_root,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaveResult {
    pub json: Value,
    pub stderr: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RaveCliError {
    #[error("rave-cli invocation failed: {0}")]
    Spawn(String),
    #[error(
        "rave-cli prebuilt binary not found at {expected_path}. Fallback compile is disabled; use upscale_request_native (native engine) or prebuild rave-cli."
    )]
    MissingPrebuiltBinary { expected_path: PathBuf },
    #[error("rave-cli exited with status {status}: {stderr}")]
    Exit { status: i32, stderr: String },
    #[error("rave-cli --json contract violated: {0}")]
    JsonContract(String),
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
            expected_path: config.bin_path.clone(),
        });
    }

    let mut cmd = Command::new(&config.bin_path);

    cmd.current_dir(&config.workspace_root)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

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
    Ok(RaveResult { json, stderr })
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
    use super::{assert_required_contract_fields, parse_json_stdout_contract};

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
}
