use std::path::PathBuf;

use crate::rave_cli::{run_benchmark, run_upscale, run_validate, RaveCliConfig};

fn workspace_root() -> Result<PathBuf, String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .map_err(|e| format!("Failed to resolve workspace root: {e}"))?;
    Ok(p)
}

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

#[tauri::command]
pub async fn rave_validate(
    fixture: Option<String>,
    profile: Option<String>,
    best_effort: Option<bool>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
) -> Result<serde_json::Value, String> {
    let root = workspace_root()?;
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
    .map_err(|e| e.to_string())?;

    Ok(res.json)
}

#[tauri::command]
pub async fn rave_upscale(
    args: Vec<String>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
) -> Result<serde_json::Value, String> {
    let profile = resolve_profile()?;
    let root = workspace_root()?;
    let config = RaveCliConfig::from_workspace_root(root);
    let args = ensure_profile_arg(args, &profile);
    let res = run_upscale(
        &config,
        &args,
        strict_audit.unwrap_or(true),
        mock_run.unwrap_or(false),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(res.json)
}

#[tauri::command]
pub async fn rave_benchmark(
    args: Vec<String>,
    strict_audit: Option<bool>,
    mock_run: Option<bool>,
) -> Result<serde_json::Value, String> {
    let profile = resolve_profile()?;
    let root = workspace_root()?;
    let config = RaveCliConfig::from_workspace_root(root);
    let args = ensure_profile_arg(args, &profile);
    let res = run_benchmark(
        &config,
        &args,
        strict_audit.unwrap_or(true),
        mock_run.unwrap_or(false),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(res.json)
}
