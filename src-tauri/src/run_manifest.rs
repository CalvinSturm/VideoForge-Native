use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::python_env::WorkerCaps;

pub const RUN_MANIFEST_SCHEMA_V1: &str = "videoforge.run_manifest.v1";
const RUN_MANIFEST_FILENAME_V1: &str = "videoforge.run_manifest.v1.json";

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerCapsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    pub use_typed_ipc: bool,
    pub use_events: bool,
    pub prealloc_tensors: bool,
    pub deterministic: bool,
}

impl From<&WorkerCaps> for WorkerCapsSnapshot {
    fn from(value: &WorkerCaps) -> Self {
        Self {
            log_level: value.log_level.clone(),
            use_typed_ipc: value.use_typed_ipc,
            use_events: value.use_events,
            prealloc_tensors: value.prealloc_tensors,
            deterministic: value.deterministic,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ProtocolSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_protocol_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_protocol_version: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RunManifestV1 {
    pub schema_version: String,
    pub created_at_utc: String,
    pub job_id: String,
    pub input_path: String,
    pub output_path: String,
    pub scale: u32,
    pub precision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    pub worker_caps: WorkerCapsSnapshot,
    pub protocol: ProtocolSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunManifestInputs<'a> {
    pub input_path: &'a str,
    pub output_path: &'a str,
    pub scale: u32,
    pub precision: &'a str,
    pub model_key: Option<&'a str>,
    pub worker_caps: &'a WorkerCaps,
    pub ipc_protocol_version: Option<u32>,
    pub shm_protocol_version: Option<u32>,
    pub app_version: Option<&'a str>,
}

pub fn normalize_path_for_id(path: &str) -> String {
    path.replace('\\', "/")
}

pub fn compute_job_id(
    input_path: &str,
    output_path: &str,
    scale: u32,
    precision: &str,
    model_key: Option<&str>,
    worker_caps: Option<&WorkerCapsSnapshot>,
) -> String {
    let mut hasher = Sha256::new();
    let caps_json = worker_caps
        .and_then(|c| serde_json::to_string(c).ok())
        .unwrap_or_default();
    let payload = format!(
        "input={}\noutput={}\nscale={}\nprecision={}\nmodel={}\nworker_caps={}\n",
        normalize_path_for_id(input_path),
        normalize_path_for_id(output_path),
        scale,
        precision,
        model_key.unwrap_or_default(),
        caps_json
    );
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

pub fn build_run_manifest_v1(inputs: &RunManifestInputs<'_>) -> Result<RunManifestV1> {
    let worker_caps = WorkerCapsSnapshot::from(inputs.worker_caps);
    let created_at_utc = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("formatting manifest timestamp")?;
    let job_id = compute_job_id(
        inputs.input_path,
        inputs.output_path,
        inputs.scale,
        inputs.precision,
        inputs.model_key,
        Some(&worker_caps),
    );

    Ok(RunManifestV1 {
        schema_version: RUN_MANIFEST_SCHEMA_V1.to_string(),
        created_at_utc,
        job_id,
        input_path: inputs.input_path.to_string(),
        output_path: inputs.output_path.to_string(),
        scale: inputs.scale,
        precision: inputs.precision.to_string(),
        model_key: inputs.model_key.map(str::to_string),
        worker_caps,
        protocol: ProtocolSnapshot {
            ipc_protocol_version: inputs.ipc_protocol_version,
            shm_protocol_version: inputs.shm_protocol_version,
        },
        app_version: inputs.app_version.map(str::to_string),
    })
}

pub fn maybe_write_run_manifest(
    enable: bool,
    inputs: &RunManifestInputs<'_>,
) -> Result<Option<PathBuf>> {
    if !enable {
        return Ok(None);
    }

    let manifest = build_run_manifest_v1(inputs)?;
    let output_dir = Path::new(inputs.output_path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let artifacts_dir = output_dir.join(".videoforge_runs").join(&manifest.job_id);
    fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("creating artifacts dir '{}'", artifacts_dir.display()))?;

    let final_path = artifacts_dir.join(RUN_MANIFEST_FILENAME_V1);
    let tmp_path = artifacts_dir.join(format!("{}.tmp", RUN_MANIFEST_FILENAME_V1));
    let json = serde_json::to_vec_pretty(&manifest).context("serializing run manifest")?;
    fs::write(&tmp_path, json)
        .with_context(|| format!("writing temp manifest '{}'", tmp_path.display()))?;
    fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "renaming temp manifest '{}' -> '{}'",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    Ok(Some(final_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_caps() -> WorkerCaps {
        WorkerCaps::default()
    }

    fn test_inputs<'a>(output_path: &'a str, caps: &'a WorkerCaps) -> RunManifestInputs<'a> {
        RunManifestInputs {
            input_path: r"C:\input\clip.mp4",
            output_path,
            scale: 4,
            precision: "fp16",
            model_key: Some("RCAN_x4"),
            worker_caps: caps,
            ipc_protocol_version: Some(1),
            shm_protocol_version: None,
            app_version: Some("0.1.0"),
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    #[test]
    fn test_manifest_schema_and_required_fields() {
        let caps = test_caps();
        let manifest =
            build_run_manifest_v1(&test_inputs("C:/out/final.mp4", &caps)).expect("manifest");
        let value = serde_json::to_value(&manifest).expect("serialize manifest");

        assert_eq!(manifest.schema_version, RUN_MANIFEST_SCHEMA_V1);
        assert_eq!(
            value.get("schema_version").and_then(Value::as_str),
            Some(RUN_MANIFEST_SCHEMA_V1)
        );
        for key in [
            "schema_version",
            "created_at_utc",
            "job_id",
            "input_path",
            "output_path",
        ] {
            assert!(value.get(key).is_some(), "missing key: {key}");
        }
    }

    #[test]
    fn test_artifacts_gated_off_creates_nothing() {
        let caps = test_caps();
        let root = unique_temp_dir("vf_manifest_off");
        let output_dir = root.join("renders");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let output_file = output_dir.join("out.mp4");
        let output_file_str = output_file.to_string_lossy().to_string();
        let inputs = test_inputs(&output_file_str, &caps);

        let written = maybe_write_run_manifest(false, &inputs).expect("helper returns");
        assert!(written.is_none());
        assert!(!output_dir.join(".videoforge_runs").exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_job_id_deterministic() {
        let caps = WorkerCapsSnapshot::default();
        let a = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            4,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );
        let b = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            4,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );
        let c = compute_job_id(
            r"C:\in.mp4",
            r"C:\out.mp4",
            2,
            "fp32",
            Some("RCAN_x4"),
            Some(&caps),
        );

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
