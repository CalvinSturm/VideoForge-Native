//! VideoForge IPC Protocol — typed Zenoh message envelopes.
//!
//! All Rust ↔ Python Zenoh messages use these types as the canonical
//! representation.  Serialization is always UTF-8 JSON.
//!
//! # Forward compatibility
//!
//! - **Deserialization**: serde ignores unknown fields by default (no
//!   `deny_unknown_fields`).  Adding new fields to either side is safe.
//! - **Missing fields on receive**: `#[serde(default)]` on all envelope
//!   fields ensures old Python workers (which omit protocol-level fields)
//!   still deserialize without error.
//!
//! # Protocol version
//!
//! Current: [`PROTOCOL_VERSION`] = 1.  Bump when a breaking schema change
//! is introduced; add a new migration path before removing the old one.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// Current protocol version written into every outgoing envelope.
pub const PROTOCOL_VERSION: u32 = 1;

static REQUEST_SEQ: AtomicU64 = AtomicU64::new(1);

/// Return a per-process-lifetime monotonically increasing request ID string.
pub fn next_request_id() -> String {
    REQUEST_SEQ
        .fetch_add(1, Ordering::Relaxed)
        .to_string()
}

// ─── Request (Rust → Python) ─────────────────────────────────────────────────

/// Envelope for every outgoing Rust → Python command.
///
/// Serialized to a JSON string and published on `videoforge/ipc/{port}/req`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestEnvelope {
    /// Protocol version — always [`PROTOCOL_VERSION`].
    pub version: u32,
    /// Per-request monotonic ID for response correlation.
    pub request_id: String,
    /// Job identifier — links this command to a Tauri invocation.
    pub job_id: String,
    /// Command kind — matches the Python `on_request` dispatcher keys.
    pub kind: String,
    /// Command-specific parameters (command-defined schema).
    pub payload: serde_json::Value,
}

impl RequestEnvelope {
    /// Construct a new request with a fresh [`next_request_id`].
    pub fn new(
        kind: impl Into<String>,
        job_id: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            request_id: next_request_id(),
            job_id: job_id.into(),
            kind: kind.into(),
            payload,
        }
    }

    /// Serialize to a JSON string ready for `publisher.put()`.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("RequestEnvelope is always serializable")
    }
}

// ─── Response (Python → Rust) ────────────────────────────────────────────────

/// Structured error carried inside a [`ResponseEnvelope`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcError {
    /// Machine-readable code (SCREAMING_SNAKE_CASE).
    pub code: String,
    /// Human-readable description.
    pub message: String,
}

impl IpcError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Envelope for every incoming Python → Rust response.
///
/// `#[serde(default)]` on protocol-level fields ensures backward
/// compatibility with Python workers that do not yet emit them.
/// Unknown JSON fields are silently ignored (serde default behaviour).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEnvelope {
    /// Protocol version of the sender (0 if old worker — pre-protocol).
    #[serde(default)]
    pub version: u32,
    /// Echo of the `request_id` this response is for ("" if old worker).
    #[serde(default)]
    pub request_id: String,
    /// Echo of the `job_id` ("" if old worker).
    #[serde(default)]
    pub job_id: String,
    /// Response kind: "status" | "progress" | "error" ("" if old worker).
    #[serde(default)]
    pub kind: String,
    /// Human-readable status string (e.g. "MODEL_LOADED", "ok").
    pub status: Option<String>,
    /// Present only on error responses.
    pub error: Option<IpcError>,
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── RequestEnvelope ──────────────────────────────────────────────────────

    #[test]
    fn request_roundtrip() {
        let req = RequestEnvelope::new(
            "load_model",
            "job-1",
            json!({"model_name": "RCAN_x4"}),
        );
        let json_str = req.to_json();
        let parsed: RequestEnvelope = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.version, PROTOCOL_VERSION);
        assert_eq!(parsed.kind, "load_model");
        assert_eq!(parsed.job_id, "job-1");
        assert_eq!(parsed.payload["model_name"], "RCAN_x4");
    }

    #[test]
    fn request_ignores_unknown_fields() {
        // A future Python version might add fields Rust does not know about.
        let json_str = r#"{
            "version": 1,
            "request_id": "42",
            "job_id": "job-x",
            "kind": "load_model",
            "payload": {"model_name": "EDSR_x2"},
            "future_field": "should be ignored"
        }"#;
        let parsed: RequestEnvelope = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.kind, "load_model");
        assert_eq!(parsed.payload["model_name"], "EDSR_x2");
    }

    #[test]
    fn request_id_is_monotonic() {
        let a = next_request_id().parse::<u64>().unwrap();
        let b = next_request_id().parse::<u64>().unwrap();
        let c = next_request_id().parse::<u64>().unwrap();
        assert!(b > a, "request IDs must increase");
        assert!(c > b, "request IDs must increase");
    }

    #[test]
    fn request_version_is_current() {
        let req = RequestEnvelope::new("shutdown", "j", json!({}));
        assert_eq!(req.version, PROTOCOL_VERSION);
    }

    // ── ResponseEnvelope ─────────────────────────────────────────────────────

    #[test]
    fn response_happy_path() {
        // A fully-conformant new Python worker response.
        let json_str = r#"{
            "version": 1,
            "request_id": "7",
            "job_id": "job-1",
            "kind": "status",
            "status": "MODEL_LOADED",
            "model": "RCAN_x4",
            "scale": 4
        }"#;
        let resp: ResponseEnvelope = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.version, 1);
        assert_eq!(resp.request_id, "7");
        assert_eq!(resp.status.as_deref(), Some("MODEL_LOADED"));
        assert!(resp.error.is_none());
    }

    #[test]
    fn response_backward_compat_missing_protocol_fields() {
        // Old Python worker that omits version/request_id/job_id/kind.
        // Must deserialize cleanly using serde defaults.
        let json_str = r#"{"status": "MODEL_LOADED", "model": "RCAN_x4", "scale": 4}"#;
        let resp: ResponseEnvelope = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.version, 0);       // default
        assert_eq!(resp.request_id, "");   // default
        assert_eq!(resp.job_id, "");       // default
        assert_eq!(resp.status.as_deref(), Some("MODEL_LOADED"));
        assert!(resp.error.is_none());
    }

    #[test]
    fn response_ignores_unknown_fields() {
        // Future Python worker adds fields Rust does not yet know.
        let json_str = r#"{
            "version": 1,
            "request_id": "3",
            "job_id": "job-2",
            "kind": "status",
            "status": "SHM_CREATED",
            "shm_path": "/tmp/vf_buffer_123.bin",
            "future_field": true,
            "another_future_field": {"nested": "data"}
        }"#;
        let resp: ResponseEnvelope = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.status.as_deref(), Some("SHM_CREATED"));
        // No panic — unknown fields silently discarded.
    }

    #[test]
    fn response_with_structured_error() {
        let json_str = r#"{
            "version": 1,
            "request_id": "5",
            "job_id": "job-3",
            "kind": "error",
            "status": "error",
            "error": {"code": "MODEL_NOT_FOUND", "message": "No weights found for RCAN_x8"}
        }"#;
        let resp: ResponseEnvelope = serde_json::from_str(json_str).unwrap();
        let err = resp.error.expect("error field should be present");
        assert_eq!(err.code, "MODEL_NOT_FOUND");
        assert_eq!(err.message, "No weights found for RCAN_x8");
    }

    #[test]
    fn response_null_error_is_ok() {
        let json_str = r#"{"version":1,"request_id":"1","job_id":"j","kind":"status","status":"ok","error":null}"#;
        let resp: ResponseEnvelope = serde_json::from_str(json_str).unwrap();
        assert!(resp.error.is_none());
    }
}
