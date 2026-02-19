//! IPC module — Zenoh transport helpers and typed protocol envelopes.
//!
//! All Rust ↔ Python communication goes through this module.
//! See [`protocol`] for message schema and [`put_request`] for the
//! canonical send path.

pub mod protocol;

pub use protocol::{IpcError, RequestEnvelope, ResponseEnvelope, PROTOCOL_VERSION};

use zenoh::pubsub::Publisher;

/// Serialize `req` to JSON and publish it on `publisher`.
///
/// This is the single Rust → Python send site.  All Zenoh `publisher.put()`
/// calls MUST go through this function to guarantee envelope conformance.
pub async fn put_request(
    publisher: &Publisher<'_>,
    req: RequestEnvelope,
) -> Result<(), zenoh::Error> {
    publisher.put(req.to_json()).await
}
