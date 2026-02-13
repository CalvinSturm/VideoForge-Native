//! Spatial Map Subscriber — Zenoh consumer + Tauri IPC bridge.
//!
//! Subscribes to `videoforge/research/spatial_map`, validates incoming binary
//! payloads, gates on an `AtomicBool` to prevent frame queueing, and exposes
//! the latest frame to the frontend via a binary-optimized Tauri command.
//!
//! IPC flow:
//!   Zenoh frame → validate → if !busy { store + emit signal } else { drop }
//!   Frontend:  listen("spatial-frame-ready") → invoke("fetch_spatial_frame")
//!              → render → invoke("mark_frame_complete")

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::{AppHandle, Emitter};

use crate::spatial_publisher::SPATIAL_MAP_TOPIC;

const HEADER_SIZE: usize = 8; // 2 × u32

// ===========================================================================
// STATE
// ===========================================================================

/// Thread-safe state shared between the Zenoh subscriber task and Tauri commands.
pub struct SpatialMapState {
    /// The most recent validated spatial map payload (header + mask).
    /// `None` when no frame is pending or it has been consumed.
    pending_frame: std::sync::Mutex<Option<Vec<u8>>>,

    /// When `true`, the frontend is still rendering the previous frame.
    /// New frames are **dropped** (not queued) while this is set.
    is_frontend_busy: AtomicBool,
}

impl SpatialMapState {
    pub fn new() -> Self {
        Self {
            pending_frame: std::sync::Mutex::new(None),
            is_frontend_busy: AtomicBool::new(false),
        }
    }
}

// ===========================================================================
// ZENOH SUBSCRIBER
// ===========================================================================

/// Start the spatial-map Zenoh subscriber.
///
/// This spawns a tokio task that:
/// 1. Subscribes to [`SPATIAL_MAP_TOPIC`]
/// 2. Validates each incoming binary payload
/// 3. Atomically checks `is_frontend_busy`
/// 4. If not busy: stores the frame, sets busy, emits `"spatial-frame-ready"`
/// 5. If busy: drops the frame (newest-frame / no-queue policy)
///
/// Returns a `JoinHandle` that can be used for shutdown.
pub async fn init_spatial_subscriber(
    session: &zenoh::Session,
    state: Arc<SpatialMapState>,
    app: AppHandle,
) -> Result<tokio::task::JoinHandle<()>, String> {
    let subscriber = session
        .declare_subscriber(SPATIAL_MAP_TOPIC)
        .await
        .map_err(|e| format!("Zenoh spatial subscribe failed: {e}"))?;

    println!("[SpatialMap] Subscribed to {}", SPATIAL_MAP_TOPIC);

    let handle = tokio::spawn(async move {
        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let raw = sample.payload().to_bytes().to_vec();

                    // --- Validate header ---
                    if raw.len() < HEADER_SIZE {
                        eprintln!(
                            "[SpatialMap] Payload too small: {} bytes (need >= {})",
                            raw.len(),
                            HEADER_SIZE
                        );
                        continue;
                    }

                    let width = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
                    let height = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
                    let expected_len = HEADER_SIZE + width * height;

                    if raw.len() != expected_len {
                        eprintln!(
                            "[SpatialMap] Bad payload: expected {} bytes for {}×{}, got {}",
                            expected_len, width, height, raw.len()
                        );
                        continue;
                    }

                    // --- Frame gate (AtomicBool CAS) ---
                    // Only proceed if the frontend is NOT busy.
                    // compare_exchange: if current == false, set to true atomically.
                    if state
                        .is_frontend_busy
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        // Store the validated payload (header + mask)
                        *state.pending_frame.lock().unwrap() = Some(raw);

                        // Lightweight signal — no payload in the event itself
                        let _ = app.emit("spatial-frame-ready", ());
                    }
                    // else: frontend is busy → frame is dropped (no queue)
                }
                Err(e) => {
                    eprintln!("[SpatialMap] Zenoh recv error: {e}");
                    break;
                }
            }
        }
    });

    Ok(handle)
}

// ===========================================================================
// TAURI COMMANDS
// ===========================================================================

/// Fetch the latest spatial map frame as raw binary.
///
/// Returns the full binary payload (8-byte header + mask) as an `ArrayBuffer`
/// on the frontend side. Returns an empty response if no frame is pending.
///
/// Wire format received by JS:
/// ```text
/// [0..4)  u32  width   (LE)
/// [4..8)  u32  height  (LE)
/// [8..)   u8[] mask    (width × height)
/// ```
#[tauri::command]
pub fn fetch_spatial_frame(
    state: tauri::State<'_, Arc<SpatialMapState>>,
) -> tauri::ipc::Response {
    let frame = state.pending_frame.lock().unwrap().take();
    match frame {
        Some(data) => tauri::ipc::Response::new(data),
        None => tauri::ipc::Response::new(Vec::<u8>::new()),
    }
}

/// Signal that the frontend has finished rendering the current frame.
///
/// This clears the `IS_FRONTEND_BUSY` gate, allowing the Zenoh subscriber
/// to forward the next incoming frame.
#[tauri::command]
pub fn mark_frame_complete(state: tauri::State<'_, Arc<SpatialMapState>>) {
    state.is_frontend_busy.store(false, Ordering::Release);
}
