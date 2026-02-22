//! Python environment resolution — locate and spawn the AI sidecar.
//!
//! Resolves the Python binary and `shm_worker.py` script from:
//! 1. The installed VideoForge Python runtime (production path).
//! 2. A dev venv path from `VIDEOFORGE_DEV_PYTHON` env var.
//! 3. Well-known dev venv locations (dev fallback).

use anyhow::{anyhow, Result};
use dirs::data_local_dir;
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Child;

// Global tracker for spawned Python PIDs (for targeted cleanup via reset_engine).
lazy_static::lazy_static! {
    pub static ref PYTHON_PIDS: Arc<std::sync::Mutex<HashSet<u32>>> =
        Arc::new(std::sync::Mutex::new(HashSet::new()));
}

/// Canonical base arguments for spawning `python/shm_worker.py`.
pub struct BaseWorkerArgs<'a> {
    pub script_path: &'a str,
    pub port: u16,
    pub parent_pid: u32,
    pub precision: &'a str,
}

/// Optional worker capability flags.
///
/// Parsed/plumbed only; not active yet.
#[derive(Debug, Clone, Default)]
pub struct WorkerCaps {
    pub log_level: Option<String>,
    pub use_typed_ipc: bool,
    pub use_shm_proto_v2: bool,
    pub use_events: bool,
    pub prealloc_tensors: bool,
    pub deterministic: bool,
}

/// Build the canonical argv passed to the Python SHM worker.
pub fn build_worker_argv(base: &BaseWorkerArgs<'_>, caps: &WorkerCaps) -> Vec<String> {
    let mut argv = vec![
        base.script_path.to_string(),
        "--port".to_string(),
        base.port.to_string(),
        "--parent-pid".to_string(),
        base.parent_pid.to_string(),
        "--precision".to_string(),
        base.precision.to_string(),
    ];

    // Parsed/plumbed only; not active yet.
    if let Some(level) = &caps.log_level {
        argv.push("--log-level".to_string());
        argv.push(level.clone());
    }
    if caps.use_typed_ipc {
        argv.push("--use-typed-ipc".to_string());
    }
    if caps.use_shm_proto_v2 {
        argv.push("--shm-proto-v2".to_string());
    }
    if caps.use_events {
        argv.push("--use-events".to_string());
    }
    if caps.prealloc_tensors {
        argv.push("--prealloc-tensors".to_string());
    }
    if caps.deterministic {
        argv.push("--deterministic".to_string());
    }

    argv
}

// ─── Path resolution ──────────────────────────────────────────────────────────

pub fn get_python_install_dir() -> PathBuf {
    let mut path = data_local_dir().expect("Could not find AppData/Local");
    path.push("VideoForge");
    path.push("python");
    path
}

/// Resolve the Python binary and `shm_worker.py` script paths.
///
/// Search order:
/// 1. Installed VideoForge Python runtime under AppData.
/// 2. `VIDEOFORGE_DEV_PYTHON` env var pointing to a Python executable.
/// 3. Known dev venv locations relative to CWD.
pub fn resolve_python_environment() -> Result<(String, String)> {
    let install_dir = get_python_install_dir();
    let installed_python = install_dir.join("python.exe");
    let installed_script = install_dir.join("shm_worker.py");

    if installed_python.exists() && installed_script.exists() {
        tracing::info!(
            python = %installed_python.display(),
            script = %installed_script.display(),
            "Using installed Python environment"
        );
        return Ok((
            installed_python.to_string_lossy().to_string(),
            installed_script.to_string_lossy().to_string(),
        ));
    }

    // Dev fallback — VIDEOFORGE_DEV_PYTHON env var or known venv locations.
    let dev_python_paths: Vec<PathBuf> = if let Ok(dev_python) = env::var("VIDEOFORGE_DEV_PYTHON") {
        vec![PathBuf::from(dev_python)]
    } else {
        vec![
            PathBuf::from(r"C:\Users\Calvin\VideoForge\venv310\Scripts\python.exe"),
            PathBuf::from(r".\venv\Scripts\python.exe"),
            PathBuf::from(r"..\venv\Scripts\python.exe"),
        ]
    };

    for local_venv in dev_python_paths {
        if local_venv.exists() {
            if let Ok(cwd) = env::current_dir() {
                let script_local = cwd.join("python").join("shm_worker.py");
                if script_local.exists() {
                    tracing::info!(
                        python = %local_venv.display(),
                        script = %script_local.display(),
                        "Using dev Python environment"
                    );
                    return Ok((
                        local_venv.to_string_lossy().to_string(),
                        script_local.to_string_lossy().to_string(),
                    ));
                }
                let script_up = cwd
                    .parent()
                    .unwrap_or(Path::new(".."))
                    .join("python")
                    .join("shm_worker.py");
                if script_up.exists() {
                    return Ok((
                        local_venv.to_string_lossy().to_string(),
                        script_up.to_string_lossy().to_string(),
                    ));
                }
            }
        }
    }

    Err(anyhow!(
        "AI Engine not found. Install it via the app installer or set \
         VIDEOFORGE_DEV_PYTHON to your Python executable path."
    ))
}

// ─── Free port ───────────────────────────────────────────────────────────────

/// Bind to port 0 to get an OS-assigned free port for the Zenoh listener.
pub fn get_free_port() -> u16 {
    use std::net::TcpListener;
    if let Ok(listener) = TcpListener::bind("127.0.0.1:0") {
        if let Ok(addr) = listener.local_addr() {
            return addr.port();
        }
    }
    7447 // fallback
}

// ─── Process guard ───────────────────────────────────────────────────────────

/// RAII guard that kills the Python child process when dropped.
///
/// Call [`ProcessGuard::disarm`] when the pipeline completes normally to
/// take ownership and perform a graceful shutdown instead.
pub struct ProcessGuard {
    child: Option<Child>,
}

impl ProcessGuard {
    pub fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    /// Disarm the guard and take ownership of the child for graceful shutdown.
    pub fn disarm(&mut self) -> Option<Child> {
        self.child.take()
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            tracing::warn!("ProcessGuard: killing Python process on drop (cleanup path)");
            let _ = child.start_kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{build_worker_argv, BaseWorkerArgs, WorkerCaps};

    #[test]
    fn default_worker_argv_matches_previous_spawn_order() {
        let base = BaseWorkerArgs {
            script_path: "python/shm_worker.py",
            port: 7447,
            parent_pid: 12345,
            precision: "fp16",
        };

        let argv = build_worker_argv(&base, &WorkerCaps::default());

        let expected = vec![
            "python/shm_worker.py".to_string(),
            "--port".to_string(),
            "7447".to_string(),
            "--parent-pid".to_string(),
            "12345".to_string(),
            "--precision".to_string(),
            "fp16".to_string(),
        ];

        assert_eq!(argv, expected);
    }

    #[test]
    fn shm_proto_v2_flag_is_opt_in() {
        let base = BaseWorkerArgs {
            script_path: "python/shm_worker.py",
            port: 7447,
            parent_pid: 12345,
            precision: "fp16",
        };
        let caps = WorkerCaps {
            use_shm_proto_v2: true,
            ..WorkerCaps::default()
        };

        let argv = build_worker_argv(&base, &caps);
        assert!(argv.iter().any(|a| a == "--shm-proto-v2"));
    }
}
