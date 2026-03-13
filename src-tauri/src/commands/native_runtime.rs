use std::path::{Path, PathBuf};

#[cfg(feature = "native_engine")]
use std::sync::OnceLock;

#[cfg(all(feature = "native_engine", windows))]
unsafe extern "system" {
    fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
}

#[cfg(all(feature = "native_engine", windows))]
fn preload_windows_dll(path: &Path) {
    use std::os::windows::ffi::OsStrExt;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: Calling system loader with a null-terminated UTF-16 path.
    let _ = unsafe { LoadLibraryW(wide.as_ptr()) };
}

pub(crate) fn workspace_root() -> Option<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .ok()
}

fn prepend_path_dirs(dirs: &[PathBuf]) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut paths: Vec<PathBuf> = std::env::split_paths(&current).collect();
    for dir in dirs.iter().filter(|d| d.is_dir()) {
        if !paths.iter().any(|p| p == dir) {
            paths.insert(0, dir.clone());
        }
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        // SAFETY: process-local env mutation before worker/process launches.
        unsafe { std::env::set_var("PATH", joined) };
    }
}

pub fn configure_repo_tool_runtime_path() {
    let runtime_paths = resolve_native_runtime_paths(workspace_root().as_deref(), None);
    prepend_path_dirs(&runtime_paths.path_additions);
}

pub(crate) fn find_file_under(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > max_depth {
            continue;
        }
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
                return Some(path);
            }
            if path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub(crate) struct NativeRuntimePaths {
    #[cfg(feature = "native_engine")]
    pub ffmpeg_cmd: String,
    #[cfg(feature = "native_engine")]
    pub ffprobe_cmd: String,
    pub path_additions: Vec<PathBuf>,
    #[cfg(feature = "native_engine")]
    pub tensorrt_bin: Option<PathBuf>,
}

pub(crate) fn resolve_native_runtime_paths(
    workspace_root: Option<&Path>,
    extra_path_dir: Option<&Path>,
) -> NativeRuntimePaths {
    let ffmpeg_exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let ffprobe_exe = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };

    let mut ffmpeg_bin: Option<PathBuf> = None;
    let mut tensorrt_bin: Option<PathBuf> = None;

    if let Some(root) = workspace_root {
        let known_ffmpeg = [
            root.join("third_party").join("ffmpeg").join("bin"),
            root.join("third_party").join("ffmpeg"),
        ];
        for dir in &known_ffmpeg {
            if dir.join(ffmpeg_exe).exists() && dir.join(ffprobe_exe).exists() {
                ffmpeg_bin = Some(dir.clone());
                break;
            }
        }

        if ffmpeg_bin.is_none() {
            for scan_root in [root.join("artifacts"), root.join("third_party")] {
                if !scan_root.exists() {
                    continue;
                }
                if let Some(ffmpeg_path) = find_file_under(&scan_root, ffmpeg_exe, 5) {
                    let bin_dir = ffmpeg_path.parent().map(|p| p.to_path_buf());
                    if let Some(bin_dir) = bin_dir {
                        if bin_dir.join(ffprobe_exe).exists() {
                            ffmpeg_bin = Some(bin_dir);
                            break;
                        }
                    }
                }
            }
        }

        let known_trt = root.join("third_party").join("tensorrt");
        if known_trt.join("nvinfer_10.dll").exists() {
            tensorrt_bin = Some(known_trt);
        } else if let Some(nvinfer) =
            find_file_under(&root.join("third_party"), "nvinfer_10.dll", 4)
        {
            tensorrt_bin = nvinfer.parent().map(|p| p.to_path_buf());
        }
    }

    let mut path_additions = Vec::new();
    let mut push_unique = |path: PathBuf| {
        if path.exists() && !path_additions.iter().any(|p| p == &path) {
            path_additions.push(path);
        }
    };

    if let Some(dir) = &tensorrt_bin {
        push_unique(dir.clone());
    }
    if let Some(dir) = &ffmpeg_bin {
        push_unique(dir.clone());
    }
    if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
        push_unique(
            PathBuf::from(vcpkg_root)
                .join("installed")
                .join("x64-windows")
                .join("bin"),
        );
    }
    push_unique(PathBuf::from(r"C:\tools\vcpkg\installed\x64-windows\bin"));
    if let Some(extra_dir) = extra_path_dir {
        push_unique(extra_dir.to_path_buf());
    }

    #[cfg(feature = "native_engine")]
    let ffmpeg_cmd = ffmpeg_bin
        .as_ref()
        .map(|bin| bin.join(ffmpeg_exe).to_string_lossy().to_string())
        .unwrap_or_else(|| "ffmpeg".to_string());
    #[cfg(feature = "native_engine")]
    let ffprobe_cmd = ffmpeg_bin
        .as_ref()
        .map(|bin| bin.join(ffprobe_exe).to_string_lossy().to_string())
        .unwrap_or_else(|| "ffprobe".to_string());

    NativeRuntimePaths {
        #[cfg(feature = "native_engine")]
        ffmpeg_cmd,
        #[cfg(feature = "native_engine")]
        ffprobe_cmd,
        path_additions,
        #[cfg(feature = "native_engine")]
        tensorrt_bin,
    }
}

#[cfg(feature = "native_engine")]
struct NativeRuntimeEnv {
    ffmpeg_cmd: String,
    ffprobe_cmd: String,
}

#[cfg(feature = "native_engine")]
static NATIVE_RUNTIME_ENV: OnceLock<NativeRuntimeEnv> = OnceLock::new();

#[cfg(feature = "native_engine")]
fn discover_native_runtime_env() -> NativeRuntimeEnv {
    let runtime_paths = resolve_native_runtime_paths(workspace_root().as_deref(), None);

    if let Some(dir) = &runtime_paths.tensorrt_bin {
        #[cfg(windows)]
        {
            // Preload core TensorRT DLLs from absolute paths so ORT provider
            // registration does not depend on PATH search behavior.
            for dll in [
                "nvinfer_10.dll",
                "nvinfer_plugin_10.dll",
                "nvinfer_dispatch_10.dll",
                "nvonnxparser_10.dll",
                "cudnn64_9.dll",
            ] {
                let p = dir.join(dll);
                if p.exists() {
                    preload_windows_dll(&p);
                }
            }
        }

        // Stage TensorRT runtime DLLs next to the executable as a robust loader path.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let src = entry.path();
                        let is_dll = src
                            .extension()
                            .and_then(|e| e.to_str())
                            .is_some_and(|e| e.eq_ignore_ascii_case("dll"));
                        if !is_dll {
                            continue;
                        }
                        let Some(name) = src.file_name() else {
                            continue;
                        };
                        let dst = exe_dir.join(name);
                        if !dst.exists() {
                            let _ = std::fs::copy(&src, &dst);
                        }
                    }
                }
            }
        }
    }

    prepend_path_dirs(&runtime_paths.path_additions);

    NativeRuntimeEnv {
        ffmpeg_cmd: runtime_paths.ffmpeg_cmd,
        ffprobe_cmd: runtime_paths.ffprobe_cmd,
    }
}

#[cfg(feature = "native_engine")]
fn native_runtime_env() -> &'static NativeRuntimeEnv {
    NATIVE_RUNTIME_ENV.get_or_init(discover_native_runtime_env)
}

#[cfg(feature = "native_engine")]
pub(crate) fn configure_native_runtime_env() -> String {
    native_runtime_env().ffmpeg_cmd.clone()
}

#[cfg(feature = "native_engine")]
pub(crate) fn configure_native_probe_cmd() -> String {
    native_runtime_env().ffprobe_cmd.clone()
}
