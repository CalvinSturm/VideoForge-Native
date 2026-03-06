use dirs::data_local_dir;
use std::{collections::HashSet, env, fs, path::PathBuf};

/// Represents a discovered model with its identifier and scale factor.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    /// Identifier derived from filename (sans extension).
    /// This is what gets sent to the Python backend as `model_key`.
    pub id: String,
    /// Upscaling factor (2, 3, 4, or 8). Defaults to 4 if not detectable.
    pub scale: u32,
    /// The actual filename on disk (for backend resolution)
    pub filename: String,
    /// Weight format: "onnx" or "pytorch"
    pub format: String,
    /// Absolute path to the weight file on disk
    pub path: String,
}

/// Strip weight-file extensions from a filename.
fn strip_weight_ext(name: &str) -> String {
    name.replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
        .replace(".bin", "")
        .replace(".onnx", "")
}

/// Extract scale factor from filename using explicit pattern matching.
///
/// Recognises patterns:
///   `_x4`, `x4plus`, `4x_`, `^4x` (prefix), `_4x`, `_x4_`
fn extract_scale(name: &str) -> u32 {
    let lower = name.to_lowercase();

    // Check for explicit scale markers — order: most specific first
    for scale in [2u32, 3, 4, 8] {
        let patterns = [
            format!("_x{}", scale),    // RCAN_x4, EDSR_x3
            format!("x{}plus", scale), // RealESRGAN_x4plus
            format!("_{}x", scale),    // model_4x
            format!("{}x_", scale),    // 4x_model
            format!("{}x-", scale),    // 4x-SwinIR
        ];
        for pattern in &patterns {
            if lower.contains(pattern) {
                return scale;
            }
        }
    }

    // Check for leading digit-x prefix: "4xFFHQDAT", "2xModel"
    if lower.len() >= 2 {
        let first = lower.as_bytes()[0];
        let second = lower.as_bytes()[1];
        if second == b'x' && (first == b'2' || first == b'3' || first == b'4' || first == b'8') {
            return (first - b'0') as u32;
        }
    }

    // Default to 4x when scale is undetectable
    4
}

/// Helper to get the AppData installation path
fn get_installed_weights_dir() -> Option<PathBuf> {
    let mut path = data_local_dir()?;
    path.push("VideoForge");
    path.push("python");
    path.push("weights");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Check if file has a valid weight extension
fn is_weight_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.ends_with(".pth")
        || lower.ends_with(".pt")
        || lower.ends_with(".safetensors")
        || lower.ends_with(".bin")
        || lower.ends_with(".onnx")
}

/// Return a reason for weights we intentionally hide from normal discovery.
fn excluded_weight_reason(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        // Known-bad FP16 export: ORT rejects it during model load because the
        // graph mixes float and float16 tensor types inconsistently.
        "4xnomos2_hq_dat2_fp32.fp16.onnx" => Some("invalid_onnx_graph"),
        _ => None,
    }
}

pub fn list_models() -> Vec<ModelInfo> {
    let mut models = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    // 1. Define all possible places where 'weights' might be
    let mut search_paths = Vec::new();

    // A. Check AppData (Installed Engine)
    if let Some(p) = get_installed_weights_dir() {
        search_paths.push(p);
    }

    // B. Check Local Dev Paths
    if let Ok(cwd) = env::current_dir() {
        search_paths.push(cwd.join("weights")); // ./weights
        search_paths.push(cwd.join("python").join("weights")); // ./python/weights

        // If running inside src-tauri, look up
        if let Some(parent) = cwd.parent() {
            search_paths.push(parent.join("weights")); // ../weights
            search_paths.push(parent.join("python").join("weights")); // ../python/weights
        }
    }

    // 2. Scan all paths
    for root in search_paths {
        if !root.exists() {
            continue;
        }

        if let Ok(entries) = fs::read_dir(&root) {
            for entry in entries.flatten() {
                let path = entry.path();
                let filename = entry.file_name().to_string_lossy().to_string();

                // Skip non-weight files
                if path.is_file() && !is_weight_file(&filename) {
                    continue;
                }

                // Handle flat weight files (e.g., weights/RCAN_x4.pt)
                if path.is_file() && is_weight_file(&filename) {
                    if let Some(model) = process_weight_file(&path, &filename, &seen_ids) {
                        seen_ids.insert(model.id.clone());
                        models.push(model);
                    }
                }
                // Handle subdirectories (e.g., weights/SwinIR_x4/SwinIR_x4.pth)
                else if path.is_dir() {
                    if let Ok(sub_entries) = fs::read_dir(&path) {
                        for sub_entry in sub_entries.flatten() {
                            let sub_path = sub_entry.path();
                            let sub_name = sub_entry.file_name().to_string_lossy().to_string();
                            if is_weight_file(&sub_name) {
                                if let Some(model) =
                                    process_weight_file(&sub_path, &sub_name, &seen_ids)
                                {
                                    seen_ids.insert(model.id.clone());
                                    models.push(model);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort alphabetically by ID for consistent ordering
    models.sort_by(|a, b| a.id.to_lowercase().cmp(&b.id.to_lowercase()));

    models
}

/// Process a single weight file and return ModelInfo if valid
fn process_weight_file(
    full_path: &std::path::Path,
    filename: &str,
    seen_ids: &HashSet<String>,
) -> Option<ModelInfo> {
    if excluded_weight_reason(filename).is_some() {
        return None;
    }

    let id = strip_weight_ext(filename);
    if id.is_empty() {
        return None;
    }

    // Skip if we've already seen this ID
    if seen_ids.contains(&id) {
        return None;
    }

    let scale = extract_scale(filename);

    // Validate scale is in supported range
    if ![2, 3, 4, 8].contains(&scale) {
        return None;
    }

    let format = if filename.to_lowercase().ends_with(".onnx") {
        "onnx".to_string()
    } else {
        "pytorch".to_string()
    };

    Some(ModelInfo {
        id,
        scale,
        filename: filename.to_string(),
        format,
        path: full_path.to_string_lossy().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_known_bad_onnx_export() {
        let seen = HashSet::new();
        let model = process_weight_file(
            std::path::Path::new("weights/4xNomos2_hq_dat2_fp32.fp16.onnx"),
            "4xNomos2_hq_dat2_fp32.fp16.onnx",
            &seen,
        );
        assert!(model.is_none());
    }

    #[test]
    fn keeps_valid_fp32_transformer_export() {
        let seen = HashSet::new();
        let model = process_weight_file(
            std::path::Path::new("weights/4xNomos2_hq_dat2_fp32.onnx"),
            "4xNomos2_hq_dat2_fp32.onnx",
            &seen,
        )
        .expect("valid FP32 export should remain discoverable");
        assert_eq!(model.scale, 4);
        assert_eq!(model.format, "onnx");
    }
}
