use dirs::data_local_dir;
use std::{collections::HashSet, env, fs, path::PathBuf};

/// Represents a discovered model with its identifier and scale factor.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    /// Canonical identifier: e.g., "RCAN_x4", "EDSR_x3"
    pub id: String,
    /// Upscaling factor (2, 3, 4, or 8)
    pub scale: u32,
    /// The actual filename on disk (for backend resolution)
    pub filename: String,
}

/// Supported model families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    RCAN,
    EDSR,
    RealESRGAN,
    Unknown,
}

/// Parse model family from filename
fn parse_model_family(name: &str) -> ModelFamily {
    let upper = name.to_uppercase();
    if upper.starts_with("RCAN") {
        ModelFamily::RCAN
    } else if upper.starts_with("EDSR") {
        ModelFamily::EDSR
    } else if upper.contains("REALESRGAN") || upper.contains("ESRGAN") {
        ModelFamily::RealESRGAN
    } else {
        ModelFamily::Unknown
    }
}

/// Extract scale factor from filename using explicit pattern matching
fn extract_scale(name: &str) -> Option<u32> {
    let lower = name.to_lowercase();

    // Check explicit patterns: x2, x3, x4, x8, _2x, _3x, _4x, _8x
    for scale in [2u32, 3, 4, 8] {
        let patterns = [
            format!("x{}", scale),
            format!("_{}x", scale),
            format!("{}x_", scale),
        ];
        for pattern in &patterns {
            if lower.contains(pattern) {
                return Some(scale);
            }
        }
    }

    None
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

/// Check if file has a valid weight extension (.pth or .pt)
fn is_weight_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.ends_with(".pth") || lower.ends_with(".pt")
}

/// Generate canonical model ID from filename and scale
fn generate_canonical_id(family: ModelFamily, scale: u32, filename: &str) -> String {
    match family {
        ModelFamily::RCAN => format!("RCAN_x{}", scale),
        ModelFamily::EDSR => format!("EDSR_x{}", scale),
        ModelFamily::RealESRGAN => {
            // Keep original name for RealESRGAN to preserve anime/plus variants
            filename.replace(".pth", "").replace(".pt", "")
        }
        ModelFamily::Unknown => filename.replace(".pth", "").replace(".pt", ""),
    }
}

pub fn list_models() -> Vec<ModelInfo> {
    let mut models = Vec::new();
    // Track seen canonical IDs to avoid duplicates (prefer first found)
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

                // Handle flat weight files (e.g., weights/RCAN_2x.pt)
                if path.is_file() && is_weight_file(&filename) {
                    if let Some(model) = process_weight_file(&filename, &seen_ids) {
                        seen_ids.insert(model.id.clone());
                        models.push(model);
                    }
                }
                // Handle subdirectories (e.g., weights/RCAN_x4/RCAN_x4.pth)
                else if path.is_dir() {
                    // Look for .pth or .pt file inside
                    if let Ok(sub_entries) = fs::read_dir(&path) {
                        for sub_entry in sub_entries.flatten() {
                            let sub_name = sub_entry.file_name().to_string_lossy().to_string();
                            if is_weight_file(&sub_name) {
                                if let Some(model) = process_weight_file(&sub_name, &seen_ids) {
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

    // Sort models by family, then by scale for consistent ordering
    models.sort_by(|a, b| {
        let family_a = parse_model_family(&a.id);
        let family_b = parse_model_family(&b.id);
        match (family_a, family_b) {
            (ModelFamily::RCAN, ModelFamily::EDSR) => std::cmp::Ordering::Less,
            (ModelFamily::EDSR, ModelFamily::RCAN) => std::cmp::Ordering::Greater,
            (ModelFamily::RealESRGAN, _) => std::cmp::Ordering::Greater,
            (_, ModelFamily::RealESRGAN) => std::cmp::Ordering::Less,
            _ => a.scale.cmp(&b.scale),
        }
    });

    models
}

/// Process a single weight file and return ModelInfo if valid
fn process_weight_file(filename: &str, seen_ids: &HashSet<String>) -> Option<ModelInfo> {
    let family = parse_model_family(filename);
    let scale = extract_scale(filename)?;

    // Only process known model families with valid scales
    if family == ModelFamily::Unknown {
        return None;
    }

    // Validate scale is in supported range
    if ![2, 3, 4, 8].contains(&scale) {
        return None;
    }

    let canonical_id = generate_canonical_id(family, scale, filename);

    // Skip if we've already seen this canonical ID
    if seen_ids.contains(&canonical_id) {
        return None;
    }

    Some(ModelInfo {
        id: canonical_id,
        scale,
        filename: filename.to_string(),
    })
}
