use dirs::data_local_dir;
use std::{collections::HashSet, env, fs, path::PathBuf};

#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub scale: u32,
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

pub fn list_models() -> Vec<ModelInfo> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();

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
        // println!("Scanning for models in: {:?}", root); // Uncomment for debugging

        if let Ok(entries) = fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                // Determine scale from filename (default to 4 if not found)
                let scale = if name.to_lowercase().contains("x2") {
                    2
                } else if name.to_lowercase().contains("x8") {
                    8
                } else {
                    4
                };

                // Check 1: Flat file (e.g., weights/RealESRGAN_x4plus.pth)
                if path.is_file() && name.ends_with(".pth") {
                    if !seen.contains(&name) {
                        models.push(ModelInfo {
                            id: name.clone(),
                            scale,
                        });
                        seen.insert(name);
                    }
                }
                // Check 2: Subfolder (e.g., weights/RealESRGAN_x4plus/RealESRGAN_x4plus.pth)
                else if path.is_dir() {
                    let nested_pth = path.join(format!("{}.pth", name));
                    if nested_pth.exists() && !seen.contains(&name) {
                        models.push(ModelInfo {
                            id: name.clone(),
                            scale,
                        });
                        seen.insert(name);
                    }
                }
            }
        }
    }

    models
}
