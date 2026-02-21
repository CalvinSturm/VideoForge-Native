use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Re-run if protocol definition changes
    println!("cargo:rerun-if-changed=../ipc/shm_protocol.json");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("shm_constants.rs");

    let protocol_json = fs::read_to_string("../ipc/shm_protocol.json")
        .expect("Failed to read shm_protocol.json");
    
    let json: serde_json::Value = serde_json::from_str(&protocol_json)
        .expect("Failed to parse shm_protocol.json");

    let mut content = String::from("// Auto-generated SHM constants from ipc/shm_protocol.json\n\n");

    // Helper to extract values
    let ring_size = json["ring_size"].as_u64().expect("ring_size missing");
    content.push_str(&format!("pub const RING_SIZE: usize = {};\n", ring_size));

    let magic = json["magic"].as_str().expect("magic missing");
    content.push_str(&format!("pub const MAGIC: &[u8; 8] = b\"{}\";\n", magic));

    let version = json["version"].as_u64().expect("version missing");
    content.push_str(&format!("pub const SHM_VERSION: u32 = {};\n", version));

    let pixel_fmt = json["pixel_format_rgb24"].as_u64().expect("pixel_format_rgb24 missing");
    content.push_str(&format!("pub const PIXEL_FORMAT_RGB24: u32 = {};\n", pixel_fmt));

    let global_hdr = json["global_header_size"].as_u64().expect("global_header_size missing");
    content.push_str(&format!("pub const GLOBAL_HEADER_SIZE: usize = {};\n", global_hdr));

    let slot_hdr = json["slot_header_size"].as_u64().expect("slot_header_size missing");
    content.push_str(&format!("pub const SLOT_HEADER_SIZE: usize = {};\n", slot_hdr));

    // Slot states
    if let Some(states) = json["slot_states"].as_object() {
        for (name, val) in states {
            let val_u32 = val.as_u64().expect("state value invalid");
            content.push_str(&format!("pub const SLOT_{}: u32 = {};\n", name, val_u32));
        }
    }

    // Offsets
    if let Some(offsets) = json["offsets"].as_object() {
        for (name, val) in offsets {
            let val_usize = val.as_u64().expect("offset value invalid");
            content.push_str(&format!("pub const {}_OFFSET: usize = {};\n", name.to_uppercase(), val_usize));
        }
    }
    
    // Derived constant
    let slot_hdr_region = slot_hdr * ring_size;
    content.push_str(&format!("pub const SLOT_HEADER_REGION: usize = {};\n", slot_hdr_region));
    
    let header_region_size = global_hdr + slot_hdr_region;
    content.push_str(&format!("pub const HEADER_REGION_SIZE: usize = {};\n", header_region_size));

    fs::write(&dest_path, content).unwrap();
    
    tauri_build::build();
}
