use anyhow::{Context, Result};
use base64::engine::general_purpose;
use base64::Engine;
use serde_json::Value;
use tokio::io::{AsyncReadExt, BufReader};
use tokio::process::ChildStdout;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Reads a full PNG image from the given async buffered reader.
/// Returns None if EOF is reached.
pub async fn read_png_frame(reader: &mut BufReader<ChildStdout>) -> Result<Option<Vec<u8>>> {
    let mut png = Vec::new();

    // PNG signature is 8 bytes
    let mut signature = [0u8; 8];
    match reader.read_exact(&mut signature).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None), // EOF reached
        Err(e) => return Err(e).context("Failed to read PNG signature"),
    }
    png.extend_from_slice(&signature);

    loop {
        // Read chunk length (4 bytes)
        let mut length_bytes = [0u8; 4];
        match reader.read_exact(&mut length_bytes).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // EOF, stop reading frame
            Err(e) => return Err(e).context("Failed to read PNG chunk length"),
        }
        png.extend_from_slice(&length_bytes);
        let length = u32::from_be_bytes(length_bytes) as usize;

        // Read chunk type (4 bytes)
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type).await.context("Failed to read PNG chunk type")?;
        png.extend_from_slice(&chunk_type);

        // Read chunk data
        let mut chunk_data = vec![0u8; length];
        reader.read_exact(&mut chunk_data).await.context("Failed to read PNG chunk data")?;
        png.extend_from_slice(&chunk_data);

        // Read chunk CRC (4 bytes)
        let mut crc = [0u8; 4];
        reader.read_exact(&mut crc).await.context("Failed to read PNG chunk CRC")?;
        png.extend_from_slice(&crc);

        if &chunk_type == b"IEND" {
            break;
        }
    }

    Ok(Some(png))
}
/// NOTE:
/// This is used only for image/batch IPC.
/// Video paths must NEVER use base64 due to overhead.

/// Decodes a base64 string array from a JSON value by key,
/// returning a vector of decoded byte buffers (`Vec<u8>`).
///
/// This is the sequential version.
pub fn decode_base64_list(value: &Value, key: &str) -> Result<Vec<Vec<u8>>> {
    let arr = value
        .get(key)
        .and_then(|v| v.as_array())
        .context(format!("Missing or invalid '{}' array", key))?;

    arr.iter()
        .map(|v| {
            let s = v.as_str().context("Invalid base64 string in array")?;
            general_purpose::STANDARD
                .decode(s)
                .context("Failed to decode base64 string")
        })
        .collect()
}

/// Decodes a base64 string array from a JSON value by key,
/// returning a vector of decoded byte buffers (`Vec<u8>`).
///
/// This is the parallel version using Rayon.
/// Enabled only when `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub fn decode_base64_list_parallel(value: &Value, key: &str) -> Result<Vec<Vec<u8>>> {
    let arr = value
        .get(key)
        .and_then(|v| v.as_array())
        .context(format!("Missing or invalid '{}' array", key))?;

    arr.par_iter()
        .map(|v| {
            let s = v.as_str().context("Invalid base64 string in array")?;
            general_purpose::STANDARD
                .decode(s)
                .context("Failed to decode base64 string")
        })
        .collect()
}

pub fn generate_unique_path(input: &str) -> String {
    let input_path = Path::new(input);
    let parent = input_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let extension = input_path.extension().unwrap_or_default().to_string_lossy();

    let mut counter = 0;
    loop {
        let suffix = if counter == 0 {
            ".upscale4x".to_string()
        } else {
            format!(".upscale4x({})", counter)
        };

        let new_filename = format!("{}{}.{}", stem, suffix, extension);
        let target_path = parent.join(new_filename);

        if !target_path.exists() {
            return target_path.to_string_lossy().to_string();
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const EXAMPLE_BASE64_1: &str = "aGVsbG8="; // "hello"
    const EXAMPLE_BASE64_2: &str = "d29ybGQ="; // "world"

    #[test]
    fn test_decode_base64_list_success() {
        let value = json!({
            "data": [EXAMPLE_BASE64_1, EXAMPLE_BASE64_2]
        });

        let decoded = decode_base64_list(&value, "data").expect("Decoding failed");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], b"hello");
        assert_eq!(decoded[1], b"world");
    }

    #[test]
    fn test_decode_base64_list_invalid_base64() {
        let value = json!({
            "data": ["not_base64!!"]
        });

        let result = decode_base64_list(&value, "data");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_base64_list_missing_key() {
        let value = json!({});

        let result = decode_base64_list(&value, "data");
        assert!(result.is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_decode_base64_list_parallel_success() {
        let value = json!({
            "data": [EXAMPLE_BASE64_1, EXAMPLE_BASE64_2]
        });

        let decoded = decode_base64_list_parallel(&value, "data").expect("Parallel decoding failed");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], b"hello");
        assert_eq!(decoded[1], b"world");
    }
}