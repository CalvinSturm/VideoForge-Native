use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::watch;
use crate::python_worker::{PythonWorker, Request, Response};
use crate::utils::decode_base64_list;

/// Run the main request loop reading JSON lines from stdin.
/// Dispatch commands to PythonWorker and send responses to stdout.
/// Supports cancellation and optional progress callback.
pub async fn run_request_loop(
    worker: &mut PythonWorker,
    mut cancel_rx: watch::Receiver<bool>,
    mut on_progress: Option<impl FnMut(&str) + Send>,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if *cancel_rx.borrow() {
            eprintln!("Request loop cancelled.");
            break;
        }

        let req: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                // Structured error response for parse errors
                let err_resp = Response {
                    id: "".to_string(),
                    status: "error".to_string(),
                    result: json!(null),
                    error: Some(format!("Failed to parse request JSON: {}", e)),
                };
                println!("{}", serde_json::to_string(&err_resp)?);
                continue;
            }
        };

        let req_id = req.id.clone();

        // Dispatch commands
        let resp_result = match req.command.as_str() {
            "cancel" => {
                if let Some(target_id) = req.params.get("id").and_then(|v| v.as_str()) {
                    match worker.cancel(target_id) {
                        Ok(_) => Ok(Response {
                            id: req_id.clone(),
                            status: "ok".to_string(),
                            result: json!(null),
                            error: None,
                        }),
                        Err(e) => Err(e.into()),
                    }
                } else {
                    Err(anyhow::anyhow!("Missing 'id' parameter for cancel command"))
                }
            }

            "upscale_image" => {
                if let Some(b64) = req.params.get("image_base64").and_then(|v| v.as_str()) {
                    let request = Request {
                        id: req_id.clone(),
                        command: "upscale_image".to_string(),
                        params: json!({"image_base64": b64}),
                    };
                    worker.send(request, on_progress.as_mut(), cancel_rx.clone()).await
                } else {
                    Err(anyhow::anyhow!("Missing 'image_base64' parameter"))
                }
            }

            "upscale_frame" => {
                let frame_index = req.params.get("frame_index").and_then(|v| v.as_u64());
                let image_base64 = req.params.get("image_base64").and_then(|v| v.as_str());

                if frame_index.is_none() || image_base64.is_none() {
                    Err(anyhow::anyhow!("Missing 'frame_index' or 'image_base64' parameter"))
                } else {
                    let request = Request {
                        id: req_id.clone(),
                        command: "upscale_frame".to_string(),
                        params: json!({
                            "frame_index": frame_index.unwrap(),
                            "image_base64": image_base64.unwrap(),
                        }),
                    };
                    worker.send(request, on_progress.as_mut(), cancel_rx.clone()).await
                }
            }

            "upscale_batch" => {
                match decode_base64_list(&req.params, "images_base64") {
                    Ok(images) => {
                        let request = Request {
                            id: req_id.clone(),
                            command: "upscale_batch".to_string(),
                            params: json!({ "images_base64": images }),
                        };
                        worker.send(request, on_progress.as_mut(), cancel_rx.clone()).await
                    }
                    Err(e) => Err(anyhow::anyhow!("Error decoding 'images_base64': {}", e)),
                }
            }

            "upscale_video" => {
                match decode_base64_list(&req.params, "frames_base64") {
                    Ok(frames) => {
                        let request = Request {
                            id: req_id.clone(),
                            command: "upscale_video".to_string(),
                            params: json!({ "frames_base64": frames }),
                        };
                        worker.send(request, on_progress.as_mut(), cancel_rx.clone()).await
                    }
                    Err(e) => Err(anyhow::anyhow!("Error decoding 'frames_base64': {}", e)),
                }
            }

            "load_model" => {
                if let Some(model_name) = req.params.get("model_name").and_then(|v| v.as_str()) {
                    let request = Request {
                        id: req_id.clone(),
                        command: "load_model".to_string(),
                        params: json!({"model_name": model_name}),
                    };
                    worker.send(request, on_progress.as_mut(), cancel_rx.clone()).await
                } else {
                    Err(anyhow::anyhow!("Missing 'model_name' parameter"))
                }
            }

            cmd => {
                Err(anyhow::anyhow!("Unknown command: {}", cmd))
            }
        };

        match resp_result {
            Ok(response) => {
                println!("{}", serde_json::to_string(&response)?);
                if let Some(cb) = on_progress.as_mut() {
                    cb(&format!("Command '{}' completed (id: {})", req.command, req_id));
                }
            }
            Err(e) => {
                let err_resp = Response {
                    id: req_id.clone(),
                    status: "error".to_string(),
                    result: json!(null),
                    error: Some(format!("{}", e)),
                };
                println!("{}", serde_json::to_string(&err_resp)?);
                eprintln!("Error handling command '{}' (id: {}): {}", req.command, req_id, e);
            }
        }
    }

    Ok(())
}