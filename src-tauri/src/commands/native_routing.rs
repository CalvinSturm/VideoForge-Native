#[cfg(feature = "native_engine")]
use std::path::{Path, PathBuf};

#[cfg(feature = "native_engine")]
use crate::run_manifest::{
    maybe_finalize_run_artifacts, maybe_write_run_manifest, run_artifacts_enabled_from_env,
    RunArtifactFinalizeInputs, RunManifestInputs, WorkerCapsSnapshot,
};
#[cfg(feature = "native_engine")]
use crate::runtime_truth::{
    log_run_observed_metrics, log_runtime_config_snapshot, NativeRuntimeConfigExtension,
    NativeRuntimeMetricsExtension, RunObservedMetrics, RunStatus, RuntimeConfigExtensions,
    RuntimeConfigSnapshot, RuntimeEngineFamily, RuntimeFallbackInfo, RuntimeMetricsExtensions,
    RuntimeSnapshotKind,
};

#[cfg(feature = "native_engine")]
use crate::commands::native_direct_pipeline::run_native_pipeline;
#[cfg(feature = "native_engine")]
use crate::commands::native_engine::{
    native_engine_direct_enabled, native_engine_runtime_enabled, native_temp_token,
    NativePerfReport, NativeUpscaleError, NativeUpscaleResult,
};
#[cfg(feature = "native_engine")]
use crate::commands::native_probe::NativeDirectPlan;

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub(crate) struct NativeExecutionRoute {
    pub requested_executor: &'static str,
    pub executed_executor: &'static str,
    pub direct_attempted: bool,
    pub fallback_reason_code: Option<String>,
    pub fallback_reason_message: Option<String>,
}

#[cfg(feature = "native_engine")]
impl NativeExecutionRoute {
    pub(crate) fn direct() -> Self {
        Self {
            requested_executor: "direct",
            executed_executor: "direct",
            direct_attempted: true,
            fallback_reason_code: None,
            fallback_reason_message: None,
        }
    }

    pub(crate) fn cli_requested() -> Self {
        Self {
            requested_executor: "cli",
            executed_executor: "cli",
            direct_attempted: false,
            fallback_reason_code: None,
            fallback_reason_message: None,
        }
    }

    pub(crate) fn cli_fallback(err: &NativeUpscaleError) -> Self {
        Self {
            requested_executor: "direct",
            executed_executor: "cli",
            direct_attempted: true,
            fallback_reason_code: Some(err.code.clone()),
            fallback_reason_message: Some(err.message.clone()),
        }
    }
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeRequestedExecutor {
    Direct,
    Cli,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeOutputPathStyle {
    DirectTemp,
    CliStable,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
struct NativeCliInvocation {
    output_path: String,
    args: Vec<String>,
}

#[cfg(feature = "native_engine")]
struct NativeCliExecutionPlan {
    output_path: String,
    prepared_command: crate::commands::rave::PreparedRaveCliCommand,
}

#[cfg(feature = "native_engine")]
#[derive(Debug, Clone)]
pub(crate) struct NativeJobSpec {
    pub run_id: String,
    pub input_path: String,
    pub requested_output_path: String,
    pub model_path: String,
    pub scale: u32,
    pub precision: String,
    pub preserve_audio: bool,
    pub max_batch: u32,
    pub trt_cache_enabled: bool,
    pub trt_cache_dir: Option<String>,
}

#[cfg(feature = "native_engine")]
impl NativeJobSpec {
    pub(crate) fn resolve(
        input_path: String,
        output_path: String,
        model_path: String,
        requested_scale: u32,
        precision: Option<String>,
        audio: Option<bool>,
        requested_max_batch: Option<u32>,
    ) -> Result<Self, String> {
        let make_err = |code: &str, msg: &str| {
            serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap()
        };

        if !Path::new(&model_path).exists() {
            return Err(make_err(
                "MODEL_NOT_FOUND",
                &format!("Model not found: {}", model_path),
            ));
        }

        let batch_policy = crate::models::native_batch_policy_for_path(&model_path);
        let max_batch = requested_max_batch.unwrap_or(batch_policy.default_max_batch);
        if !(1..=8).contains(&max_batch) {
            return Err(make_err(
                "INVALID_BATCH",
                &format!("Invalid max_batch value '{max_batch}'. Must be in range 1-8."),
            ));
        }

        let effective_scale = infer_model_scale(&model_path).unwrap_or(requested_scale);
        if effective_scale != requested_scale {
            tracing::warn!(
                requested_scale,
                inferred_scale = effective_scale,
                model = %model_path,
                "Requested native scale does not match model filename; overriding to inferred model scale"
            );
        }
        if requested_max_batch.is_none() {
            tracing::info!(
                model = %model_path,
                default_max_batch = batch_policy.default_max_batch,
                max_validated_batch = batch_policy.max_validated_batch,
                "Applying model-aware native batching default"
            );
        } else {
            tracing::info!(
                model = %model_path,
                requested_max_batch = max_batch,
                default_max_batch = batch_policy.default_max_batch,
                max_validated_batch = batch_policy.max_validated_batch,
                "Using explicit native batching override"
            );
        }

        let (trt_cache_enabled, trt_cache_dir) = trt_cache_runtime(&model_path);

        Ok(Self {
            run_id: crate::ipc::protocol::next_request_id(),
            input_path,
            requested_output_path: output_path,
            model_path,
            scale: effective_scale,
            precision: precision.unwrap_or_else(|| "fp32".to_string()),
            preserve_audio: audio.unwrap_or(true),
            max_batch,
            trt_cache_enabled,
            trt_cache_dir,
        })
    }

    pub(crate) fn resolved_output_path(&self, style: NativeOutputPathStyle) -> String {
        if !self.requested_output_path.trim().is_empty() {
            return self.requested_output_path.clone();
        }

        match style {
            NativeOutputPathStyle::CliStable => self.default_cli_output_path(),
            NativeOutputPathStyle::DirectTemp => self.default_direct_output_path(),
        }
    }

    fn default_cli_output_path(&self) -> String {
        let p = Path::new(&self.input_path);
        let stem = p
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = p
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if ext.is_empty() {
            format!("{stem}_rave_upscaled.mp4")
        } else {
            self.input_path
                .replace(&format!(".{ext}"), "_rave_upscaled.mp4")
        }
    }

    fn default_direct_output_path(&self) -> String {
        let stem = Path::new(&self.input_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let dir = Path::new(&self.input_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let file_name = format!("{}_{}_native_upscaled.mp4", stem, native_temp_token());
        dir.join(file_name).to_string_lossy().to_string()
    }

    fn prepare_cli_invocation(&self) -> NativeCliInvocation {
        let output_path = self.resolved_output_path(NativeOutputPathStyle::CliStable);
        let mut args = vec![
            "-i".to_string(),
            self.input_path.clone(),
            "-m".to_string(),
            self.model_path.clone(),
            "-o".to_string(),
            output_path.clone(),
            "--precision".to_string(),
            self.precision.clone(),
            "--progress".to_string(),
            "jsonl".to_string(),
        ];
        if self.max_batch > 1 {
            args.push("--max-batch".to_string());
            args.push(self.max_batch.to_string());
        }
        NativeCliInvocation { output_path, args }
    }

    fn prepare_cli_execution(&self) -> Result<NativeCliExecutionPlan, String> {
        let invocation = self.prepare_cli_invocation();
        let prepared_command = crate::commands::rave::prepare_rave_upscale_command(
            invocation.args,
            true,
            false,
            true,
        )?;
        Ok(NativeCliExecutionPlan {
            output_path: invocation.output_path,
            prepared_command,
        })
    }

    pub(crate) fn prepare_direct_plan(
        &self,
        ffmpeg_cmd: String,
    ) -> Result<NativeDirectPlan, String> {
        use videoforge_engine::codecs::sys::cudaVideoCodec as CudaCodec;

        NativeDirectPlan::prepare(self, ffmpeg_cmd, CudaCodec::H264)
    }

    pub(crate) fn resolved_output_path_for_route(&self, route: &NativeExecutionRoute) -> String {
        match route.executed_executor {
            "direct" => self.resolved_output_path(NativeOutputPathStyle::DirectTemp),
            _ => self.resolved_output_path(NativeOutputPathStyle::CliStable),
        }
    }

    pub(crate) fn base_perf(&self, frames_processed: u64) -> NativePerfReport {
        NativePerfReport {
            frames_processed,
            effective_max_batch: self.max_batch,
            trt_cache_enabled: self.trt_cache_enabled,
            trt_cache_dir: self.trt_cache_dir.clone(),
            requested_executor: None,
            executed_executor: None,
            direct_attempted: false,
            fallback_used: false,
            fallback_reason_code: None,
            fallback_reason_message: None,
            total_elapsed_ms: None,
            frames_decoded: None,
            frames_preprocessed: None,
            frames_inferred: None,
            frames_encoded: None,
            preprocess_avg_us: None,
            inference_frame_avg_us: None,
            inference_dispatch_avg_us: None,
            postprocess_frame_avg_us: None,
            postprocess_dispatch_avg_us: None,
            encode_avg_us: None,
            vram_current_mb: None,
            vram_peak_mb: None,
        }
    }

    pub(crate) fn build_result(
        &self,
        output_path: String,
        engine: impl Into<String>,
        encoder_mode: impl Into<String>,
        encoder_detail: Option<String>,
        mut perf: NativePerfReport,
        route: NativeExecutionRoute,
    ) -> NativeUpscaleResult {
        perf.requested_executor = Some(route.requested_executor.to_string());
        perf.executed_executor = Some(route.executed_executor.to_string());
        perf.direct_attempted = route.direct_attempted;
        perf.fallback_used = route.fallback_reason_code.is_some();
        perf.fallback_reason_code = route.fallback_reason_code;
        perf.fallback_reason_message = route.fallback_reason_message;

        NativeUpscaleResult {
            output_path,
            engine: engine.into(),
            encoder_mode: encoder_mode.into(),
            encoder_detail,
            audio_preserved: self.preserve_audio,
            runtime_snapshot: None,
            observed_metrics: None,
            perf,
        }
    }
}

#[cfg(feature = "native_engine")]
fn route_id_for_executor(executed_executor: &str) -> &'static str {
    match executed_executor {
        "direct" => "native_direct",
        _ => "native_via_rave_cli",
    }
}

#[cfg(feature = "native_engine")]
pub(crate) fn build_native_runtime_snapshot(
    job: &NativeJobSpec,
    route: &NativeExecutionRoute,
) -> RuntimeConfigSnapshot {
    let route_id = route_id_for_executor(route.executed_executor);
    let mut snapshot = RuntimeConfigSnapshot {
        requested_executor: Some(route.requested_executor.to_string()),
        executed_executor: Some(route.executed_executor.to_string()),
        model_path: Some(job.model_path.clone()),
        model_format: Some("onnx".to_string()),
        scale: Some(job.scale),
        precision: Some(job.precision.clone()),
        fallback: route
            .fallback_reason_code
            .as_ref()
            .map(|_| RuntimeFallbackInfo {
                from_route_id: "native_direct".to_string(),
                to_route_id: route_id.to_string(),
                reason_code: route.fallback_reason_code.clone(),
                reason_message: route.fallback_reason_message.clone(),
            }),
        extensions: RuntimeConfigExtensions {
            python: None,
            native: Some(NativeRuntimeConfigExtension {
                requested_output_path: (!job.requested_output_path.trim().is_empty())
                    .then(|| job.requested_output_path.clone()),
                native_runtime_enabled: native_engine_runtime_enabled(),
                native_direct_enabled: native_engine_direct_enabled(),
                preserve_audio: job.preserve_audio,
                max_batch: Some(job.max_batch),
                trt_cache_enabled: job.trt_cache_enabled,
                trt_cache_dir: job.trt_cache_dir.clone(),
            }),
        },
        ..RuntimeConfigSnapshot::new(
            job.run_id.clone(),
            route_id,
            RuntimeEngineFamily::Native,
            job.input_path.clone(),
            job.resolved_output_path_for_route(route),
        )
    };

    if snapshot.fallback.is_some() {
        snapshot.snapshot_kind = RuntimeSnapshotKind::RouteFallback;
    }

    snapshot
}

#[cfg(feature = "native_engine")]
pub(crate) fn build_native_observed_metrics(
    run_id: &str,
    route_id: &str,
    status: RunStatus,
    perf: Option<&NativePerfReport>,
    error: Option<&NativeUpscaleError>,
) -> RunObservedMetrics {
    let native_metrics = perf.map(|perf| NativeRuntimeMetricsExtension {
        frames_decoded: perf.frames_decoded,
        frames_preprocessed: perf.frames_preprocessed,
        frames_inferred: perf.frames_inferred,
        frames_encoded: perf.frames_encoded,
        preprocess_avg_us: perf.preprocess_avg_us,
        inference_frame_avg_us: perf.inference_frame_avg_us,
        inference_dispatch_avg_us: perf.inference_dispatch_avg_us,
        postprocess_frame_avg_us: perf.postprocess_frame_avg_us,
        postprocess_dispatch_avg_us: perf.postprocess_dispatch_avg_us,
        encode_avg_us: perf.encode_avg_us,
        vram_current_mb: perf.vram_current_mb,
        vram_peak_mb: perf.vram_peak_mb,
    });
    let mut metrics = RunObservedMetrics::new(run_id, route_id, status);
    if let Some(perf) = perf {
        metrics.total_elapsed_ms = perf.total_elapsed_ms;
        metrics.work_units_processed = perf
            .frames_encoded
            .or_else(|| (perf.frames_processed > 0).then_some(perf.frames_processed));
    }
    if let Some(error) = error {
        metrics.error_code = Some(error.code.clone());
        metrics.error_message = Some(error.message.clone());
    }
    if let Some(native_metrics) =
        native_metrics.filter(|m: &NativeRuntimeMetricsExtension| !m.is_empty())
    {
        metrics.extensions = RuntimeMetricsExtensions {
            python: None,
            native: Some(native_metrics),
        };
    }
    metrics
}

#[cfg(feature = "native_engine")]
fn maybe_write_native_run_manifest(
    job: &NativeJobSpec,
    result: &NativeUpscaleResult,
) -> Result<(), String> {
    if let Some(manifest_path) = maybe_write_run_manifest(
        run_artifacts_enabled_from_env(),
        &RunManifestInputs {
            input_path: &job.input_path,
            output_path: &result.output_path,
            engine_family: Some("native"),
            route_id: result
                .perf
                .executed_executor
                .as_deref()
                .map(route_id_for_executor),
            scale: job.scale,
            precision: &job.precision,
            model_key: None,
            model_path: Some(&job.model_path),
            worker_caps: WorkerCapsSnapshot::default(),
            ipc_protocol_version: None,
            shm_protocol_version: None,
            requested_executor: result.perf.requested_executor.as_deref(),
            executed_executor: result.perf.executed_executor.as_deref(),
            audio_preserved: Some(result.audio_preserved),
            trt_cache_enabled: Some(result.perf.trt_cache_enabled),
            trt_cache_dir: result.perf.trt_cache_dir.as_deref(),
            app_version: Some(env!("CARGO_PKG_VERSION")),
        },
    )
    .map_err(|e| format!("Failed to write native run manifest: {e}"))?
    {
        tracing::info!(path = %manifest_path.display(), "Native run manifest written");
        maybe_finalize_run_artifacts(
            manifest_path.parent(),
            &RunArtifactFinalizeInputs {
                runtime_snapshot: result
                    .runtime_snapshot
                    .as_ref()
                    .ok_or_else(|| "Native result missing runtime snapshot".to_string())?,
                observed_metrics: result.observed_metrics.as_ref(),
            },
        )
        .map_err(|e| format!("Failed to finalize native run artifacts: {e}"))?;
    }

    Ok(())
}

#[cfg(feature = "native_engine")]
fn trt_cache_runtime(model_path: &str) -> (bool, Option<String>) {
    let enabled = std::env::var("VIDEOFORGE_TRT_ENABLE_ENGINE_CACHE")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false);
    if !enabled {
        return (false, None);
    }

    let cache_root = std::env::var_os("VIDEOFORGE_TRT_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("videoforge").join("trt_cache"));
    let model_tag = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let cache_dir = cache_root.join(model_tag);
    (true, Some(cache_dir.to_string_lossy().to_string()))
}

#[cfg(feature = "native_engine")]
fn infer_model_scale(model_path: &str) -> Option<u32> {
    let stem = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())?
        .to_ascii_lowercase();

    for scale in [8u32, 4, 3, 2] {
        let prefix = format!("{scale}x");
        if stem.starts_with(&prefix) {
            return Some(scale);
        }
    }

    None
}

#[cfg(feature = "native_engine")]
fn build_cli_perf_report(
    job: &NativeJobSpec,
    res: &serde_json::Value,
    progress: Option<&crate::rave_cli::RaveProgressSummary>,
) -> NativePerfReport {
    let mut perf = job.base_perf(progress.map(|p| p.frames_encoded).unwrap_or(0));
    perf.total_elapsed_ms = res
        .get("elapsed_ms")
        .and_then(|v| v.as_u64())
        .or_else(|| progress.map(|p| p.elapsed_ms));
    perf.frames_decoded = progress.map(|p| p.frames_decoded);
    perf.frames_inferred = progress.map(|p| p.frames_inferred);
    perf.frames_encoded = progress.map(|p| p.frames_encoded);
    perf.vram_current_mb = res.get("vram_current_mb").and_then(|v| v.as_u64());
    perf.vram_peak_mb = res.get("vram_peak_mb").and_then(|v| v.as_u64());
    perf
}

#[cfg(feature = "native_engine")]
pub(crate) fn build_direct_perf_report(
    job: &NativeJobSpec,
    metrics: &videoforge_engine::engine::pipeline::PipelineMetrics,
    total_elapsed_ms: u64,
    vram_current_bytes: usize,
    vram_peak_bytes: usize,
) -> NativePerfReport {
    use std::sync::atomic::Ordering;

    let frames_decoded = metrics.frames_decoded.load(Ordering::Relaxed);
    let frames_preprocessed = metrics.frames_preprocessed.load(Ordering::Relaxed);
    let frames_inferred = metrics.frames_inferred.load(Ordering::Relaxed);
    let frames_encoded = metrics.frames_encoded.load(Ordering::Relaxed);
    let inference_dispatches = metrics.inference_dispatches.load(Ordering::Relaxed);
    let postprocess_dispatches = metrics.postprocess_dispatches.load(Ordering::Relaxed);
    let avg = |total: &std::sync::atomic::AtomicU64, count: u64| -> Option<u64> {
        if count > 0 {
            Some(total.load(Ordering::Relaxed) / count)
        } else {
            None
        }
    };

    let mut perf = job.base_perf(frames_encoded);
    perf.total_elapsed_ms = Some(total_elapsed_ms);
    perf.frames_decoded = Some(frames_decoded);
    perf.frames_preprocessed = Some(frames_preprocessed);
    perf.frames_inferred = Some(frames_inferred);
    perf.frames_encoded = Some(frames_encoded);
    perf.preprocess_avg_us = avg(&metrics.preprocess_total_us, frames_preprocessed);
    perf.inference_frame_avg_us = avg(&metrics.inference_total_us, frames_inferred);
    perf.inference_dispatch_avg_us = avg(&metrics.inference_total_us, inference_dispatches);
    perf.postprocess_frame_avg_us = avg(&metrics.postprocess_total_us, frames_inferred);
    perf.postprocess_dispatch_avg_us = avg(&metrics.postprocess_total_us, postprocess_dispatches);
    perf.encode_avg_us = avg(&metrics.encode_total_us, frames_encoded);
    perf.vram_current_mb = Some((vram_current_bytes / (1024 * 1024)) as u64);
    perf.vram_peak_mb = Some((vram_peak_bytes / (1024 * 1024)) as u64);
    perf
}

#[cfg(feature = "native_engine")]
fn requested_native_executor() -> NativeRequestedExecutor {
    if native_engine_direct_enabled() {
        NativeRequestedExecutor::Direct
    } else {
        NativeRequestedExecutor::Cli
    }
}

#[cfg(feature = "native_engine")]
async fn run_native_via_rave_cli(
    job: &NativeJobSpec,
    route: NativeExecutionRoute,
) -> Result<NativeUpscaleResult, String> {
    let make_err =
        |code: &str, msg: &str| serde_json::to_string(&NativeUpscaleError::new(code, msg)).unwrap();
    let cli = job.prepare_cli_execution()?;
    let res = crate::commands::rave::run_prepared_rave_upscale(cli.prepared_command).await?;
    let output = res
        .json
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            make_err(
                "RAVE_CONTRACT",
                "rave_upscale did not return a valid output path",
            )
        })?
        .to_string();
    if output != cli.output_path {
        tracing::warn!(
            expected_output = %cli.output_path,
            actual_output = %output,
            "CLI-native returned an output path that differs from the prepared native job output"
        );
    }

    let perf = build_cli_perf_report(job, &res.json, res.progress.as_ref());
    let runtime_snapshot = build_native_runtime_snapshot(job, &route);
    let observed_metrics = build_native_observed_metrics(
        &job.run_id,
        route_id_for_executor(route.executed_executor),
        RunStatus::Succeeded,
        Some(&perf),
        None,
    );
    let mut result = job.build_result(output, "native_via_rave_cli", "rave_cli", None, perf, route);
    result.runtime_snapshot = Some(runtime_snapshot);
    result.observed_metrics = Some(observed_metrics.clone());
    log_run_observed_metrics(&observed_metrics);
    maybe_write_native_run_manifest(job, &result)?;

    Ok(result)
}

#[cfg(feature = "native_engine")]
fn decode_native_error(err_json: &str) -> Option<NativeUpscaleError> {
    serde_json::from_str(err_json).ok()
}

#[cfg(feature = "native_engine")]
fn should_fallback_to_rave_cli(err: &NativeUpscaleError) -> bool {
    matches!(err.code.as_str(), "ENCODER_INIT" | "PIPELINE")
        && (err.message.contains("NVENC")
            || err.message.contains("nvEnc")
            || err.message.contains("Software fallback"))
}

#[cfg(feature = "native_engine")]
pub(crate) async fn run_native_job(job: NativeJobSpec) -> Result<NativeUpscaleResult, String> {
    match requested_native_executor() {
        NativeRequestedExecutor::Direct => {
            log_runtime_config_snapshot(&build_native_runtime_snapshot(
                &job,
                &NativeExecutionRoute::direct(),
            ));
            run_direct_with_fallback(job).await
        }
        NativeRequestedExecutor::Cli => {
            let route = NativeExecutionRoute::cli_requested();
            log_runtime_config_snapshot(&build_native_runtime_snapshot(&job, &route));
            let result = run_native_via_rave_cli(&job, route.clone()).await;
            if let Err(err_json) = &result {
                if let Some(err) = decode_native_error(err_json) {
                    log_run_observed_metrics(&build_native_observed_metrics(
                        &job.run_id,
                        route_id_for_executor(route.executed_executor),
                        RunStatus::Failed,
                        None,
                        Some(&err),
                    ));
                }
            }
            result
        }
    }
}

#[cfg(feature = "native_engine")]
async fn run_direct_with_fallback(job: NativeJobSpec) -> Result<NativeUpscaleResult, String> {
    let direct_result = run_native_pipeline(&job).await;

    match direct_result {
        Ok(result) => {
            maybe_write_native_run_manifest(&job, &result)?;
            Ok(result)
        }
        Err(err_json) => {
            let Some(err) = decode_native_error(&err_json) else {
                return Err(err_json);
            };
            if should_fallback_to_rave_cli(&err) {
                let route = NativeExecutionRoute::cli_fallback(&err);
                log_runtime_config_snapshot(&build_native_runtime_snapshot(&job, &route));
                tracing::warn!(
                    run_id = %job.run_id,
                    code = %err.code,
                    message = %err.message,
                    "Direct native path failed; falling back to CLI-backed native path"
                );
                let cli_result = run_native_via_rave_cli(&job, route.clone()).await;
                if let Err(cli_err_json) = &cli_result {
                    if let Some(cli_err) = decode_native_error(cli_err_json) {
                        log_run_observed_metrics(&build_native_observed_metrics(
                            &job.run_id,
                            route_id_for_executor(route.executed_executor),
                            RunStatus::Failed,
                            None,
                            Some(&cli_err),
                        ));
                    }
                }
                cli_result
            } else {
                log_run_observed_metrics(&build_native_observed_metrics(
                    &job.run_id,
                    "native_direct",
                    RunStatus::Failed,
                    None,
                    Some(&err),
                ));
                Err(err_json)
            }
        }
    }
}
