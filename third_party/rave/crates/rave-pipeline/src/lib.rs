#![doc = include_str!("../README.md")]

#[cfg(feature = "nvidia-inference")]
pub mod inference;
#[cfg(feature = "cuda-pipeline")]
pub mod pipeline;
#[cfg(not(feature = "cuda-pipeline"))]
#[path = "pipeline_stub.rs"]
pub mod pipeline;
pub mod stage_graph;

pub use pipeline::{
    enforce_determinism_policy, DeterminismObserved, DeterminismPolicy, DeterminismSkipReason,
    PipelineConfig, PipelineMetrics, TileConfig, UpscalePipeline,
};
pub use stage_graph::{
    validate_batch_config, AuditItem, AuditLevel, BatchConfig, EnhanceConfig, PipelineReport,
    PrecisionPolicyConfig, ProfilePreset, RunContract, StageConfig, StageGraph, StageId, StageKind,
    StageTimingReport, GRAPH_SCHEMA_VERSION,
};

#[cfg(test)]
mod tests {
    use super::{PipelineConfig, ProfilePreset, RunContract};

    #[test]
    fn strict_no_host_copies_defaults_to_off() {
        assert!(!PipelineConfig::default().strict_no_host_copies);
    }

    #[test]
    fn production_strict_profile_enables_strict_contract() {
        let contract = RunContract::for_profile(ProfilePreset::ProductionStrict);
        assert!(contract.deterministic_output);
        assert!(contract.fail_on_audit_warn);
    }

    #[cfg(feature = "audit-no-host-copies")]
    #[test]
    fn audit_feature_wires_to_core_guard() {
        let baseline = rave_core::host_copy_audit::is_strict_mode();
        {
            let _guard = rave_core::host_copy_audit::push_strict_mode(true);
            assert!(rave_core::host_copy_audit::is_strict_mode());
        }
        assert_eq!(rave_core::host_copy_audit::is_strict_mode(), baseline);
    }
}
