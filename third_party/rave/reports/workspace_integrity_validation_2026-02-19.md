# Workspace Integrity Validation (2026-02-19)

## Environment
```text
date: 2026-02-19T00:12:07Z
pwd: /home/calvin/src/rave
branch: fix/workspace-integrity-pass-root-cargo
rustc: rustc 1.93.1 (01f6ddf75 2026-02-11)
cargo: cargo 1.93.1 (083ac5135 2025-12-15)
cuda_lib_dir_exists: yes
```

## git rev-parse --show-toplevel
```text
/home/calvin/src/rave
[exit_code=0]
```

## pwd
```text
/home/calvin/src/rave
[exit_code=0]
```

## git status -sb
```text
## fix/workspace-integrity-pass-root-cargo
 M Cargo.lock
 M Cargo.toml
 M crates/rave-core/Cargo.toml
 M crates/rave-core/src/lib.rs
 M crates/rave-cuda/Cargo.toml
 M crates/rave-cuda/src/lib.rs
 M crates/rave-ffmpeg/Cargo.toml
 M crates/rave-ffmpeg/src/lib.rs
 M crates/rave-nvcodec/Cargo.toml
 M crates/rave-nvcodec/src/lib.rs
 M crates/rave-pipeline/Cargo.toml
 M crates/rave-pipeline/src/lib.rs
 M crates/rave-tensorrt/Cargo.toml
 M crates/rave-tensorrt/src/lib.rs
 M crates/rave-tensorrt/tests/provider_bridge_smoke.rs
 M rave-cli/Cargo.toml
?? .cargo/
?? crates/rave-core/src/backend.rs
?? crates/rave-core/src/codec_traits.rs
?? crates/rave-core/src/debug_alloc.rs
?? crates/rave-core/src/error.rs
?? crates/rave-core/src/ffi_types.rs
?? crates/rave-cuda/src/sys.rs
?? crates/rave-ffmpeg/src/ffmpeg_sys.rs
?? crates/rave-ffmpeg/src/file_source.rs
?? crates/rave-ffmpeg/src/probe.rs
?? crates/rave-nvcodec/build.rs
?? crates/rave-nvcodec/src/sys.rs
?? crates/rave-pipeline/src/inference.rs
?? reports/workspace_integrity_validation_2026-02-19.md
[exit_code=0]
```

## cargo fmt --check
```text
[exit_code=0]
```

## cargo clippy --workspace --all-targets -- -D warnings
```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s
[exit_code=0]
```

## cargo test --workspace
```text
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.10s
     Running unittests src/main.rs (target/debug/deps/rave-7cb6d15ca8207752)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/cli_subcommands.rs (target/debug/deps/cli_subcommands-afaf2e38a561896f)

running 16 tests
test devices_help_lists_json ... ok
test probe_help_lists_all_and_json ... ok
test help_lists_subcommands ... ok
test probe_json_emits_schema_and_command_fields ... ok
test devices_json_emits_schema_and_command_fields ... ok
test upscale_help_lists_progress_and_jsonl ... ok
test benchmark_help_lists_skip_encode_and_json_out ... ok
test benchmark_dry_run_without_json_is_human_readable ... ok
test benchmark_dry_run_json_mode_is_clean_stdout_even_with_progress_flags ... ok
test upscale_dry_run_json_mode_is_clean_stdout_even_with_progress_flags ... ok
test benchmark_dry_run_emits_valid_json_shape ... ok
test benchmark_dry_run_accepts_jsonl_progress_flag ... ok
test benchmark_json_mode_emits_structured_error_on_failure ... ok
test upscale_dry_run_json_emits_valid_json_shape ... ok
test upscale_json_mode_emits_structured_error_on_failure ... ok
test upscale_dry_run_accepts_subcommand_args ... ok

test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s

     Running unittests src/lib.rs (target/debug/deps/rave_core-6638d2a0004cbde7)

running 3 tests
test context::tests::bucket_sizing_large ... ok
test context::tests::bucket_sizing_small ... ok
test context::tests::bucket_sizing_zero ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_cuda-3448694888477fa6)

running 3 tests
test kernels::tests::falls_back_to_cuda_home_include_when_cuda_path_missing ... ok
test kernels::tests::falls_back_to_usr_local_cuda_include ... ok
test kernels::tests::prefers_cuda_path_include_when_present ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_ffmpeg-f6e4776497c1aa76)

running 1 test
test ffmpeg_demuxer::tests::empty_packet_data_is_safe ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_nvcodec-26ae7a8258501faf)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_pipeline-8475821b1cf50e24)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_tensorrt-1e7ebe5f004e21c8)

running 2 tests
test tensorrt::tests::ort_registers_tensorrt_ep_smoke ... ignored, requires model + full ORT/TensorRT runtime
test tensorrt::tests::providers_load_with_bridge_preloaded ... ignored, requires ORT TensorRT provider libs on host

test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/provider_bridge_smoke.rs (target/debug/deps/provider_bridge_smoke-2d57d83f1b546e15)

running 2 tests
test ort_tensorrt_ep_registration_smoke ... ok
test providers_shared_then_tensorrt_dlopen_smoke ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s

   Doc-tests rave_core

running 1 test
test crates/rave-core/src/debug_alloc.rs - debug_alloc (line 8) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

all doctests ran in 0.31s; merged doctests compilation took 0.31s
   Doc-tests rave_cuda

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_ffmpeg

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_nvcodec

running 1 test
test crates/rave-nvcodec/src/nvdec.rs - nvdec::wait_for_event (line 624) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

all doctests ran in 0.31s; merged doctests compilation took 0.31s
   Doc-tests rave_pipeline

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_tensorrt

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

[exit_code=0]
```

## cargo build --workspace --release
```text
    Finished `release` profile [optimized] target(s) in 0.06s
[exit_code=0]
```

