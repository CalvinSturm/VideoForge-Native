use std::env;
use std::path::PathBuf;

fn main() {
    let workspace = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent().unwrap().parent().unwrap().to_path_buf();
    let header = workspace.join("engine-v2/third-party/nvidia-sdk/include/nvEncodeAPI.h");
    let out = workspace.join("engine-v2/src/codecs/nvenc_bindings_generated.rs");

    let mut b = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_type("GUID")
        .allowlist_type("NV_ENC_.*")
        .allowlist_type("NVENC_.*")
        .allowlist_var("NV_ENC_.*")
        .allowlist_var("NVENCAPI_.*")
        .allowlist_function("NvEncodeAPI.*")
        .layout_tests(false)
        .derive_default(true)
        .generate_comments(false)
        .clang_arg("-DNV_WINDOWS")
        .clang_arg("-D_MSC_VER=1930")
        .clang_arg("-fparse-all-comments");

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let include = PathBuf::from(cuda_path).join("include");
        b = b.clang_arg(format!("-I{}", include.display()));
    }

    let bindings = b.generate().expect("failed to generate nvenc bindings");
    bindings.write_to_file(&out).expect("failed to write bindings");
    println!("wrote {}", out.display());
}
