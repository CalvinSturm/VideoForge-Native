use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/ffmpeg_accessors.c");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");

    let mut build = cc::Build::new();
    build.file("src/ffmpeg_accessors.c");

    if let Ok(ffmpeg_dir) = env::var("FFMPEG_DIR") {
        let include = PathBuf::from(ffmpeg_dir).join("include");
        if include.exists() {
            build.include(include);
        }
    }

    if let Ok(vcpkg_root) = env::var("VCPKG_ROOT") {
        let include = PathBuf::from(vcpkg_root)
            .join("installed")
            .join("x64-windows")
            .join("include");
        if include.exists() {
            build.include(include);
        }
    }

    // Common local default for this repo on Windows.
    let fallback_vcpkg = PathBuf::from(r"C:\tools\vcpkg\installed\x64-windows\include");
    if fallback_vcpkg.exists() {
        build.include(fallback_vcpkg);
    }

    build.compile("rave_ffmpeg_accessors");
}
