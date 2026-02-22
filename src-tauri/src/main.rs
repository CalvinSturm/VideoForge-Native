// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// FIX: Explicitly link the library defined in Cargo.toml
extern crate app_lib;

fn main() {
    app_lib::run();
}
