# VideoForge Development & Build Setup Documentation

## Overview

This document outlines the development and build setup for **VideoForge**, a hybrid Rust + Electron + React project. It covers how to work with Cargo features for parallelism, how to manage builds and watch tasks, and recommended workflows.

---

## 1. Rust Cargo Features & Parallelism

### Why use features?

Rust features allow conditional compilation of optional functionality. For VideoForge, the `"parallel"` feature enables parallel processing using the Rayon crate for base64 decoding and other performance gains.

### Cargo.toml

- The `parallel` feature is defined as:
  
  ```toml
  [features]
  parallel = ["rayon"]
````

* This means that when `"parallel"` is enabled, the `rayon` crate is included.

### When to use

* **Development (debug build):**

  ```bash
  cargo build --features parallel
  ```

* **Release (optimized build):**

  ```bash
  cargo build --release --features parallel
  ```

---

## 2. Rust Build & Watch Scripts in package.json

### Scripts for Rust compilation and watch

In your `package.json`, add the following scripts to integrate Rust compilation with your Node/Electron workflow:

```json
{
  "scripts": {
    "rust:build:debug": "cargo build --features parallel",
    "rust:build:release": "cargo build --release --features parallel",
    "rust:watch": "cargo watch -q -c -w src -x \"build --features parallel\"",
    "electron-dev": "electron-vite dev",
    "dev": "npm-run-all -p rust:watch electron-dev",
    "build": "npm run rust:build:release && electron-vite build && electron-builder",
    "preview": "electron-vite preview",
    "electron-start": "electron ."
  }
}
```

* `rust:watch` uses `cargo-watch` to automatically rebuild Rust code on source changes.
* `dev` runs Rust watch and Electron dev concurrently.
* `build` compiles Rust in release mode, builds the frontend, then packages with electron-builder.

---

## 3. Installing cargo-watch

`cargo-watch` is a separate Rust utility to watch your code and run commands.

### Install globally (once):

```bash
cargo install cargo-watch
```

### Verify installation:

```bash
cargo watch --version
```

---

## 4. Running Watch Commands on Windows

**Important**: On Windows (cmd or PowerShell), quoting rules differ from Unix shells.

### Correct `cargo-watch` command syntax for Windows:

```bash
cargo watch -q -c -w src -x "build --features parallel"
```

* Use double quotes `"` around the full `-x` command argument.
* Single quotes `'` do **not** work properly on Windows cmd.

---

## 5. Base64 Decoding in Rust Utilities

We use Rust utility functions to decode base64 strings from JSON:

### decode_base64_list

* Input: JSON value and key to an array of base64-encoded strings.
* Output: `Vec<Vec<u8>>` with each element decoded from base64.
* Uses serde_json + base64 crates.
* Includes error context for better debugging.

### decode_base64_list_parallel

* Same as above, but uses Rayon’s parallel iterator for concurrent decoding.
* Enabled with `parallel` feature.
* Guard with feature flag if desired.

---

## 6. Cargo Feature Usage Example

Add to your Rust source to conditionally compile parallel decoding:

```rust
#[cfg(feature = "parallel")]
pub fn decode_parallel(...) { ... }

#[cfg(not(feature = "parallel"))]
pub fn decode_serial(...) { ... }
```

Use the appropriate function depending on whether `"parallel"` feature is enabled.

---

## 7. How to Build & Develop

### Development

Run the dev script that watches Rust source and runs Electron hot reload:

```bash
npm run dev
```

### Build for production

Build Rust in release mode, then bundle Electron app:

```bash
npm run build
```

### Run Electron app manually

```bash
npm run electron-start
```

---

## 8. Additional Tips

* Keep Rust and Node builds separate but coordinated via npm scripts.
* Use cargo features for optional functionality to keep compile times manageable.
* For Windows users, always verify quoting styles when running shell commands.
* `cargo-watch` drastically improves Rust development iteration by rebuilding on save.

---

## 9. Troubleshooting

* **`cargo-watch` not found?**

  Ensure it's installed globally with `cargo install cargo-watch`.

* **Quotes not working on Windows?**

  Use double quotes `"` around multi-word commands in `-x`.

* **Build fails due to missing features?**

  Confirm you pass `--features parallel` when building.

---

## 10. Sample Dependencies (package.json excerpt)

```json
"devDependencies": {
  "@vitejs/plugin-react": "^5.1.2",
  "electron": "^39.2.7",
  "electron-vite": "^5.0.0",
  "electron-builder": "^24.9.1",
  "npm-run-all": "^4.1.5"
},
"dependencies": {
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-mosaic-component": "^6.1.1",
  "systeminformation": "^5.30.0",
  "uuid": "^13.0.0"
}
```