# Release Hygiene Checklist

Use this before cutting a release or changing package/app metadata.

## Version Source Of Truth

- Keep the app version aligned across:
  - `package.json`
  - `ui/package.json`
  - `src-tauri/Cargo.toml`
  - `src-tauri/tauri.conf.json`
- If only one version should change, update all four in the same PR.

## Naming And Packaging

- Keep package ids lowercase where required:
  - `package.json` -> `videoforge`
  - `ui/package.json` -> `videoforge-ui`
- Keep user-facing product naming consistent:
  - `src-tauri/tauri.conf.json` -> `VideoForge`
  - window/app title strings should match `VideoForge`

## Metadata

- Confirm descriptions still reflect the current app shape:
  - root package describes workspace/shell
  - UI package describes the React/Vite frontend
  - Rust crate describes the desktop app/backend
- Remove placeholder metadata before release.
- Keep proprietary licensing labels consistent when no public OSS license file exists.

## Docs

- Verify README commands still match actual scripts and validation commands.
- Verify native-engine commands and feature flags still match current behavior.
- Add any new release-critical docs to the README docs index if they become canonical.
