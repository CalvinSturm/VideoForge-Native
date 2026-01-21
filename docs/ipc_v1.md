# IPC v1 – Unified Backend Protocol

This document defines the **canonical IPC protocol** shared between:

- Renderer (TypeScript)
- Electron Main (TypeScript)
- Python Engine
- Rust Engine (future)

Transport:
- JSON over STDIN / STDOUT
- One JSON object per line
- UTF-8 encoded
- No framing beyond newline

---

## Message Types

All messages fall into **exactly one** of the following categories:

- Request (Renderer → Backend)
- Event (Backend → Renderer)
- Response (Backend → Renderer)

---

## Request Envelope

```json
{
  "id": "string (uuid)",
  "command": "string",
  "params": { }
}
