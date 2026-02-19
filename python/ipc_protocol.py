"""
VideoForge IPC Protocol — Python dataclasses for typed Zenoh messages.

Mirrors src-tauri/src/ipc/protocol.rs.

Forward compatibility rules (both sides must follow):
  - Receivers MUST ignore unknown fields.
  - Senders SHOULD include all required fields.
  - Default values are used when a field is absent (backward compat).

Usage::

    # Parse an incoming request
    raw = json.loads(sample.payload.to_bytes().decode())
    req = RequestEnvelope.from_dict(raw)
    cmd = req.kind  # "load_model", "create_shm", etc.

    # Build and send a response
    resp = ResponseEnvelope.ok("MODEL_LOADED", req, extra={"model": "RCAN_x4", "scale": 4})
    pub.put(resp.to_json().encode("utf-8"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

PROTOCOL_VERSION: int = 1


# ─── IpcError ────────────────────────────────────────────────────────────────


@dataclass
class IpcError:
    """Structured error payload."""
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IpcError":
        return cls(
            code=d.get("code", "UNKNOWN"),
            message=d.get("message", ""),
        )


# ─── RequestEnvelope (Python receives) ───────────────────────────────────────


@dataclass
class RequestEnvelope:
    """Incoming Rust → Python command envelope.

    ``from_dict`` accepts both the new protocol format (with ``kind``) and
    the legacy format (with ``command``) for backward compatibility during
    the transition period.
    """
    version: int = 0
    request_id: str = ""
    job_id: str = ""
    # The command kind — falls back to the old "command" key for compat.
    kind: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RequestEnvelope":
        """Parse from a raw JSON dict.

        Unknown keys are silently ignored (forward compatibility).
        Falls back to ``d["command"]`` if ``d["kind"]`` is absent
        (backward compatibility with pre-protocol Rust callers).
        """
        kind = d.get("kind") or d.get("command", "")
        # For legacy messages the payload is the entire dict minus "command".
        payload = d.get("payload", {k: v for k, v in d.items() if k != "command"})
        return cls(
            version=d.get("version", 0),
            request_id=d.get("request_id", ""),
            job_id=d.get("job_id", ""),
            kind=kind,
            payload=payload,
        )


# ─── ResponseEnvelope (Python sends) ─────────────────────────────────────────


@dataclass
class ResponseEnvelope:
    """Outgoing Python → Rust response envelope."""
    version: int = PROTOCOL_VERSION
    request_id: str = ""
    job_id: str = ""
    kind: str = "status"
    status: Optional[str] = None
    error: Optional[IpcError] = None
    # Extra top-level fields merged in on serialisation (for backward compat
    # with Rust callers that check raw response strings).
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "version": self.version,
            "request_id": self.request_id,
            "job_id": self.job_id,
            "kind": self.kind,
        }
        if self.status is not None:
            d["status"] = self.status
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["error"] = None
        # Merge extra fields at top level for backward compat
        d.update(self._extra)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def ok(
        cls,
        status: str,
        req: Optional[RequestEnvelope] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ResponseEnvelope":
        """Build a success response."""
        return cls(
            version=PROTOCOL_VERSION,
            request_id=req.request_id if req else "",
            job_id=req.job_id if req else "",
            kind="status",
            status=status,
            _extra=extra or {},
        )

    @classmethod
    def progress(
        cls,
        req: Optional[RequestEnvelope] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ResponseEnvelope":
        """Build a progress response."""
        return cls(
            version=PROTOCOL_VERSION,
            request_id=req.request_id if req else "",
            job_id=req.job_id if req else "",
            kind="progress",
            status="progress",
            _extra=extra or {},
        )

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        req: Optional[RequestEnvelope] = None,
    ) -> "ResponseEnvelope":
        """Build an error response."""
        return cls(
            version=PROTOCOL_VERSION,
            request_id=req.request_id if req else "",
            job_id=req.job_id if req else "",
            kind="error",
            status="error",
            error=IpcError(code=code, message=message),
        )


# ─── Self-test ───────────────────────────────────────────────────────────────


def _selftest() -> None:
    """Run basic conformance checks.  Call with: python -c 'from ipc_protocol import _selftest; _selftest()'"""
    import sys

    # 1. RequestEnvelope — new protocol format
    raw = {
        "version": 1,
        "request_id": "42",
        "job_id": "job-x",
        "kind": "load_model",
        "payload": {"model_name": "RCAN_x4"},
        "future_field": "ignored",
    }
    req = RequestEnvelope.from_dict(raw)
    assert req.version == 1
    assert req.kind == "load_model"
    assert req.payload["model_name"] == "RCAN_x4"

    # 2. RequestEnvelope — legacy format (backward compat)
    raw_legacy = {"command": "load_model", "params": {"model_name": "RCAN_x4"}}
    req_legacy = RequestEnvelope.from_dict(raw_legacy)
    assert req_legacy.kind == "load_model"

    # 3. ResponseEnvelope — ok response
    resp = ResponseEnvelope.ok("MODEL_LOADED", req, extra={"model": "RCAN_x4", "scale": 4})
    d = resp.to_dict()
    assert d["status"] == "MODEL_LOADED"
    assert d["model"] == "RCAN_x4"
    assert d["version"] == PROTOCOL_VERSION
    assert d["request_id"] == "42"

    # 4. ResponseEnvelope — error response
    err_resp = ResponseEnvelope.error("MODEL_NOT_FOUND", "No weights for RCAN_x8", req)
    d = err_resp.to_dict()
    assert d["error"]["code"] == "MODEL_NOT_FOUND"
    assert d["kind"] == "error"

    # 5. ResponseEnvelope — progress response
    prog = ResponseEnvelope.progress(req, extra={"current": 5, "total": 20})
    d = prog.to_dict()
    assert d["status"] == "progress"
    assert d["current"] == 5

    # 6. JSON round-trip
    json_str = resp.to_json()
    parsed = json.loads(json_str)
    assert parsed["status"] == "MODEL_LOADED"

    print("[ipc_protocol] Self-test passed.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    _selftest()
