"""Typed IPC envelopes for Rust <-> Python Zenoh messages.

Must mirror: src-tauri/src/ipc/protocol.rs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Must match src-tauri/src/ipc/protocol.rs
PROTOCOL_VERSION: int = 1


@dataclass
class IpcError:
    code: str
    message: str

    @classmethod
    def from_json(cls, obj: Any) -> "IpcError":
        if not isinstance(obj, dict):
            raise ValueError("error must be an object")
        code = obj.get("code")
        message = obj.get("message")
        if not isinstance(code, str) or not code:
            raise ValueError("error.code must be a non-empty string")
        if not isinstance(message, str):
            raise ValueError("error.message must be a string")
        return cls(code=code, message=message)

    def to_json(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass
class RequestEnvelope:
    version: int
    request_id: str
    job_id: str
    kind: str
    payload: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: Any) -> "RequestEnvelope":
        if not isinstance(obj, dict):
            raise ValueError("request envelope must be an object")

        required = ("version", "request_id", "job_id", "kind", "payload")
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f"missing required request fields: {', '.join(missing)}")

        version = obj["version"]
        request_id = obj["request_id"]
        job_id = obj["job_id"]
        kind = obj["kind"]
        payload = obj["payload"]

        if not isinstance(version, int):
            raise ValueError("version must be an integer")
        if not isinstance(request_id, str):
            raise ValueError("request_id must be a string")
        if not isinstance(job_id, str):
            raise ValueError("job_id must be a string")
        if not isinstance(kind, str):
            raise ValueError("kind must be a string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        extra = {k: v for k, v in obj.items() if k not in required}
        return cls(
            version=version,
            request_id=request_id,
            job_id=job_id,
            kind=kind,
            payload=payload,
            extra=extra,
        )


@dataclass
class ResponseEnvelope:
    version: int = PROTOCOL_VERSION
    request_id: str = ""
    job_id: str = ""
    kind: str = ""
    status: Optional[str] = None
    error: Optional[IpcError] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "version": self.version,
            "request_id": self.request_id,
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "error": self.error.to_json() if self.error is not None else None,
        }
        # Preserve unknown/extra top-level fields for compatibility.
        out.update(self.extra)
        return out

    @classmethod
    def status_response(
        cls,
        status: str,
        req: Optional[RequestEnvelope] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ResponseEnvelope":
        return cls(
            version=PROTOCOL_VERSION,
            request_id=req.request_id if req else "",
            job_id=req.job_id if req else "",
            kind="status",
            status=status,
            error=None,
            extra=extra or {},
        )

    @classmethod
    def error_response(
        cls,
        code: str,
        message: str,
        req: Optional[RequestEnvelope] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ResponseEnvelope":
        return cls(
            version=PROTOCOL_VERSION,
            request_id=req.request_id if req else "",
            job_id=req.job_id if req else "",
            kind="error",
            status="error",
            error=IpcError(code=code, message=message),
            extra=extra or {},
        )

