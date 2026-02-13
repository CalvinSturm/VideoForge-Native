"""
VideoForge SR Settings Node — Dynamic model registry, feature-gated UI, production dispatch.

Orchestrates the SR pipeline parameters from UI sliders / socket overrides
through a debounced, single-flight dispatch to ``model_manager.process_frame()``.

Key architecture:
  - **Dynamic Registry**: Scans ``weights/`` directories for model files at startup
    and on-demand via ``refresh_model_registry()``.  No hard-coded model lists.
  - **Feature Gating**: Each model declares capabilities (temporal, ADR, etc.).
    Parameters outside the model's capability set are silently dropped from the
    dispatch payload, preventing the engine from receiving unsupported flags.
  - **State Shadowing**: ``_state`` holds UI-set values, ``_socket_overrides``
    holds socket-injected values.  Effective value = socket override ?? UI state.
  - **Dirty Tracking**: Only changed keys are transmitted; dirty set cleared on
    successful dispatch.
  - **Single-Flight Dispatch**: At most one ``process_frame()`` in-flight.
    New state overwrites pending — guarantees newest-wins semantics.
  - **Debounce**: 45ms coalescing for slider chatter.
  - **Thread Safety**: ``threading.Lock`` guards all state mutations and engine access.

Author: VideoForge Team
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("videoforge.sr_settings_node")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHTS_DIRS: List[str] = [
    os.path.join(PROJECT_ROOT, "weights"),
    os.path.join(SCRIPT_DIR, "weights"),
]

# Recognised weight file extensions (case-insensitive matching)
_WEIGHT_EXTENSIONS: FrozenSet[str] = frozenset({".safetensors", ".pth", ".pt", ".bin"})

# Debounce window for slider coalescing (seconds)
_DEBOUNCE_MS = 45
_DEBOUNCE_SEC = _DEBOUNCE_MS / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CAPABILITIES — Feature Gating
# ═══════════════════════════════════════════════════════════════════════════════
# Each capability flag corresponds to a set of parameters that are only sent
# to the engine when the selected model supports them.  Unknown models default
# to the conservative "_BASE" capability set.

class ModelCapability:
    """Feature flags for what a model family supports."""
    TEMPORAL = "temporal"        # EMA temporal stabilization
    ADR = "adr"                  # Adaptive Detail Residual (needs GAN texture)
    EDGE_AWARE = "edge_aware"    # Sobel edge-aware blending
    LUMA_BLEND = "luma_blend"    # YCbCr luminance-only blend
    SHARPEN = "sharpen"          # Unsharp mask post-processing
    SECONDARY = "secondary"      # Can act as a secondary model for blending


# Family → capability set.  Looked up by prefix-matching model_key.
_FAMILY_CAPABILITIES: Dict[str, FrozenSet[str]] = {
    "realesrgan": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.ADR,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
        ModelCapability.SECONDARY,
    }),
    "rcan": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
    }),
    "edsr": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
    }),
    "swinir": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.ADR,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
        ModelCapability.SECONDARY,
    }),
    "swin2sr": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.ADR,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
        ModelCapability.SECONDARY,
    }),
    "hat": frozenset({
        ModelCapability.TEMPORAL,
        ModelCapability.ADR,
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
        ModelCapability.SECONDARY,
    }),
    "easr": frozenset({
        ModelCapability.EDGE_AWARE,
        ModelCapability.LUMA_BLEND,
        ModelCapability.SHARPEN,
        ModelCapability.SECONDARY,
    }),
}

# Conservative fallback for unknown families
_BASE_CAPABILITIES: FrozenSet[str] = frozenset({
    ModelCapability.SHARPEN,
})

# Map: capability → parameter keys gated by that capability
_CAPABILITY_PARAM_MAP: Dict[str, FrozenSet[str]] = {
    ModelCapability.TEMPORAL: frozenset({"temporal_enabled", "temporal_alpha"}),
    ModelCapability.ADR: frozenset({"adr_enabled", "detail_strength"}),
    ModelCapability.EDGE_AWARE: frozenset({"edge_strength"}),
    ModelCapability.LUMA_BLEND: frozenset({"luma_only"}),
    ModelCapability.SHARPEN: frozenset({"sharpen_strength"}),
    ModelCapability.SECONDARY: frozenset({"secondary_model"}),
}

# Parameters that are always sent (not gated)
_UNGATED_PARAMS: FrozenSet[str] = frozenset({
    "primary_model",
    "blend_alpha",
    "return_gpu_tensor",
})


def _get_capabilities(model_key: str) -> FrozenSet[str]:
    """Resolve capability set for a model key via prefix matching."""
    key_lower = model_key.lower()
    for family, caps in _FAMILY_CAPABILITIES.items():
        if key_lower.startswith(family):
            return caps
    return _BASE_CAPABILITIES


def _gated_params(model_key: str) -> FrozenSet[str]:
    """
    Return the set of parameter keys that are ALLOWED for the given model.

    Combines ungated params + params whose capability the model supports.
    """
    caps = _get_capabilities(model_key)
    allowed = set(_UNGATED_PARAMS)
    for cap, param_keys in _CAPABILITY_PARAM_MAP.items():
        if cap in caps:
            allowed.update(param_keys)
    return frozenset(allowed)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MODEL REGISTRY — File-system discovery
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Scans ``weights/`` directories and caches discovered model keys.

    Thread-safe.  Call ``refresh()`` to re-scan disk; results are cached until
    the next explicit refresh.
    """

    def __init__(self, weight_dirs: Optional[List[str]] = None) -> None:
        self._dirs = weight_dirs or list(WEIGHTS_DIRS)
        self._lock = threading.Lock()
        self._cache: List[str] = []
        self._last_refresh: float = 0.0
        # Initial scan
        self.refresh()

    def refresh(self) -> List[str]:
        """
        Re-scan weight directories and update the cached model list.

        Discovery logic:
          1. Walk each directory in ``self._dirs``
          2. Match files with extensions in ``_WEIGHT_EXTENSIONS``
          3. Strip the extension to produce a model key
          4. Deduplicate (first occurrence wins, preserving order)
          5. Sort alphabetically for deterministic UI ordering

        Returns the refreshed model key list.
        """
        seen: Set[str] = set()
        found: List[str] = []

        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in sorted(os.listdir(d)):
                path = os.path.join(d, entry)

                # Direct file: weights/model.pth → "model"
                if os.path.isfile(path):
                    stem, ext = os.path.splitext(entry)
                    if ext.lower() in _WEIGHT_EXTENSIONS and stem not in seen:
                        seen.add(stem)
                        found.append(stem)

                # Nested directory: weights/model/model.pth → "model"
                elif os.path.isdir(path):
                    for ext in _WEIGHT_EXTENSIONS:
                        nested = os.path.join(path, entry + ext)
                        if os.path.isfile(nested) and entry not in seen:
                            seen.add(entry)
                            found.append(entry)
                            break

        found.sort()

        with self._lock:
            self._cache = found
            self._last_refresh = time.monotonic()

        logger.info("Model registry refreshed: %d models found", len(found))
        print(f"[SRSettingsNode] Registry refreshed: {found}", flush=True)
        return found

    @property
    def models(self) -> List[str]:
        """Return the cached model key list (read-only copy)."""
        with self._lock:
            return list(self._cache)

    def contains(self, model_key: str) -> bool:
        """Check if a model key exists in the registry."""
        with self._lock:
            return model_key in self._cache

    def capabilities_for(self, model_key: str) -> FrozenSet[str]:
        """Return the capability set for a model key."""
        return _get_capabilities(model_key)


# Module-level singleton registry
_registry = ModelRegistry()


def refresh_model_registry() -> List[str]:
    """Public API: re-scan weights directories and return updated model list."""
    return _registry.refresh()


def get_available_models() -> List[str]:
    """Public API: return cached list of discovered model keys."""
    return _registry.models


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SCHEMA — Validation & Defaults
# ═══════════════════════════════════════════════════════════════════════════════

# (default, min, max, type)  — type is used for coercion + validation
_PARAM_SCHEMA: Dict[str, Tuple[Any, Any, Any, type]] = {
    "primary_model":     ("RCAN_x4",  None,  None,  str),
    "secondary_model":   ("None",     None,  None,  str),
    "blend_alpha":       (0.3,        0.0,   1.0,   float),
    "adr_enabled":       (False,      None,  None,  bool),
    "detail_strength":   (0.5,        0.0,   1.0,   float),
    "luma_only":         (True,       None,  None,  bool),
    "edge_strength":     (0.3,        0.0,   3.0,   float),
    "sharpen_strength":  (0.0,        0.0,   2.0,   float),
    "temporal_enabled":  (True,       None,  None,  bool),
    "temporal_alpha":    (0.9,        0.0,   1.0,   float),
    "return_gpu_tensor": (True,       None,  None,  bool),
}


def _validate_param(key: str, value: Any) -> Any:
    """
    Validate and coerce a parameter value against the schema.

    Returns the validated value.
    Raises ``ValueError`` if the key is unknown or value is out of range.
    """
    if key not in _PARAM_SCHEMA:
        raise ValueError(f"Unknown parameter: {key}")

    default, vmin, vmax, vtype = _PARAM_SCHEMA[key]

    # Type coercion
    try:
        value = vtype(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot coerce {key}={value!r} to {vtype.__name__}: {e}")

    # Model validation: check registry if it's a model selection param
    if key in ("primary_model", "secondary_model") and isinstance(value, str):
        if value != "None" and not _registry.contains(value):
            raise ValueError(
                f"Model '{value}' not found in weights directory.  "
                f"Available: {_registry.models}"
            )

    # Range clamp for numeric types
    if vmin is not None and isinstance(value, (int, float)):
        value = max(vmin, value)
    if vmax is not None and isinstance(value, (int, float)):
        value = min(vmax, value)

    return value


def _get_defaults() -> Dict[str, Any]:
    """Return a dict of all parameter defaults."""
    return {k: v[0] for k, v in _PARAM_SCHEMA.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# SR SETTINGS NODE
# ═══════════════════════════════════════════════════════════════════════════════

class SRSettingsNode:
    """
    Production-grade settings node for the VideoForge SR pipeline.

    Thread-safe state management with UI / socket dual-source resolution,
    dirty tracking, debounced coalescing, feature-gated payload, and
    single-flight dispatch to the sync engine.

    Parameters
    ----------
    engine : object
        Must expose ``process_frame(**kwargs)`` (synchronous, GPU-locked).
    debounce_ms : int
        Debounce window for UI slider coalescing.  Default 45ms.
    """

    def __init__(
        self,
        engine: Any,
        debounce_ms: int = _DEBOUNCE_MS,
    ) -> None:
        self.engine = engine
        self._debounce_sec = debounce_ms / 1000.0

        # ── State shadowing ──────────────────────────────────────────────
        # _state: values set by UI sliders (persistent across socket disconnect)
        # _socket_overrides: values injected by socket connections (take priority)
        self._state: Dict[str, Any] = _get_defaults()
        self._socket_overrides: Dict[str, Any] = {}

        # ── Dirty tracking ───────────────────────────────────────────────
        # Set of keys whose effective value changed since last dispatch
        self._dirty: Set[str] = set()

        # ── Dispatch control ─────────────────────────────────────────────
        self._lock = threading.Lock()
        self._in_flight = False
        self._pending_payload: Optional[Dict[str, Any]] = None

        # ── Debounce timer ───────────────────────────────────────────────
        self._debounce_timer: Optional[threading.Timer] = None

        logger.info("SRSettingsNode initialised (debounce=%dms)", debounce_ms)
        print(f"[SRSettingsNode] Initialised, debounce={debounce_ms}ms", flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # STATE RESOLUTION
    # ─────────────────────────────────────────────────────────────────────

    def _effective(self, key: str) -> Any:
        """
        Resolve effective value: socket override takes priority over UI state.

        Thread-safe state resolution — caller must hold self._lock.
        """
        if key in self._socket_overrides:
            return self._socket_overrides[key]
        return self._state.get(key)

    def get_effective_state(self) -> Dict[str, Any]:
        """Return a snapshot of all effective parameter values."""
        with self._lock:
            return {k: self._effective(k) for k in _PARAM_SCHEMA}

    # ─────────────────────────────────────────────────────────────────────
    # PARAMETER SETTERS
    # ─────────────────────────────────────────────────────────────────────

    def set_param(
        self,
        key: str,
        value: Any,
        *,
        from_socket: bool = False,
    ) -> None:
        """
        Set a parameter value from UI or socket source.

        - Validates against schema (type coercion, range clamp, model existence).
        - Socket writes go to ``_socket_overrides`` only (UI state unchanged).
        - UI writes go to ``_state`` only.
        - Marks key dirty if effective value changed (or if socket newly connected).
        - Triggers debounced dispatch.

        Raises ``ValueError`` for unknown keys or invalid values.
        """
        validated = _validate_param(key, value)

        with self._lock:
            old_effective = self._effective(key)
            was_socket = key in self._socket_overrides

            if from_socket:
                # Socket writes go ONLY to overrides — UI state preserved
                self._socket_overrides[key] = validated
                # A new socket connection is always meaningful even if numeric
                # value matches current state (socket presence changes semantics)
                if not was_socket or old_effective != validated:
                    self._dirty.add(key)
            else:
                # UI writes go ONLY to base state
                self._state[key] = validated
                # Only dirty if UI value actually changes the effective output
                # (socket override may mask this change)
                new_effective = self._effective(key)
                if old_effective != new_effective:
                    self._dirty.add(key)

        self._schedule_dispatch()

    def disconnect_socket(self, key: str) -> None:
        """
        Remove a socket override, reverting to UI state for this key.

        If the effective value changes (socket value != UI value), marks dirty.
        """
        with self._lock:
            if key not in self._socket_overrides:
                return
            old_effective = self._effective(key)
            del self._socket_overrides[key]
            new_effective = self._effective(key)
            if old_effective != new_effective:
                self._dirty.add(key)

        self._schedule_dispatch()

    def disconnect_all_sockets(self) -> None:
        """Remove all socket overrides, reverting fully to UI state."""
        with self._lock:
            if not self._socket_overrides:
                return
            for key in list(self._socket_overrides):
                old_eff = self._effective(key)
                del self._socket_overrides[key]
                new_eff = self._effective(key)
                if old_eff != new_eff:
                    self._dirty.add(key)

        self._schedule_dispatch()

    # ─────────────────────────────────────────────────────────────────────
    # DYNAMIC REGISTRY INTEGRATION
    # ─────────────────────────────────────────────────────────────────────

    def refresh_models(self) -> List[str]:
        """
        Re-scan weights directories and return the updated model list.

        If the currently selected primary/secondary model no longer exists,
        revert to the default and mark dirty.
        """
        models = refresh_model_registry()

        with self._lock:
            # Validate current selections against refreshed registry
            for key in ("primary_model", "secondary_model"):
                current = self._effective(key)
                if current and current != "None" and current not in models:
                    logger.warning(
                        "Model '%s' no longer exists after refresh, reverting %s to default",
                        current, key
                    )
                    print(
                        f"[SRSettingsNode] WARNING: '{current}' not found, "
                        f"reverting {key} to default",
                        flush=True,
                    )
                    # Clear socket override if it was the source
                    if key in self._socket_overrides:
                        del self._socket_overrides[key]
                    # Revert UI state to default
                    self._state[key] = _PARAM_SCHEMA[key][0]
                    self._dirty.add(key)

        if self._dirty:
            self._schedule_dispatch()

        return models

    def get_available_models(self) -> List[str]:
        """Return the cached list of discovered model keys."""
        return _registry.models

    def get_model_capabilities(self, model_key: Optional[str] = None) -> FrozenSet[str]:
        """
        Return the capability set for a model.

        If *model_key* is None, uses the current effective primary_model.
        """
        if model_key is None:
            with self._lock:
                model_key = self._effective("primary_model") or "RCAN_x4"
        return _get_capabilities(model_key)

    def is_param_enabled(self, key: str) -> bool:
        """
        Check if a parameter is enabled for the current primary model.

        Used by UI to grey out / disable controls that the model doesn't support.
        Returns True for ungated params, True if the param's capability is
        supported, False otherwise.
        """
        if key in _UNGATED_PARAMS:
            return True

        with self._lock:
            model_key = self._effective("primary_model") or "RCAN_x4"

        allowed = _gated_params(model_key)
        return key in allowed

    # ─────────────────────────────────────────────────────────────────────
    # DISPATCH — Debounce + Single-Flight + Feature Gating
    # ─────────────────────────────────────────────────────────────────────

    def _schedule_dispatch(self) -> None:
        """
        Schedule a debounced dispatch.

        Cancels any pending timer and starts a new one.  When the timer fires,
        ``_do_dispatch()`` runs on a background thread.
        """
        # Cancel existing timer (coalesce rapid slider changes)
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()

        self._debounce_timer = threading.Timer(self._debounce_sec, self._do_dispatch)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _do_dispatch(self) -> None:
        """
        Build a feature-gated payload from dirty keys and dispatch to engine.

        Single-flight guarantee: if an engine call is already in-flight, the
        payload is stored as ``_pending_payload`` and will be dispatched when
        the current call completes (newest-wins).
        """
        with self._lock:
            if not self._dirty:
                return

            # Build full effective state snapshot
            full_state = {k: self._effective(k) for k in _PARAM_SCHEMA}

            # Feature gating: filter params based on primary model capabilities
            primary_key = full_state.get("primary_model", "RCAN_x4")
            allowed_keys = _gated_params(primary_key)

            # Deterministic payload: stable sorted keys, no None values
            payload: Dict[str, Any] = {}
            for k in sorted(full_state.keys()):
                if k in allowed_keys and full_state[k] is not None:
                    payload[k] = full_state[k]

            # Clear dirty set — these changes are now captured in payload
            self._dirty.clear()

            # Single-flight: if engine busy, store as pending (newest wins)
            if self._in_flight:
                self._pending_payload = payload
                logger.debug("Dispatch queued (engine busy), payload will overwrite pending")
                return

            self._in_flight = True

        # Dispatch outside lock — engine.process_frame() is sync & GPU-locked
        self._execute_dispatch(payload)

    def _execute_dispatch(self, payload: Dict[str, Any]) -> None:
        """
        Execute the engine dispatch in a background thread.

        On completion, checks for pending payload and dispatches again if needed.
        """
        def _run() -> None:
            try:
                logger.debug("Dispatching to engine: %s", list(payload.keys()))
                print(f"[SRSettingsNode] Dispatch: {list(payload.keys())}", flush=True)
                self.engine.process_frame(**payload)
            except Exception as e:
                logger.error("Engine dispatch failed: %s", e)
                print(f"[SRSettingsNode] ERROR: Engine dispatch failed: {e}", flush=True)
            finally:
                # Check for pending payload (newest-wins)
                next_payload: Optional[Dict[str, Any]] = None
                with self._lock:
                    self._in_flight = False
                    if self._pending_payload is not None:
                        next_payload = self._pending_payload
                        self._pending_payload = None
                        self._in_flight = True

                # Tail-call: dispatch pending payload if any
                if next_payload is not None:
                    self._execute_dispatch(next_payload)

        thread = threading.Thread(target=_run, daemon=True, name="sr-dispatch")
        thread.start()

    # ─────────────────────────────────────────────────────────────────────
    # FORCE DISPATCH (bypass debounce)
    # ─────────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """
        Immediately dispatch all dirty state, bypassing the debounce timer.

        Useful for preset application or programmatic batch updates.
        """
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None
        self._do_dispatch()

    # ─────────────────────────────────────────────────────────────────────
    # PRESETS
    # ─────────────────────────────────────────────────────────────────────

    def apply_preset(self, preset_name: str) -> None:
        """
        Batch-apply a named preset.  Bypasses debounce for immediate effect.

        Available presets:
          - ``"performance"``: minimal post-processing, fastest render
          - ``"balanced"``: moderate ADR + sharpening
          - ``"quality"``: full pipeline with temporal + ADR
        """
        presets: Dict[str, Dict[str, Any]] = {
            "performance": {
                "adr_enabled": False,
                "detail_strength": 0.0,
                "luma_only": False,
                "edge_strength": 0.0,
                "sharpen_strength": 0.0,
                "temporal_enabled": False,
                "temporal_alpha": 0.9,
            },
            "balanced": {
                "adr_enabled": True,
                "detail_strength": 0.3,
                "luma_only": True,
                "edge_strength": 0.3,
                "sharpen_strength": 0.15,
                "temporal_enabled": True,
                "temporal_alpha": 0.85,
            },
            "quality": {
                "adr_enabled": True,
                "detail_strength": 0.5,
                "luma_only": True,
                "edge_strength": 0.5,
                "sharpen_strength": 0.25,
                "temporal_enabled": True,
                "temporal_alpha": 0.9,
            },
        }

        preset = presets.get(preset_name.lower())
        if preset is None:
            raise ValueError(
                f"Unknown preset '{preset_name}'.  "
                f"Available: {list(presets.keys())}"
            )

        logger.info("Applying preset: %s", preset_name)
        print(f"[SRSettingsNode] Applying preset: {preset_name}", flush=True)

        for key, value in preset.items():
            self.set_param(key, value)

        # Bypass debounce for immediate batch dispatch
        self.flush()

    # ─────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all parameters to defaults and clear socket overrides."""
        with self._lock:
            self._socket_overrides.clear()
            old_state = dict(self._state)
            self._state = _get_defaults()
            # Mark any changed keys as dirty
            for key in _PARAM_SCHEMA:
                if old_state.get(key) != self._state.get(key):
                    self._dirty.add(key)

        if self._dirty:
            self.flush()

        logger.info("SRSettingsNode reset to defaults")
        print("[SRSettingsNode] Reset to defaults", flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # INTROSPECTION
    # ─────────────────────────────────────────────────────────────────────

    def get_dirty_keys(self) -> Set[str]:
        """Return the current set of dirty keys (for testing / debug)."""
        with self._lock:
            return set(self._dirty)

    def is_in_flight(self) -> bool:
        """Check if an engine dispatch is currently in-flight."""
        with self._lock:
            return self._in_flight

    def __repr__(self) -> str:
        with self._lock:
            model = self._effective("primary_model")
            dirty_count = len(self._dirty)
            override_count = len(self._socket_overrides)
        return (
            f"SRSettingsNode(model={model!r}, "
            f"dirty={dirty_count}, overrides={override_count})"
        )
