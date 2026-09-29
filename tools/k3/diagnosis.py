"""Stable profiler-to-diagnosis records used by optimization agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import util

LIMITERS = (
    "memory_bandwidth",
    "memory_latency",
    "register_pressure",
    "occupancy",
    "shared_memory",
    "instruction_throughput",
    "launch_overhead",
    "unknown",
)


def normalize(
    payload: Dict[str, Any],
    *,
    profile_hash: Optional[str] = None,
    diagnosis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a profiler parser result without inventing missing metrics."""
    if not isinstance(payload, dict):
        raise util.ToolError("diagnosis payload must be an object")
    out: Dict[str, Any] = {
        "schema_version": 1,
        "diagnosis_id": diagnosis_id or str(payload.get("diagnosis_id") or "unknown"),
        "kernel": str(payload.get("kernel") or "unknown"),
        "phase": str(payload.get("phase") or "unknown"),
        "limiter_class": str(payload.get("limiter_class") or payload.get("limiter") or "unknown"),
        "profile_hash": profile_hash or payload.get("profile_hash"),
        "source": payload.get("source") or "profiler",
        "metrics": dict(payload.get("metrics") or {}),
        "hypothesis": payload.get("hypothesis"),
        "created_at": payload.get("created_at") or util.utcnow(),
    }
    if out["limiter_class"] not in LIMITERS:
        raise util.ToolError("unsupported limiter_class: %s" % out["limiter_class"])
    if not out["kernel"] or not out["phase"]:
        raise util.ToolError("kernel and phase are required")
    if not isinstance(out["metrics"], dict):
        raise util.ToolError("metrics must be an object")
    return out


def validate(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if record.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for key in ("diagnosis_id", "kernel", "phase", "limiter_class", "created_at"):
        if not str(record.get(key) or "").strip():
            errors.append("missing %s" % key)
    if record.get("limiter_class") not in LIMITERS:
        errors.append("unsupported limiter_class")
    if not isinstance(record.get("metrics"), dict):
        errors.append("metrics must be an object")
    return errors


def write(path: Path, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    record = normalize(payload, **kwargs)
    errors = validate(record)
    if errors:
        raise util.ToolError("invalid diagnosis: %s" % "; ".join(errors))
    util.write_json(path, record)
    return record
