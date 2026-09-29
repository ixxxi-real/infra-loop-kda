"""Persistent campaign state for kernel optimization.

The model agent proposes edits; this module owns the campaign lifecycle.  It
is intentionally stdlib-only and does not launch a process or make a
promotion decision from benchmark output.  A runner may persist the returned
record next to an experiment and resume it after a disconnected session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import util

PREPARED = "prepared"
RUNNING = "running"
CANDIDATE_READY = "candidate_ready"
VERIFIED = "verified"
PROMOTED = "promoted"
REJECTED = "rejected"
BLOCKED = "blocked"

TERMINAL = (PROMOTED, REJECTED)
KNOWN = (
    PREPARED,
    RUNNING,
    CANDIDATE_READY,
    VERIFIED,
    PROMOTED,
    REJECTED,
    BLOCKED,
)

# Promotion is deliberately only reachable from verified evidence.  The
# agent can create a candidate, but it cannot skip the evaluator-owned gate.
TRANSITIONS = {
    PREPARED: (RUNNING, BLOCKED),
    RUNNING: (CANDIDATE_READY, BLOCKED),
    CANDIDATE_READY: (RUNNING, VERIFIED, BLOCKED),
    VERIFIED: (PROMOTED, REJECTED, BLOCKED),
    BLOCKED: (PREPARED, RUNNING),
    PROMOTED: (),
    REJECTED: (),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new(
    campaign_id: str,
    task_id: str,
    base_commit: str,
    workload_hash: str,
    gate_version: str,
    max_candidates: int = 4,
) -> Dict[str, Any]:
    """Create a campaign record with immutable experiment identity."""
    if not campaign_id or "/" in campaign_id or ".." in campaign_id:
        raise util.ToolError("campaign_id must be a single safe identifier")
    if not task_id or not base_commit or not workload_hash or not gate_version:
        raise util.ToolError("task, base, workload and gate identities are required")
    if max_candidates < 1:
        raise util.ToolError("max_candidates must be positive")
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "task_id": task_id,
        "base_commit": base_commit,
        "workload_hash": workload_hash,
        "gate_version": gate_version,
        "max_candidates": int(max_candidates),
        "status": PREPARED,
        "candidate_count": 0,
        "candidates": [],
        "history": [
            {"from": None, "to": PREPARED, "at": _now(), "actor": "orchestrator"}
        ],
    }


def validate(record: Dict[str, Any]) -> List[str]:
    """Return invariant violations without changing the record."""
    errors: List[str] = []
    if record.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    status = record.get("status")
    if status not in KNOWN:
        errors.append("unknown campaign status: %r" % status)
    for key in ("campaign_id", "task_id", "base_commit", "workload_hash", "gate_version"):
        if not str(record.get(key) or "").strip():
            errors.append("missing %s" % key)
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be a list")
    else:
        if record.get("candidate_count") != len(candidates):
            errors.append("candidate_count does not match candidates")
        limit = int(record.get("max_candidates") or 0)
        if limit < 1 or len(candidates) > limit:
            errors.append("candidate count exceeds max_candidates")
    if not isinstance(record.get("history"), list) or not record["history"]:
        errors.append("history must be a non-empty list")
    return errors


def transition(
    record: Dict[str, Any],
    target: str,
    actor: str = "orchestrator",
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Move a campaign through an explicit, auditable transition."""
    errors = validate(record)
    if errors:
        raise util.ToolError("invalid campaign: %s" % "; ".join(errors))
    current = str(record["status"])
    if target not in KNOWN:
        raise util.ToolError("unknown campaign status: %s" % target)
    if target not in TRANSITIONS[current]:
        raise util.ToolError("invalid campaign transition: %s -> %s" % (current, target))
    item = dict(record)
    item["status"] = target
    event: Dict[str, Any] = {"from": current, "to": target, "at": _now(), "actor": actor}
    if reason:
        event["reason"] = reason
    item["history"] = list(record["history"]) + [event]
    return item


def add_candidate(record: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a candidate without granting it verified or promoted status."""
    errors = validate(record)
    if errors:
        raise util.ToolError("invalid campaign: %s" % "; ".join(errors))
    if record["status"] not in (RUNNING, CANDIDATE_READY):
        raise util.ToolError("candidates can only be attached while running")
    if len(record["candidates"]) >= int(record["max_candidates"]):
        raise util.ToolError("candidate budget exhausted")
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise util.ToolError("candidate_id is required")
    for key in ("search_lane", "diagnosis_ref", "limiter_class"):
        if not str(candidate.get(key) or "").strip():
            raise util.ToolError("candidate.%s is required" % key)
    candidate_gate = str(candidate.get("gate_version") or "").strip()
    if candidate_gate != record["gate_version"]:
        raise util.ToolError("candidate gate_version must match campaign gate_version")
    candidate.setdefault("repro_count", 0)
    if any(c.get("candidate_id") == candidate_id for c in record["candidates"]):
        raise util.ToolError("duplicate candidate_id: %s" % candidate_id)
    item = dict(record)
    item["candidates"] = list(record["candidates"]) + [dict(candidate)]
    item["candidate_count"] = len(item["candidates"])
    return item


def load(path: Path) -> Dict[str, Any]:
    record = util.read_json(path)
    if not isinstance(record, dict):
        raise util.ToolError("campaign file must contain a JSON object: %s" % path)
    errors = validate(record)
    if errors:
        raise util.ToolError("invalid campaign %s: %s" % (path, "; ".join(errors)))
    return record


def save(path: Path, record: Dict[str, Any]) -> None:
    errors = validate(record)
    if errors:
        raise util.ToolError("invalid campaign: %s" % "; ".join(errors))
    util.write_json(path, record)
