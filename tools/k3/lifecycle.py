#!/usr/bin/env python3
"""Task lifecycle states and the gates that guard process launch.

A fresh scaffold is deliberately unstartable, and an accepted or closed task
can only be reproduced or inspected. Reopening one for a new optimization
requires creating a new task, never a silent state change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCAFFOLDED = "scaffolded"
PREPARED = "prepared"
READY = "ready"
CANDIDATE_RUNNING = "candidate_running"
IN_PROGRESS = "in_progress"
CANDIDATE_VERIFIED = "candidate_verified"
ACCEPTED = "accepted"
DELIVERED = "delivered"
REJECTED = "rejected"
CLOSED = "closed"
PAUSED = "paused"
BLOCKED = "blocked"

#: States from which a new optimization loop may be started.
#:
#: ``prepared`` and ``ready`` are equivalent for this purpose: the documented
#: lifecycle calls the "inputs frozen" state ``prepared``, and ``ready`` is
#: accepted as a synonym. ``candidate_running`` and ``in_progress`` are likewise
#: equivalent, and permit resuming an interrupted loop.
STARTABLE = (PREPARED, READY, CANDIDATE_RUNNING, IN_PROGRESS)

#: States that are immutable for optimization purposes. Reopening one requires
#: creating a new task, never a silent state change.
TERMINAL = (ACCEPTED, DELIVERED, REJECTED, CLOSED)

#: States that stop progress deliberately, with a recorded reason. Not terminal:
#: the task resumes once the blocker is resolved and the state is moved back.
HELD = (PAUSED, BLOCKED)

#: Evidence exists but has not been accepted. Not startable as new optimization
#: work, and not terminal either.
VERIFIED = (CANDIDATE_VERIFIED,)

#: Every state this module recognises. An unrecognised value is refused rather
#: than guessed at, so a typo can never be treated as startable.
KNOWN = (
    SCAFFOLDED,
    PREPARED,
    READY,
    CANDIDATE_RUNNING,
    IN_PROGRESS,
    CANDIDATE_VERIFIED,
    ACCEPTED,
    DELIVERED,
    REJECTED,
    CLOSED,
    PAUSED,
    BLOCKED,
)

UNRESOLVED = "unresolved"
PLACEHOLDER = "placeholder"
RESOLVED = "resolved"


def task_status(task: Dict[str, Any]) -> str:
    return str(task.get("status") or SCAFFOLDED)


def is_terminal(task: Dict[str, Any]) -> bool:
    return task_status(task) in TERMINAL


def start_blockers(task: Dict[str, Any]) -> List[str]:
    """Reasons a task must not spawn a new optimization loop.

    An empty list means the lifecycle state permits a start; it is not by
    itself a readiness verdict.
    """
    reasons: List[str] = []
    status = task_status(task)

    if status in TERMINAL:
        reasons.append(
            "task status is %r: an accepted, delivered, rejected or closed task "
            "is immutable. To reproduce or inspect it, run 'k3ctl workspace "
            "prepare' (which materialises its exact base without starting a "
            "loop). For new optimization work, run 'k3ctl task-create'." % status
        )
    elif status == SCAFFOLDED:
        reasons.append(
            "task status is 'scaffolded': resolve the workload and write the "
            "plan input, then set status to 'prepared'"
        )
    elif status in HELD:
        reasons.append(
            "task status is %r: progress was stopped deliberately. Resolve the "
            "recorded blocker and move the status back to 'prepared' before "
            "starting a loop." % status
        )
    elif status in VERIFIED:
        reasons.append(
            "task status is %r: candidate evidence already exists and is awaiting "
            "review. Accept or reject it, or create a new task for further "
            "optimization." % status
        )
    elif status not in STARTABLE:
        known = "known states: %s" % ", ".join(KNOWN)
        reasons.append(
            "task status %r is not startable (expected one of: %s). %s"
            % (status, ", ".join(STARTABLE), known)
        )

    # Allow-list, not deny-list: any value other than an explicit "resolved"
    # blocks the start, so an unknown or misspelled status cannot pass.
    workload_status = str(task.get("workload_status") or UNRESOLVED)
    if workload_status != RESOLVED:
        reasons.append(
            "workload_status is %r, expected %r: only an explicitly resolved "
            "workload may drive a loop" % (workload_status, RESOLVED)
        )

    if task.get("optimization_status") == ACCEPTED:
        reasons.append(
            "optimization_status is 'accepted': the accepted result is frozen "
            "and must not be reopened implicitly"
        )
    return reasons


def describe(task: Dict[str, Any]) -> Dict[str, Any]:
    """Compact lifecycle view for status output."""
    return {
        "status": task_status(task),
        "optimization_status": task.get("optimization_status", "unknown"),
        "integration_status": task.get("integration_status", "unknown"),
        "serving_validation": task.get("serving_validation", "not_performed"),
        "workload_status": task.get("workload_status", UNRESOLVED),
        "terminal": is_terminal(task),
        "start_blockers": start_blockers(task),
    }


def require_startable(task: Dict[str, Any], task_id: str) -> None:
    """Raise when the lifecycle state forbids starting a loop."""
    from . import util

    reasons = start_blockers(task)
    if reasons:
        raise util.ToolError(
            "task %s cannot start an optimization loop:\n  - %s"
            % (task_id, "\n  - ".join(reasons))
        )


def base_commit(task: Dict[str, Any]) -> Optional[str]:
    source = task.get("source")
    if isinstance(source, dict) and source.get("commit"):
        return str(source["commit"])
    if task.get("base_commit"):
        return str(task["base_commit"])
    return None
