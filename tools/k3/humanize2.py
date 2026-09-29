#!/usr/bin/env python3
"""Humanize2 adapter for Infra Loop-KDA.

Humanize2 is intentionally a separate block from :mod:`tools.k3.agent`.
The old adapter owns the Claude plugin and ``.humanize/rlcr`` state; this
module only prepares a reproducible ``hmz exec`` invocation in the task clone
and records the process that was explicitly started by the operator.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as config_mod
from . import gitq, lifecycle, paths, util, workspace

SESSION_STARTING = "starting"
SESSION_RUNNING = "running"
SESSION_EXITED = "exited"
SESSION_STOPPED = "stopped"

# Humanize2's cleanup flow treats paths outside ``work_paths`` as disposable
# scratch.  The task control directory contains the candidate ledger, gate
# reports, failure reasons and profiler provenance, so it must always be in the
# protected set even when an older local config only listed ``python``.
PROTECTED_EVIDENCE_PATHS = (".kda-task",)
REVIEW_SESSION_POLICY = "fresh-per-turn"
REVIEW_MODE = "advisory-only"
FINAL_REVIEW_BACKEND = "claude"


def runtime_dir(task_dir: Path) -> Path:
    return task_dir / "runtime" / "humanize2"


def plan_path(task_dir: Path) -> Path:
    return runtime_dir(task_dir) / "plan.json"


def session_path(task_dir: Path) -> Path:
    return runtime_dir(task_dir) / "session.json"


def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return config_mod.workflow_settings(config).get("humanize2") or {}


def _read_task_text(control: Optional[Path], task: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Build a prompt that points at frozen files instead of copying them.

    Control files can contain private paths or runtime metadata.  Humanize2
    runs in the workspace and can read those files itself, so the invocation
    record never needs to duplicate their contents.
    """
    files: List[str] = []
    if control is not None:
        for name in ("plan.md", "contract.md", "source-trace.md"):
            target = control / name
            if target.is_file():
                files.append(name)
    objective = str(task.get("objective") or "Optimize the configured kernel task.")
    policy = (
        "Optimization workflow rules: every coding/review turn must start a fresh "
        "agent session; the second chaser is an independent reviewer. The reviewer "
        "may inspect the candidate and write a review record, but its pass/reject "
        "reason never substitutes for correctness, precision or performance gates. "
        "Run every candidate against the immutable baseline with interleaved paired "
        "trials and repeated samples; a smoke run is diagnostic only. Keep the "
        "candidate ledger, benchmark summaries, rejected-candidate reasons, failure "
        "logs and profiler originals under .kda-task/evidence (or another path under "
        ".kda-task). A cleanup turn must never delete or rewrite those records. "
        "Before a final promotion, request an independent Claude review or human "
        "review using the frozen evidence bundle."
    )
    if files:
        references = ", ".join(".kda-task/%s" % name for name in files)
        prompt = (
            "%s\n\nRead the frozen task files %s before editing. Work only inside "
            "the configured Humanize2 work_paths. Preserve the task contract, "
            "run its correctness and performance gates, and record evidence "
            "for every candidate.\n\n%s" % (objective, references, policy)
        )
    else:
        prompt = (
            "%s\n\nNo frozen control files were found. Inspect the task tree and "
            "establish the required correctness and performance gates before editing.\n\n%s"
            % (objective, policy)
        )
        files.append("task.json:objective")
    return prompt, files


def _session_state(session: Dict[str, Any]) -> Dict[str, Any]:
    state = session.get("state")
    if state in (SESSION_EXITED, SESSION_STOPPED):
        return {"state": str(state), "detail": "recorded as %s" % state}
    pid = session.get("pid")
    identity = session.get("process_identity")
    if not isinstance(pid, int):
        return {"state": SESSION_EXITED, "detail": "no pid recorded"}
    matched, detail = util.process_matches(pid, identity)
    return {
        "state": SESSION_RUNNING if matched else SESSION_EXITED,
        "detail": detail,
    }


def _load_session(task_dir: Path) -> Optional[Dict[str, Any]]:
    target = session_path(task_dir)
    if not target.is_file():
        return None
    try:
        value = config_mod.load_json(target)
    except config_mod.ValidationError:
        return None
    return value


def _workspace_inputs(
    root: Path, config: Dict[str, Any], task_id: str, resume: bool
) -> Tuple[Dict[str, Any], Path, Optional[Path], List[str]]:
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    blockers = list(lifecycle.start_blockers(task))
    # Humanize1 and Humanize2 must never edit the same candidate tree at the
    # same time.  Read the old adapter's session record without importing its
    # private state machine; a PID is accepted only with its recorded identity.
    old_session_path = task_dir / "runtime" / "agent" / "session.json"
    if old_session_path.is_file():
        try:
            old_session = config_mod.load_json(old_session_path)
        except config_mod.ValidationError:
            old_session = {}
        old_pid = old_session.get("pid")
        if isinstance(old_pid, int):
            matched, _ = util.process_matches(old_pid, old_session.get("process_identity"))
            if matched:
                blockers.append(
                    "Humanize1 agent session is still running (pid %s); stop it before "
                    "starting Humanize2" % old_pid
                )
    try:
        record = workspace.load_record(root, config, task_id)
    except util.ToolError as exc:
        return task, task_dir, None, blockers + [str(exc)]
    info = record.get("workspace") or {}
    raw_candidate = str(info.get("absolute_path") or "").strip()
    candidate: Optional[Path] = Path(raw_candidate) if raw_candidate else None
    if candidate is None or not candidate.is_dir():
        blockers.append("prepared workspace is missing from disk: %s" % candidate)
        return task, task_dir, candidate, blockers
    if record.get("mode") != workspace.MODE_CLONE:
        blockers.append("Humanize2 requires a clone workspace with commit history")
    control = candidate / workspace.CONTROL_DIR
    if not control.is_dir():
        blockers.append("workspace control directory is missing: %s" % control)
    if not resume:
        visible = [line for line in (gitq.porcelain(candidate) or []) if ".humanize" not in line]
        if visible:
            blockers.append(
                "workspace working tree is not clean (%d entries); use humanize2 resume "
                "for an existing run" % len(visible)
            )
    return task, task_dir, candidate, blockers


def build_invocation(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    resume: bool = False,
) -> Dict[str, Any]:
    """Create a dry-run record.  This function never spawns ``hmz``."""
    task, task_dir, candidate, blockers = _workspace_inputs(root, config, task_id, resume)
    settings = _settings(config)
    runtime = str(settings.get("runtime_dir") or "external/humanize2")
    command = str(settings.get("command") or "hmz")
    flow = str(settings.get("flow") or "flame_chase_agent_cleanup")
    # A checked-in flowverse path is resolved from the control-plane root, not
    # from the candidate clone.  This keeps the run offline/reproducible while
    # still allowing Humanize2's normal git+URL#flow refs in other configs.
    flow_ref = Path(flow).expanduser()
    if not flow_ref.is_absolute():
        candidate_flow = root / flow_ref
        if candidate_flow.exists():
            flow = str(candidate_flow.resolve())
    runtime_path = paths.safe_join(root, runtime)
    if not (runtime_path / "pyproject.toml").is_file() or not (runtime_path / "src" / "hmz").is_dir():
        blockers.append("Humanize2 runtime is not initialized: %s" % runtime)
    executable = util.which(command)
    if executable is None:
        blockers.append("Humanize2 command %r is not on PATH" % command)
    work_paths = settings.get("work_paths") or ["python"]
    if not isinstance(work_paths, list):
        work_paths = [str(work_paths)]
    safe_work_paths: List[str] = []
    if candidate is not None and candidate.is_dir():
        for raw in work_paths:
            relative = str(raw).strip()
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                blockers.append("workflow.humanize2.work_paths contains an unsafe path: %r" % raw)
                continue
            safe_work_paths.append(relative)
            if not (candidate / relative).exists():
                blockers.append("Humanize2 work path is missing: %s" % relative)
    else:
        safe_work_paths = [str(item) for item in work_paths]
    # Protect the control/evidence directory even for a stale local config that
    # predates this policy. It already exists in every prepared workspace and
    # contains the frozen task files read above.
    for relative in PROTECTED_EVIDENCE_PATHS:
        if candidate is not None and candidate.is_dir() and not (candidate / relative).exists():
            blockers.append("Humanize2 protected evidence path is missing: %s" % relative)
        if relative not in safe_work_paths:
            safe_work_paths.append(relative)

    control = candidate / workspace.CONTROL_DIR if candidate is not None else None
    task_text, prompt_files = _read_task_text(control if control and control.is_dir() else None, task)
    argv: List[str] = [command, "exec", "-f", flow]
    for role in ("first_chaser", "second_chaser", "cleaner"):
        value = str(settings.get(role) or "").strip()
        if value:
            argv += ["-a", "%s=%s" % (role, value)]
    argv += ["-p", "work_paths=%s" % ",".join(safe_work_paths)]
    budget = str(settings.get("budget") or "duration=12h,cost=100").strip()
    if budget:
        argv += ["-b", budget]
    if resume:
        argv.append("--resume")
    argv.append(task_text)
    codex_home = settings.get("codex_home")
    env: Dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if isinstance(codex_home, str) and codex_home.strip():
        env["CODEX_HOME"] = str(Path(os.path.expandvars(os.path.expanduser(codex_home))).resolve())
    identity = {
        "argv": argv[:-1] + ["<task-text>"],
        "cwd": str(candidate) if candidate else None,
        "env": env,
        "runtime": runtime,
        "flow": flow,
        "prompt_files": prompt_files,
        "resume": bool(resume),
    }
    record = {
        "schema": "k3ctl/humanize2-plan/1",
        "generated_at": util.utcnow(),
        "task_id": task_id,
        "config": str(config_path),
        "workflow": {
            "runtime_dir": runtime,
            "command": command,
            "flow": flow,
            "first_chaser": settings.get("first_chaser"),
            "second_chaser": settings.get("second_chaser"),
            "cleaner": settings.get("cleaner"),
            "work_paths": safe_work_paths,
            "reviewer_session_policy": REVIEW_SESSION_POLICY,
            "review_mode": REVIEW_MODE,
            "protected_evidence_paths": list(PROTECTED_EVIDENCE_PATHS),
            "final_review": {"required": True, "backend": FINAL_REVIEW_BACKEND},
            "budget": settings.get("budget") or "duration=12h,cost=100",
            "resume": bool(resume),
        },
        "lifecycle": lifecycle.describe(task),
        "invocation": {
            "argv": argv,
            "cwd": str(candidate) if candidate else None,
            "env": env,
            "shell": False,
        },
        "identity": {"sha256": util.sha256_json(identity)},
        "ready": not blockers,
        "not_ready_reasons": blockers,
        "would_spawn": False,
        "note": "Dry run only. Humanize2 owns flow state, traces and agent sessions.",
    }
    util.write_json(plan_path(task_dir), record)
    return record


def plan(root: Path, config_path: Path, config: Dict[str, Any], task_id: str, resume: bool = False) -> Dict[str, Any]:
    return build_invocation(root, config_path, config, task_id, resume=resume)


def start(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    resume: bool = False,
) -> Dict[str, Any]:
    task_dir = config_mod.task_dir(root, config, task_id)
    existing = _load_session(task_dir)
    if existing and _session_state(existing)["state"] == SESSION_RUNNING:
        raise util.ToolError("a Humanize2 session is already running (pid %s)" % existing.get("pid"))
    record = plan(root, config_path, config, task_id, resume=resume)
    if not record["ready"]:
        raise util.ToolError(
            "task %s is not ready for Humanize2:\n  - %s"
            % (task_id, "\n  - ".join(record["not_ready_reasons"]))
        )
    runtime = runtime_dir(task_dir)
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (util.new_uuid() + ".log")
    argv = list(record["invocation"]["argv"])
    cwd = Path(str(record["invocation"]["cwd"]))
    env = dict(os.environ)
    env.update(record["invocation"].get("env") or {})
    with log.open("wb") as handle:
        process = subprocess.Popen(  # noqa: S603 - argv is frozen above, shell=False
            argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(0.05)
    session = {
        "schema": "k3ctl/humanize2-session/1",
        "task_id": task_id,
        "config": str(config_path),
        "state": SESSION_RUNNING,
        "pid": process.pid,
        "process_identity": util.process_identity(process.pid),
        "started_at": util.utcnow(),
        "log": str(log),
        "plan_sha256": record["identity"]["sha256"],
        "resume": bool(resume),
    }
    util.write_json(session_path(task_dir), session)
    return session


def status(root: Path, config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task_dir = config_mod.task_dir(root, config, task_id)
    session = _load_session(task_dir)
    result: Dict[str, Any] = {
        "task_id": task_id,
        "runtime_dir": str(runtime_dir(task_dir)),
        "plan": str(plan_path(task_dir)) if plan_path(task_dir).is_file() else None,
        "session": None,
    }
    if session is not None:
        state = _session_state(session)
        result["session"] = dict(session, **state)
    return result


def stop(root: Path, config: Dict[str, Any], task_id: str, timeout: float = 15.0) -> Dict[str, Any]:
    task_dir = config_mod.task_dir(root, config, task_id)
    session = _load_session(task_dir)
    if session is None:
        return {"task_id": task_id, "stopped": False, "detail": "no Humanize2 session"}
    state = _session_state(session)
    if state["state"] != SESSION_RUNNING:
        return {"task_id": task_id, "stopped": False, "state": state["state"], "detail": state["detail"]}
    pid = int(session["pid"])
    matched, detail = util.process_matches(pid, session.get("process_identity"))
    if not matched:
        return {"task_id": task_id, "stopped": False, "detail": detail}
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline and util.pid_alive(pid):
        time.sleep(0.1)
    if util.pid_alive(pid):
        os.killpg(pid, signal.SIGKILL)
    session["state"] = SESSION_STOPPED
    session["stopped_at"] = util.utcnow()
    util.write_json(session_path(task_dir), session)
    return {"task_id": task_id, "stopped": True, "pid": pid}
