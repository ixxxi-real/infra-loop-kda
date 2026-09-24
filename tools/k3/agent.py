#!/usr/bin/env python3
"""Claude + Humanize adapter.

``agent plan`` is a reproducible dry run: it freezes the exact argv, cwd,
non-secret environment, toolchain identity and contract identity, and lists
every reason the task is not ready. It never spawns a process.

``agent start`` launches Claude itself, which then invokes the official
``/humanize:start-rlcr-loop`` command. This adapter never calls the Humanize
setup script directly, never fabricates loop state, and never writes a review
verdict: the pinned plugin's own hooks remain the only components that decide
outcomes.

``agent status`` reads the real Claude stream and the plugin's own loop state,
binding to the loop by session id. A zero exit code alone is never reported as
acceptance.

``agent resume`` reuses the frozen invocation from the session record, and
refuses once the loop has reached a terminal state.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as config_mod
from . import gitq, lifecycle, overlay, paths, toolchain, util, workspace

#: Non-secret environment keys this adapter declares and records.
DECLARED_ENV_KEYS = (
    "CLAUDE_PROJECT_DIR",
    "K3_NO_COMMIT_REVIEW",
    "K3_PLAN_FILE",
    "K3_PLAN_SHA256",
    "K3_PLAN_SNAPSHOT",
    "K3_PLAN_SNAPSHOT_SHA256",
    "CODEX_HOME",
    "HUMANIZE_CODEX_BYPASS_SANDBOX",
    "KDA_REVIEWER_REAL_BIN",
    "KDA_REVIEWER_MODEL",
    "KDA_REVIEWER_EFFORT",
    "KDA_REVIEWER_DISABLE_APPS",
)

#: Substrings that indicate a value must never be recorded or printed.
SECRET_HINTS = ("token", "secret", "key", "password", "credential", "cookie")

SESSION_STARTING = "starting"
SESSION_RUNNING = "running"
SESSION_EXITED = "exited"
SESSION_STOPPED = "stopped"

#: State-file names the plugin treats as an active loop.
ACTIVE_STATE_FILES = (
    ("finalize-state.md", "finalize"),
    ("methodology-analysis-state.md", "methodology-analysis"),
    ("state.md", "normal"),
)

#: ``end_loop`` renames the state file to ``<reason>-state.md`` on exit.
TERMINAL_REASONS = ("complete", "cancel", "maxiter", "stop", "unexpected")


def agent_dir(task_dir: Path) -> Path:
    return task_dir / "runtime" / "agent"


def plan_path(task_dir: Path) -> Path:
    return agent_dir(task_dir) / "plan.json"


def session_path(task_dir: Path) -> Path:
    return agent_dir(task_dir) / "session.json"


def lock_path(task_dir: Path) -> Path:
    return agent_dir(task_dir) / "task.lock"


def log_dir(task_dir: Path, uuid: str) -> Path:
    return agent_dir(task_dir) / "logs" / uuid


# --------------------------------------------------------------- identities


def contract_identity(control: Path) -> Dict[str, Any]:
    """Hash the contract inputs that define the task's scope."""
    files: Dict[str, str] = {}
    for name in (
        "task.json",
        "contract.md",
        "plan.md",
        "plan-snapshot.md",
        "source-trace.md",
        "prompt.md",
    ):
        target = control / name
        if target.is_file():
            files[name] = util.sha256_file(target)
    return {"files": files, "sha256": util.sha256_json(files)}


def _plan_files(control: Path) -> Tuple[Optional[Path], Optional[Path]]:
    live = control / "plan.md"
    snapshot = control / "plan-snapshot.md"
    return (live if live.is_file() else None, snapshot if snapshot.is_file() else None)


# ------------------------------------------------------------------- argv


def build_prompt(
    plan_relative: str,
    reviewer: Dict[str, Any],
    base_branch: Optional[str],
    max_iterations: int,
    start_command: str,
    skip_quiz: bool = False,
) -> str:
    """Build the slash-command prompt Claude runs to start the official loop."""
    parts = [start_command, plan_relative]
    model = str(reviewer.get("model") or "").strip()
    effort = str(reviewer.get("effort") or "").strip()
    if model:
        parts.append("--codex-model")
        parts.append("%s:%s" % (model, effort) if effort else model)
    if base_branch:
        parts.append("--base-branch")
        parts.append(base_branch)
    if skip_quiz:
        # Humanize owns the quiz. The adapter may disable only that advisory
        # interaction when the workflow has recorded an explicit decision;
        # compliance and setup still run normally.
        parts.append("--skip-quiz")
    parts.append("--max")
    parts.append(str(int(max_iterations)))
    return " ".join(parts)


def build_argv(
    writer: Dict[str, Any],
    plugin_dir: Path,
    prompt: str,
    session_id: Optional[str] = None,
) -> List[str]:
    """Freeze the exact Claude argv. No shell, no interpolation.

    ``session_id`` is chosen by this adapter before launch so the loop can be
    bound deterministically, and so ``resume`` never has to scrape it back out
    of a stream log.
    """
    argv = [
        str(writer.get("command") or "claude"),
        "-p",
        prompt,
        "--plugin-dir",
        str(plugin_dir),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if session_id:
        argv += ["--session-id", str(session_id)]
    model = str(writer.get("model") or "").strip()
    if model:
        argv += ["--model", model]
    # The configured writer effort is passed through. It is never dropped and
    # never silently downgraded.
    effort = str(writer.get("effort") or "").strip()
    if effort:
        argv += ["--effort", effort]
    permission_mode = str(writer.get("permission_mode") or "").strip()
    if permission_mode:
        argv += ["--permission-mode", permission_mode]
    return argv


def build_resume_argv(
    writer: Dict[str, Any],
    plugin_dir: Path,
    claude_session_id: str,
) -> List[str]:
    """Rebuild the resume argv from frozen session settings."""
    argv = [
        str(writer.get("command") or "claude"),
        "--resume",
        str(claude_session_id),
        "-p",
        "Continue the active RLCR loop. Do not start a new loop.",
        "--plugin-dir",
        str(plugin_dir),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    model = str(writer.get("model") or "").strip()
    if model:
        argv += ["--model", model]
    effort = str(writer.get("effort") or "").strip()
    if effort:
        argv += ["--effort", effort]
    permission_mode = str(writer.get("permission_mode") or "").strip()
    if permission_mode:
        argv += ["--permission-mode", permission_mode]
    return argv


def strip_session_id(argv: List[str]) -> List[str]:
    """Remove ``--session-id`` and its value as a pair.

    The session id is chosen per launch, so it must not contribute to the plan
    identity: a dry run and the real start of the same configuration have to
    produce the same hash.

    Removing only the *value* leaves a dangling ``--session-id`` flag, which the
    dry-run argv does not contain -- so the two hashes would differ anyway, and
    the documented "plan and start identities agree" property would be false.
    The flag and its argument are therefore dropped together.
    """
    result: List[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--session-id":
            index += 2  # skip the flag and its value
            continue
        result.append(argv[index])
        index += 1
    return result


def build_env(
    workspace_path: Path,
    no_commit: bool,
    live_plan: Optional[Path],
    snapshot: Optional[Path],
    reviewer: Optional[Dict[str, Any]] = None,
    reviewer_binary: Optional[str] = None,
) -> Dict[str, str]:
    """Declared, non-secret environment overrides.

    Humanize invokes ``codex`` by literal name from shell hooks.  A shell alias
    such as ``codex-bak`` is therefore invisible to it.  The reviewer settings
    are exported explicitly and the PATH shim below turns them into a checked
    launcher, so the nested reviewer cannot silently fall back to another
    Codex home or profile.
    """
    reviewer = reviewer or {}
    env: Dict[str, str] = {
        "CLAUDE_PROJECT_DIR": str(workspace_path),
        "K3_NO_COMMIT_REVIEW": "true" if no_commit else "false",
    }
    if live_plan is not None:
        env["K3_PLAN_FILE"] = str(live_plan)
        env["K3_PLAN_SHA256"] = util.sha256_file(live_plan)
    if snapshot is not None:
        env["K3_PLAN_SNAPSHOT"] = str(snapshot)
        env["K3_PLAN_SNAPSHOT_SHA256"] = util.sha256_file(snapshot)
    codex_home = reviewer.get("codex_home")
    if isinstance(codex_home, str) and codex_home.strip():
        env["CODEX_HOME"] = configured_codex_home(codex_home)
    if reviewer.get("bypass_sandbox") is True:
        # This also makes Humanize choose its supported bypass flag instead of
        # appending the obsolete --full-auto flag to modern Codex commands.
        env["HUMANIZE_CODEX_BYPASS_SANDBOX"] = "true"
    if reviewer_binary:
        env["KDA_REVIEWER_REAL_BIN"] = reviewer_binary
    model = reviewer.get("model")
    effort = reviewer.get("effort")
    if isinstance(model, str) and model.strip():
        env["KDA_REVIEWER_MODEL"] = model.strip()
    if isinstance(effort, str) and effort.strip():
        env["KDA_REVIEWER_EFFORT"] = effort.strip()
    if reviewer.get("disable_apps") is True:
        env["KDA_REVIEWER_DISABLE_APPS"] = "true"
    return env


def configured_codex_home(value: Any) -> str:
    """Expand a configured Codex home without invoking a shell."""
    return str(Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve())


def redact_env(env: Dict[str, str]) -> Dict[str, str]:
    """Drop anything that looks like a secret before recording or printing."""
    safe: Dict[str, str] = {}
    for key, value in sorted(env.items()):
        lowered = key.lower()
        if any(hint in lowered for hint in SECRET_HINTS):
            safe[key] = "<redacted>"
        else:
            safe[key] = value
    return safe


# ------------------------------------------------------------- PATH control
#
# The Humanize hooks invoke the literal names ``codex``, ``jq`` and (through
# their shebangs) ``bash``. Recording a configured reviewer command, or a
# doctor-verified bash 5, proves nothing if the hooks resolve a different
# binary from PATH at runtime. These helpers resolve the exact binaries and
# expose them through a controlled shim directory that is prepended to PATH,
# so the verified toolchain is the one that actually runs.

#: Names the pinned Humanize hooks invoke directly.
SHIM_NAMES = ("codex", "bash", "jq")


def resolve_toolchain_paths(
    config: Dict[str, Any]
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Resolve the absolute binaries the loop will actually use.

    Returns ``(resolved, problems)``. ``problems`` is non-empty when something
    the loop genuinely needs is missing or too old, so the caller can refuse
    instead of reporting a success that depends on an unverified PATH.
    """
    settings = config_mod.workflow_settings(config)
    writer = settings.get("writer") or {}
    reviewer = settings.get("reviewer") or {}
    problems: List[str] = []
    resolved: Dict[str, Optional[str]] = {
        "writer": None,
        "codex": None,
        "bash": None,
        "jq": None,
    }

    writer_command = str(writer.get("command") or "claude")
    writer_path = util.which(writer_command)
    if writer_path is None:
        problems.append("writer command %r is not on PATH" % writer_command)
    resolved["writer"] = writer_path

    # The reviewer binary is exposed to the hooks as `codex`, whatever it is
    # called locally. A differently-named command is supported through the
    # shim; it is never silently ignored.
    reviewer_command = str(reviewer.get("command") or "codex")
    reviewer_path = util.which(reviewer_command)
    if reviewer_path is None:
        problems.append(
            "reviewer command %r is not on PATH; the Humanize hooks invoke "
            "`codex` and the loop cannot review without it" % reviewer_command
        )
    resolved["codex"] = reviewer_path

    configured_home = reviewer.get("codex_home")
    if isinstance(configured_home, str) and configured_home.strip():
        home = Path(configured_codex_home(configured_home))
        if not home.is_dir():
            problems.append(
                "configured CODEX_HOME does not exist: %s" % home
            )

    bash_check = toolchain.bash_check()
    if bash_check["status"] != toolchain.OK:
        problems.append(
            "bash is unusable for the Humanize hooks: %s" % bash_check["detail"]
        )
    else:
        resolved["bash"] = bash_check.get("path")

    jq_path = util.which("jq")
    if jq_path is None:
        problems.append(
            "jq is not on PATH; the Humanize hooks require it for every gate"
        )
    resolved["jq"] = jq_path

    return resolved, problems


def shim_dir(task_dir: Path, uuid: str) -> Path:
    return agent_dir(task_dir) / "shim" / uuid


def build_shim_dir(
    task_dir: Path,
    uuid: str,
    resolved: Dict[str, Optional[str]],
    reviewer: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a PATH shim exposing exactly the resolved binaries.

    ``codex`` is linked from the configured reviewer command, so a locally
    renamed reviewer still reaches the hooks under the name they invoke.
    """
    target = shim_dir(task_dir, uuid)
    target.mkdir(parents=True, exist_ok=True)
    reviewer = reviewer or {}
    isolated_reviewer = bool(
        reviewer.get("codex_home")
        or reviewer.get("bypass_sandbox") is True
        or reviewer.get("disable_apps") is True
    )
    for name in SHIM_NAMES:
        source = resolved.get(name)
        if not source:
            continue
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if name == "codex" and isolated_reviewer:
            model = str(reviewer.get("model") or "gpt-6-astra")
            effort = str(reviewer.get("effort") or "ultra")
            lines = [
                "#!/usr/bin/env bash",
                "set -eu",
                "real_bin=${KDA_REVIEWER_REAL_BIN:?KDA_REVIEWER_REAL_BIN is not set}",
                "args=()",
                "args+=(--dangerously-bypass-approvals-and-sandbox)",
                "args+=(--model %s)" % shlex.quote(model),
                "args+=(-c %s)" % shlex.quote('model_reasoning_effort="%s"' % effort),
                "if [ \"${KDA_REVIEWER_DISABLE_APPS:-false}\" = true ]; then",
                "  args+=(--disable apps)",
                "fi",
                "exec \"$real_bin\" \"${args[@]}\" \"$@\"",
                "",
            ]
            # Bash is guaranteed by the Humanize toolchain check and is
            # exposed through the same PATH shim.  Arrays keep every flag an
            # argv element and avoid shell parsing of reviewer input.
            link.write_text("\n".join(lines), encoding="utf-8")
            link.chmod(0o700)
        else:
            os.symlink(str(source), str(link))
    return target


def verify_shim(
    shim: Path,
    resolved: Dict[str, Optional[str]],
    reviewer: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Confirm a shim still points at the recorded binaries."""
    problems: List[str] = []
    if not shim.is_dir():
        return ["PATH shim directory is missing: %s" % shim]
    reviewer = reviewer or {}
    isolated_reviewer = bool(
        reviewer.get("codex_home")
        or reviewer.get("bypass_sandbox") is True
        or reviewer.get("disable_apps") is True
    )
    for name in SHIM_NAMES:
        expected = resolved.get(name)
        if not expected:
            continue
        link = shim / name
        if not link.exists():
            problems.append("PATH shim is missing %s" % name)
            continue
        if name == "codex" and isolated_reviewer:
            try:
                content = link.read_text(encoding="utf-8")
            except OSError as exc:
                problems.append("reviewer launcher cannot be read: %s" % exc)
                continue
            required = (
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                "model_reasoning_effort",
                "--disable apps",
            )
            for marker in required:
                if marker not in content:
                    problems.append("reviewer launcher is missing %s" % marker)
            continue
        try:
            actual = os.readlink(str(link))
        except OSError:
            actual = str(link)
        if actual != expected:
            problems.append(
                "PATH shim %s points at %s, expected %s" % (name, actual, expected)
            )
    return problems


def shim_env(base: Dict[str, str], shim: Path) -> Dict[str, str]:
    """Prepend the shim directory to PATH."""
    env = dict(base)
    existing = env.get("PATH") or os.environ.get("PATH") or ""
    env["PATH"] = "%s%s%s" % (str(shim), os.pathsep, existing) if existing else str(shim)
    return env


# ------------------------------------------------------------------- plan


def plan(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    max_iterations: int = 42,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Reproducible dry run. Never spawns a process."""
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    settings = config_mod.workflow_settings(config)
    writer = settings.get("writer") or {}
    reviewer = settings.get("reviewer") or {}
    humanize = settings.get("humanize") or {}
    overlay_settings = config_mod.overlay_settings(config)

    not_ready: List[str] = list(lifecycle.start_blockers(task))

    # --- workspace -------------------------------------------------------
    workspace_record: Optional[Dict[str, Any]] = None
    workspace_path: Optional[Path] = None
    base_branch: Optional[str] = None
    control: Optional[Path] = None
    try:
        workspace_record = workspace.load_record(root, config, task_id)
    except util.ToolError as exc:
        not_ready.append(str(exc))
    else:
        info = workspace_record.get("workspace") or {}
        workspace_path = Path(str(info.get("absolute_path") or ""))
        base_branch = info.get("review_base_branch")
        if not workspace_path.is_dir():
            not_ready.append(
                "prepared workspace is missing from disk: %s" % workspace_path
            )
        elif workspace_record.get("mode") != workspace.MODE_CLONE:
            not_ready.append(
                "workspace mode is %r: the Humanize loop requires commit history, "
                "so prepare it with --mode clone"
                % workspace_record.get("mode")
            )
        else:
            if not gitq.head_exists(workspace_path):
                not_ready.append(
                    "workspace has no commit; the Humanize setup script requires "
                    "at least one commit"
                )
            if not base_branch:
                not_ready.append("workspace does not record a review base branch")
            control = workspace_path / workspace.CONTROL_DIR

    # --- plan file -------------------------------------------------------
    live_plan: Optional[Path] = None
    snapshot: Optional[Path] = None
    plan_relative: Optional[str] = None
    if control is not None:
        live_plan, snapshot = _plan_files(control)
        if live_plan is None:
            not_ready.append(
                "no plan at %s/plan.md; write the task plan input and re-run "
                "'workspace prepare'" % workspace.CONTROL_DIR
            )
        else:
            plan_relative = "%s/plan.md" % workspace.CONTROL_DIR
            # Upstream requires the plan to be untracked when --track-plan-file
            # is not used, and the working tree to be clean.
            if gitq.is_tracked(workspace_path, plan_relative):
                not_ready.append(
                    "plan file %s is tracked in the workspace; it must be "
                    "untracked/ignored for the loop to start" % plan_relative
                )
            dirty = gitq.porcelain(workspace_path) or []
            visible = [line for line in dirty if ".humanize" not in line]
            if visible:
                not_ready.append(
                    "workspace working tree is not clean (%d entries); the "
                    "Humanize setup script requires a clean tree at start"
                    % len(visible)
                )

    # --- toolchain -------------------------------------------------------
    report = toolchain.inspect(
        root, config_path, config, agent_profile=True, task_id=task_id
    )
    if report["status"] == "failed":
        not_ready.append(
            "toolchain check failed: missing=%s incompatible=%s"
            % (", ".join(report["missing"]) or "-", ", ".join(report["incompatible"]) or "-")
        )
    availability = report["source_availability"]
    if availability.get("task_base_available") is False:
        not_ready.append(
            "task base commit %s is not available locally"
            % (availability.get("task_base_commit") or "?")
        )

    # --- binaries the hooks will actually resolve -------------------------
    # Recording a verified toolchain is meaningless if the hooks resolve a
    # different binary from PATH at runtime, so the exact paths are resolved
    # here and exposed through a controlled shim at launch.
    resolved_paths, path_problems = resolve_toolchain_paths(config)
    not_ready.extend(path_problems)

    # --- plugin / overlay ------------------------------------------------
    plugin_dir: Optional[Path] = None
    provenance: Dict[str, Any] = {}
    try:
        plugin_dir, provenance = overlay.resolve_plugin_dir(root, config)
    except util.ToolError as exc:
        not_ready.append(str(exc))

    effort = str(reviewer.get("effort") or "").strip()
    if effort and effort not in config_mod.UPSTREAM_EFFORTS:
        if not overlay_settings["enabled"]:
            not_ready.append(
                "reviewer effort %r is not accepted by the pinned Humanize "
                "release (it accepts %s). Enable no_commit_overlay to widen the "
                "parser, or choose a supported effort. It is NOT downgraded "
                "silently." % (effort, ", ".join(config_mod.UPSTREAM_EFFORTS))
            )

    # --- existing session ------------------------------------------------
    existing = _load_session(task_dir)
    if existing is not None:
        state = _session_state(existing)
        if state["state"] == SESSION_RUNNING:
            not_ready.append(
                "a session is already running (uuid %s, pid %s); use 'agent "
                "status' or 'agent stop'"
                % (existing.get("uuid"), existing.get("pid"))
            )

    # --- freeze the invocation ------------------------------------------
    prompt = None
    argv: List[str] = []
    env: Dict[str, str] = {}
    if plugin_dir is not None and plan_relative is not None:
        prompt = build_prompt(
            plan_relative,
            reviewer,
            base_branch,
            max_iterations,
            str(humanize.get("start_command") or "/humanize:start-rlcr-loop"),
            skip_quiz=bool(humanize.get("skip_quiz", False)),
        )
        argv = build_argv(writer, plugin_dir, prompt, session_id=session_id)
        env = build_env(
            workspace_path,
            bool(provenance.get("no_commit_overlay")),
            live_plan,
            snapshot,
            reviewer=reviewer,
            reviewer_binary=resolved_paths.get("codex"),
        )

    contract = (
        contract_identity(control) if control is not None else {"files": {}, "sha256": None}
    )
    toolchain_identity = toolchain.identity(report)

    # The plan identity deliberately excludes the session id, so a dry run and
    # the real start of the same configuration share one identity.
    identity_payload = {
        "argv": strip_session_id(argv),
        "cwd": str(workspace_path) if workspace_path else None,
        "env": redact_env(env),
        "toolchain_sha256": toolchain_identity["sha256"],
        "contract_sha256": contract["sha256"],
        "plugin": provenance,
        # The identity covers which binaries will run, not just which were
        # found, so a PATH change produces a different plan identity.
        "toolchain_paths": resolved_paths,
    }

    record = {
        "schema": "k3ctl/agent-plan/1",
        "generated_at": util.utcnow(),
        "task_id": task_id,
        "config": str(config_path),
        "lifecycle": lifecycle.describe(task),
        "workflow": {
            "writer": writer,
            "reviewer": reviewer,
            "start_command": humanize.get("start_command"),
            "skip_quiz": bool(humanize.get("skip_quiz", False)),
            "max_iterations": max_iterations,
        },
        "invocation": {
            "argv": argv,
            "cwd": str(workspace_path) if workspace_path else None,
            "env": redact_env(env),
            "prompt": prompt,
            "shell": False,
        },
        "plugin": provenance,
        "plugin_absolute_path": str(plugin_dir) if plugin_dir else None,
        "identity": {
            "toolchain_sha256": toolchain_identity["sha256"],
            "contract_sha256": contract["sha256"],
            "plan_sha256": util.sha256_json(identity_payload),
        },
        "contract_files": contract["files"],
        "toolchain_status": report["status"],
        "source_availability": {
            "task_base_commit": availability.get("task_base_commit"),
            "task_base_available": availability.get("task_base_available"),
        },
        "ready": not not_ready,
        "not_ready_reasons": not_ready,
        "would_spawn": False,
        "note": (
            "Dry run only. No process is spawned, no loop state is created, and "
            "no verdict is produced. Claude itself invokes the official "
            "start-rlcr-loop command when 'agent start' runs."
        ),
    }
    util.write_json(plan_path(task_dir), record)
    return record


# ------------------------------------------------------------------ session


def _load_session(task_dir: Path) -> Optional[Dict[str, Any]]:
    target = session_path(task_dir)
    if not target.is_file():
        return None
    try:
        return config_mod.load_json(target)
    except config_mod.ValidationError:
        return None


def _session_state(session: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a recorded session by verifying its process identity."""
    if session.get("state") in (SESSION_EXITED, SESSION_STOPPED):
        return {
            "state": str(session["state"]),
            "detail": "recorded as %s" % session["state"],
        }
    pid = session.get("pid")
    if not isinstance(pid, int):
        return {"state": SESSION_EXITED, "detail": "no pid recorded"}
    matches, detail = util.process_matches(pid, session.get("process_identity"))
    if matches:
        return {"state": SESSION_RUNNING, "detail": detail}
    return {"state": SESSION_EXITED, "detail": detail}


def _acquire_task_lock(
    task_dir: Path, task_id: str, uuid: str
) -> util.FileLock:
    """Take the exclusive task lock, reclaiming it only when provably stale.

    The lock file persists for the session's lifetime, so a session that died
    without calling ``stop`` would otherwise block the task forever. A lock is
    reclaimed only when the recorded holder's process identity no longer
    matches a live process.
    """
    lock = util.FileLock(
        lock_path(task_dir),
        {"task_id": task_id, "uuid": uuid, "state": SESSION_STARTING},
    )
    try:
        lock.acquire()
        return lock
    except util.LockError:
        pass

    holder = lock.holder() or {}
    holder_pid = holder.get("pid")
    session = _load_session(task_dir)

    live = False
    detail = "lock holder pid is unknown"
    if isinstance(holder_pid, int):
        recorded_identity = None
        if session is not None and session.get("pid") == holder_pid:
            recorded_identity = session.get("process_identity")
        if recorded_identity:
            live, detail = util.process_matches(holder_pid, recorded_identity)
        else:
            # No verifiable identity: treat a live pid as live, fail closed.
            live = util.pid_alive(holder_pid)
            detail = (
                "pid %s is alive but has no recorded identity to verify"
                % holder_pid
                if live
                else "pid %s is not running" % holder_pid
            )

    if live:
        raise util.LockError(
            "task %s is locked by uuid %s (pid %s): %s\n"
            "Use 'agent status', or 'agent stop' to end that session."
            % (task_id, holder.get("uuid"), holder_pid, detail)
        )

    # Provably stale: reclaim it once, then take the lock.
    try:
        lock_path(task_dir).unlink()
    except FileNotFoundError:
        pass
    lock.acquire()
    return lock


# --------------------------------------------------------------- loop state


def _read_loop_dir(loop: Path) -> Dict[str, Any]:
    """Read one loop directory's state, active or terminal."""
    info: Dict[str, Any] = {
        "loop_dir": str(loop),
        "state_file": None,
        "phase": None,
        "terminal": False,
        "terminal_reason": None,
        "fields": {},
        "present": False,
    }
    for reason in TERMINAL_REASONS:
        target = loop / ("%s-state.md" % reason)
        if target.is_file():
            info.update(
                {
                    "state_file": str(target),
                    "phase": "terminal",
                    "terminal": True,
                    "terminal_reason": reason,
                    "fields": toolchain.read_frontmatter(target),
                    "present": True,
                }
            )
            return info
    for name, phase in ACTIVE_STATE_FILES:
        target = loop / name
        if target.is_file():
            info.update(
                {
                    "state_file": str(target),
                    "phase": phase,
                    "fields": toolchain.read_frontmatter(target),
                    "present": True,
                }
            )
            return info
    return info


def _loop_state(
    workspace_path: Path, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Read the plugin's own loop state, bound by session id when known.

    Loop state is never synthesised. Binding by the ``session_id`` frontmatter
    field, the same field the plugin itself matches on, avoids attributing an
    unrelated concurrent loop's state to this session.
    """
    base = workspace_path / ".humanize" / "rlcr"
    empty: Dict[str, Any] = {
        "loop_dir": None,
        "state_file": None,
        "phase": None,
        "terminal": False,
        "terminal_reason": None,
        "fields": {},
        "present": False,
        "bound_by": None,
    }
    if not base.is_dir():
        return empty

    candidates = sorted(
        (item for item in base.iterdir() if item.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )
    if not candidates:
        return empty

    fallback: Optional[Dict[str, Any]] = None
    for loop in candidates:
        info = _read_loop_dir(loop)
        if not info["present"]:
            continue
        stored = str(info["fields"].get("session_id") or "").strip()
        if session_id and stored == session_id:
            info["bound_by"] = "session-id"
            return _augment_loop(info, loop)
        if fallback is None and (not session_id or not stored):
            fallback = _augment_loop(info, loop)
            fallback["bound_by"] = "newest-unbound" if not stored else "newest"
    if fallback is not None:
        return fallback
    return empty


def _augment_loop(info: Dict[str, Any], loop: Path) -> Dict[str, Any]:
    info["review_phase_started"] = (loop / ".review-phase-started").is_file()
    info["review_result_files"] = sorted(
        item.name for item in loop.glob("round-*-review-result.md")
    )
    info["summary_files"] = sorted(
        item.name for item in loop.glob("round-*-summary.md")
    )
    return info


def _tail_stream(path: Path, limit: int = 5) -> Dict[str, Any]:
    """Summarise the tail of Claude's stream-json log."""
    import json

    info: Dict[str, Any] = {
        "present": path.is_file(),
        "bytes": 0,
        "events": 0,
        "last_types": [],
        "session_id": None,
        "result": None,
    }
    if not path.is_file():
        return info
    info["bytes"] = path.stat().st_size
    types: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                info["events"] += 1
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = str(event.get("type") or "")
                types.append(kind)
                if event.get("session_id") and not info["session_id"]:
                    info["session_id"] = event["session_id"]
                if kind == "result":
                    info["result"] = {
                        "subtype": event.get("subtype"),
                        "is_error": event.get("is_error"),
                        "num_turns": event.get("num_turns"),
                    }
    except OSError:
        return info
    info["last_types"] = types[-limit:]
    return info


# ------------------------------------------------------------------- start


def start(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    max_iterations: int = 42,
) -> Dict[str, Any]:
    """Launch Claude, which invokes the official RLCR loop command."""
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")

    # Lifecycle gates fail before any process is created.
    lifecycle.require_startable(task, task_id)

    # The session id is chosen before planning so it can be passed to Claude,
    # recorded, and used to bind the loop deterministically.
    uuid = util.new_uuid()

    # Serialise before planning, so two concurrent starts cannot both pass
    # their readiness checks and then both spawn.
    lock = _acquire_task_lock(task_dir, task_id, uuid)

    try:
        record = plan(
            root,
            config_path,
            config,
            task_id,
            max_iterations=max_iterations,
            session_id=uuid,
        )
        if not record["ready"]:
            raise util.ToolError(
                "task %s is not ready to start:\n  - %s"
                % (task_id, "\n  - ".join(record["not_ready_reasons"]))
            )

        logs = log_dir(task_dir, uuid)
        logs.mkdir(parents=True, exist_ok=True)
        stream_log = logs / "stream.jsonl"
        stderr_log = logs / "stderr.log"

        argv = list(record["invocation"]["argv"])
        cwd = Path(str(record["invocation"]["cwd"]))
        declared = {
            key: value
            for key, value in record["invocation"]["env"].items()
            if key in DECLARED_ENV_KEYS
        }
        # Resolve the binaries the hooks will invoke, and expose exactly those
        # through a controlled PATH shim. Without this, a recorded toolchain
        # says nothing about what actually runs.
        settings = config_mod.workflow_settings(config)
        resolved_paths, path_problems = resolve_toolchain_paths(config)
        if path_problems:
            raise util.ToolError(
                "cannot start %s: the loop toolchain is incomplete:\n  - %s"
                % (task_id, "\n  - ".join(path_problems))
            )
        shim = build_shim_dir(
            task_dir,
            uuid,
            resolved_paths,
            reviewer=(settings.get("reviewer") or {}),
        )
        shim_problems = verify_shim(
            shim,
            resolved_paths,
            reviewer=(settings.get("reviewer") or {}),
        )
        if shim_problems:
            raise util.ToolError(
                "reviewer PATH shim failed integrity checks:\n  - %s"
                % "\n  - ".join(shim_problems)
            )

        spawn_env = dict(os.environ)
        spawn_env.update(declared)
        spawn_env = shim_env(spawn_env, shim)

        session: Dict[str, Any] = {
            "schema": "k3ctl/agent-session/2",
            "uuid": uuid,
            "claude_session_id": uuid,
            "task_id": task_id,
            "config": str(config_path),
            "state": SESSION_STARTING,
            "started_at": util.utcnow(),
            "argv": argv,
            "cwd": str(cwd),
            "env": declared,
            "logs": {
                "stream": str(stream_log),
                "stderr": str(stderr_log),
                "dir": str(logs),
            },
            "identity": record["identity"],
            # Frozen at launch. 'resume' rebuilds its argv from these values,
            # never from whatever the configuration happens to say later.
            "frozen": {
                "writer": settings.get("writer") or {},
                "reviewer": settings.get("reviewer") or {},
                "plugin": record["plugin"],
                "plugin_absolute_path": record["plugin_absolute_path"],
                "max_iterations": max_iterations,
                "toolchain_paths": resolved_paths,
                "shim_dir": str(shim),
            },
            "plugin": record["plugin"],
            "max_iterations": max_iterations,
            "pid": None,
            "process_identity": None,
        }
        # Persist before launch so a crash still leaves an auditable record.
        util.write_json(session_path(task_dir), session)

        try:
            with open(stream_log, "wb") as out, open(stderr_log, "wb") as err:
                process = subprocess.Popen(  # noqa: S603 - argv list, shell=False
                    argv,
                    cwd=str(cwd),
                    env=spawn_env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
        except OSError as exc:
            session["state"] = SESSION_EXITED
            session["error"] = "failed to spawn: %s" % exc
            util.write_json(session_path(task_dir), session)
            raise util.ToolError("cannot start %s: %s" % (argv[0], exc)) from exc
    except BaseException:
        # Nothing was spawned, so the lock must not be left behind.
        lock.release()
        raise

    session["pid"] = process.pid
    session["state"] = SESSION_RUNNING
    session["process_identity"] = util.process_identity(process.pid)
    session["process_group"] = process.pid
    util.write_json(session_path(task_dir), session)

    # The lock file stays on disk for the session's lifetime; it is the session
    # marker that 'stop' clears after verifying process identity.
    util.write_json(
        lock_path(task_dir),
        {
            "task_id": task_id,
            "uuid": uuid,
            "pid": process.pid,
            "state": SESSION_RUNNING,
            "acquired_at": util.utcnow(),
        },
    )
    return {
        "task_id": task_id,
        "uuid": uuid,
        "claude_session_id": uuid,
        "pid": process.pid,
        "state": SESSION_RUNNING,
        "cwd": str(cwd),
        "logs": session["logs"],
        "plugin": record["plugin"],
        "identity": record["identity"],
        "note": (
            "Claude was launched and will invoke the official "
            "start-rlcr-loop command. A running process is not a result; use "
            "'agent status'."
        ),
    }


# ------------------------------------------------------------------ status


def status(root: Path, config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Report real session, process and plugin loop state."""
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    session = _load_session(task_dir)

    result: Dict[str, Any] = {
        "task_id": task_id,
        "lifecycle": lifecycle.describe(task),
        "session": None,
        "process": None,
        "loop": None,
        "stream": None,
        "outcome": "no-session",
        "accepted": False,
        "note": (
            "A zero exit code is not acceptance. Acceptance requires the "
            "plugin's own review verdict plus human review."
        ),
    }
    if session is None:
        return result

    state = _session_state(session)
    result["session"] = {
        "uuid": session.get("uuid"),
        "claude_session_id": session.get("claude_session_id"),
        "state": state["state"],
        "started_at": session.get("started_at"),
        "argv": session.get("argv"),
        "cwd": session.get("cwd"),
        "logs": session.get("logs"),
        "identity": session.get("identity"),
        "plugin": session.get("plugin"),
    }
    result["process"] = {
        "pid": session.get("pid"),
        "recorded_identity": session.get("process_identity"),
        "verified": state["state"] == SESSION_RUNNING,
        "detail": state["detail"],
    }

    stream_log = Path(str((session.get("logs") or {}).get("stream") or ""))
    result["stream"] = _tail_stream(stream_log)

    workspace_path = Path(str(session.get("cwd") or ""))
    claude_session = session.get("claude_session_id") or result["stream"].get(
        "session_id"
    )
    if workspace_path.is_dir():
        loop = _loop_state(workspace_path, claude_session)
        result["loop"] = loop
        running = state["state"] == SESSION_RUNNING
        if not loop["present"]:
            result["outcome"] = (
                "running-no-loop-state" if running else "exited-without-loop-state"
            )
        elif loop["terminal"]:
            result["outcome"] = "loop-ended-%s" % loop["terminal_reason"]
        elif running:
            result["outcome"] = "running-%s" % loop["phase"]
        else:
            result["outcome"] = "exited-in-%s-phase" % loop["phase"]
    elif state["state"] != SESSION_RUNNING:
        result["outcome"] = "exited"
    else:
        result["outcome"] = "running"

    # Acceptance is never derived here. Report the evidence, not a verdict.
    loop_info = result["loop"] or {}
    result["evidence"] = {
        "review_result_files": loop_info.get("review_result_files") or [],
        "loop_fields": loop_info.get("fields") or {},
        "loop_terminal_reason": loop_info.get("terminal_reason"),
        "stream_result": result["stream"].get("result"),
    }
    return result


# ------------------------------------------------------------------ resume


def resume(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    """Resume the same Claude session after verifying the old process exited.

    The invocation is rebuilt from the values frozen at start, not from current
    configuration. Resuming never starts a new loop, and a loop that already
    reached a terminal state is never restarted.
    """
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    lifecycle.require_startable(task, task_id)

    session = _load_session(task_dir)
    if session is None:
        raise util.ToolError(
            "no recorded session for task %s; use 'agent start'" % task_id
        )
    state = _session_state(session)
    if state["state"] == SESSION_RUNNING:
        raise util.ToolError(
            "session %s is still running (pid %s): %s\n"
            "Stop it before resuming; a second loop must never be started."
            % (session.get("uuid"), session.get("pid"), state["detail"])
        )

    claude_session = session.get("claude_session_id")
    if not claude_session:
        stream = _tail_stream(
            Path(str((session.get("logs") or {}).get("stream") or ""))
        )
        claude_session = stream.get("session_id")
    if not claude_session:
        raise util.ToolError(
            "cannot determine the Claude session id for task %s; resume requires "
            "a recorded session id" % task_id
        )

    workspace_path = Path(str(session.get("cwd") or ""))
    if not workspace_path.is_dir():
        raise util.ToolError("recorded workspace is missing: %s" % workspace_path)

    loop = _loop_state(workspace_path, str(claude_session))
    if not loop["present"]:
        raise util.ToolError(
            "no Humanize loop state in %s; there is nothing to resume. Use "
            "'agent start' to begin a loop." % workspace_path
        )
    if loop["terminal"]:
        raise util.ToolError(
            "the loop already ended (%s-state.md in %s).\n"
            "A completed or cancelled loop is never restarted automatically. "
            "Review its result, or create a new task for further work."
            % (loop["terminal_reason"], loop["loop_dir"])
        )

    # Rebuild strictly from the frozen invocation.
    frozen = session.get("frozen") or {}
    writer = frozen.get("writer") or {}
    plugin_absolute = frozen.get("plugin_absolute_path")
    if not writer or not plugin_absolute:
        raise util.ToolError(
            "session record has no frozen writer/plugin settings; it was created "
            "by an older version and cannot be resumed safely. Stop and start a "
            "new session deliberately."
        )
    plugin_dir = Path(str(plugin_absolute))
    if not plugin_dir.is_dir():
        raise util.ToolError(
            "the plugin directory frozen at start is gone: %s\n"
            "Resume requires the exact same plugin that the loop started with."
            % plugin_dir
        )

    # Report, but do not silently follow, configuration drift.
    drift: List[str] = []
    try:
        current_plugin, current_provenance = overlay.resolve_plugin_dir(root, config)
    except util.ToolError as exc:
        drift.append("current plugin cannot be resolved: %s" % exc)
    else:
        if str(current_plugin) != str(plugin_dir):
            drift.append(
                "configuration now resolves the plugin to %s, but this session is "
                "resuming with its frozen plugin %s"
                % (current_provenance.get("path"), plugin_absolute)
            )
    current_writer = (config_mod.workflow_settings(config).get("writer") or {})
    if current_writer != writer:
        drift.append(
            "writer settings changed since start; resuming with the frozen "
            "settings recorded at launch"
        )

    uuid = util.new_uuid()
    lock = _acquire_task_lock(task_dir, task_id, uuid)

    try:
        argv = build_resume_argv(writer, plugin_dir, str(claude_session))
        logs = log_dir(task_dir, uuid)
        logs.mkdir(parents=True, exist_ok=True)
        stream_out = logs / "stream.jsonl"
        stderr_out = logs / "stderr.log"

        declared = {
            key: value
            for key, value in (session.get("env") or {}).items()
            if key in DECLARED_ENV_KEYS
        }
        frozen_reviewer = frozen.get("reviewer") or {}
        frozen_paths = frozen.get("toolchain_paths") or {}
        shim = build_shim_dir(
            task_dir,
            uuid,
            frozen_paths,
            reviewer=frozen_reviewer,
        )
        shim_problems = verify_shim(
            shim,
            frozen_paths,
            reviewer=frozen_reviewer,
        )
        if shim_problems:
            raise util.ToolError(
                "reviewer PATH shim failed integrity checks while resuming:\n  - %s"
                % "\n  - ".join(shim_problems)
            )
        spawn_env = dict(os.environ)
        spawn_env.update(declared)
        spawn_env = shim_env(spawn_env, shim)

        resumed = dict(session)
        resumed.update(
            {
                "uuid": uuid,
                "resumed_from": session.get("uuid"),
                "claude_session_id": str(claude_session),
                "state": SESSION_STARTING,
                "resumed_at": util.utcnow(),
                "argv": argv,
                "logs": {
                    "stream": str(stream_out),
                    "stderr": str(stderr_out),
                    "dir": str(logs),
                },
                "pid": None,
                "process_identity": None,
                "configuration_drift": drift,
                "frozen": dict(
                    resumed.get("frozen") or {},
                    shim_dir=str(shim),
                ),
            }
        )
        util.write_json(session_path(task_dir), resumed)

        try:
            with open(stream_out, "wb") as out, open(stderr_out, "wb") as err:
                process = subprocess.Popen(  # noqa: S603
                    argv,
                    cwd=str(workspace_path),
                    env=spawn_env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
        except OSError as exc:
            resumed["state"] = SESSION_EXITED
            resumed["error"] = "failed to resume: %s" % exc
            util.write_json(session_path(task_dir), resumed)
            raise util.ToolError("cannot resume %s: %s" % (argv[0], exc)) from exc
    except BaseException:
        lock.release()
        raise

    resumed["pid"] = process.pid
    resumed["state"] = SESSION_RUNNING
    resumed["process_identity"] = util.process_identity(process.pid)
    util.write_json(session_path(task_dir), resumed)
    util.write_json(
        lock_path(task_dir),
        {
            "task_id": task_id,
            "uuid": uuid,
            "pid": process.pid,
            "state": SESSION_RUNNING,
            "acquired_at": util.utcnow(),
        },
    )
    return {
        "task_id": task_id,
        "uuid": uuid,
        "resumed_from": session.get("uuid"),
        "claude_session_id": str(claude_session),
        "pid": process.pid,
        "state": SESSION_RUNNING,
        "loop_dir": loop.get("loop_dir"),
        "loop_phase": loop.get("phase"),
        "configuration_drift": drift,
        "note": (
            "Resumed the same session and loop with the invocation frozen at "
            "start. No new loop was created."
        ),
    }


# -------------------------------------------------------------------- stop


def stop(
    root: Path,
    config: Dict[str, Any],
    task_id: str,
    signal_group: bool = False,
    timeout: float = 15.0,
    escalate: bool = False,
) -> Dict[str, Any]:
    """Signal only the recorded, identity-confirmed process."""
    task_dir = config_mod.task_dir(root, config, task_id)
    session = _load_session(task_dir)
    if session is None:
        raise util.ToolError("no recorded session for task %s" % task_id)

    pid = session.get("pid")
    if not isinstance(pid, int):
        raise util.ToolError(
            "recorded session has no pid; refusing to signal anything"
        )

    matches, detail = util.process_matches(pid, session.get("process_identity"))
    if not matches:
        raise util.ToolError(
            "refusing to signal pid %s: %s\n"
            "Only a process whose recorded start time and command still match is "
            "ever signalled. Unrelated pids and services are never touched."
            % (pid, detail)
        )

    target = -pid if signal_group else pid
    try:
        os.kill(target, signal.SIGTERM)
    except OSError as exc:
        raise util.ToolError("cannot signal pid %s: %s" % (pid, exc)) from exc

    deadline = time.time() + max(0.0, timeout)
    exited = False
    while time.time() < deadline:
        still, _ = util.process_matches(pid, session.get("process_identity"))
        if not still:
            exited = True
            break
        time.sleep(0.2)

    escalated = False
    if not exited and escalate:
        try:
            os.kill(target, signal.SIGKILL)
            escalated = True
        except OSError:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline:
            still, _ = util.process_matches(pid, session.get("process_identity"))
            if not still:
                exited = True
                break
            time.sleep(0.2)

    session["state"] = SESSION_STOPPED if exited else SESSION_RUNNING
    session["stopped_at"] = util.utcnow() if exited else None
    util.write_json(session_path(task_dir), session)

    if exited:
        lock = lock_path(task_dir)
        if lock.is_file():
            try:
                lock.unlink()
            except OSError:
                pass

    return {
        "task_id": task_id,
        "uuid": session.get("uuid"),
        "pid": pid,
        "signalled": "process-group" if signal_group else "process",
        "escalated_to_sigkill": escalated,
        "exited": exited,
        "state": session["state"],
        "note": (
            "Stopping a process is not a verdict. No loop state or review "
            "outcome was written."
        ),
    }
