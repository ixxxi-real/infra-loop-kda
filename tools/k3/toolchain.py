#!/usr/bin/env python3
"""Offline toolchain inspection.

Everything here is read-only: no installation, no network access, no model or
GPU invocation. Checks report ``missing`` or ``incompatible`` rather than
collapsing to a single "passed" verdict, and dependency dirtiness is reported
separately from version mismatches.
"""

from __future__ import annotations

import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as config_mod
from . import gitq, paths, util

OK = "ok"
MISSING = "missing"
INCOMPATIBLE = "incompatible"
DIRTY = "dirty"
SKIPPED = "skipped"

ERROR_STATUSES = {MISSING, INCOMPATIBLE}

MIN_PYTHON = (3, 9)
MIN_BASH = (4, 0)


def _check(
    name: str,
    status: str,
    detail: str,
    severity: str = "error",
    **extra: Any,
) -> Dict[str, Any]:
    record = {"name": name, "status": status, "detail": detail, "severity": severity}
    record.update(extra)
    return record


# ------------------------------------------------------------------ commands


def _version_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def command_check(
    name: str,
    argv: Optional[List[str]] = None,
    severity: str = "error",
) -> Dict[str, Any]:
    """Locate a command and capture its version line, without side effects."""
    location = util.which(name)
    if location is None:
        return _check(
            name,
            MISSING,
            "%s is not on PATH" % name,
            severity=severity,
            path=None,
            version=None,
        )
    version = None
    if argv:
        rc, out, err = util.run([location, *argv[1:]], timeout=20)
        if rc == 0:
            version = _version_line(out or err)
    return _check(
        name,
        OK,
        version or location,
        severity=severity,
        path=location,
        version=version,
    )


def python_check() -> Dict[str, Any]:
    current = sys.version_info[:3]
    status = OK if current >= MIN_PYTHON else INCOMPATIBLE
    detail = "Python %d.%d.%d" % current
    if status != OK:
        detail += " (requires >= %d.%d)" % MIN_PYTHON
    return _check(
        "python3",
        status,
        detail,
        path=sys.executable,
        version="%d.%d.%d" % current,
        implementation=platform.python_implementation(),
    )


def bash_check() -> Dict[str, Any]:
    """Bash 4 or newer is required: macOS still ships Bash 3.2 as /bin/bash."""
    location = util.which("bash")
    if location is None:
        return _check("bash", MISSING, "bash is not on PATH", path=None, version=None)
    rc, out, err = util.run([location, "--version"], timeout=20)
    text = out or err
    match = re.search(r"version (\d+)\.(\d+)", text)
    if rc != 0 or not match:
        return _check(
            "bash",
            INCOMPATIBLE,
            "cannot determine the bash version from %s" % location,
            path=location,
            version=None,
        )
    version = (int(match.group(1)), int(match.group(2)))
    status = OK if version >= MIN_BASH else INCOMPATIBLE
    detail = "bash %d.%d at %s" % (version[0], version[1], location)
    if status != OK:
        detail += " (Humanize hooks require bash >= 4.0; install a newer bash)"
    return _check(
        "bash",
        status,
        detail,
        path=location,
        version="%d.%d" % version,
    )


# ------------------------------------------------------- repository/submodule


def repository_section(root: Path) -> Dict[str, Any]:
    """Describe this repository without assuming a born HEAD."""
    head = gitq.head_commit(root) if gitq.head_exists(root) else None
    staged = gitq.staged_paths(root)
    dirty_lines = gitq.porcelain(root) or []
    return {
        "root": str(root),
        "is_own_git_root": gitq.is_own_root(root),
        "head_exists": head is not None,
        "head_commit": head,
        "branch": gitq.current_branch(root),
        "staged_file_count": len(staged),
        "worktree_entries": len(dirty_lines),
    }


def dependency_check(
    root: Path,
    name: str,
    relative_path: str,
    configured_commit: Optional[str],
    recorded_gitlinks: Dict[str, str],
) -> Dict[str, Any]:
    """Compare configured commit, recorded gitlink and the actual HEAD.

    An absent or empty dependency directory must not resolve to the parent
    repository, so the check requires the path to be its own Git root.
    """
    try:
        target = paths.safe_join(root, relative_path)
    except paths.PathError as exc:
        return _check(name, INCOMPATIBLE, str(exc), path=relative_path)

    recorded = recorded_gitlinks.get(relative_path.rstrip("/"))
    base: Dict[str, Any] = {
        "path": relative_path,
        "configured_commit": configured_commit,
        "recorded_gitlink": recorded,
        "current_head": None,
        "dirty": None,
        "sparse": None,
        "present_files": None,
        "tracked_files": None,
    }

    if not target.exists():
        return _check(
            name,
            MISSING,
            "%s does not exist; run: git submodule update --init --recursive"
            % relative_path,
            **base,
        )
    if not gitq.is_own_root(target):
        return _check(
            name,
            MISSING,
            "%s is not an initialized Git checkout (it resolves to the parent "
            "repository); run: git submodule update --init --recursive"
            % relative_path,
            **base,
        )

    head = gitq.head_commit(target)
    dirty = gitq.is_dirty(target)
    sparse = gitq.is_sparse(target)
    tracked = gitq.tracked_file_count(target)
    present = gitq.present_file_count(target)
    base.update(
        {
            "current_head": head,
            "dirty": dirty,
            "sparse": sparse,
            "tracked_files": tracked,
            "present_files": present,
        }
    )

    mismatches: List[str] = []
    if configured_commit and head and configured_commit != head:
        mismatches.append(
            "configured %s != HEAD %s" % (configured_commit[:12], head[:12])
        )
    if recorded and head and recorded != head:
        mismatches.append("recorded gitlink %s != HEAD %s" % (recorded[:12], head[:12]))
    if configured_commit and recorded and configured_commit != recorded:
        mismatches.append(
            "configured %s != recorded gitlink %s"
            % (configured_commit[:12], recorded[:12])
        )

    if mismatches:
        return _check(
            name,
            INCOMPATIBLE,
            "%s pin mismatch: %s" % (relative_path, "; ".join(mismatches)),
            **base
        )

    detail = "%s @ %s" % (relative_path, (head or "unknown")[:12])
    if sparse and tracked and present is not None and present < tracked:
        detail += " (sparse checkout: %d/%d tracked files present)" % (present, tracked)
    return _check(name, OK, detail, **base)


def dependency_dirty_check(dependency: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Report dependency dirtiness separately from pin correctness."""
    if dependency.get("status") in (MISSING,):
        return None
    if not dependency.get("dirty"):
        return None
    return _check(
        "%s-worktree" % dependency["name"],
        DIRTY,
        "%s has local modifications; a frozen base cannot be claimed from it"
        % dependency.get("path"),
        severity="warning",
        path=dependency.get("path"),
    )


# -------------------------------------------------------------- skill probing


def read_frontmatter(path: Path) -> Dict[str, str]:
    """Parse a minimal ``key: value`` YAML frontmatter block."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: Dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("'\"")
    return fields


def skill_check(root: Path, label: str, relative: str) -> Dict[str, Any]:
    """Verify a skill file exists at its project-local path and hash it."""
    try:
        target = paths.safe_join(root, relative)
    except paths.PathError as exc:
        return _check(label, INCOMPATIBLE, str(exc), path=relative)
    if not target.is_file():
        return _check(
            label,
            MISSING,
            "%s is not present; run: git submodule update --init --recursive"
            % relative,
            path=relative,
            sha256=None,
            version=None,
        )
    front = read_frontmatter(target)
    return _check(
        label,
        OK,
        "%s (%s)" % (relative, front.get("name") or target.parent.name),
        path=relative,
        sha256=util.sha256_file(target),
        version=front.get("version"),
        skill_name=front.get("name"),
    )


# ------------------------------------------------------------- kda / humanize


def kda_section(root: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Hash the pinned KDA prompt and locate the recursive kernel skills.

    KDA here is NVlabs *Kernel Design Agents* -- the workflow, prompts and
    skills. It is not the Kimi Delta Attention kernel, and it is not an SDK.
    """
    kda = settings.get("kda", {})
    section: Dict[str, Any] = {"checks": [], "prompt": None, "agent_flow": None}
    for key, label in (("prompt", "kda-prompt"), ("agent_flow", "kda-agent-flow")):
        relative = kda.get(key)
        if not relative:
            continue
        try:
            target = paths.safe_join(root, str(relative))
        except paths.PathError as exc:
            section["checks"].append(_check(label, INCOMPATIBLE, str(exc)))
            continue
        if not target.is_file():
            section["checks"].append(
                _check(
                    label,
                    MISSING,
                    "%s is not present; run: git submodule update --init --recursive"
                    % relative,
                    path=str(relative),
                )
            )
            continue
        digest = util.sha256_file(target)
        section[key] = {"path": str(relative), "sha256": digest}
        section["checks"].append(
            _check(
                label,
                OK,
                "%s sha256=%s" % (relative, digest[:12]),
                path=str(relative),
                sha256=digest,
            )
        )
    return section


def humanize_section(root: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Read the pinned Humanize plugin manifest, hooks and loop commands."""
    humanize = settings.get("humanize", {})
    relative = humanize.get("plugin_dir") or "external/humanize"
    section: Dict[str, Any] = {
        "plugin_dir": relative,
        "name": None,
        "version": None,
        "hooks": [],
        "commands": [],
        "checks": [],
    }
    try:
        plugin_dir = paths.safe_join(root, str(relative))
    except paths.PathError as exc:
        section["checks"].append(_check("humanize-plugin", INCOMPATIBLE, str(exc)))
        return section

    manifest_path = plugin_dir / ".claude-plugin" / "plugin.json"
    if not manifest_path.is_file():
        section["checks"].append(
            _check(
                "humanize-plugin",
                MISSING,
                "%s/.claude-plugin/plugin.json is not present; run: "
                "git submodule update --init --recursive" % relative,
                path=str(relative),
            )
        )
        return section
    try:
        manifest = config_mod.load_json(manifest_path)
    except config_mod.ValidationError as exc:
        section["checks"].append(_check("humanize-plugin", INCOMPATIBLE, str(exc)))
        return section
    section["name"] = manifest.get("name")
    section["version"] = manifest.get("version")
    section["checks"].append(
        _check(
            "humanize-plugin",
            OK,
            "%s %s at %s" % (manifest.get("name"), manifest.get("version"), relative),
            path=str(relative),
            version=manifest.get("version"),
        )
    )

    hooks_path = plugin_dir / "hooks" / "hooks.json"
    if not hooks_path.is_file():
        section["checks"].append(
            _check("humanize-hooks", MISSING, "%s/hooks/hooks.json is missing" % relative)
        )
    else:
        try:
            hooks = config_mod.load_json(hooks_path)
        except config_mod.ValidationError as exc:
            section["checks"].append(_check("humanize-hooks", INCOMPATIBLE, str(exc)))
        else:
            events = sorted((hooks.get("hooks") or {}).keys())
            section["hooks"] = events
            section["checks"].append(
                _check(
                    "humanize-hooks",
                    OK,
                    "hook events: %s" % ", ".join(events) if events else "no hooks",
                    events=events,
                )
            )

    commands_dir = plugin_dir / "commands"
    if commands_dir.is_dir():
        section["commands"] = sorted(
            "/humanize:%s" % item.stem for item in commands_dir.glob("*.md")
        )
    start_command = humanize.get("start_command") or "/humanize:start-rlcr-loop"
    if section["commands"] and start_command not in section["commands"]:
        section["checks"].append(
            _check(
                "humanize-start-command",
                MISSING,
                "%s is not provided by the pinned plugin (found: %s)"
                % (start_command, ", ".join(section["commands"])),
            )
        )
    elif section["commands"]:
        section["checks"].append(
            _check("humanize-start-command", OK, "%s is available" % start_command)
        )

    setup_script = plugin_dir / "scripts" / "setup-rlcr-loop.sh"
    stop_hook = plugin_dir / "hooks" / "loop-codex-stop-hook.sh"
    for label, script in (
        ("humanize-setup-script", setup_script),
        ("humanize-stop-hook", stop_hook),
    ):
        if script.is_file():
            section["checks"].append(
                _check(
                    label,
                    OK,
                    "%s sha256=%s"
                    % (
                        paths.relative_to_root(root, script),
                        util.sha256_file(script)[:12],
                    ),
                    sha256=util.sha256_file(script),
                )
            )
        else:
            section["checks"].append(
                _check(label, MISSING, "%s is missing" % paths.relative_to_root(root, script))
            )
    return section


# ------------------------------------------------------------------ assembly


def source_availability(
    root: Path, config: Dict[str, Any], task_id: Optional[str]
) -> Dict[str, Any]:
    """Report whether a task's exact base commit is reachable locally.

    A task base that differs from the public submodule pin is never silently
    substituted; a missing base commit is reported as an availability error.
    """
    source = config.get("source", {}) or {}
    relative = source.get("submodule") or "external/sglang"
    result: Dict[str, Any] = {
        "source_path": relative,
        "configured_commit": source.get("commit"),
        "task_id": task_id,
        "task_base_commit": None,
        "configured_commit_available": None,
        "task_base_available": None,
        "checks": [],
    }
    try:
        source_dir = paths.safe_join(root, str(relative))
    except paths.PathError as exc:
        result["checks"].append(_check("source-path", INCOMPATIBLE, str(exc)))
        return result

    if not gitq.is_own_root(source_dir):
        result["checks"].append(
            _check(
                "source-availability",
                MISSING,
                "%s is not an initialized Git checkout; the exact task base "
                "cannot be verified" % relative,
                path=str(relative),
            )
        )
        return result

    configured = source.get("commit")
    if isinstance(configured, str) and config_mod.SHA_RE.fullmatch(configured):
        available = gitq.has_commit(source_dir, configured)
        result["configured_commit_available"] = available
        result["checks"].append(
            _check(
                "source-configured-commit",
                OK if available else MISSING,
                "%s %s in %s"
                % (
                    configured[:12],
                    "is present" if available else "is NOT present",
                    relative,
                ),
                commit=configured,
            )
        )

    if task_id:
        try:
            task = config_mod.load_task(root, config, task_id)
        except (util.ToolError, config_mod.ValidationError) as exc:
            result["checks"].append(_check("task-source", INCOMPATIBLE, str(exc)))
            return result
        base = (task.get("source") or {}).get("commit") or task.get("base_commit")
        result["task_base_commit"] = base
        if isinstance(base, str) and config_mod.SHA_RE.fullmatch(base):
            available = gitq.has_commit(source_dir, base)
            result["task_base_available"] = available
            detail = "task %s base %s %s in %s" % (
                task_id,
                base[:12],
                "is present" if available else "is NOT present",
                relative,
            )
            if not available:
                detail += (
                    "; publish or fetch the exact base ref -- the public pin is "
                    "not a substitute"
                )
            result["checks"].append(
                _check(
                    "task-base-commit",
                    OK if available else MISSING,
                    detail,
                    commit=base,
                    severity="error" if available else "warning",
                )
            )
            if (
                available is False
                and isinstance(configured, str)
                and base != configured
            ):
                result["checks"].append(
                    _check(
                        "task-base-differs-from-pin",
                        INCOMPATIBLE,
                        "task base %s differs from the configured source pin %s; "
                        "they are different source bases and must not be "
                        "interchanged" % (base[:12], configured[:12]),
                        severity="warning",
                    )
                )
    return result


def inspect(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    agent_profile: bool = False,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect the full offline toolchain report."""
    settings = config_mod.workflow_settings(config)
    checks: List[Dict[str, Any]] = [python_check(), command_check("git", ["git", "--version"])]

    dependencies = config.get("dependencies") or {}
    gitlinks = gitq.recorded_gitlinks(root)
    dependency_checks: List[Dict[str, Any]] = []
    source = config.get("source") or {}
    dependency_checks.append(
        dependency_check(
            root,
            "source-sglang",
            str(source.get("submodule") or "external/sglang"),
            source.get("commit") if config_mod.SHA_RE.fullmatch(str(source.get("commit") or "")) else None,
            gitlinks,
        )
    )
    for name in ("kda", "humanize"):
        entry = dependencies.get(name) or {}
        relative = str(entry.get("path") or "external/%s" % name)
        commit = entry.get("commit")
        dependency_checks.append(
            dependency_check(
                root,
                "dependency-%s" % name,
                relative,
                commit if config_mod.SHA_RE.fullmatch(str(commit or "")) else None,
                gitlinks,
            )
        )
    dirty_checks = [
        record
        for record in (dependency_dirty_check(item) for item in dependency_checks)
        if record is not None
    ]

    skills = config.get("skills") or {}
    skill_checks = [
        skill_check(root, "skill-%s" % label, str(relative))
        for label, relative in sorted(skills.items())
        if isinstance(relative, str) and relative
    ]

    kda = kda_section(root, settings)
    humanize = humanize_section(root, settings)

    agent_checks: List[Dict[str, Any]] = []
    if agent_profile:
        writer = settings.get("writer", {})
        reviewer = settings.get("reviewer", {})
        agent_checks.append(bash_check())
        agent_checks.append(command_check("jq", ["jq", "--version"]))
        agent_checks.append(
            command_check(str(writer.get("command") or "claude"), ["claude", "--version"])
        )
        agent_checks.append(
            command_check(str(reviewer.get("command") or "codex"), ["codex", "--version"])
        )

    availability = source_availability(root, config, task_id)

    all_checks = (
        checks
        + dependency_checks
        + dirty_checks
        + skill_checks
        + kda["checks"]
        + humanize["checks"]
        + agent_checks
        + availability["checks"]
    )

    errors = [
        item
        for item in all_checks
        if item["status"] in ERROR_STATUSES and item.get("severity") == "error"
    ]
    warnings = [
        item
        for item in all_checks
        if item["status"] in (ERROR_STATUSES | {DIRTY})
        and item.get("severity") == "warning"
    ]
    if errors:
        status = "failed"
    elif warnings:
        status = "degraded"
    else:
        status = "ok"

    report = {
        "schema": "k3ctl/doctor/1",
        "generated_at": util.utcnow(),
        "status": status,
        "config": str(config_path),
        "agent_profile": bool(agent_profile),
        "repository": repository_section(root),
        "environment": checks,
        "dependencies": dependency_checks,
        "dependency_state": dirty_checks,
        "skills": skill_checks,
        "kda": kda,
        "humanize": humanize,
        "agents": agent_checks,
        "source_availability": availability,
        "missing": [item["name"] for item in all_checks if item["status"] == MISSING],
        "incompatible": [
            item["name"] for item in all_checks if item["status"] == INCOMPATIBLE
        ],
        "workflow": {
            "writer": settings.get("writer"),
            "reviewer": settings.get("reviewer"),
        },
    }
    return report


def identity(report: Dict[str, Any]) -> Dict[str, Any]:
    """Stable, hashable toolchain identity for run and plan manifests."""
    value = {
        "python": next(
            (item.get("version") for item in report["environment"] if item["name"] == "python3"),
            None,
        ),
        "dependencies": {
            item["name"]: {
                "path": item.get("path"),
                "configured_commit": item.get("configured_commit"),
                "recorded_gitlink": item.get("recorded_gitlink"),
                "current_head": item.get("current_head"),
                "status": item.get("status"),
            }
            for item in report["dependencies"]
        },
        "skills": {
            item["name"]: {"path": item.get("path"), "sha256": item.get("sha256")}
            for item in report["skills"]
        },
        "kda_prompt": report["kda"].get("prompt"),
        "humanize": {
            "name": report["humanize"].get("name"),
            "version": report["humanize"].get("version"),
            "hooks": report["humanize"].get("hooks"),
        },
    }
    return {"toolchain": value, "sha256": util.sha256_json(value)}


# ------------------------------------------------------------------- rendering


_SYMBOL = {OK: "ok  ", MISSING: "miss", INCOMPATIBLE: "bad ", DIRTY: "dirty", SKIPPED: "skip"}


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    repo = report["repository"]
    lines.append("project root : %s" % repo["root"])
    lines.append(
        "git state    : HEAD %s, branch %s, %d staged entries"
        % (
            repo["head_commit"][:12] if repo["head_commit"] else "unborn",
            repo["branch"] or "-",
            repo["staged_file_count"],
        )
    )
    lines.append("config       : %s" % report["config"])
    lines.append("")

    def section(title: str, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        lines.append("%s:" % title)
        for item in items:
            lines.append(
                "  %s %-28s %s"
                % (_SYMBOL.get(item["status"], item["status"]), item["name"], item["detail"])
            )
        lines.append("")

    section("environment", report["environment"])
    section("dependencies", report["dependencies"])
    section("dependency state", report["dependency_state"])
    section("skills", report["skills"])
    section("kda workflow", report["kda"]["checks"])
    section("humanize plugin", report["humanize"]["checks"])
    if report["agent_profile"]:
        section("agent commands", report["agents"])
    section("source availability", report["source_availability"]["checks"])

    workflow = report.get("workflow") or {}
    writer = workflow.get("writer") or {}
    reviewer = workflow.get("reviewer") or {}
    lines.append(
        "workflow     : writer %s/%s, reviewer %s/%s"
        % (
            writer.get("command"),
            writer.get("model"),
            reviewer.get("command"),
            reviewer.get("model"),
        )
    )
    lines.append("status       : %s" % report["status"])
    if report["missing"]:
        lines.append("missing      : %s" % ", ".join(report["missing"]))
    if report["incompatible"]:
        lines.append("incompatible : %s" % ", ".join(report["incompatible"]))
    return "\n".join(lines)
