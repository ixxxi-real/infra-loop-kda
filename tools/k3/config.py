#!/usr/bin/env python3
"""Project and task configuration: loading, validation and defaults.

The validation rules from the original ``tools/k3ctl.py`` are preserved exactly,
including their message wording, so existing tests and CI keep working. New
sections (``workflow``, ``runner``, ``no_commit_overlay``) are optional and are
only validated when present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import paths, util

SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")

#: Reasoning efforts the pinned Humanize release accepts natively.
UPSTREAM_EFFORTS = ("xhigh", "high", "medium", "low")

#: Defaults for both adapters. The active public workflow is Humanize2; its
#: three roles below all use the isolated Codex profile. The legacy Humanize1
#: fields remain available for old local configurations.
DEFAULT_WORKFLOW: Dict[str, Any] = {
    # Backwards-compatible default for local configurations that predate the
    # Humanize2 block. The public project config opts into Humanize2 explicitly.
    "humanize_backend": "humanize",
    "writer": {
        "command": "claude",
        "model": "opus",
        "effort": "max",
        "permission_mode": "acceptEdits",
    },
    "reviewer": {
        "command": "codex",
        "model": "gpt-6-astra",
        "effort": "ultra",
        # The project deliberately isolates reviewer credentials from the
        # default Codex profile.  The agent adapter exports this path to the
        # Humanize child process and its reviewer launcher.
        "codex_home": "~/.codex-bak",
        "bypass_sandbox": True,
        "disable_apps": True,
    },
    "humanize": {
        "plugin_dir": "external/humanize",
        "start_command": "/humanize:start-rlcr-loop",
        "cancel_command": "/humanize:cancel-rlcr-loop",
    },
    # Humanize2 is a separate Python/CLI runtime.  It deliberately lives next
    # to the Humanize1 plugin instead of replacing it: existing RLCR sessions
    # and their state remain owned by the old adapter.
    "humanize2": {
        "runtime_dir": "external/humanize2",
        "command": "hmz",
        "flow": "flows/infra_loop_kda_flame_chase",
        "first_chaser": "codex/gpt-6-astra:ultra",
        "second_chaser": "claude/claude-opus-5:max",
        "cleaner": "codex/gpt-6-astra:ultra",
        "work_paths": ["python", ".kda-task"],
        "budget": "duration=12h,cost=100",
        "codex_home": "~/.codex-bak",
        "resume": True,
    },
    "kda": {
        "prompt": "external/kda/prompts/basic-flow.md",
        "agent_flow": "external/kda/docs/agent-flow.md",
    },
}

DEFAULT_RUNNER: Dict[str, Any] = {
    "adapter": "local-command",
    "workdir": None,
    "environment": {},
}


class ValidationError(ValueError):
    """Raised when configuration cannot be loaded at all."""


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object, raising :class:`ValidationError` on any problem."""
    try:
        value = util.read_json(Path(path))
    except util.ToolError as exc:
        raise ValidationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ValidationError("top-level JSON value must be an object: %s" % path)
    return value


def resolve_config(raw: str, root: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    """Resolve a config path (relative to the project root) and load it."""
    base = paths.find_root() if root is None else Path(root)
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path, load_json(path)


def is_example(path: Path) -> bool:
    return "example" in Path(path).name


def find_placeholders(value: Any, prefix: str = "") -> List[str]:
    """Return paths whose string values still contain a template placeholder."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = "%s.%s" % (prefix, key) if prefix else str(key)
            found.extend(find_placeholders(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, "%s[%d]" % (prefix, index)))
    elif isinstance(value, str) and "<" in value and ">" in value:
        found.append(prefix)
    return found


# --------------------------------------------------------------- validation


def _validate_source(config_path: Path, config: Dict[str, Any], errors: List[str]) -> None:
    source = config.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
        return
    if source.get("submodule") != "external/sglang":
        errors.append("source.submodule must be external/sglang")
    if not source.get("repository"):
        errors.append("source.repository is required")
    commit = source.get("commit")
    if not isinstance(commit, str):
        errors.append("source.commit must be a string")
    elif not is_example(config_path) and not SHA_RE.fullmatch(commit):
        errors.append("local source.commit must be a complete 40-character Git SHA")
    elif is_example(config_path) and not (
        SHA_RE.fullmatch(commit) or commit.startswith("<")
    ):
        errors.append(
            "example source.commit must be a 40-character SHA or an explicit placeholder"
        )


def _validate_dependencies(
    config_path: Path, config: Dict[str, Any], errors: List[str]
) -> None:
    dependencies = config.get("dependencies")
    if not isinstance(dependencies, dict):
        errors.append("dependencies must be an object")
        return
    # humanize2 is optional for backwards-compatible local configurations.  A
    # config that declares the block gets the same strict pin validation as the
    # two original dependencies.
    names = ("kda", "humanize")
    for optional in ("humanize2", "flowverse"):
        if optional in dependencies:
            names += (optional,)
    for name in names:
        dependency = dependencies.get(name)
        if not isinstance(dependency, dict):
            errors.append("dependencies.%s must be an object" % name)
            continue
        for field in ("path", "repository"):
            if not isinstance(dependency.get(field), str) or not dependency.get(field):
                errors.append("dependencies.%s.%s is required" % (name, field))
        commit = dependency.get("commit")
        if not isinstance(commit, str):
            errors.append("dependencies.%s.commit must be a string" % name)
        elif not is_example(config_path) and not SHA_RE.fullmatch(commit):
            errors.append(
                "dependencies.%s.commit must be a complete 40-character Git SHA" % name
            )
        elif is_example(config_path) and not (
            SHA_RE.fullmatch(commit) or commit.startswith("<")
        ):
            errors.append(
                "example dependencies.%s.commit must be a 40-character SHA or placeholder"
                % name
            )


def _validate_task_entry(
    root: Path,
    config: Dict[str, Any],
    item: Dict[str, Any],
    task_id: str,
    errors: List[str],
) -> None:
    task_path = item.get("path")
    if not isinstance(task_path, str):
        errors.append("task %s needs a path" % task_id)
        return
    task_root = (root / task_path).resolve()
    try:
        task_root.relative_to(root)
    except ValueError:
        errors.append("task %s path escapes the repository: %s" % (task_id, task_path))
        return
    task_file = task_root / "task.json"
    if not task_file.is_file():
        errors.append("task %s is missing %s/task.json" % (task_id, task_path))
        return
    try:
        task_config = load_json(task_file)
    except ValidationError as exc:
        errors.append(str(exc))
        return

    if task_config.get("task_id") != task_id:
        errors.append("task %s has mismatched task.json task_id" % task_id)
    if task_config.get("project_id") != config.get("project_id"):
        errors.append("task %s has a mismatched project_id" % task_id)
    task_source = task_config.get("source", {})
    if not isinstance(task_source, dict):
        task_source = {}
    task_commit = task_source.get("commit", task_config.get("base_commit"))
    if not isinstance(task_commit, str) or not SHA_RE.fullmatch(task_commit):
        errors.append("task %s needs a complete source commit" % task_id)

    for field in ("workload_file", "model_profile_file", "patch_file"):
        relative = task_config.get(field)
        if relative:
            artifact = (task_root / relative).resolve()
            try:
                artifact.relative_to(root)
            except ValueError:
                errors.append(
                    "task %s %s escapes the repository: %s" % (task_id, field, relative)
                )
                continue
            if not artifact.is_file():
                errors.append("task %s is missing %s: %s" % (task_id, field, relative))

    patch_file = task_config.get("patch_file")
    expected_patch_sha = task_config.get("patch_sha256")
    if patch_file and expected_patch_sha:
        artifact = (task_root / patch_file).resolve()
        escaped = False
        try:
            artifact.relative_to(root)
        except ValueError:
            errors.append(
                "task %s patch_file escapes the repository: %s" % (task_id, patch_file)
            )
            escaped = True
        if not escaped and artifact.is_file():
            actual = util.sha256_file(artifact)
            if actual != expected_patch_sha:
                errors.append(
                    "task %s patch_sha256 does not match %s" % (task_id, patch_file)
                )

    evidence_manifest = task_config.get("evidence_manifest")
    if evidence_manifest:
        artifact = (task_root / evidence_manifest).resolve()
        try:
            artifact.relative_to(root)
        except ValueError:
            errors.append(
                "task %s evidence_manifest escapes the repository: %s"
                % (task_id, evidence_manifest)
            )
            return
        if not artifact.is_file():
            errors.append(
                "task %s is missing evidence_manifest: %s" % (task_id, evidence_manifest)
            )


def _validate_workflow(config: Dict[str, Any], errors: List[str]) -> None:
    workflow = config.get("workflow")
    if workflow is None:
        return
    if not isinstance(workflow, dict):
        errors.append("workflow must be an object")
        return
    backend = workflow.get("humanize_backend")
    if backend is not None and backend not in ("humanize", "humanize2"):
        errors.append("workflow.humanize_backend must be 'humanize' or 'humanize2'")
    for role in ("writer", "reviewer"):
        section = workflow.get(role)
        if section is None:
            continue
        if not isinstance(section, dict):
            errors.append("workflow.%s must be an object" % role)
            continue
        command = section.get("command")
        if command is not None and (not isinstance(command, str) or not command.strip()):
            errors.append("workflow.%s.command must be a non-empty string" % role)
        model = section.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            errors.append("workflow.%s.model must be a non-empty string" % role)
        effort = section.get("effort")
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            errors.append("workflow.%s.effort must be a non-empty string" % role)
        if role == "reviewer":
            codex_home = section.get("codex_home")
            if codex_home is not None and (
                not isinstance(codex_home, str) or not codex_home.strip()
            ):
                errors.append("workflow.reviewer.codex_home must be a non-empty string")
            for key in ("bypass_sandbox", "disable_apps"):
                value = section.get(key)
                if value is not None and not isinstance(value, bool):
                    errors.append("workflow.reviewer.%s must be a boolean" % key)
    for key in ("secret", "token", "api_key", "password"):
        if key in workflow:
            errors.append(
                "workflow.%s must not be stored in project configuration" % key
            )
    humanize = workflow.get("humanize")
    if humanize is not None and not isinstance(humanize, dict):
        errors.append("workflow.humanize must be an object")
    elif isinstance(humanize, dict):
        skip_quiz = humanize.get("skip_quiz")
        if skip_quiz is not None and not isinstance(skip_quiz, bool):
            errors.append("workflow.humanize.skip_quiz must be a boolean")
    humanize2 = workflow.get("humanize2")
    if humanize2 is not None:
        if not isinstance(humanize2, dict):
            errors.append("workflow.humanize2 must be an object")
        else:
            for key in ("runtime_dir", "command", "flow"):
                value = humanize2.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append("workflow.humanize2.%s must be a non-empty string" % key)
            for key in ("first_chaser", "second_chaser", "cleaner"):
                value = humanize2.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append("workflow.humanize2.%s must be a non-empty string" % key)
            budget = humanize2.get("budget")
            if budget is not None and (not isinstance(budget, str) or not budget.strip()):
                errors.append("workflow.humanize2.budget must be a non-empty string")
            work_paths = humanize2.get("work_paths")
            if work_paths is not None and (
                not isinstance(work_paths, list)
                or not work_paths
                or any(not isinstance(item, str) or not item.strip() for item in work_paths)
            ):
                errors.append("workflow.humanize2.work_paths must be a non-empty string list")
            resume = humanize2.get("resume")
            if resume is not None and not isinstance(resume, bool):
                errors.append("workflow.humanize2.resume must be a boolean")
            codex_home = humanize2.get("codex_home")
            if codex_home is not None and (not isinstance(codex_home, str) or not codex_home.strip()):
                errors.append("workflow.humanize2.codex_home must be a non-empty string")


def _validate_overlay(config: Dict[str, Any], errors: List[str]) -> None:
    overlay = config.get("no_commit_overlay")
    if overlay is None:
        return
    if not isinstance(overlay, dict):
        errors.append("no_commit_overlay must be an object")
        return
    enabled = overlay.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append("no_commit_overlay.enabled must be a boolean")
    workspace = overlay.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        errors.append("no_commit_overlay.workspace must be a string")


def _validate_runner(config: Dict[str, Any], errors: List[str]) -> None:
    runner = config.get("runner")
    if runner is None:
        return
    if not isinstance(runner, dict):
        errors.append("runner must be an object")
        return
    adapter = runner.get("adapter")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        errors.append("runner.adapter must be a non-empty string")
    environment = runner.get("environment")
    if environment is not None:
        if not isinstance(environment, dict):
            errors.append("runner.environment must be an object")
        else:
            for key, value in environment.items():
                if not isinstance(value, str):
                    errors.append(
                        "runner.environment.%s must be a string" % key
                    )
    commands = runner.get("commands")
    if commands is not None and not isinstance(commands, dict):
        errors.append("runner.commands must be an object")


def validate(
    config_path: Path,
    config: Dict[str, Any],
    root: Optional[Path] = None,
) -> List[str]:
    """Validate a project configuration and return a list of error strings."""
    base = paths.find_root() if root is None else Path(root).resolve()
    errors: List[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not config.get("project_id"):
        errors.append("project_id is required")

    _validate_source(config_path, config, errors)
    _validate_dependencies(config_path, config, errors)

    skills = config.get("skills")
    if not isinstance(skills, dict):
        errors.append("skills must be an object")
    else:
        for name in ("project", "kernelwiki", "ncu_report", "humanize"):
            if not isinstance(skills.get(name), str) or not skills.get(name):
                errors.append("skills.%s is required" % name)

    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        tasks = []
    task_ids: set = set()
    for item in tasks:
        if not isinstance(item, dict):
            errors.append("each task must be an object")
            continue
        task_id = item.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("each task needs a non-empty id")
            continue
        if task_id in task_ids:
            errors.append("duplicate task id: %s" % task_id)
        task_ids.add(task_id)
        _validate_task_entry(base, config, item, task_id, errors)

    _validate_workflow(config, errors)
    _validate_runner(config, errors)
    _validate_overlay(config, errors)

    if not is_example(config_path):
        for field in find_placeholders(config):
            errors.append("local config still contains a placeholder at %s" % field)

    active = config.get("active_task")
    if active and active not in task_ids:
        errors.append("active_task does not refer to a configured task: %s" % active)
    return errors


# ------------------------------------------------------------------ accessors


def _merge_defaults(defaults: Dict[str, Any], override: Any) -> Dict[str, Any]:
    merged = {}
    for key, value in defaults.items():
        if isinstance(value, dict):
            child = override.get(key) if isinstance(override, dict) else None
            merged[key] = _merge_defaults(value, child)
        else:
            merged[key] = value
    if isinstance(override, dict):
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                continue
            merged[key] = value
    return merged


def workflow_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Workflow settings with project defaults applied."""
    return _merge_defaults(DEFAULT_WORKFLOW, config.get("workflow"))


def humanize_backend(config: Dict[str, Any]) -> str:
    """Return the selected KDA writer/reviewer workflow backend."""
    value = (config.get("workflow") or {}).get("humanize_backend", "humanize")
    return str(value or "humanize")


def runner_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_defaults(DEFAULT_RUNNER, config.get("runner"))


def overlay_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    overlay = config.get("no_commit_overlay")
    if not isinstance(overlay, dict):
        overlay = {}
    return {
        "enabled": bool(overlay.get("enabled", False)),
        "workspace": overlay.get("workspace") or "external/.overlay/humanize-no-commit",
        "plan_relpath": overlay.get("plan_relpath"),
        "snapshot_relpath": overlay.get("snapshot_relpath"),
    }


def task_entry(config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for item in config.get("tasks", []):
        if isinstance(item, dict) and item.get("id") == task_id:
            return item
    raise util.ToolError(
        "task %r is not registered in the project configuration" % task_id
    )


def task_dir(root: Path, config: Dict[str, Any], task_id: str) -> Path:
    entry = task_entry(config, task_id)
    return paths.safe_join(root, str(entry.get("path")))


def load_task(root: Path, config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    return load_json(task_dir(root, config, task_id) / "task.json")


def active_task_id(config: Dict[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    active = config.get("active_task")
    if isinstance(active, str) and active:
        return active
    raise util.ToolError(
        "no task selected: pass --task or set active_task in the configuration"
    )
