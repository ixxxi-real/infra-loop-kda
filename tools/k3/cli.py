#!/usr/bin/env python3
"""Command-line interface for the Kimi K3 KDA lab control plane.

One canonical entrypoint. The legacy ``validate``, ``status`` and ``task-list``
commands keep their original behaviour and output format; everything else is
new and prints JSON by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import agent as agent_mod
from . import config as config_mod
from . import export as export_mod
from . import lifecycle, overlay, paths, runner, taskfactory, toolchain, util
from . import workspace as workspace_mod

DEFAULT_CONFIG = "config/project.example.json"


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _root() -> Path:
    return paths.find_root()


def _load(args: argparse.Namespace) -> tuple:
    root = _root()
    path, config = config_mod.resolve_config(args.config, root=root)
    return root, path, config


def _validated(args: argparse.Namespace) -> tuple:
    """Load a configuration and fail when it does not validate."""
    root, path, config = _load(args)
    errors = config_mod.validate(path, config, root=root)
    if errors:
        _emit({"status": "failed", "config": str(path), "errors": errors})
        raise SystemExit(1)
    return root, path, config


def _task_id(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    return config_mod.active_task_id(config, getattr(args, "task", None))


# ------------------------------------------------------------ legacy commands


def command_validate(args: argparse.Namespace) -> int:
    root, path, config = _load(args)
    errors = config_mod.validate(path, config, root=root)
    _emit(
        {
            "status": "passed" if not errors else "failed",
            "config": str(path),
            "errors": errors,
        }
    )
    return 0 if not errors else 1


def command_status(args: argparse.Namespace) -> int:
    root, path, config = _load(args)
    errors = config_mod.validate(path, config, root=root)
    if errors:
        _emit({"status": "failed", "errors": errors})
        return 1
    source = config["source"]
    print("project: %s" % config["project_id"])
    print("model: %s" % config.get("model", {}).get("id", "<unset>"))
    print("source: %s @ %s" % (source["repository"], source["commit"]))
    print("active task: %s" % config.get("active_task", "<unset>"))
    for task in config["tasks"]:
        task_config = config_mod.load_json(root / task["path"] / "task.json")
        kind = task.get("kind", task_config.get("kind", "unknown"))
        print(
            "- %s [%s]: %s (optimization=%s, integration=%s)"
            % (
                task["id"],
                kind,
                task_config.get("status", "unknown"),
                task_config.get("optimization_status", "unknown"),
                task_config.get("integration_status", "unknown"),
            )
        )
    return 0


def command_task_list(args: argparse.Namespace) -> int:
    root, path, config = _load(args)
    errors = config_mod.validate(path, config, root=root)
    if errors:
        _emit({"status": "failed", "errors": errors})
        return 1
    for task in config["tasks"]:
        task_config = config_mod.load_json(root / task["path"] / "task.json")
        kind = task.get("kind", task_config.get("kind", "unknown"))
        print(
            "%s\t%s\t%s\toptimization=%s\tintegration=%s\t%s"
            % (
                task["id"],
                kind,
                task_config.get("status", "unknown"),
                task_config.get("optimization_status", "unknown"),
                task_config.get("integration_status", "unknown"),
                task["path"],
            )
        )
    return 0


# -------------------------------------------------------------------- doctor


def command_doctor(args: argparse.Namespace) -> int:
    root, path, config = _load(args)
    task_id = getattr(args, "task", None) or config.get("active_task")
    report = toolchain.inspect(
        root,
        path,
        config,
        agent_profile=bool(args.agent_profile),
        task_id=task_id if isinstance(task_id, str) else None,
    )
    if args.json:
        _emit(report)
    else:
        print(toolchain.render_text(report))
    return 0 if report["status"] in ("ok", "degraded") else 1


# --------------------------------------------------------------- task-create


def command_task_create(args: argparse.Namespace) -> int:
    root, path, config = _load(args)
    result = taskfactory.scaffold(
        root=root,
        config_path=path,
        config=config,
        task_id=args.id,
        kind=args.kind,
        base_commit=args.base_commit,
        objective=args.objective,
        validation_command=args.validation_command,
        performance_command=args.performance_command,
        tasks_dir=args.tasks_dir,
        repository=args.repository,
        branch=args.branch,
        register=bool(args.register),
        dry_run=bool(args.dry_run),
    )
    _emit(result)
    return 0


# ----------------------------------------------------------------- workspace


def command_workspace_prepare(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    result = workspace_mod.prepare(
        root=root,
        config_path=path,
        config=config,
        task_id=task_id,
        mode=args.mode,
        source_repo=args.source_repo,
        allow_dirty_source=bool(args.allow_dirty_source),
        include_harness=not args.no_harness,
        hardlinks=bool(args.hardlinks),
        dry_run=bool(args.dry_run),
    )
    _emit(result)
    return 0


def command_workspace_status(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(workspace_mod.status(root, config, task_id))
    return 0


# ------------------------------------------------------------------- overlay


def command_overlay_build(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    _emit(overlay.build(root, path, config))
    return 0


def command_overlay_verify(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    result = overlay.verify(root, config)
    _emit(result)
    return 0 if result["status"] == "ok" else 1


# --------------------------------------------------------------------- agent


def command_agent_plan(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    result = agent_mod.plan(
        root, path, config, task_id, max_iterations=args.max_iterations
    )
    _emit(result)
    return 0 if result["ready"] else 1


def command_agent_start(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(
        agent_mod.start(
            root, path, config, task_id, max_iterations=args.max_iterations
        )
    )
    return 0


def command_agent_status(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(agent_mod.status(root, config, task_id))
    return 0


def command_agent_resume(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(agent_mod.resume(root, path, config, task_id))
    return 0


def command_agent_stop(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    result = agent_mod.stop(
        root,
        config,
        task_id,
        signal_group=bool(args.process_group),
        timeout=args.timeout,
        escalate=bool(args.escalate),
    )
    _emit(result)
    return 0 if result["exited"] else 1


# ----------------------------------------------------------------------- run


def command_run_plan(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    result = runner.plan(
        root, path, config, task_id, args.operation, run_id=args.run_id
    )
    _emit(result)
    return 0 if result["ready"] else 1


def command_run_start(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(
        runner.start(
            root, path, config, task_id, args.operation, run_id=args.run_id
        )
    )
    return 0


def command_run_status(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(runner.status(root, config, task_id, args.run_id))
    return 0


def command_run_cancel(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(runner.cancel(root, config, task_id, args.run_id))
    return 0


def command_run_fetch(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    _emit(runner.fetch(root, config, task_id, args.run_id, args.destination))
    return 0


# -------------------------------------------------------------------- export


def command_export_bundle(args: argparse.Namespace) -> int:
    root, path, config = _validated(args)
    task_id = _task_id(args, config)
    result = export_mod.bundle(
        root, path, config, task_id, destination=args.destination, label=args.label
    )
    _emit(result)
    return 0 if result["complete"] else 1


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k3ctl",
        description=(
            "Control plane for Kimi K3 KDA kernel experiments: toolchain "
            "inspection, task scaffolding, isolated workspaces, the "
            "Claude+Humanize loop adapter, runner adapters and evidence export."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str, config: bool = True):
        sub = subparsers.add_parser(name, help=help_text)
        if config:
            sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.set_defaults(func=func)
        return sub

    # legacy, unchanged behaviour
    add("validate", command_validate, "validate project metadata")
    add("status", command_status, "show project and task status")
    add("task-list", command_task_list, "list configured tasks")

    doctor = add("doctor", command_doctor, "offline toolchain inspection")
    doctor.add_argument("--task", help="task whose source base is checked")
    doctor.add_argument(
        "--agent-profile",
        action="store_true",
        help="also check bash>=4, jq and the writer/reviewer commands",
    )
    doctor.add_argument("--json", action="store_true", help="emit JSON")

    create = add("task-create", command_task_create, "scaffold a new task")
    create.add_argument("--id", required=True)
    create.add_argument("--kind", default="kernel")
    create.add_argument("--base-commit", required=True)
    create.add_argument("--objective", required=True)
    create.add_argument("--validation-command", required=True)
    create.add_argument("--performance-command")
    create.add_argument("--tasks-dir", default="tasks")
    create.add_argument("--repository")
    create.add_argument("--branch")
    create.add_argument(
        "--register",
        action="store_true",
        help="append the task to a local (non-example) configuration",
    )
    create.add_argument("--dry-run", action="store_true")

    workspace = subparsers.add_parser(
        "workspace", help="prepare and inspect isolated source workspaces"
    )
    workspace_sub = workspace.add_subparsers(dest="workspace_command", required=True)

    prepare = workspace_sub.add_parser(
        "prepare", help="materialise a task's exact base commit"
    )
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.add_argument("--task")
    prepare.add_argument(
        "--mode",
        default=workspace_mod.MODE_CLONE,
        choices=list(workspace_mod.MODES),
        help="clone keeps history (required by the loop); archive is tree-only",
    )
    prepare.add_argument(
        "--source-repo", help="local source repository containing the base commit"
    )
    prepare.add_argument("--allow-dirty-source", action="store_true")
    prepare.add_argument("--no-harness", action="store_true")
    prepare.add_argument("--hardlinks", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=command_workspace_prepare)

    ws_status = workspace_sub.add_parser("status", help="report workspace state")
    ws_status.add_argument("--config", default=DEFAULT_CONFIG)
    ws_status.add_argument("--task")
    ws_status.set_defaults(func=command_workspace_status)

    overlay_parser = subparsers.add_parser(
        "overlay", help="build and verify the no-commit Humanize compat overlay"
    )
    overlay_sub = overlay_parser.add_subparsers(
        dest="overlay_command", required=True
    )
    ov_build = overlay_sub.add_parser("build", help="build the overlay")
    ov_build.add_argument("--config", default=DEFAULT_CONFIG)
    ov_build.set_defaults(func=command_overlay_build)
    ov_verify = overlay_sub.add_parser("verify", help="verify the built overlay")
    ov_verify.add_argument("--config", default=DEFAULT_CONFIG)
    ov_verify.set_defaults(func=command_overlay_verify)

    agent = subparsers.add_parser(
        "agent", help="Claude + Humanize loop adapter"
    )
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)

    def add_agent(name: str, func, help_text: str):
        sub = agent_sub.add_parser(name, help=help_text)
        sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.add_argument("--task")
        sub.set_defaults(func=func)
        return sub

    ag_plan = add_agent("plan", command_agent_plan, "dry run; never spawns")
    ag_plan.add_argument(
        "--max-iterations",
        type=int,
        default=42,
    )
    ag_start = add_agent("start", command_agent_start, "launch the loop")
    ag_start.add_argument(
        "--max-iterations",
        type=int,
        default=42,
    )
    add_agent("status", command_agent_status, "report real loop/process state")
    add_agent("resume", command_agent_resume, "resume the same session")
    ag_stop = add_agent("stop", command_agent_stop, "stop the recorded process")
    ag_stop.add_argument("--process-group", action="store_true")
    ag_stop.add_argument("--timeout", type=float, default=15.0)
    ag_stop.add_argument("--escalate", action="store_true")

    run = subparsers.add_parser("run", help="runner adapter operations")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    def add_run(name: str, func, help_text: str):
        sub = run_sub.add_parser(name, help=help_text)
        sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.add_argument("--task")
        sub.set_defaults(func=func)
        return sub

    rn_plan = add_run("plan", command_run_plan, "freeze a complete run plan")
    rn_plan.add_argument("--operation", default="correctness")
    rn_plan.add_argument(
        "--run-id",
        help=(
            "stable run id (single path component). Reuse it to keep one "
            "identity across a disconnect instead of spawning a duplicate job."
        ),
    )
    rn_start = add_run("start", command_run_start, "start through the adapter")
    rn_start.add_argument("--operation", default="correctness")
    rn_start.add_argument("--run-id", help="stable run id (single path component)")
    rn_status = add_run("status", command_run_status, "run state and artifacts")
    rn_status.add_argument("--run-id")
    rn_cancel = add_run("cancel", command_run_cancel, "cancel a recorded run")
    rn_cancel.add_argument("--run-id")
    rn_fetch = add_run("fetch", command_run_fetch, "fetch run artifacts")
    rn_fetch.add_argument("--run-id")
    rn_fetch.add_argument("--destination")

    export = subparsers.add_parser("export", help="delivery and evidence export")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    ex_bundle = export_sub.add_parser("bundle", help="write a fresh export bundle")
    ex_bundle.add_argument("--config", default=DEFAULT_CONFIG)
    ex_bundle.add_argument("--task")
    ex_bundle.add_argument("--destination")
    ex_bundle.add_argument("--label")
    ex_bundle.set_defaults(func=command_export_bundle)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except config_mod.ValidationError as exc:
        _emit({"status": "failed", "errors": [str(exc)]})
        return 1
    except paths.PathError as exc:
        _emit({"status": "failed", "errors": [str(exc)]})
        return 1
    except util.ToolError as exc:
        _emit({"status": "failed", "errors": [str(exc)]})
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
