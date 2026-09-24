#!/usr/bin/env python3
"""Isolated source workspace preparation.

``workspace prepare`` materialises a task's **exact** base commit from a
configured local source repository into a new, project-owned clone. It is local
setup only: it never contacts a network, never writes to the source repository
or to this repository's index, and it is explicitly *not* a readiness or
numerical verdict.

Two equivalent materialisation modes are supported:

``clone``
    ``git clone --no-checkout`` from the local source, then a checkout owned
    entirely by the new clone. Retains commit history, which the Humanize loop
    requires.

``archive``
    ``git archive`` of the exact commit. Produces an identical file tree with
    no history.

If the exact base commit is unreachable, preparation fails with a source
availability error. The public submodule pin is a different source base and is
never substituted.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as config_mod
from . import gitq, lifecycle, paths, util

#: Task control files live here inside the owned clone. Excluded locally so the
#: clone's working tree stays clean without any commit or staging.
CONTROL_DIR = ".kda-task"

#: Immutable ref pinned to the exact base commit, created in the owned clone only.
REVIEW_BASE_BRANCH = "kda-review-base"

WORK_BRANCH_PREFIX = "kda-work"

MODE_CLONE = "clone"
MODE_ARCHIVE = "archive"
MODES = (MODE_CLONE, MODE_ARCHIVE)

#: Harness directories snapshotted into the workspace when they exist.
HARNESS_DIRS = ("bench", "precision", "scripts")


class SourceUnavailable(util.ToolError):
    """The exact base commit cannot be reached from the local source."""


def runtime_dir(task_dir: Path) -> Path:
    return task_dir / "runtime"


def workspace_root(task_dir: Path) -> Path:
    """The writable candidate tree. Claude edits this one."""
    return runtime_dir(task_dir) / "workspace"


def baseline_root(task_dir: Path) -> Path:
    """The immutable baseline tree, materialised at the same base commit.

    A/B harnesses (``benchmark.py``, ``precision/diagnose.py``) take a
    ``--baseline-root`` and a ``--candidate-root``. Pointing both at the
    candidate would compare a tree with itself and report a meaningless
    zero delta, so the baseline is materialised separately and never written
    to after preparation.
    """
    return runtime_dir(task_dir) / "baseline"


def record_path(task_dir: Path) -> Path:
    return runtime_dir(task_dir) / "workspace.json"


def resolve_source_repo(
    root: Path, config: Dict[str, Any], override: Optional[str] = None
) -> Path:
    """Locate the local source repository to read the base commit from."""
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        source = config.get("source") or {}
        local = source.get("local_path")
        relative = str(local or source.get("submodule") or "external/sglang")
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
    candidate = candidate.expanduser()
    if not candidate.is_dir():
        raise SourceUnavailable(
            "local source repository does not exist: %s\n"
            "Configure source.local_path in your local configuration, or pass "
            "--source-repo, pointing at a clone that contains the task base "
            "commit." % candidate
        )
    if not gitq.is_own_root(candidate):
        raise SourceUnavailable(
            "%s is not an initialized Git repository of its own.\n"
            "An empty submodule directory resolves to its parent repository and "
            "must not be used as a source base. Run:\n"
            "  git submodule update --init --recursive\n"
            "or point --source-repo at a real local clone." % candidate
        )
    return candidate


def require_base_commit(
    source_repo: Path, base_commit: str, configured_pin: Optional[str]
) -> None:
    """Fail closed when the exact base commit is not reachable."""
    if not config_mod.SHA_RE.fullmatch(base_commit or ""):
        raise util.ToolError(
            "task base commit must be a complete 40-character Git SHA, got %r"
            % base_commit
        )
    if gitq.has_commit(source_repo, base_commit):
        return
    message = [
        "source availability error: base commit %s is not present in %s"
        % (base_commit, source_repo),
        "",
        "The exact task base is required. Publish or fetch that ref into a local",
        "clone and retry, for example:",
        "  git -C <clone> fetch <remote> %s" % base_commit,
    ]
    if configured_pin and configured_pin != base_commit:
        available = gitq.has_commit(source_repo, configured_pin)
        message += [
            "",
            "The configured source pin %s %s in this repository, but it is a"
            % (configured_pin[:12], "IS present" if available else "is not present"),
            "different source base. It is not a substitute for the task base and",
            "will not be used.",
        ]
    raise SourceUnavailable("\n".join(message))


def _refuse_unsafe_destination(destination: Path) -> None:
    """Validate the destination. Never deletes or replaces anything.

    There is deliberately no ``--force``: a prepared workspace may hold real
    candidate work, so this milestone always refuses an existing output and
    requires a fresh workspace or a fresh task. Removal is the operator's
    explicit, manual decision.
    """
    if destination.is_symlink():
        raise util.ToolError(
            "refusing to prepare into a symbolic link: %s" % destination
        )
    if destination.exists():
        if not destination.is_dir():
            raise util.ToolError(
                "refusing to overwrite an existing file: %s" % destination
            )
        if any(destination.iterdir()):
            raise util.ToolError(
                "refusing to reuse an existing workspace: %s\n"
                "It may contain candidate work. This command never deletes or "
                "replaces a workspace. Move or remove it yourself, or prepare a "
                "different task." % destination
            )

    # Reject symlinks anywhere along the parent chain, including components
    # that do not exist yet.
    current = destination.parent
    while True:
        if current.is_symlink():
            raise util.ToolError(
                "refusing to prepare beneath a symbolic link: %s" % current
            )
        if current == current.parent:
            break
        current = current.parent


def _check_source_cleanliness(source_repo: Path, allow_dirty: bool) -> Dict[str, Any]:
    """A frozen base must not be claimed from a dirty source by default."""
    dirty_lines = gitq.porcelain(source_repo) or []
    state = {
        "dirty": bool(dirty_lines),
        "dirty_entries": len(dirty_lines),
        "allow_dirty_source": bool(allow_dirty),
        "sparse": gitq.is_sparse(source_repo),
    }
    if dirty_lines and not allow_dirty:
        raise util.ToolError(
            "source repository %s has %d uncommitted entries.\n"
            "\n"
            "Materialisation always reads the exact base commit, so those local "
            "changes would NOT appear in the prepared workspace. The refusal is "
            "about provenance, not contents: if you expected uncommitted work to "
            "be included, this workspace would silently not contain it.\n"
            "\n"
            "This command never modifies the source repository. Either point "
            "--source-repo at a clean clone, or pass --allow-dirty-source to "
            "proceed and record the dirty state in the workspace record."
            % (source_repo, len(dirty_lines))
        )
    return state


def _materialise_clone(
    source_repo: Path,
    destination: Path,
    base_commit: str,
    task_id: str,
    hardlinks: bool = False,
) -> Dict[str, Any]:
    """Clone locally without checkout, then check out inside the owned clone."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = ["git", "clone", "--no-checkout", "--local"]
    if not hardlinks:
        argv.append("--no-hardlinks")
    argv += [str(source_repo), str(destination)]
    rc, _, err = util.run(argv, env=util.git_env(), timeout=1800)
    if rc != 0:
        raise util.ToolError(
            "git clone from %s failed: %s" % (source_repo, err.strip())
        )

    # Every write below targets the new, project-owned clone only.
    rc, _, err = util.run(
        ["git", "branch", "--force", REVIEW_BASE_BRANCH, base_commit],
        cwd=destination,
        env=util.git_env(),
        timeout=120,
    )
    if rc != 0:
        raise util.ToolError(
            "cannot create the local review base ref in the owned clone: %s"
            % err.strip()
        )
    work_branch = "%s/%s" % (WORK_BRANCH_PREFIX, task_id)
    rc, _, err = util.run(
        ["git", "checkout", "-b", work_branch, base_commit],
        cwd=destination,
        env=util.git_env(),
        timeout=600,
    )
    if rc != 0:
        raise util.ToolError(
            "checkout of %s in the owned clone failed: %s" % (base_commit, err.strip())
        )
    return {
        "mode": MODE_CLONE,
        "has_history": True,
        "review_base_branch": REVIEW_BASE_BRANCH,
        "work_branch": work_branch,
        "head_commit": gitq.head_commit(destination),
    }


def _materialise_archive(
    source_repo: Path, destination: Path, base_commit: str
) -> Dict[str, Any]:
    gitq.archive_commit(source_repo, base_commit, destination)
    return {
        "mode": MODE_ARCHIVE,
        "has_history": False,
        "review_base_branch": None,
        "work_branch": None,
        "head_commit": None,
    }


def _exclude_control_dir(destination: Path) -> bool:
    """Exclude the control directory locally via ``.git/info/exclude``.

    This keeps the owned clone's working tree clean for the Humanize loop
    without touching the index, staging anything, or creating a commit.
    """
    git_dir = gitq.git_dir(destination)
    if git_dir is None:
        return False
    info = Path(git_dir) / "info"
    info.mkdir(parents=True, exist_ok=True)
    exclude = info / "exclude"
    line = "/%s/" % CONTROL_DIR
    existing = ""
    if exclude.is_file():
        existing = exclude.read_text(encoding="utf-8")
        if line in existing.splitlines():
            return True
    with open(exclude, "a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write("# infra-loop-kda task control files (local only)\n")
        handle.write(line + "\n")
    return True


def _snapshot_control_files(
    root: Path,
    task_dir: Path,
    task: Dict[str, Any],
    task_id: str,
    control: Path,
    include_harness: bool,
) -> Dict[str, Any]:
    """Copy contract, plan and workload inputs into the owned clone.

    The live plan and an independent frozen snapshot are both written so goal
    drift can be detected without trusting a single mutable file.
    """
    control.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}

    def copy(relative: str, target_name: Optional[str] = None) -> None:
        source = task_dir / relative
        if not source.is_file():
            return
        if source.is_symlink():
            raise util.ToolError(
                "refusing to snapshot a symlinked task file: %s" % relative
            )
        target = control / (target_name or Path(relative).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        copied[str(target.relative_to(control))] = util.sha256_file(target)

    util.write_json(control / "task.json", task)
    copied["task.json"] = util.sha256_file(control / "task.json")

    copy(str(task.get("contract_file") or "contract.md"))
    copy(str(task.get("source_trace_file") or "source-trace.md"))

    plan_input = str(task.get("plan_input_file") or "plan-input.md")
    if (task_dir / plan_input).is_file():
        copy(plan_input, "plan.md")
        copy(plan_input, "plan-snapshot.md")
    copy(str(task.get("prompt_file") or "prompt.md"))

    workload = task.get("workload_file")
    if workload:
        copy(str(workload), "workloads.json")
    profile = task.get("model_profile_file")
    if profile:
        copy(str(profile), "model-profile.json")

    harness: Dict[str, int] = {}
    if include_harness:
        for name in HARNESS_DIRS:
            source = task_dir / name
            if not source.is_dir():
                continue
            target = control / "harness" / name
            count = 0
            for current, dirnames, filenames in os.walk(source):
                dirnames[:] = [
                    d for d in sorted(dirnames) if d not in {"__pycache__", ".git"}
                ]
                for filename in sorted(filenames):
                    full = Path(current) / filename
                    if full.is_symlink() or not full.is_file():
                        continue
                    if filename.endswith((".pyc", ".pyo")):
                        continue
                    relative = full.relative_to(source)
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(full), str(destination))
                    copied[str(destination.relative_to(control))] = util.sha256_file(
                        destination
                    )
                    count += 1
            if count:
                harness[name] = count
    return {"files": copied, "harness": harness}


def prepare(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    mode: str = MODE_CLONE,
    source_repo: Optional[str] = None,
    allow_dirty_source: bool = False,
    include_harness: bool = True,
    hardlinks: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Prepare an isolated workspace at the task's exact base commit.

    There is deliberately no force/replace option: a prepared workspace may
    contain real candidate work, so an existing output is always refused.
    """
    if mode not in MODES:
        raise util.ToolError(
            "unknown workspace mode %r (expected one of: %s)" % (mode, ", ".join(MODES))
        )

    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    base = lifecycle.base_commit(task)
    if not base:
        raise util.ToolError("task %s does not record a base commit" % task_id)

    # Validate both destinations before any source work, so a dry run and a
    # real run agree on whether preparation is possible.
    destination = workspace_root(task_dir)
    baseline = baseline_root(task_dir)
    _refuse_unsafe_destination(destination)
    _refuse_unsafe_destination(baseline)

    source_path = resolve_source_repo(root, config, source_repo)
    configured_pin = (config.get("source") or {}).get("commit")
    require_base_commit(source_path, base, configured_pin if isinstance(configured_pin, str) else None)
    source_state = _check_source_cleanliness(source_path, allow_dirty_source)

    record: Dict[str, Any] = {
        "schema": "k3ctl/workspace/1",
        "generated_at": util.utcnow(),
        "task_id": task_id,
        "config": str(config_path),
        "mode": mode,
        "source": {
            "repository_path": str(source_path),
            "base_commit": base,
            "configured_pin": configured_pin,
            "base_differs_from_pin": bool(
                configured_pin and configured_pin != base
            ),
            "state": source_state,
        },
        "workspace": {
            "path": paths.relative_to_root(root, destination),
            "absolute_path": str(destination),
            "control_dir": CONTROL_DIR,
            "role": "candidate (writable; the agent edits this tree)",
        },
        "baseline": {
            "path": paths.relative_to_root(root, baseline),
            "absolute_path": str(baseline),
            "role": "baseline (immutable reference at the same base commit)",
        },
        "is_ready_verdict": False,
        "note": (
            "Local setup only. This record does not assert correctness, "
            "performance, or that the task is READY."
        ),
    }

    if dry_run:
        record["dry_run"] = True
        return record

    if mode == MODE_CLONE:
        materialised = _materialise_clone(
            source_path, destination, base, task_id, hardlinks=hardlinks
        )
    else:
        materialised = _materialise_archive(source_path, destination, base)
    record["workspace"].update(materialised)

    excluded = _exclude_control_dir(destination) if mode == MODE_CLONE else False
    record["workspace"]["control_dir_excluded_locally"] = excluded

    control = destination / CONTROL_DIR
    snapshot = _snapshot_control_files(
        root, task_dir, task, task_id, control, include_harness
    )
    record["control_files"] = snapshot["files"]
    record["harness"] = snapshot["harness"]

    hashes = util.tree_hashes(destination, skip=(".git", CONTROL_DIR, "__pycache__"))
    record["base_tree"] = {
        "file_count": len(hashes),
        "digest": util.tree_digest(hashes),
    }
    if mode == MODE_CLONE:
        record["workspace"]["clean"] = gitq.is_dirty(destination) is False

    # Materialise the immutable baseline at the same commit. ``git archive``
    # is used regardless of mode: the baseline needs no history, only the exact
    # tree. Both trees are hashed so a reviewer can confirm they started
    # identical -- and so a comparison of a tree against itself is impossible.
    gitq.archive_commit(source_path, base, baseline)
    baseline_hashes = util.tree_hashes(baseline, skip=(".git", "__pycache__"))
    record["baseline"].update(
        {
            "mode": MODE_ARCHIVE,
            "file_count": len(baseline_hashes),
            "digest": util.tree_digest(baseline_hashes),
        }
    )
    record["base_tree"]["baseline_matches_candidate"] = (
        record["baseline"]["digest"] == record["base_tree"]["digest"]
    )
    if not record["base_tree"]["baseline_matches_candidate"]:
        raise util.ToolError(
            "baseline and candidate trees differ immediately after preparation "
            "(baseline %s, candidate %s at commit %s).\n"
            "They must be identical before any candidate edit. Refusing to "
            "record an inconsistent workspace."
            % (
                record["baseline"]["digest"][:12],
                record["base_tree"]["digest"][:12],
                base,
            )
        )

    util.write_json(record_path(task_dir), record)
    return record


def load_record(root: Path, config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task_dir = config_mod.task_dir(root, config, task_id)
    target = record_path(task_dir)
    if not target.is_file():
        raise util.ToolError(
            "no workspace prepared for task %s; run: k3ctl workspace prepare "
            "--task %s" % (task_id, task_id)
        )
    return config_mod.load_json(target)


def status(root: Path, config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Report the current state of a prepared workspace."""
    task_dir = config_mod.task_dir(root, config, task_id)
    target = record_path(task_dir)
    if not target.is_file():
        return {
            "task_id": task_id,
            "prepared": False,
            "detail": "no workspace record; run 'k3ctl workspace prepare'",
        }
    record = config_mod.load_json(target)
    destination = Path(record.get("workspace", {}).get("absolute_path") or "")
    present = destination.is_dir()
    result: Dict[str, Any] = {
        "task_id": task_id,
        "prepared": True,
        "mode": record.get("mode"),
        "path": record.get("workspace", {}).get("path"),
        "present": present,
        "base_commit": record.get("source", {}).get("base_commit"),
        "base_differs_from_pin": record.get("source", {}).get("base_differs_from_pin"),
        "is_ready_verdict": False,
    }
    if present and record.get("mode") == MODE_CLONE:
        result["head_commit"] = gitq.head_commit(destination)
        result["branch"] = gitq.current_branch(destination)
        result["dirty"] = gitq.is_dirty(destination)
        result["review_base_present"] = bool(
            gitq.has_commit(destination, REVIEW_BASE_BRANCH)
        )
    if present:
        hashes = util.tree_hashes(
            destination, skip=(".git", CONTROL_DIR, "__pycache__")
        )
        current = util.tree_digest(hashes)
        recorded = record.get("base_tree", {}).get("digest")
        result["tree_digest"] = current
        result["base_tree_unmodified"] = bool(recorded) and current == recorded
    return result
