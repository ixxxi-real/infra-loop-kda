#!/usr/bin/env python3
"""Delivery and evidence export.

``export bundle`` writes a **fresh** directory containing an inventory, the
current task's declared deliverables, the dependency/toolchain/source identity,
and a diff of local work.

Rules this module holds to:

* No Git index writes. Nothing is staged, committed, stashed or cleaned.
* This repository's HEAD is unborn. The bundle therefore diffs the working tree
  against the **staged index** and lists untracked files separately. It never
  invents a commit, and it never claims work is "uncommitted" as a permanent
  property -- it records the observed state at export time.
* Every staged path is preserved and recorded, so a reviewer can confirm the
  user's staged work is intact.
* Missing local proof is explicit. An absent report, an unavailable source base
  or an unverifiable artifact is reported as missing, never omitted.
* An existing accepted patch may be read and copied, but only when its hash
  matches the recorded value exactly. No formal review is ever fabricated.
* A local export performs no GPU work and says so.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as config_mod
from . import gitq, lifecycle, paths, toolchain, util, workspace

#: Default export location. Must be Git-ignored: bundles contain local paths.
DEFAULT_EXPORT_ROOT = ".infra/exports"

#: Task files copied into a bundle when they exist.
DELIVERABLE_CANDIDATES = (
    "task.json",
    "contract.md",
    "source-trace.md",
    "plan.md",
    "plan-input.md",
    "README.md",
    "model-profile.json",
)


def export_root(root: Path, override: Optional[str] = None) -> Path:
    return paths.safe_join(root, str(override or DEFAULT_EXPORT_ROOT))


def _require_ignored(root: Path, relative: str) -> None:
    if not gitq.is_ignored(root, relative.rstrip("/") + "/"):
        raise util.ToolError(
            "export destination %s is not ignored by Git.\n"
            "Bundles contain local absolute paths and must never be tracked. "
            "Add it to .gitignore first." % relative
        )


def _copy_recorded(
    source: Path, destination: Path, inventory: Dict[str, Any], label: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(destination))
    inventory[label] = {
        "path": str(destination.name),
        "sha256": util.sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _patch_entry(
    task_dir: Path, task: Dict[str, Any], bundle: Path, missing: List[str]
) -> Optional[Dict[str, Any]]:
    """Copy an accepted patch only when its recorded hash matches exactly."""
    relative = task.get("patch_file")
    if not relative:
        return None
    source = task_dir / str(relative)
    if not source.is_file():
        missing.append("declared patch_file is absent: %s" % relative)
        return None
    actual = util.sha256_file(source)
    expected = task.get("patch_sha256")
    if expected and actual != expected:
        missing.append(
            "patch_file hash mismatch: recorded %s, actual %s -- not copied"
            % (expected, actual)
        )
        return {
            "path": str(relative),
            "copied": False,
            "recorded_sha256": expected,
            "actual_sha256": actual,
            "hash_verified": False,
        }
    target = bundle / "deliverables" / Path(str(relative)).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(target))
    return {
        "path": "deliverables/%s" % target.name,
        "source": str(relative),
        "copied": True,
        "sha256": actual,
        "recorded_sha256": expected,
        "hash_verified": bool(expected) and actual == expected,
        "note": (
            "Copied verbatim from the task package. This is an existing "
            "artifact; no review of it was performed by this export."
        ),
    }


def _reports(
    task_dir: Path, bundle: Path, missing: List[str]
) -> Dict[str, Any]:
    """Copy the task's reports directory when present."""
    source = task_dir / "reports"
    result: Dict[str, Any] = {"present": source.is_dir(), "files": {}}
    if not source.is_dir():
        missing.append("task has no reports/ directory")
        return result
    target = bundle / "reports"
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.is_file() and not item.is_symlink():
            shutil.copyfile(str(item), str(target / item.name))
            result["files"][item.name] = util.sha256_file(target / item.name)
    if not result["files"]:
        missing.append("task reports/ directory contains no files")
    return result


def _local_diff(
    root: Path, bundle: Path, task_relative: Optional[str]
) -> Dict[str, Any]:
    """Diff the working tree against the staged index, plus untracked files.

    This repository has an unborn HEAD, so there is no commit to diff against.
    The staged index is the authoritative base.
    """
    staged = gitq.staged_entries(root)
    untracked = gitq.untracked_paths_z(root)

    # The bundle itself must never appear in its own diff.
    bundle_relative = paths.relative_to_root(root, bundle)
    untracked = [
        item
        for item in untracked
        if not item.startswith(bundle_relative.rstrip("/") + "/")
        and item != bundle_relative
    ]

    # The index holds the staged baseline and new work arrives untracked, so a
    # plain worktree-vs-index diff would omit every new file's content. The
    # bundle must be applicable, so untracked files are included as additions.
    index_before = (
        util.sha256_file(root / ".git" / "index")
        if (root / ".git" / "index").is_file()
        else None
    )
    combined = gitq.diff_including_untracked(root, untracked)
    diff_text = combined["diff"]
    diff_path = bundle / "local-changes.diff"
    util.write_text(diff_path, diff_text)

    # The diff is built against a temporary copy of the index; prove the real
    # one is untouched rather than asserting it.
    index_after = (
        util.sha256_file(root / ".git" / "index")
        if (root / ".git" / "index").is_file()
        else None
    )
    if index_before != index_after:
        raise util.ToolError(
            "the Git index changed while building the export diff "
            "(%s -> %s). Refusing to emit a bundle: the staged baseline must "
            "never be modified."
            % ((index_before or "absent")[:12], (index_after or "absent")[:12])
        )

    # Per-file record for the files the diff carries as additions, so a reviewer
    # can verify the patch reconstructs exactly these bytes.
    untracked_inventory: Dict[str, Any] = {}
    for relative in combined["included_untracked"]:
        target = root / relative
        try:
            if target.is_symlink():
                untracked_inventory[relative] = {
                    "kind": "symlink",
                    "target": os.readlink(str(target)),
                }
            else:
                untracked_inventory[relative] = {
                    "kind": "file",
                    "sha256": util.sha256_file(target),
                    "bytes": target.stat().st_size,
                    "executable": bool(target.stat().st_mode & 0o111),
                }
        except OSError as exc:
            untracked_inventory[relative] = {"kind": "unreadable", "error": str(exc)}

    staged_listing = "".join(
        "%s %s %s\t%s\n" % (e["mode"], e["object"], e["stage"], e["path"])
        for e in staged
    )
    staged_path = bundle / "staged-index.txt"
    util.write_text(staged_path, staged_listing)

    untracked_path = bundle / "untracked-files.txt"
    util.write_text(untracked_path, "".join("%s\n" % item for item in untracked))

    binary_markers = diff_text.count("GIT binary patch")

    return {
        "base": "staged-index",
        "head_exists_at_export": gitq.head_exists(root),
        "observed_at": util.utcnow(),
        "staged_entry_count": len(staged),
        "staged_listing_sha256": util.sha256_text(staged_listing),
        "staged_index_file": "staged-index.txt",
        "diff_file": "local-changes.diff",
        "diff_sha256": util.sha256_text(diff_text),
        "diff_bytes": len(diff_text.encode("utf-8")),
        "diff_is_empty": not diff_text.strip(),
        "binary_patch_sections": binary_markers,
        "diff_method": combined["method"],
        "diff_includes_untracked_content": True,
        "tracked_modifications_in_diff": diff_text.count("diff --git")
        - len(combined["included_untracked"]),
        "untracked_file_count": len(untracked),
        "untracked_file_list": "untracked-files.txt",
        "untracked_included_in_diff": len(combined["included_untracked"]),
        "untracked_inventory": untracked_inventory,
        "untracked_skipped": combined["skipped_untracked"],
        "index_sha256_before": index_before,
        "index_sha256_after": index_after,
        "index_unchanged": index_before == index_after,
        "excluded_from_diff": [bundle_relative],
        "apply_with": (
            "git apply local-changes.diff  (from a tree materialised with "
            "'git checkout-index -a --prefix=<dir>/' using staged-index.txt)"
        ),
        "note": (
            "Diffed against the staged index because this repository has no "
            "commit yet. Untracked files are included as additions with their "
            "full content, so the patch reconstructs new work rather than only "
            "naming it. The diff was built against a temporary copy of the "
            "index; nothing was staged, committed or cleaned, and the recorded "
            "before/after index hashes prove the staged baseline is intact."
        ),
    }


def bundle(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    destination: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a fresh export bundle and return its inventory."""
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")

    base_root = export_root(root, destination)
    _require_ignored(root, paths.relative_to_root(root, base_root))

    name = "%s-%s" % (task_id, label or util.stamp())
    target = base_root / name
    if target.exists():
        raise util.ToolError(
            "export destination already exists: %s\n"
            "Each export writes a fresh directory; it never overwrites one."
            % paths.relative_to_root(root, target)
        )
    target.mkdir(parents=True)

    missing: List[str] = []
    inventory: Dict[str, Any] = {}

    # --- task package ----------------------------------------------------
    task_bundle = target / "task"
    for candidate in DELIVERABLE_CANDIDATES:
        source = task_dir / candidate
        if source.is_file() and not source.is_symlink():
            _copy_recorded(
                source, task_bundle / candidate, inventory, "task/%s" % candidate
            )
        elif candidate == "task.json":
            raise util.ToolError("task %s has no task.json" % task_id)

    workload = task.get("workload_file")
    if workload:
        source = task_dir / str(workload)
        if source.is_file():
            _copy_recorded(
                source,
                task_bundle / Path(str(workload)).name,
                inventory,
                "task/%s" % Path(str(workload)).name,
            )
        else:
            missing.append("declared workload_file is absent: %s" % workload)

    patch = _patch_entry(task_dir, task, target, missing)
    reports = _reports(task_dir, target, missing)

    # --- identity --------------------------------------------------------
    report = toolchain.inspect(root, config_path, config, task_id=task_id)
    identity = toolchain.identity(report)
    availability = report["source_availability"]

    base_commit = lifecycle.base_commit(task)
    source_identity: Dict[str, Any] = {
        "task_base_commit": base_commit,
        "configured_pin": availability.get("configured_commit"),
        "base_differs_from_pin": bool(
            base_commit
            and availability.get("configured_commit")
            and base_commit != availability.get("configured_commit")
        ),
        "task_base_available_locally": availability.get("task_base_available"),
        "configured_pin_available_locally": availability.get(
            "configured_commit_available"
        ),
    }
    if availability.get("task_base_available") is False:
        missing.append(
            "task base commit %s is NOT available in the local source; the "
            "bundle cannot prove reproduction from that base"
            % (base_commit or "?")
        )

    # --- workspace / runs ------------------------------------------------
    workspace_state: Dict[str, Any]
    try:
        workspace_state = workspace.status(root, config, task_id)
    except util.ToolError as exc:
        workspace_state = {"prepared": False, "detail": str(exc)}
        missing.append("no prepared workspace: %s" % exc)

    # --- evidence manifests ----------------------------------------------
    evidence: Dict[str, Any] = {"declared": None, "copied": False}
    declared = task.get("evidence_manifest")
    if declared:
        source = (task_dir / str(declared)).resolve()
        evidence["declared"] = str(declared)
        try:
            source.relative_to(root.resolve())
        except ValueError:
            missing.append("evidence_manifest escapes the repository: %s" % declared)
        else:
            if source.is_file():
                _copy_recorded(
                    source,
                    target / "evidence" / source.name,
                    inventory,
                    "evidence/%s" % source.name,
                )
                evidence["copied"] = True
                evidence["sha256"] = util.sha256_file(source)
                evidence["historical"] = True
                evidence["note"] = (
                    "Pre-existing manifest, copied unchanged. It documents an "
                    "earlier accepted result and is historical evidence; this "
                    "export neither re-validated nor re-reviewed it."
                )
            else:
                missing.append("declared evidence_manifest is absent: %s" % declared)

    # --- local diff ------------------------------------------------------
    local = _local_diff(root, target, paths.relative_to_root(root, task_dir))

    manifest = {
        "schema": "k3ctl/export/1",
        "generated_at": util.utcnow(),
        "bundle": paths.relative_to_root(root, target),
        "task_id": task_id,
        "config": str(config_path),
        "lifecycle": lifecycle.describe(task),
        "files": inventory,
        "patch": patch,
        "reports": reports,
        "evidence_manifest": evidence,
        "source_identity": source_identity,
        "dependencies": identity["toolchain"]["dependencies"],
        "toolchain": {
            "sha256": identity["sha256"],
            "status": report["status"],
            "missing": report["missing"],
            "incompatible": report["incompatible"],
        },
        "skills": identity["toolchain"]["skills"],
        "humanize": identity["toolchain"]["humanize"],
        "kda_prompt": identity["toolchain"]["kda_prompt"],
        "workspace": workspace_state,
        "local_changes": local,
        "gpu": {
            "performed": False,
            "statement": (
                "No GPU, SSH, model server or profiler run was performed by this "
                "export. Any timing or profiling numbers referenced in copied "
                "reports come from earlier recorded runs, not from this bundle."
            ),
        },
        "review": {
            "formal_review_performed": False,
            "statement": (
                "This export performs no review and produces no verdict. It "
                "packages existing artifacts and records their identity."
            ),
        },
        "missing_proof": missing,
        "complete": not missing,
        "publication_safety": {
            "contains_local_absolute_paths": True,
            "statement": (
                "This bundle is local output in an ignored directory. It may "
                "contain machine-specific absolute paths and must not be "
                "committed or published as-is. Tracked example files never "
                "contain private paths."
            ),
        },
        "git_writes": {
            "index_modified": False,
            "commits_created": 0,
            "statement": "No git add/commit/reset/stash/clean was performed.",
        },
    }
    util.write_json(target / "manifest.json", manifest)

    # A plain-text inventory for reviewers who read the directory directly.
    lines = [
        "# Export bundle: %s" % name,
        "",
        "task            : %s" % task_id,
        "generated at    : %s" % manifest["generated_at"],
        "task base commit: %s" % (base_commit or "unrecorded"),
        "base available  : %s" % source_identity["task_base_available_locally"],
        "toolchain sha256: %s" % identity["sha256"],
        "toolchain status: %s" % report["status"],
        "staged entries  : %d" % local["staged_entry_count"],
        "diff bytes      : %d" % local["diff_bytes"],
        "untracked files : %d" % local["untracked_file_count"],
        "GPU performed   : no",
        "formal review   : no",
        "",
        "## Contents",
    ]
    for key in sorted(inventory):
        lines.append("- %s" % key)
    if patch and patch.get("copied"):
        lines.append("- %s" % patch["path"])
    for name_ in sorted(reports.get("files") or {}):
        lines.append("- reports/%s" % name_)
    lines += ["- local-changes.diff", "- staged-index.txt", "- untracked-files.txt"]
    if missing:
        lines += ["", "## Missing proof"]
        lines += ["- %s" % item for item in missing]
    lines += [
        "",
        "## Notes",
        "- Local output in an ignored directory; do not commit or publish as-is.",
        "- Diffed against the staged index: this repository has no commit yet.",
        "- No git index write, no GPU run, no review verdict.",
    ]
    util.write_text(target / "INVENTORY.md", "\n".join(lines) + "\n")

    return manifest
