#!/usr/bin/env python3
"""Task scaffolding.

``task-create`` writes a complete, self-consistent task package derived from
the Kernel Design Agents basic-flow contract fields: objective, correctness
requirements, validation command, evaluation command, promotion criteria, and
candidate lineage/evidence locations.

The pinned KDA prompt is referenced and hashed rather than copied, so
attribution stays accurate and the upstream licence is respected. No accepted
package, acceptance field, patch, report or evidence manifest is ever copied
into a new task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as config_mod
from . import gitq, lifecycle, paths, util

#: Fields that describe a completed acceptance. They must never be scaffolded.
FORBIDDEN_SCAFFOLD_FIELDS = (
    "patch_file",
    "patch_sha256",
    "evidence_manifest",
    "accepted_at",
    "acceptance",
    "reports",
)


def validate_task_id(task_id: str) -> None:
    if not config_mod.TASK_ID_RE.fullmatch(task_id or ""):
        raise util.ToolError(
            "invalid task id %r: use lowercase letters, digits, dot, dash or "
            "underscore (2-64 characters)" % task_id
        )


def validate_commit(commit: str) -> None:
    if not config_mod.SHA_RE.fullmatch(commit or ""):
        raise util.ToolError(
            "--base-commit must be a complete 40-character Git SHA, got %r" % commit
        )


def _kda_reference(root: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Locate, hash and describe the pinned KDA prompt when it is available."""
    kda = settings.get("kda", {}) or {}
    relative = kda.get("prompt") or "external/kda/prompts/basic-flow.md"
    reference: Dict[str, Any] = {
        "prompt_path": str(relative),
        "prompt_sha256": None,
        "prompt_available": False,
        "kda_commit": None,
        "attribution": (
            "NVlabs Kernel Design Agents (Kernel Design Agents workflow, not an "
            "SDK). See THIRD_PARTY_NOTICES.md and external/kda/LICENSE."
        ),
    }
    try:
        target = paths.safe_join(root, str(relative))
    except paths.PathError:
        return reference
    if target.is_file():
        reference["prompt_available"] = True
        reference["prompt_sha256"] = util.sha256_file(target)
    kda_dir_raw = str(relative).split("/")[0:2]
    if len(kda_dir_raw) == 2:
        try:
            kda_dir = paths.safe_join(root, "/".join(kda_dir_raw))
        except paths.PathError:
            return reference
        if gitq.is_own_root(kda_dir):
            reference["kda_commit"] = gitq.head_commit(kda_dir)
    return reference


def build_task_record(
    task_id: str,
    project_id: str,
    kind: str,
    base_commit: str,
    repository: str,
    branch: Optional[str],
    objective: str,
    validation_command: str,
    performance_command: Optional[str],
    kda_reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical ``task.json`` for a fresh, unstartable scaffold."""
    return {
        "schema_version": 1,
        "task_id": task_id,
        "project_id": project_id,
        "kind": kind,
        "status": lifecycle.SCAFFOLDED,
        "created_at": util.utcnow(),
        "objective": objective,
        "source": {
            "repository": repository,
            "branch": branch or "<base branch>",
            "commit": base_commit,
            "availability": "verify locally before use; see source-trace.md",
        },
        "base_commit": base_commit,
        "allowed_source_files": [],
        "workload_file": "workloads.placeholder.json",
        "model_profile_file": "model-profile.json",
        "workload_status": lifecycle.PLACEHOLDER,
        "optimization_status": lifecycle.UNRESOLVED,
        "integration_status": "not_applied",
        "serving_validation": "not_performed",
        "contract_file": "contract.md",
        "source_trace_file": "source-trace.md",
        "plan_input_file": "plan-input.md",
        "prompt_file": "prompt.md",
        "commands": {
            "validation": validation_command,
            "performance": performance_command or "",
        },
        "promotion": {
            "rules": [
                "Validation command passes with zero failures and a non-zero "
                "case count.",
                "Performance evidence is recorded with the exact source commit "
                "and workload definition.",
                "A rejected candidate records the reason instead of being "
                "silently discarded.",
            ],
            "requires_human_review": True,
        },
        "lineage": {
            "candidates_file": "candidates.jsonl",
            "benchmark_log": "benchmark.csv",
            "draft": "docs/draft.md",
            "plan": "docs/plan.md",
            "evidence_dir": "local-evidence",
        },
        "workflow": {
            "source": "kda-basic-flow",
            "kda_prompt": kda_reference.get("prompt_path"),
            "kda_prompt_sha256": kda_reference.get("prompt_sha256"),
            "kda_commit": kda_reference.get("kda_commit"),
            "attribution": kda_reference.get("attribution"),
        },
        "next_actions": [
            "Resolve the workload definition and set workload_status to 'resolved'.",
            "Write plan-input.md into an executable plan.",
            "Run 'k3ctl workspace prepare' to materialise the exact base commit.",
            "Set status to 'ready' only when the above are complete.",
        ],
    }


CONTRACT_TEMPLATE = """# Task contract: {task_id}

Derived from the Kernel Design Agents basic flow. The upstream prompt is
referenced, not copied: `{kda_prompt}`
(sha256 `{kda_sha}`, KDA commit `{kda_commit}`).
{attribution}

## Objective

{objective}

## Inputs and outputs

Describe the exact entrypoint, tensor layouts or interfaces this task may
change. Keep the public signature and gate semantics unchanged unless the
objective says otherwise.

## Correctness requirements

State the required behaviour, tolerances and invariants. A correctness gate
must define a bounded case count; zero executed cases is never a pass.

## Constraints

- Allowed source files: list them explicitly in `task.json`.
- Implementation language, dependencies and APIs permitted.
- Deployment constraints that the change must not break.

## Validation command

```
{validation_command}
```

## Evaluation command

```
{performance_command}
```

## Promotion criteria

- The validation command passes with a non-zero, bounded case count.
- Performance evidence records the exact source commit and workload file.
- A rejected candidate records why, in `candidates.jsonl`.
- Human review is required before the task status becomes `accepted`.

## Evidence

Candidate lineage lives in `candidates.jsonl`; measurements in
`benchmark.csv`; raw artifacts stay outside Git with a manifest checked in.
"""

SOURCE_TRACE_TEMPLATE = """# Source trace: {task_id}

| Field | Value |
| --- | --- |
| Repository | `{repository}` |
| Base branch | `{branch}` |
| Base commit | `{base_commit}` |
| Availability | verify locally before use |

## Verifying the base

```bash
k3ctl doctor --task {task_id}
k3ctl workspace prepare --task {task_id}
```

`workspace prepare` refuses to substitute any other commit. If the exact base
commit is not reachable from the configured local source repository, it fails
with a source-availability error. Publish or fetch the exact ref; the public
submodule pin is a different source base and is not an acceptable stand-in.

## Notes

Record here how the base commit was obtained, whether it is published, and any
divergence from the public pin.
"""

PLAN_INPUT_TEMPLATE = """# Plan input: {task_id}

This file is the human-authored input to planning. It is not the executable
plan and it is not loop state.

## Goal

{objective}

## Acceptance criteria

- AC-1: The validation command `{validation_command}` passes with a bounded,
  non-zero case count.
- AC-2: Performance evidence is recorded against base commit
  `{base_commit}` with the resolved workload file.
- AC-3: The change stays within the allowed source files listed in `task.json`.

## Baseline and validation path

Describe the current behaviour and how it is validated today.

## Risks and unknowns

List the main risks, unknown layouts, and anything that must be measured
before a candidate can be promoted.

## Candidate directions

Rank candidate implementation directions by expected value and risk.

## First concrete steps

1. Inspect the baseline implementation and its tests.
2. Reproduce the validation command on the frozen base.
3. Implement one candidate at a time.
"""

PROMPT_TEMPLATE = """# Implementation prompt: {task_id}

You are working in an isolated task workspace prepared at base commit
`{base_commit}`. Produce the best correct implementation for the contract
below, one candidate at a time.

This prompt follows the Kernel Design Agents basic flow. The upstream prompt
is referenced rather than reproduced: `{kda_prompt}`
(sha256 `{kda_sha}`).
{attribution}

## Task contract

- Task name: `{task_id}`
- Objective: {objective}
- Correctness requirements: see `contract.md`
- Performance or quality target: see `contract.md`
- Allowed implementation approaches: see `contract.md` constraints
- Validation command: `{validation_command}`
- Evaluation command: `{performance_command}`
- Promotion criteria: see `contract.md`

## Workflow

1. Read the workspace structure, baseline implementation, tests and contract.
2. Identify the baseline behaviour and the validation path.
3. Research only the references needed for this task.
4. Write the implementation-plan draft to `docs/draft.md`.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time.
7. Run validation after each meaningful candidate.
8. Record candidate results, parent relationships and evidence.
9. Keep the final change scoped to the contract.

Do not start implementation until the draft exists.
"""

README_TEMPLATE = """# Task: {task_id}

Fresh scaffold created by `k3ctl task-create`. It is intentionally
**unstartable**: the workload is a placeholder and no plan has been written.

| File | Purpose |
| --- | --- |
| `task.json` | Canonical task state. Nothing else overrides it. |
| `contract.md` | Objective, correctness, commands, promotion rules. |
| `source-trace.md` | Exact base commit and how to verify it. |
| `plan-input.md` | Human input to planning. |
| `prompt.md` | Implementation prompt derived from the KDA basic flow. |
| `workloads.placeholder.json` | Replace with a resolved workload. |
| `model-profile.json` | Model facts this task depends on. |

## Making it startable

```bash
# 1. Resolve the workload, then set workload_status to "resolved" in task.json.
# 2. Write plan-input.md into an executable plan.
# 3. Materialise the exact base commit.
k3ctl workspace prepare --task {task_id}
# 4. Set "status" to "ready" in task.json.
k3ctl agent plan --task {task_id}
```

No acceptance, patch, report or evidence manifest is scaffolded. Those are
produced by real runs and reviewed separately.
"""


def scaffold(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    kind: str,
    base_commit: str,
    objective: str,
    validation_command: str,
    performance_command: Optional[str],
    tasks_dir: str = "tasks",
    repository: Optional[str] = None,
    branch: Optional[str] = None,
    register: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create a task package. Validates everything before writing anything."""
    validate_task_id(task_id)
    validate_commit(base_commit)

    existing = {
        item.get("id")
        for item in config.get("tasks", [])
        if isinstance(item, dict)
    }
    if task_id in existing:
        raise util.ToolError(
            "task id %r is already registered in %s" % (task_id, config_path)
        )

    relative = "%s/%s" % (tasks_dir.rstrip("/"), task_id)
    target = paths.safe_join(root, relative)
    if target.exists():
        raise util.ToolError(
            "refusing to overwrite an existing path: %s" % relative
        )

    source = config.get("source") or {}
    repository = repository or str(source.get("repository") or "")
    if not repository:
        raise util.ToolError(
            "source.repository is not configured and --repository was not given"
        )

    settings = config_mod.workflow_settings(config)
    kda_reference = _kda_reference(root, settings)

    record = build_task_record(
        task_id=task_id,
        project_id=str(config.get("project_id") or ""),
        kind=kind,
        base_commit=base_commit,
        repository=repository,
        branch=branch,
        objective=objective,
        validation_command=validation_command,
        performance_command=performance_command,
        kda_reference=kda_reference,
    )
    for field in FORBIDDEN_SCAFFOLD_FIELDS:
        if field in record:
            raise util.ToolError(
                "internal error: scaffold must not contain acceptance field %r" % field
            )

    attribution = kda_reference.get("attribution") or ""
    fields = {
        "task_id": task_id,
        "objective": objective,
        "validation_command": validation_command,
        "performance_command": performance_command or "(none)",
        "base_commit": base_commit,
        "repository": repository,
        "branch": branch or "<base branch>",
        "kda_prompt": kda_reference.get("prompt_path"),
        "kda_sha": kda_reference.get("prompt_sha256") or "unavailable (submodule not initialized)",
        "kda_commit": kda_reference.get("kda_commit") or "unavailable",
        "attribution": attribution,
    }

    files: Dict[str, str] = {
        "contract.md": CONTRACT_TEMPLATE.format(**fields),
        "source-trace.md": SOURCE_TRACE_TEMPLATE.format(**fields),
        "plan-input.md": PLAN_INPUT_TEMPLATE.format(**fields),
        "prompt.md": PROMPT_TEMPLATE.format(**fields),
        "README.md": README_TEMPLATE.format(**fields),
    }

    placeholder_workload = {
        "schema_version": 1,
        "task_id": task_id,
        "status": lifecycle.PLACEHOLDER,
        "note": (
            "Replace with a resolved workload definition, then set "
            "workload_status to 'resolved' in task.json."
        ),
        "cases": [],
    }
    model_profile = {
        "schema_version": 1,
        "task_id": task_id,
        "model": config.get("model", {}).get("id", "<model id>"),
        "note": "Record the model facts this task depends on.",
        "fields": {},
    }

    plan: Dict[str, Any] = {
        "task_id": task_id,
        "task_path": relative,
        "files": sorted(list(files) + [
            "task.json",
            "workloads.placeholder.json",
            "model-profile.json",
        ]),
        "kda_reference": kda_reference,
        "registered": False,
        "dry_run": bool(dry_run),
        "status": record["status"],
        "startable": not lifecycle.start_blockers(record),
    }
    if dry_run:
        return plan

    target.mkdir(parents=True)
    util.write_json(target / "task.json", record)
    util.write_json(target / "workloads.placeholder.json", placeholder_workload)
    util.write_json(target / "model-profile.json", model_profile)
    for name, text in files.items():
        util.write_text(target / name, text)

    if register:
        plan["registered"] = _register_task(config_path, config, task_id, kind, relative)
    return plan


def _register_task(
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    kind: str,
    relative: str,
) -> bool:
    """Append a task entry to a local configuration file.

    Tracked example configurations are never modified: registration is a local
    operation only.
    """
    if config_mod.is_example(config_path):
        raise util.ToolError(
            "refusing to modify the tracked example configuration %s; copy it to "
            "config/project.local.json and pass --config that file" % config_path
        )
    updated = dict(config)
    tasks: List[Any] = list(updated.get("tasks") or [])
    tasks.append({"id": task_id, "kind": kind, "path": relative})
    updated["tasks"] = tasks
    util.write_json(config_path, updated)
    return True
