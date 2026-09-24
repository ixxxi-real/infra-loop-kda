#!/usr/bin/env python3
"""Runner adapter protocol.

``run plan`` freezes a complete argv, environment and run identity from local
configuration plus the task. ``run start/status/cancel/fetch`` delegate to the
configured environment adapter. Nothing is passed through a shell, so a
workload value can never become a shell command.

Two adapters ship here:

``local-command``
    Runs the frozen argv on this machine under a supervisor that durably
    records the real exit status. Intended for offline work and mock testing,
    and as the reference a site wrapper can copy.

``site-wrapper``
    A thin delegation boundary. Every operation is forwarded to the configured
    wrapper argv; when no wrapper is configured, each operation fails with an
    explicit statement of what is missing. This CLI never implements SSH,
    container routing, GPU scheduling, or GPU resource locking -- those belong
    to the configured adapter.

Execution and evidence are kept strictly separate:

* **execution_state** -- did the command run, and how did it end (exit code,
  signal). ``exit_code == 0`` means "ran to completion", nothing more.
* **evidence_state** -- are the declared artifacts present, non-empty and
  hashed. Never inferred from an exit code.
* **gate results** -- evaluated only by the task's own harness against that
  evidence. This module never computes one.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as config_mod
from . import lifecycle, paths, toolchain, util, workspace

ADAPTER_LOCAL = "local-command"
ADAPTER_SITE = "site-wrapper"
ADAPTERS = (ADAPTER_LOCAL, ADAPTER_SITE)

RUN_STARTING = "starting"
RUN_RUNNING = "running"
RUN_EXITED = "exited"
RUN_SIGNALLED = "signalled"
RUN_CANCELLED = "cancelled"
RUN_UNKNOWN = "unknown"

#: Seconds to keep retrying the process-identity capture after spawning. A
#: single ``ps`` can miss a just-forked process, and an unrecorded identity
#: would leave the run unverifiable for its whole lifetime.
IDENTITY_CAPTURE_TIMEOUT = 5.0

#: Seconds to wait for the supervisor's first status write, so a run is
#: observably registered before ``start`` returns.
STATUS_REGISTER_TIMEOUT = 10.0

#: Contract gates are declared metadata, preserved from the task contract.
#: They are evaluated by the task harness against real evidence, never here.
PREFILL_GATES: Dict[str, Any] = {
    "correctness": {
        "oracle_cases": 11,
        "grid_cases": 51,
        "requirement": "11/11 oracle and 51/51 grid cases, no violations",
        "zero_cases_is_failure": True,
    },
    "precision": {
        "status": "advisory",
        "requirement": (
            "advisory diagnostic over frozen inputs and one device; not a proof "
            "over all layouts"
        ),
    },
    "timing": {
        "trials": 5,
        "samples": 30,
        "final_measurements": 2,
        "requirement": "two independent final 5x30 measurements",
    },
}

#: Ordered pipelines. A run executes one stage; the plan records the whole
#: pipeline and this stage's position in it, so a partial run is never mistaken
#: for a complete acceptance sequence.
PIPELINES: Dict[str, List[str]] = {
    "prefill": ["preflight", "correctness", "precision", "benchmark"],
}

#: Stages that the harness itself gates on.
#:
#: There is deliberately no cross-run prerequisite check here. ``benchmark.py``
#: already executes the real correctness gate itself unless it is handed a
#: verified ``--correctness-report`` from the same frozen snapshot. A scan for
#: "some earlier run left a correctness.json somewhere" would call a failed run,
#: or a run against a different source tree, a satisfied prerequisite -- which is
#: exactly the false assurance this module must not manufacture.
HARNESS_GATED_STAGES: Dict[str, str] = {
    "benchmark": (
        "benchmark.py runs the correctness gate itself unless given a verified "
        "--correctness-report from the same frozen snapshot"
    ),
    "precision": (
        "precision/diagnose.py is advisory and evaluates its own comparison"
    ),
}


#: A run id is a single path component: letters, digits, dot, dash, underscore.
#: Anything else -- a separator, ``..``, an absolute path -- is refused, so a
#: caller-supplied id can never place run state outside the task's runtime area.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    """Validate a run id as a single, non-traversing path component."""
    if not isinstance(run_id, str) or not run_id:
        raise util.ToolError("run id must be a non-empty string")
    if run_id in (".", ".."):
        raise util.ToolError("run id must not be %r" % run_id)
    if not RUN_ID_RE.fullmatch(run_id):
        raise util.ToolError(
            "invalid run id %r.\n"
            "A run id must be a single path component: start with a letter or "
            "digit, then letters, digits, dot, dash or underscore (max 128). "
            "Separators and parent references are refused so run state cannot "
            "escape the task's runtime directory." % run_id
        )
    return run_id


#: Runs live at this fixed location beneath the task directory.
RUNS_RELATIVE = "runtime/runs"


def runs_dir(task_dir: Path) -> Path:
    """Resolve the runs directory. Read-only: never creates anything.

    Anchored at *task_dir* and validated component by component, so a symlinked
    ``runtime`` or ``runs`` is rejected rather than followed.
    """
    return paths.safe_join(task_dir, RUNS_RELATIVE)


def run_dir(task_dir: Path, run_id: str) -> Path:
    """Resolve one run's directory. Read-only: never creates anything.

    The whole relative path is resolved from *task_dir* in a single
    ``safe_join``. Anchoring at ``runs_dir`` instead would be unsound:
    ``safe_join`` resolves its root first, so a symlinked ``runtime`` or
    ``runs`` would become the new root and the containment check would pass
    against the symlink's target -- permitting exactly the escape it is meant
    to prevent. Anchoring at *task_dir* makes ``runtime``, ``runs`` and the run
    id itself all checked components.

    Status and read helpers must never create directories: ``k3ctl run status``
    for an unknown run is a query, and a query that materialises
    ``runtime/runs/`` leaves state behind on a pure read.
    """
    return paths.safe_join(task_dir, "%s/%s" % (RUNS_RELATIVE, validate_run_id(run_id)))


def reserve_run_dir(task_dir: Path, run_id: str) -> Path:
    """Create a run directory, refusing to reuse an existing one.

    This is the *only* function here that writes to the filesystem. Two plans
    sharing a run id would overwrite each other's ``run.json`` and
    ``manifest.json``, so the second would silently inherit the first's
    recorded identity. ``mkdir`` without ``exist_ok`` makes that collision
    atomic and fatal, while ``parents=True`` still allows the shared
    ``runtime/runs`` ancestors to exist already.
    """
    target = run_dir(task_dir, run_id)
    try:
        target.mkdir(parents=True)
    except FileExistsError:
        raise util.ToolError(
            "run id %r already exists at %s.\n"
            "Run records are never overwritten: a second run under the same id "
            "would inherit the first run's recorded identity and artifacts. "
            "Choose a different --run-id, or inspect the existing run with "
            "'k3ctl run status --run-id %s'." % (run_id, target, run_id)
        )
    return target


def record_path(task_dir: Path, run_id: str) -> Path:
    return run_dir(task_dir, run_id) / "run.json"


def manifest_path(task_dir: Path, run_id: str) -> Path:
    return run_dir(task_dir, run_id) / "manifest.json"


def status_file(task_dir: Path, run_id: str) -> Path:
    return run_dir(task_dir, run_id) / "status.json"


def latest_pointer(task_dir: Path) -> Path:
    return runs_dir(task_dir) / "latest.json"


# --------------------------------------------------------------------- roots


def _canonical(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def verify_baseline_integrity(
    root: Path,
    config: Dict[str, Any],
    task_id: str,
    task: Dict[str, Any],
    baseline: Optional[Path],
    candidate: Optional[Path],
    observed_digest: Optional[str],
) -> Dict[str, Any]:
    """Bind the run to the workspace that was actually prepared for this task.

    The run plan claims the task's base commit as its provenance. A digest check
    alone does not establish that claim:

    * editing ``base_commit`` in ``task.json`` while leaving the prepared
      baseline untouched keeps the digest identical, so the run would report a
      base commit the baseline was never materialised from;
    * a workspace record belonging to a different task, or pointing at different
      paths than the roots being measured, would likewise pass a digest-only
      check.

    So identity is verified too: recorded task id, recorded base commit versus
    the task's current base commit, and the canonical baseline/candidate paths
    versus the roots actually being used. The candidate *tree* is expected to
    change -- that is the point -- so only the baseline tree is hashed.
    """
    problems: List[str] = []
    result: Dict[str, Any] = {
        "checked": False,
        "matches_prepared": None,
        "identity_bound": None,
        "recorded_digest": None,
        "observed_digest": observed_digest,
        "recorded_task_id": None,
        "recorded_base_commit": None,
        "task_base_commit": lifecycle.base_commit(task),
        "problems": problems,
        "problem": None,
    }
    try:
        record = workspace.load_record(root, config, task_id)
    except util.ToolError as exc:
        problems.append(str(exc))
        result["problem"] = str(exc)
        return result

    # --- identity -------------------------------------------------------
    recorded_task = record.get("task_id")
    result["recorded_task_id"] = recorded_task
    if recorded_task != task_id:
        problems.append(
            "the workspace record belongs to task %r, not %r; it cannot provide "
            "provenance for this run" % (recorded_task, task_id)
        )

    recorded_base = (record.get("source") or {}).get("base_commit")
    current_base = lifecycle.base_commit(task)
    result["recorded_base_commit"] = recorded_base
    if recorded_base != current_base:
        problems.append(
            "task.json now records base commit %s but the workspace was prepared "
            "from %s.\n"
            "The prepared trees were materialised from the old commit, so this "
            "run cannot claim the new one. Prepare a fresh workspace for the new "
            "base."
            % (
                (current_base or "<unset>")[:12],
                (recorded_base or "<unset>")[:12],
            )
        )

    recorded_baseline = _canonical(
        Path(str((record.get("baseline") or {}).get("absolute_path") or ""))
        if (record.get("baseline") or {}).get("absolute_path")
        else None
    )
    recorded_candidate = _canonical(
        Path(str((record.get("workspace") or {}).get("absolute_path") or ""))
        if (record.get("workspace") or {}).get("absolute_path")
        else None
    )
    if recorded_baseline and _canonical(baseline) != recorded_baseline:
        problems.append(
            "the baseline root being measured (%s) is not the prepared baseline "
            "(%s)" % (_canonical(baseline), recorded_baseline)
        )
    if recorded_candidate and _canonical(candidate) != recorded_candidate:
        problems.append(
            "the candidate root being measured (%s) is not the prepared "
            "workspace (%s)" % (_canonical(candidate), recorded_candidate)
        )
    result["identity_bound"] = not problems

    # --- baseline tree ---------------------------------------------------
    recorded_digest = (record.get("baseline") or {}).get("digest")
    result["recorded_digest"] = recorded_digest
    if not recorded_digest:
        problems.append(
            "the workspace record has no baseline digest, so baseline integrity "
            "cannot be verified; re-run 'k3ctl workspace prepare'"
        )
    elif not observed_digest:
        problems.append(
            "the baseline tree could not be hashed, so its integrity cannot be "
            "verified"
        )
    else:
        result["checked"] = True
        result["matches_prepared"] = observed_digest == recorded_digest
        if not result["matches_prepared"]:
            problems.append(
                "the baseline tree has changed since it was prepared (recorded "
                "%s, now %s).\n"
                "The baseline must stay immutable: a measurement against a "
                "modified baseline cannot claim the task's base commit as its "
                "provenance. Restore it, or prepare a fresh workspace."
                % (recorded_digest[:12], observed_digest[:12])
            )

    result["problem"] = problems[0] if problems else None
    return result


def resolve_roots(
    root: Path, config: Dict[str, Any], task_id: str
) -> Tuple[Path, Path]:
    """Return ``(baseline_root, candidate_root)`` as two distinct trees.

    The A/B harnesses take ``--baseline-root`` and ``--candidate-root``.
    Passing the same path twice would compare a tree with itself and report a
    meaningless zero delta, so identical roots are refused outright.
    """
    record = workspace.load_record(root, config, task_id)
    candidate = Path(
        str((record.get("workspace") or {}).get("absolute_path") or "")
    )
    baseline = Path(str((record.get("baseline") or {}).get("absolute_path") or ""))

    if not candidate.is_dir():
        raise util.ToolError(
            "candidate workspace is missing: %s\nRun 'k3ctl workspace prepare'."
            % candidate
        )
    if not baseline or not baseline.is_dir():
        raise util.ToolError(
            "baseline tree is missing: %s\n"
            "A/B measurement needs an immutable baseline distinct from the "
            "candidate. Re-run 'k3ctl workspace prepare' to materialise it."
            % (baseline or "<unrecorded>")
        )
    if baseline.resolve() == candidate.resolve():
        raise util.ToolError(
            "baseline and candidate roots are the same path (%s).\n"
            "Comparing a tree against itself always reports zero difference and "
            "is never valid evidence. Refusing to build this command." % candidate
        )
    return baseline, candidate


# ------------------------------------------------------------ command build


def _python_executable(settings: Dict[str, Any]) -> str:
    commands = settings.get("commands")
    if isinstance(commands, dict) and commands.get("python"):
        return str(commands["python"])
    return "python3"


def builtin_commands(
    task: Dict[str, Any],
    task_dir: Path,
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    scratch_dir: Path,
    python: str,
) -> Dict[str, List[str]]:
    """Real argv for the prefill harness that already exists in the task.

    The measurement scripts are used exactly as they are; no kernel or
    measurement behaviour is changed. Arguments match each script's own
    ``argparse`` definition.
    """
    if str(task.get("kind") or "") != "prefill":
        return {}

    workload = task.get("workload_file") or "workloads.resolved.json"
    workloads = task_dir / str(workload)
    bench = task_dir / "bench"
    precision = task_dir / "precision"

    commands: Dict[str, List[str]] = {}

    preflight = bench / "preflight.py"
    if preflight.is_file():
        commands["preflight"] = [
            python,
            str(preflight),
            "--source-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--report",
            str(output_dir / "preflight.json"),
        ]
        commands["preflight-static"] = [
            python,
            str(preflight),
            "--source-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--report",
            str(output_dir / "preflight-static.json"),
            "--static-check",
        ]

    correctness = bench / "correctness.py"
    if correctness.is_file():
        commands["correctness"] = [
            python,
            str(correctness),
            "--source-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--report",
            str(output_dir / "correctness.json"),
        ]

    benchmark = bench / "benchmark.py"
    if benchmark.is_file():
        commands["benchmark"] = [
            python,
            str(benchmark),
            "--baseline-root",
            str(baseline),
            "--candidate-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--out",
            str(output_dir / "benchmark"),
            "--trials",
            str(PREFILL_GATES["timing"]["trials"]),
            "--warmup",
            "5",
            "--samples",
            str(PREFILL_GATES["timing"]["samples"]),
        ]
        commands["correctness-only"] = [
            python,
            str(benchmark),
            "--baseline-root",
            str(baseline),
            "--candidate-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--out",
            str(output_dir / "correctness-only"),
            "--correctness-only",
        ]

    diagnose = precision / "diagnose.py"
    if diagnose.is_file():
        commands["precision"] = [
            python,
            str(diagnose),
            "--baseline-root",
            str(baseline),
            "--candidate-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--out",
            str(output_dir / "precision"),
            "--scratch",
            str(scratch_dir / "precision"),
        ]

    profile = bench / "profile_one.py"
    if profile.is_file():
        commands["profile"] = [
            python,
            str(profile),
            "--source-root",
            str(candidate),
            "--workloads",
            str(workloads),
            "--id",
            "{workload_id}",
            "--iters",
            "5",
            "--warmup",
            "5",
        ]
    return commands


def expected_artifacts(operation: str, output_dir: Path) -> List[str]:
    """Artifacts a completed operation must produce."""
    mapping = {
        "preflight": [output_dir / "preflight.json"],
        "preflight-static": [output_dir / "preflight-static.json"],
        "correctness": [output_dir / "correctness.json"],
        "correctness-only": [output_dir / "correctness-only"],
        "benchmark": [output_dir / "benchmark"],
        "precision": [output_dir / "precision"],
    }
    return [str(item) for item in mapping.get(operation, [])]


def resolve_command(
    settings: Dict[str, Any],
    task: Dict[str, Any],
    task_dir: Path,
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    scratch_dir: Path,
    operation: str,
) -> Tuple[List[str], str]:
    """Resolve the argv for an operation.

    Configured commands win over builtins so a site can override anything
    without editing this module. A configured command must be a list of
    strings: a single string would require shell parsing.
    """
    python = _python_executable(settings)
    configured = settings.get("commands")
    if isinstance(configured, dict) and operation in configured:
        value = configured[operation]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise util.ToolError(
                "runner.commands.%s must be a list of strings (argv), not a "
                "single string: a string would require shell parsing" % operation
            )
        if not value:
            raise util.ToolError("runner.commands.%s is empty" % operation)
        substitutions = {
            "{baseline_root}": str(baseline),
            "{candidate_root}": str(candidate),
            # Kept for convenience; resolves to the candidate tree.
            "{source_root}": str(candidate),
            "{output_dir}": str(output_dir),
            "{scratch_dir}": str(scratch_dir),
            "{task_dir}": str(task_dir),
            "{python}": python,
        }
        argv = []
        for item in value:
            for token, replacement in substitutions.items():
                item = item.replace(token, replacement)
            argv.append(item)
        return argv, "configured"

    builtins = builtin_commands(
        task, task_dir, baseline, candidate, output_dir, scratch_dir, python
    )
    if operation in builtins:
        return builtins[operation], "builtin"
    raise util.ToolError(
        "no command for operation %r.\n"
        "Available builtins: %s\n"
        "Define runner.commands.%s in your local configuration as an argv list."
        % (operation, ", ".join(sorted(builtins)) or "none", operation)
    )


# ----------------------------------------------------------------- pipelines


def pipeline_for(task: Dict[str, Any]) -> List[str]:
    return list(PIPELINES.get(str(task.get("kind") or ""), []))


def build_stages(
    task: Dict[str, Any], operation: str, task_dir: Path
) -> Dict[str, Any]:
    """Describe the whole pipeline and this run's position in it."""
    pipeline = pipeline_for(task)
    stages: List[Dict[str, Any]] = []
    for name in pipeline:
        stages.append(
            {
                "stage": name,
                "is_this_run": name == operation,
                "harness_gate": HARNESS_GATED_STAGES.get(name),
                "gate": _stage_gate(name),
            }
        )
    position = pipeline.index(operation) + 1 if operation in pipeline else None
    return {
        "pipeline": pipeline,
        "stages": stages,
        "this_stage": operation,
        "position": position,
        "total": len(pipeline) or None,
        "in_pipeline": operation in pipeline,
        "note": (
            "This run executes one stage. Completing one stage is not an "
            "acceptance sequence; every stage's evidence is required, and the "
            "gates are evaluated by the task harness."
        ),
    }


def _stage_gate(stage: str) -> Optional[str]:
    if stage == "correctness":
        return "correctness"
    if stage == "benchmark":
        return "timing"
    if stage == "precision":
        return "precision"
    return None


def execution_inputs(
    task_dir: Path,
    task: Dict[str, Any],
    baseline: Optional[Path],
    candidate: Optional[Path],
) -> Dict[str, Any]:
    """Hash everything a measurement actually depends on.

    The run identity must change when the measured inputs change. The candidate
    tree is the one the agent edits, so a measurement taken before an edit and
    one taken after must never share an identity -- otherwise two different
    results would be filed under the same identity and a stale artifact could be
    mistaken for a current one.

    Hashed here: the full candidate tree, the full baseline tree, the resolved
    workload, and the harness sources that perform the measurement.
    """
    warnings: List[str] = []
    payload: Dict[str, Any] = {}

    def digest_tree(label: str, target: Optional[Path]) -> Optional[Dict[str, Any]]:
        if target is None or not target.is_dir():
            warnings.append("%s tree is unavailable; identity is incomplete" % label)
            return None
        hashes = util.tree_hashes(
            target, skip=(".git", workspace.CONTROL_DIR, "__pycache__")
        )
        return {"file_count": len(hashes), "digest": util.tree_digest(hashes)}

    payload["candidate_tree"] = digest_tree("candidate", candidate)
    payload["baseline_tree"] = digest_tree("baseline", baseline)

    workload_file = task.get("workload_file")
    if workload_file and (task_dir / str(workload_file)).is_file():
        payload["workload"] = {
            "path": str(workload_file),
            "sha256": util.sha256_file(task_dir / str(workload_file)),
        }
    else:
        payload["workload"] = None
        warnings.append("resolved workload file is unavailable")

    # The harness sources are part of the measurement: changing correctness.py
    # changes what "the correctness gate passed" means.
    harness: Dict[str, str] = {}
    for name in ("bench", "precision", "scripts"):
        directory = task_dir / name
        if not directory.is_dir():
            continue
        hashes = util.tree_hashes(directory, skip=("__pycache__",))
        if hashes:
            harness[name] = util.tree_digest(hashes)
    payload["harness"] = harness or None
    if not harness:
        warnings.append("no harness sources found in the task package")

    if payload["candidate_tree"] and payload["baseline_tree"]:
        payload["candidate_differs_from_baseline"] = (
            payload["candidate_tree"]["digest"] != payload["baseline_tree"]["digest"]
        )
        if not payload["candidate_differs_from_baseline"]:
            warnings.append(
                "candidate and baseline trees are byte-identical: an A/B "
                "measurement would report no difference because there is no "
                "candidate change yet"
            )
    else:
        payload["candidate_differs_from_baseline"] = None

    return {
        "candidate_tree": payload["candidate_tree"],
        "baseline_tree": payload["baseline_tree"],
        "workload": payload["workload"],
        "harness": payload["harness"],
        "candidate_differs_from_baseline": payload["candidate_differs_from_baseline"],
        "inputs_sha256": util.sha256_json(payload),
        "warnings": warnings,
        "note": (
            "The run identity is bound to these input digests, so a measurement "
            "taken before a candidate edit and one taken after are distinct."
        ),
    }


def earlier_stage_history(task_dir: Path, operation: str) -> Dict[str, Any]:
    """List earlier runs of related stages, for orientation only.

    This is **not** a prerequisite check and must never be read as one. Finding
    a ``correctness.json`` in some earlier run directory says nothing about
    whether that run passed, whether it targeted this candidate tree, or whether
    its snapshot matches the current one.

    Stage gating is the harness's job: ``benchmark.py`` re-runs the correctness
    gate itself unless it is handed a verified ``--correctness-report`` from the
    same frozen snapshot. That verified-reuse path is the only sound way to
    skip it, and it is opted into explicitly through ``runner.commands``.
    """
    base = runs_dir(task_dir)
    observed: Dict[str, List[str]] = {}
    if base.is_dir():
        for stage in ("preflight", "correctness", "precision", "benchmark"):
            found: List[str] = []
            for candidate_run in sorted(base.iterdir()):
                if not candidate_run.is_dir():
                    continue
                for expected in expected_artifacts(
                    stage, candidate_run / "artifacts"
                ):
                    target = Path(expected)
                    if target.is_file() or (
                        target.is_dir() and any(target.iterdir())
                    ):
                        found.append(str(target))
            if found:
                observed[stage] = found
    return {
        "observed_artifacts_by_stage": observed,
        "is_prerequisite_check": False,
        "harness_gate": HARNESS_GATED_STAGES.get(operation),
        "note": (
            "Historical orientation only. The presence of an earlier artifact "
            "does not mean that run passed, targeted this candidate tree, or "
            "shares this snapshot. Correctness gating is performed by the task "
            "harness, not by this scan."
        ),
    }


# ------------------------------------------------------------------- planning


def plan(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    operation: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze a complete run plan. Never starts anything."""
    task_dir = config_mod.task_dir(root, config, task_id)
    task = config_mod.load_json(task_dir / "task.json")
    settings = config_mod.runner_settings(config)
    adapter = str(settings.get("adapter") or ADAPTER_LOCAL)
    if adapter not in ADAPTERS:
        raise util.ToolError(
            "unknown runner.adapter %r (expected one of: %s)"
            % (adapter, ", ".join(ADAPTERS))
        )

    blockers: List[str] = []
    baseline: Optional[Path] = None
    candidate: Optional[Path] = None
    try:
        baseline, candidate = resolve_roots(root, config, task_id)
    except util.ToolError as exc:
        blockers.append(str(exc))

    # A caller-supplied run id is validated as a single path component, and its
    # directory is reserved atomically so two plans can never share one id and
    # overwrite each other's recorded identity.
    if run_id is None:
        run_id = "%s-%s" % (util.stamp(), util.new_uuid()[:8])
    else:
        validate_run_id(run_id)
    target = reserve_run_dir(task_dir, run_id)
    output_dir = target / "artifacts"
    scratch_dir = target / "scratch"

    argv: List[str] = []
    origin = None
    if baseline is not None and candidate is not None:
        argv, origin = resolve_command(
            settings,
            task,
            task_dir,
            baseline,
            candidate,
            output_dir,
            scratch_dir,
            operation,
        )
        unresolved = [item for item in argv if "{" in item and "}" in item]
        if unresolved:
            blockers.append(
                "command still contains unresolved placeholders: %s"
                % ", ".join(unresolved)
            )

    environment = settings.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    env = {str(key): str(value) for key, value in environment.items()}

    report = toolchain.inspect(root, config_path, config, task_id=task_id)
    toolchain_identity = toolchain.identity(report)

    workload_file = task.get("workload_file")
    workload_hash = None
    if workload_file and (task_dir / str(workload_file)).is_file():
        workload_hash = util.sha256_file(task_dir / str(workload_file))

    # Bind the identity to the actual measured inputs, not just to the argv.
    # The candidate tree is what the agent edits, so a measurement taken before
    # an edit and one taken after must not share an identity.
    inputs = execution_inputs(task_dir, task, baseline, candidate)

    # The run claims the task's base commit as provenance. Verify that claim
    # against the workspace that was actually prepared: recorded task id,
    # recorded base commit, the prepared paths, and the baseline tree digest.
    baseline_digest = (inputs.get("baseline_tree") or {}).get("digest")
    baseline_integrity = verify_baseline_integrity(
        root, config, task_id, task, baseline, candidate, baseline_digest
    )
    blockers.extend(baseline_integrity["problems"])

    identity_payload = {
        "argv": argv,
        "cwd": str(candidate) if candidate else None,
        "env": env,
        "operation": operation,
        "task_id": task_id,
        "base_commit": lifecycle.base_commit(task),
        "workload_sha256": workload_hash,
        "toolchain_sha256": toolchain_identity["sha256"],
        "inputs_sha256": inputs["inputs_sha256"],
    }

    site_boundary = None
    if adapter == ADAPTER_SITE and not settings.get("wrapper"):
        site_boundary = (
            "runner.adapter is 'site-wrapper' but runner.wrapper is not "
            "configured. This CLI does not implement SSH, container routing or "
            "GPU scheduling, and does not perform GPU resource locking. "
            "Provide a site adapter that owns UUID/resource locking, or use "
            "'local-command' for offline work."
        )
        blockers.append(site_boundary)

    stages = build_stages(task, operation, task_dir)
    history = earlier_stage_history(task_dir, operation)

    record: Dict[str, Any] = {
        "schema": "k3ctl/run-plan/2",
        "generated_at": util.utcnow(),
        "run_id": run_id,
        "task_id": task_id,
        "config": str(config_path),
        "operation": operation,
        "adapter": adapter,
        "command_origin": origin,
        "roots": {
            "baseline": str(baseline) if baseline else None,
            "candidate": str(candidate) if candidate else None,
            "distinct": bool(
                baseline and candidate and baseline.resolve() != candidate.resolve()
            ),
        },
        "invocation": {
            "argv": argv,
            "cwd": str(candidate) if candidate else None,
            "env": env,
            "shell": False,
        },
        "identity": {
            "run_sha256": util.sha256_json(identity_payload),
            "toolchain_sha256": toolchain_identity["sha256"],
            "workload_sha256": workload_hash,
            "base_commit": lifecycle.base_commit(task),
            "inputs_sha256": inputs["inputs_sha256"],
        },
        "inputs": inputs,
        "baseline_integrity": baseline_integrity,
        "expected_artifacts": expected_artifacts(operation, output_dir),
        # Relative names too, so 'run fetch' can check the same expectations
        # against a destination directory that is not output_dir.
        "expected_artifact_names": [
            Path(item).name for item in expected_artifacts(operation, output_dir)
        ],
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch_dir),
        "pipeline": stages,
        "stage_history": history,
        "contract_gates": PREFILL_GATES if task.get("kind") == "prefill" else {},
        "gate_evaluation": (
            "Gates are evaluated by the task harness against real evidence. "
            "This CLI records commands and artifacts; it never infers a gate "
            "result, and a zero exit code is not a pass."
        ),
        "gpu": {
            "performed_by_this_cli": False,
            "resource_locking": (
                "required of the configured GPU adapter; the generic CLI does "
                "not claim or hold any GPU/UUID lock"
            ),
        },
        "site_boundary": site_boundary,
        "ready": not blockers,
        "blockers": blockers,
        "warnings": inputs.get("warnings") or [],
        "would_spawn": False,
    }
    util.write_json(record_path(task_dir, run_id), record)
    return record


# ------------------------------------------------------------------ adapters


class LocalCommandAdapter:
    """Runs the frozen argv on this machine under a status supervisor."""

    name = ADAPTER_LOCAL

    def start(self, record: Dict[str, Any], task_dir: Path) -> Dict[str, Any]:
        argv = list(record["invocation"]["argv"])
        cwd = Path(str(record["invocation"]["cwd"]))
        target = run_dir(task_dir, record["run_id"])
        Path(record["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(record["scratch_dir"]).mkdir(parents=True, exist_ok=True)
        stdout_path = target / "stdout.log"
        stderr_path = target / "stderr.log"
        status_path = status_file(task_dir, record["run_id"])

        env = dict(os.environ)
        env.update(record["invocation"]["env"])

        # The supervisor is the direct parent, so the real exit status survives
        # this CLI exiting. Invoked by absolute path: the run's cwd is the
        # candidate tree, not this repository.
        supervisor = Path(__file__).resolve().parent / "supervise.py"
        wrapped = [sys.executable, str(supervisor), str(status_path)] + argv

        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            process = subprocess.Popen(  # noqa: S603 - argv list, shell=False
                wrapped,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )

        # Capture the process identity, retrying briefly. A single `ps` call can
        # miss a process that has only just been forked, and an absent identity
        # would make the run unverifiable for its whole lifetime: `status` could
        # never confirm liveness and would report `unknown` for a healthy run.
        identity = None
        deadline = time.monotonic() + IDENTITY_CAPTURE_TIMEOUT
        while time.monotonic() < deadline:
            identity = util.process_identity(process.pid)
            if identity and identity.get("lstart") and identity.get("command"):
                break
            if process.poll() is not None:
                # Already finished; the status file is then the authority.
                break
            time.sleep(0.05)

        # Wait for the supervisor's first status write, so a run is observably
        # registered before `start` returns. Without this, an immediate `status`
        # call can find neither a status file nor a verifiable pid and report
        # `unknown` purely because it asked too early.
        registered = False
        deadline = time.monotonic() + STATUS_REGISTER_TIMEOUT
        while time.monotonic() < deadline:
            if status_path.is_file():
                registered = True
                break
            time.sleep(0.02)

        return {
            "pid": process.pid,
            "process_identity": identity,
            "identity_captured": bool(
                identity and identity.get("lstart") and identity.get("command")
            ),
            "supervisor": str(supervisor),
            "status_file": str(status_path),
            "status_registered": registered,
            "logs": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
        }

    def status(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Report how the command ended, from the durable status file.

        Liveness and identity are deliberately separate questions here.

        For *status* -- a read-only query -- the question is "is this pid still
        running", answered by ``pid_alive``. Identity is corroboration: it
        upgrades the answer's confidence, and a mismatch means the pid was
        recycled so the state is genuinely unknown. But a *missing* identity must
        not be read as "not running": a fast command can exit before the
        post-spawn ``ps`` capture succeeds, and treating unverifiable as dead
        made healthy runs permanently ``unknown``.

        For *signalling* (``cancel``) the asymmetry is reversed: a strict
        identity match is required, because signalling the wrong pid is
        unrecoverable. That strictness lives in ``cancel``, not here.
        """
        status_path = Path(str(manifest.get("status_file") or ""))
        recorded: Dict[str, Any] = {}
        if status_path.is_file():
            try:
                recorded = config_mod.load_json(status_path)
            except config_mod.ValidationError:
                recorded = {}

        pid = manifest.get("pid")
        alive = False
        identity_verified = False
        detail = "no pid recorded"
        if isinstance(pid, int):
            alive = util.pid_alive(pid)
            recorded_identity = manifest.get("process_identity")
            if recorded_identity:
                identity_verified, identity_detail = util.process_matches(
                    pid, recorded_identity
                )
                if alive and not identity_verified:
                    # Alive, but not the process we started: pid recycled.
                    alive = False
                    detail = identity_detail
                elif alive:
                    detail = identity_detail
                else:
                    detail = "pid %s is not running" % pid
            elif alive:
                detail = (
                    "pid %s is running, but no process identity was captured at "
                    "spawn, so this is unverified" % pid
                )
            else:
                detail = "pid %s is not running" % pid

        if recorded:
            state = str(recorded.get("state") or RUN_UNKNOWN)
            # States the supervisor writes while the child is still alive.
            in_flight = {"starting", "running", "cancelling"}
            if state in in_flight and not alive:
                # The status file was read *before* the pid check, so the
                # supervisor may have written its terminal status in between.
                # Re-read once before declaring the outcome unknown: reporting
                # a completed run as unknown would be a false negative.
                if status_path.is_file():
                    try:
                        recheck = config_mod.load_json(status_path)
                    except config_mod.ValidationError:
                        recheck = {}
                    if recheck and str(recheck.get("state") or "") not in in_flight:
                        recorded = recheck
                        state = str(recorded.get("state") or RUN_UNKNOWN)

            if state in in_flight and not alive:
                # The supervisor really is gone with no terminal status written.
                return {
                    "state": RUN_UNKNOWN,
                    "detail": (
                        "the supervisor is gone but recorded no terminal status "
                        "(last state %r); the run's outcome is unknown, not "
                        "successful" % state
                    ),
                    "exit_code": None,
                    "signal": None,
                    "exited_normally": False,
                }
            mapping = {
                "starting": RUN_STARTING,
                "running": RUN_RUNNING,
                "cancelling": RUN_RUNNING,
                "cancelled": RUN_CANCELLED,
                "signalled": RUN_SIGNALLED,
                "exited": RUN_EXITED,
                "failed-to-spawn": RUN_EXITED,
            }
            mapped = mapping.get(state, RUN_UNKNOWN)
            if state == "cancelling":
                detail = "termination forwarded to the child; waiting for it to exit"
            return {
                "state": mapped,
                "detail": detail,
                "exit_code": recorded.get("exit_code"),
                "signal": recorded.get("signal"),
                "exited_normally": recorded.get("exited_normally"),
                "cancelled": bool(recorded.get("cancelled")),
                "received_signal": recorded.get("received_signal"),
                "escalated_to_sigkill": bool(recorded.get("escalated_to_sigkill")),
                "child_pid": recorded.get("child_pid"),
                "started_at": recorded.get("started_at"),
                "finished_at": recorded.get("finished_at"),
            }

        if alive:
            return {
                "state": RUN_RUNNING,
                "detail": detail,
                "exit_code": None,
                "signal": None,
                "exited_normally": None,
            }
        return {
            "state": RUN_UNKNOWN,
            "detail": (
                "no status file and no live process: the outcome was never "
                "recorded and cannot be assumed"
            ),
            "exit_code": None,
            "signal": None,
            "exited_normally": False,
        }

    def cancel(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        pid = manifest.get("pid")
        if not isinstance(pid, int):
            raise util.ToolError("no pid recorded; refusing to signal anything")
        matches, detail = util.process_matches(pid, manifest.get("process_identity"))
        if not matches:
            raise util.ToolError(
                "refusing to signal pid %s: %s\n"
                "Only the recorded, identity-confirmed process is ever signalled."
                % (pid, detail)
            )
        os.kill(pid, signal.SIGTERM)
        return {"signalled": pid, "signal": "SIGTERM"}

    def fetch(self, manifest: Dict[str, Any], destination: Path) -> Dict[str, Any]:
        """Artifacts are already local; record what exists."""
        output = Path(str(manifest.get("output_dir") or ""))
        files: Dict[str, str] = {}
        if output.is_dir():
            files = util.tree_hashes(output)
        return {
            "location": str(output),
            "files": files,
            "transferred": False,
            "detail": "local adapter: artifacts are already on this machine",
        }


class SiteWrapperAdapter:
    """Delegates every operation to a configured site wrapper.

    The wrapper owns SSH, containers, GPU scheduling and GPU/UUID resource
    locking. This class only forwards and records; it never simulates a result.
    """

    name = ADAPTER_SITE

    def __init__(self, wrapper: Optional[Any] = None):
        if wrapper is not None and (
            not isinstance(wrapper, list)
            or not all(isinstance(item, str) for item in wrapper)
        ):
            raise util.ToolError(
                "runner.wrapper must be a list of strings (argv), so no value is "
                "ever passed through a shell"
            )
        self.wrapper = list(wrapper) if wrapper else None

    def _require(self, operation: str) -> List[str]:
        if not self.wrapper:
            raise util.ToolError(
                "runner operation %r requires a configured site adapter.\n"
                "This CLI deliberately does not implement SSH, container "
                "routing, GPU scheduling, or GPU resource locking. Configure "
                "runner.wrapper as an argv list that owns those concerns, or "
                "use the 'local-command' adapter for offline and mock work."
                % operation
            )
        return list(self.wrapper)

    def start(self, record: Dict[str, Any], task_dir: Path) -> Dict[str, Any]:
        argv = self._require("start") + [
            "start",
            "--run-id",
            str(record["run_id"]),
            "--cwd",
            str(record["invocation"]["cwd"]),
            "--",
        ] + list(record["invocation"]["argv"])
        result = _delegate(argv, record["invocation"]["env"], task_dir, record["run_id"], "start")
        # A site runner owns the remote job id; keep whatever it reported so a
        # disconnect can resume the same job instead of spawning a duplicate.
        return {
            "delegated": True,
            "site_run_id": result.get("stdout_first_line") or str(record["run_id"]),
            "logs": {"delegate": result["log"]},
            "detail": "start delegated to the configured site adapter",
        }

    def status(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        argv = self._require("status") + [
            "status",
            "--run-id",
            str(manifest.get("site_run_id") or manifest.get("run_id")),
        ]
        result = _delegate(
            argv, manifest.get("env") or {}, None, manifest.get("run_id"), "status",
            log_path=Path(str(manifest.get("output_dir") or ".")).parent / "status-delegate.log",
        )
        reported = (result.get("stdout_first_line") or "").strip().lower()
        allowed = {
            RUN_RUNNING,
            RUN_EXITED,
            RUN_SIGNALLED,
            RUN_CANCELLED,
            RUN_UNKNOWN,
        }
        if reported not in allowed:
            return {
                "state": RUN_UNKNOWN,
                "detail": (
                    "site adapter reported %r, which is not one of %s; treating "
                    "the run state as unknown rather than guessing"
                    % (reported or "<empty>", ", ".join(sorted(allowed)))
                ),
                "exit_code": None,
                "signal": None,
                "exited_normally": None,
            }
        return {
            "state": reported,
            "detail": "reported by the configured site adapter",
            "exit_code": None,
            "signal": None,
            "exited_normally": None,
        }

    def cancel(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        argv = self._require("cancel") + [
            "cancel",
            "--run-id",
            str(manifest.get("site_run_id") or manifest.get("run_id")),
        ]
        result = _delegate(
            argv, manifest.get("env") or {}, None, manifest.get("run_id"), "cancel",
            log_path=Path(str(manifest.get("output_dir") or ".")).parent / "cancel-delegate.log",
        )
        return {"delegated": True, "log": result["log"]}

    def fetch(self, manifest: Dict[str, Any], destination: Path) -> Dict[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        argv = self._require("fetch") + [
            "fetch",
            "--run-id",
            str(manifest.get("site_run_id") or manifest.get("run_id")),
            "--destination",
            str(destination),
        ]
        result = _delegate(
            argv, manifest.get("env") or {}, None, manifest.get("run_id"), "fetch",
            log_path=destination.parent / "fetch-delegate.log",
        )
        files = util.tree_hashes(destination) if destination.is_dir() else {}
        return {
            "location": str(destination),
            "files": files,
            "transferred": bool(files),
            "log": result["log"],
            "detail": "fetch delegated to the configured site adapter",
        }


def _delegate(
    argv: List[str],
    env_overrides: Dict[str, Any],
    task_dir: Optional[Path],
    run_id: Optional[str],
    operation: str,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a site wrapper argv, record its output, and fail loudly."""
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in (env_overrides or {}).items()})
    rc, out, err = util.run(argv, env=env, timeout=3600)
    target = log_path
    if target is None and task_dir is not None and run_id:
        target = run_dir(task_dir, str(run_id)) / ("%s-delegate.log" % operation)
    if target is not None:
        util.write_text(target, "# argv: %s\n%s%s" % (" ".join(argv), out, err))
    if rc != 0:
        raise util.ToolError(
            "site adapter %s failed (rc=%d): %s"
            % (operation, rc, err.strip() or out.strip() or "no output")
        )
    first = ""
    for line in out.splitlines():
        if line.strip():
            first = line.strip()
            break
    return {
        "log": str(target) if target is not None else None,
        "stdout_first_line": first,
    }


def get_adapter(settings: Dict[str, Any]):
    adapter = str(settings.get("adapter") or ADAPTER_LOCAL)
    if adapter == ADAPTER_LOCAL:
        return LocalCommandAdapter()
    if adapter == ADAPTER_SITE:
        return SiteWrapperAdapter(settings.get("wrapper"))
    raise util.ToolError("unknown runner.adapter %r" % adapter)


def adapter_from_manifest(manifest: Dict[str, Any]):
    """Rebuild the adapter a run was actually started with.

    ``status``/``cancel``/``fetch`` must never be built from current
    configuration: if the configured adapter changed after the run started, a
    local adapter would inspect a pid that belongs to a remote job, or a site
    adapter would be asked about a local one. The adapter and its frozen wrapper
    argv are recorded at start and reused verbatim here.
    """
    name = str(manifest.get("adapter") or "")
    frozen = manifest.get("adapter_settings")
    if not isinstance(frozen, dict):
        frozen = {}
    if name == ADAPTER_LOCAL:
        return LocalCommandAdapter()
    if name == ADAPTER_SITE:
        wrapper = frozen.get("wrapper")
        if not wrapper:
            raise util.ToolError(
                "run %s was started by the site adapter but its manifest records "
                "no wrapper argv, so it cannot be managed.\n"
                "The wrapper that owns this run must be recorded at start."
                % manifest.get("run_id")
            )
        return SiteWrapperAdapter(wrapper)
    raise util.ToolError(
        "run %s records an unknown adapter %r; refusing to guess how to manage it"
        % (manifest.get("run_id"), name)
    )


def report_adapter_drift(
    manifest: Dict[str, Any], settings: Dict[str, Any]
) -> Optional[str]:
    """Note when configuration no longer matches the run's own adapter."""
    configured = str(settings.get("adapter") or ADAPTER_LOCAL)
    recorded = str(manifest.get("adapter") or "")
    if recorded and configured != recorded:
        return (
            "configuration now selects the %r adapter, but this run was started "
            "by %r and is being managed by its own adapter"
            % (configured, recorded)
        )
    return None


# ------------------------------------------------------- start/status/cancel


def start(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
    task_id: str,
    operation: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze a plan, then delegate the start to the configured adapter.

    ``run_id`` is optional and, when supplied, is reserved atomically by
    ``plan``: reusing an existing id fails rather than overwriting that run's
    recorded identity.
    """
    task_dir = config_mod.task_dir(root, config, task_id)
    record = plan(root, config_path, config, task_id, operation, run_id=run_id)
    if not record["ready"]:
        raise util.ToolError(
            "run %s cannot start:\n  - %s"
            % (record["run_id"], "\n  - ".join(record["blockers"]))
        )
    settings = config_mod.runner_settings(config)
    adapter = get_adapter(settings)

    manifest: Dict[str, Any] = {
        "schema": "k3ctl/run-manifest/2",
        "run_id": record["run_id"],
        "task_id": task_id,
        "operation": operation,
        "adapter": adapter.name,
        "state": RUN_STARTING,
        "started_at": util.utcnow(),
        "argv": record["invocation"]["argv"],
        "cwd": record["invocation"]["cwd"],
        "env": record["invocation"]["env"],
        "roots": record["roots"],
        "identity": record["identity"],
        "output_dir": record["output_dir"],
        "scratch_dir": record["scratch_dir"],
        "expected_artifacts": record["expected_artifacts"],
        "pipeline": record["pipeline"],
        "stage_history": record["stage_history"],
        "inputs": record["inputs"],
        "contract_gates": record["contract_gates"],
        # Frozen so status/cancel/fetch are managed by the adapter this run
        # actually started with, not by whatever configuration says later.
        "adapter_settings": {
            "adapter": adapter.name,
            "wrapper": settings.get("wrapper"),
        },
        "gpu_performed": False,
    }
    util.write_json(manifest_path(task_dir, record["run_id"]), manifest)

    started = adapter.start(record, task_dir)
    manifest.update(started)
    manifest["state"] = RUN_RUNNING
    util.write_json(manifest_path(task_dir, record["run_id"]), manifest)
    util.write_json(
        latest_pointer(task_dir),
        {"run_id": record["run_id"], "updated_at": util.utcnow()},
    )
    return {
        "run_id": record["run_id"],
        "task_id": task_id,
        "operation": operation,
        "adapter": adapter.name,
        "state": RUN_RUNNING,
        "pid": manifest.get("pid"),
        "site_run_id": manifest.get("site_run_id"),
        "warnings": record.get("warnings") or [],
        "note": (
            "The command was started. Completion is not success: use "
            "'run status' for the exit status and artifact completeness."
        ),
    }


def _load_manifest(task_dir: Path, run_id: str) -> Dict[str, Any]:
    target = manifest_path(task_dir, run_id)
    if not target.is_file():
        raise util.ToolError("no run manifest for %s" % run_id)
    return config_mod.load_json(target)


def resolve_run_id(task_dir: Path, run_id: Optional[str]) -> str:
    """Resolve a run id, validating any caller-supplied value.

    ``--run-id`` reaches ``status``/``cancel``/``fetch`` too, so it is validated
    here as well as in ``plan``: an unvalidated id would let those commands read
    or write outside the task's runtime directory.
    """
    if run_id:
        return validate_run_id(run_id)
    pointer = latest_pointer(task_dir)
    if not pointer.is_file():
        raise util.ToolError("no runs recorded; pass --run-id")
    value = config_mod.load_json(pointer).get("run_id")
    if not value:
        raise util.ToolError("run pointer does not record a run_id")
    return validate_run_id(str(value))


#: Summary file the measurement harnesses write into a report directory.
SUMMARY_NAME = "summary.json"

#: Gate file whose ``reports_sha256`` is an authoritative filename -> sha256
#: manifest of the worker reports the gate depends on.
GATE_NAME = "correctness.json"

#: Suffixes that identify a sidecar artifact by name.
SIDECAR_SUFFIXES = (".json", ".csv")


def _collect_sidecars(
    directory: Path, payload: Any, required: Dict[str, Optional[str]],
    provenance: Dict[str, str], key: str = "",
) -> None:
    """Walk a harness report, separating local sidecars from provenance.

    Two kinds of path-shaped string appear in these reports and they must not be
    treated the same way:

    * **Local sidecars** -- a bare filename such as ``trial-01.candidate.json``
      or ``per-field.csv``, written next to the summary. These are part of the
      evidence and must exist.
    * **Provenance pointers** -- an *absolute* path such as
      ``reused_correctness_report`` (the gate this run reused) or
      ``baseline_cache.directory``. These deliberately point outside this report
      directory, and they legitimately do not resolve on another machine or
      after a fetch. Existence-checking them would report a complete run as
      incomplete.

    The distinction is structural: absolute means provenance, relative means a
    local sidecar.
    """
    if isinstance(payload, dict):
        for name, value in payload.items():
            # `reports_sha256` is a filename -> sha256 manifest, so the required
            # files are its KEYS. A value-only walk misses them entirely, which
            # is how a deleted worker report slipped through.
            if name == "reports_sha256" and isinstance(value, dict):
                for filename, digest in value.items():
                    if isinstance(filename, str):
                        required[filename] = (
                            digest if isinstance(digest, str) else None
                        )
                continue
            _collect_sidecars(directory, value, required, provenance, name)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_sidecars(directory, item, required, provenance, key)
        return
    if not isinstance(payload, str) or not payload:
        return

    candidate = Path(payload)
    if candidate.is_absolute():
        # Provenance: recorded as lineage, never existence-checked.
        provenance[key or payload] = payload
        return
    if payload.endswith(SIDECAR_SUFFIXES):
        required.setdefault(payload, None)


#: Summary key that identifies the precision diagnostic's schema.
PRECISION_KIND = "accuracy_only_precision_diagnostic"


def _schema_requirements(summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Files a report directory must contain, derived from the harness schema.

    Collecting only what a summary *names* is not sufficient: a directory holding
    nothing but ``summary.json`` names nothing, so nothing is missing and it
    reads as complete. The schema itself says what must be there.

    ``bench/benchmark.py`` always writes ``correctness.json`` -- on the
    ``--correctness-only`` path it is the gate it produced, and on the timing
    path it is the verified gate it copied in. It also writes one worker report
    per executed trial, named ``trial-<NN>.<implementation>.json``. Those names
    are derived from the summary's own ``execution_order``, which records exactly
    which trials ran, rather than assumed from a numbering convention.

    ``precision/diagnose.py`` always writes ``per-field.csv``; its per-trial
    reports appear by relative name in ``trial_reports`` and are picked up by the
    sidecar walk.
    """
    required: List[str] = []

    if summary.get("kind") == PRECISION_KIND:
        required.append("per-field.csv")
        return "precision", required

    # The benchmark schema is identified by fields only it writes.
    if any(key in summary for key in ("execution_order", "phase", "sources")):
        required.append(GATE_NAME)
        for entry in summary.get("execution_order") or []:
            if not isinstance(entry, dict):
                continue
            trial = entry.get("trial")
            implementation = entry.get("implementation")
            if isinstance(trial, int) and isinstance(implementation, str):
                required.append("trial-%02d.%s.json" % (trial, implementation))
        return "benchmark", sorted(set(required))

    return "unknown", required


def _inspect_summary(directory: Path) -> Dict[str, Any]:
    """Verify a harness report directory is complete, passing evidence.

    The prefill harnesses (``bench/benchmark.py``, ``precision/diagnose.py``)
    write ``summary.json`` with ``status`` initialised to ``"failed"`` and set to
    ``"passed"`` only after every phase succeeds. A crashed or killed run leaves
    a directory full of partial reports and a ``"failed"`` summary, which must
    not read as complete evidence.

    Required sidecars are derived from the reports' own schema rather than
    guessed at, and hashes are verified wherever the harness recorded one.
    """
    info: Dict[str, Any] = {
        "summary_present": False,
        "gate_present": False,
        "schema": None,
        "status": None,
        "gate_status": None,
        "case_counts": {},
        "required_sidecars": {},
        "verified_hashes": 0,
        "referenced_missing": [],
        "hash_mismatches": [],
        "provenance": {},
        "problems": [],
    }

    required: Dict[str, Optional[str]] = {}
    provenance: Dict[str, str] = {}

    summary_path = directory / SUMMARY_NAME
    if not summary_path.is_file():
        info["problems"].append(
            "%s is absent: the harness did not reach the point where it writes "
            "its summary, so the directory is partial output, not evidence"
            % SUMMARY_NAME
        )
        return info

    info["summary_present"] = True
    try:
        summary = config_mod.load_json(summary_path)
    except config_mod.ValidationError as exc:
        info["problems"].append("%s is unreadable: %s" % (SUMMARY_NAME, exc))
        return info

    status_value = summary.get("status")
    info["status"] = status_value
    if status_value != "passed":
        info["problems"].append(
            "%s records status=%r, so this run is not passing evidence"
            % (SUMMARY_NAME, status_value)
        )

    # A gate that executed zero cases is never a pass.
    for key, value in summary.items():
        if key.endswith("case_count") and isinstance(value, int):
            info["case_counts"][key] = value
            if value <= 0:
                info["problems"].append(
                    "%s records %s=%d: a gate that executed no cases is never a "
                    "pass" % (SUMMARY_NAME, key, value)
                )

    _collect_sidecars(directory, summary, required, provenance)

    # Files the schema *requires*, independent of what the summary happens to
    # name. Without this, a directory holding only summary.json reads as
    # complete: nothing is missing because nothing was ever required.
    schema, schema_required = _schema_requirements(summary)
    info["schema"] = schema
    for name in schema_required:
        required.setdefault(name, None)

    # The gate file, when present, is the authoritative manifest of the worker
    # reports the correctness evidence rests on.
    gate_path = directory / GATE_NAME
    if gate_path.is_file():
        info["gate_present"] = True
        try:
            gate = config_mod.load_json(gate_path)
        except config_mod.ValidationError as exc:
            info["problems"].append("%s is unreadable: %s" % (GATE_NAME, exc))
        else:
            info["gate_status"] = gate.get("status")
            if gate.get("status") not in (None, "passed"):
                info["problems"].append(
                    "%s records status=%r, so the correctness gate did not pass"
                    % (GATE_NAME, gate.get("status"))
                )
            for key, value in gate.items():
                if key.endswith("case_count") and isinstance(value, int):
                    info["case_counts"].setdefault(key, value)
                    if value <= 0:
                        info["problems"].append(
                            "%s records %s=%d: a gate that executed no cases is "
                            "never a pass" % (GATE_NAME, key, value)
                        )
            _collect_sidecars(directory, gate, required, provenance)

    info["required_sidecars"] = {
        name: ("sha256 known" if digest else "existence only")
        for name, digest in sorted(required.items())
    }
    info["provenance"] = provenance

    for name, digest in sorted(required.items()):
        target = directory / name
        if not target.is_file():
            info["referenced_missing"].append(name)
            continue
        if target.stat().st_size == 0:
            info["referenced_missing"].append("%s (empty)" % name)
            continue
        if digest:
            actual = util.sha256_file(target)
            if actual != digest:
                info["hash_mismatches"].append(
                    "%s (recorded %s, actual %s)" % (name, digest[:12], actual[:12])
                )
            else:
                info["verified_hashes"] += 1

    if info["referenced_missing"]:
        info["problems"].append(
            "report references sidecar files that are absent or empty: %s"
            % ", ".join(sorted(set(info["referenced_missing"])))
        )
    if info["hash_mismatches"]:
        info["problems"].append(
            "sidecar files do not match the hashes the harness recorded: %s"
            % ", ".join(sorted(set(info["hash_mismatches"])))
        )
    return info


def artifact_report(
    manifest: Dict[str, Any], base: Optional[Path] = None
) -> Dict[str, Any]:
    """Check artifact completeness and hashes, independently of exit status.

    ``base`` remaps the declared artifacts onto another directory, so ``fetch``
    can apply the same expectations to a destination that is not ``output_dir``.

    Completeness here means: the declared artifact exists, is non-empty, and --
    for a report directory -- carries a harness summary that records a pass with
    a non-zero case count and no missing referenced reports. A directory that
    merely contains files is *not* complete.
    """
    declared = [Path(item) for item in manifest.get("expected_artifacts") or []]
    if base is not None:
        declared = [Path(base) / item.name for item in declared]

    present: Dict[str, Any] = {}
    missing: List[str] = []
    incomplete: List[str] = []
    summaries: Dict[str, Any] = {}

    for item in declared:
        if item.is_file():
            if item.stat().st_size == 0:
                missing.append("%s (file is empty)" % item)
                continue
            entry: Dict[str, Any] = {
                "kind": "file",
                "sha256": util.sha256_file(item),
                "bytes": item.stat().st_size,
            }
            # A single-file report carries its own status.
            if item.suffix == ".json":
                try:
                    payload = config_mod.load_json(item)
                except config_mod.ValidationError as exc:
                    incomplete.append("%s is unreadable: %s" % (item, exc))
                else:
                    status_value = payload.get("status")
                    entry["status"] = status_value
                    if status_value is not None and status_value != "passed":
                        incomplete.append(
                            "%s records status=%r, so it is not passing evidence"
                            % (item, status_value)
                        )
            present[str(item)] = entry
        elif item.is_dir():
            hashes = util.tree_hashes(item)
            if not hashes:
                missing.append("%s (directory is empty)" % item)
                continue
            summary_info = _inspect_summary(item)
            summaries[str(item)] = summary_info
            present[str(item)] = {
                "kind": "directory",
                "file_count": len(hashes),
                "digest": util.tree_digest(hashes),
                "summary_status": summary_info["status"],
            }
            incomplete.extend(
                "%s: %s" % (item, problem) for problem in summary_info["problems"]
            )
        else:
            missing.append(str(item))

    return {
        "expected_count": len(declared),
        "base": str(base) if base is not None else None,
        "present": present,
        "missing": missing,
        "incomplete": incomplete,
        "summaries": summaries,
        # Complete requires declared artifacts, all present, and none partial.
        "complete": bool(declared) and not missing and not incomplete,
        "no_expectations_declared": not declared,
        "note": (
            "Completeness describes artifacts only. It is not a gate result and "
            "not acceptance: a present, passing summary still has to be reviewed "
            "against the task contract."
        ),
    }


def status(
    root: Path,
    config: Dict[str, Any],
    task_id: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Report execution state and evidence state -- never a gate verdict."""
    task_dir = config_mod.task_dir(root, config, task_id)
    resolved = resolve_run_id(task_dir, run_id)
    manifest = _load_manifest(task_dir, resolved)
    settings = config_mod.runner_settings(config)
    # Managed by the adapter this run started with, never by current config.
    adapter = adapter_from_manifest(manifest)
    drift = report_adapter_drift(manifest, settings)
    execution = adapter.status(manifest)
    artifacts = artifact_report(manifest)

    # A cancelled run is never a completion, whatever exit code it carries.
    ran_to_completion = (
        execution.get("state") == RUN_EXITED
        and execution.get("exit_code") == 0
        and bool(execution.get("exited_normally"))
        and not execution.get("cancelled")
    )
    evidence_complete = bool(artifacts["complete"])

    return {
        "run_id": resolved,
        "task_id": task_id,
        "operation": manifest.get("operation"),
        "adapter": manifest.get("adapter"),
        "roots": manifest.get("roots"),
        # Execution: did it run, and how did it end.
        "execution": {
            "state": execution.get("state"),
            "detail": execution.get("detail"),
            "exit_code": execution.get("exit_code"),
            "signal": execution.get("signal"),
            "exited_normally": execution.get("exited_normally"),
            "started_at": execution.get("started_at"),
            "finished_at": execution.get("finished_at"),
            "ran_to_completion": ran_to_completion,
        },
        # Evidence: is there anything usable on disk.
        "evidence": {
            "complete": evidence_complete,
            "artifacts": artifacts,
        },
        "pid": manifest.get("pid"),
        "site_run_id": manifest.get("site_run_id"),
        "identity": manifest.get("identity"),
        "inputs": manifest.get("inputs"),
        "pipeline": manifest.get("pipeline"),
        "stage_history": manifest.get("stage_history"),
        "contract_gates": manifest.get("contract_gates") or {},
        "gpu_performed": bool(manifest.get("gpu_performed")),
        "gates_evaluated": False,
        "accepted": False,
        "adapter_drift": drift,
        "note": (
            "'ran_to_completion' describes the process only. 'evidence.complete' "
            "describes artifacts only. Neither is acceptance: the task harness "
            "must evaluate the contract gates against the evidence, and a human "
            "must review the result."
        ),
    }


def cancel(
    root: Path,
    config: Dict[str, Any],
    task_id: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    task_dir = config_mod.task_dir(root, config, task_id)
    resolved = resolve_run_id(task_dir, run_id)
    manifest = _load_manifest(task_dir, resolved)
    settings = config_mod.runner_settings(config)
    adapter = adapter_from_manifest(manifest)
    drift = report_adapter_drift(manifest, settings)
    result = adapter.cancel(manifest)
    # Record that cancellation was requested. The terminal outcome is written by
    # the supervisor (local) or reported by the site adapter, not asserted here.
    manifest["cancel_requested_at"] = util.utcnow()
    util.write_json(manifest_path(task_dir, resolved), manifest)
    return {
        "run_id": resolved,
        "cancel_requested": True,
        "adapter": manifest.get("adapter"),
        "adapter_drift": drift,
        "detail": result,
        "note": (
            "Cancellation was requested and forwarded to the child. Use "
            "'run status' for the terminal outcome; a cancelled run is never a "
            "completion."
        ),
    }


def fetch(
    root: Path,
    config: Dict[str, Any],
    task_id: str,
    run_id: Optional[str] = None,
    destination: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch artifacts through the adapter and record their hashes."""
    task_dir = config_mod.task_dir(root, config, task_id)
    resolved = resolve_run_id(task_dir, run_id)
    manifest = _load_manifest(task_dir, resolved)
    settings = config_mod.runner_settings(config)
    adapter = adapter_from_manifest(manifest)
    drift = report_adapter_drift(manifest, settings)

    # A caller-supplied destination is confined to the repository, so a fetch
    # cannot write outside it.
    if destination:
        candidate = Path(destination)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(Path(root).resolve())
            except ValueError:
                raise util.ToolError(
                    "fetch destination must be inside the project root: %s"
                    % destination
                )
            target = candidate
        else:
            target = paths.safe_join(root, destination)
    else:
        target = run_dir(task_dir, resolved) / "fetched"

    result = adapter.fetch(manifest, target)

    # Check the expectations against wherever the artifacts actually landed.
    # The local adapter leaves them in output_dir; a site adapter downloads them
    # into the destination. Reporting on output_dir after a remote fetch would
    # describe a directory the fetch never touched.
    location = result.get("location")
    fetch_base: Optional[Path] = None
    if location:
        located = Path(str(location))
        if located.resolve() != Path(str(manifest.get("output_dir") or "")).resolve():
            fetch_base = located
    artifacts = artifact_report(manifest, base=fetch_base)

    fetched_files = result.get("files") or {}
    inventory = {
        "schema": "k3ctl/fetch-inventory/1",
        "run_id": resolved,
        "task_id": task_id,
        "fetched_at": util.utcnow(),
        "adapter": manifest.get("adapter"),
        "location": str(location) if location else None,
        "checked_against": str(fetch_base) if fetch_base else manifest.get("output_dir"),
        "transferred": bool(result.get("transferred", False)),
        "file_count": len(fetched_files),
        "files": fetched_files,
        "evidence_complete": bool(artifacts["complete"]),
        "missing": artifacts["missing"],
        "incomplete": artifacts["incomplete"],
    }
    util.write_json(run_dir(task_dir, resolved) / "fetch-inventory.json", inventory)

    return {
        "run_id": resolved,
        "task_id": task_id,
        "adapter": manifest.get("adapter"),
        "adapter_drift": drift,
        "location": str(location) if location else None,
        "checked_against": inventory["checked_against"],
        "transferred": bool(result.get("transferred", False)),
        "fetched_file_count": len(fetched_files),
        "inventory": "fetch-inventory.json",
        "evidence": {"complete": artifacts["complete"], "artifacts": artifacts},
        "note": (
            "Artifact presence and hashes are recorded against the directory the "
            "fetch actually populated. Completeness is not a gate result."
        ),
    }
