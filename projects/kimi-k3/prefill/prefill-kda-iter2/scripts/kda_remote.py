#!/usr/bin/env python3
"""Durable, single-GPU KDA execution over Mac -> <gateway> -> GB300 -> Docker."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

DEFAULT_TASK = "projects/kimi-k3/prefill/prefill-kda-iter2"
BASE_COMMIT = "9ac2710bd37622f38edb078cc753244a3c38c334"
DEFAULT_GPU = "<redacted-gpu-uuid>"
EXCLUDED = {".git", ".humanize", "__pycache__", ".build", ".venv", "artifacts", "runs", ".cache", "build", ".baseline"}
EXCLUDED_SUFFIXES = {".so", ".o", ".pyc", ".ncu-rep", ".nsys-rep", ".sqlite"}
REFERENCE_ARTIFACT_PARENTS = ("benchmark", "benchmark_repeat", "correctness", "profile-correctness")
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "gpu_busy"}
LOCK_DIRECTORY = Path("/data/kda-locks")
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def checked(cmd, **kwargs):
    strip_output = kwargs.pop("strip_output", True)
    output = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=kwargs.pop("timeout", 60), **kwargs).stdout
    return output.strip() if strip_output else output


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"expected a relative path without parent traversal: {value}")
    return str(path)


def run_root(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain only letters, digits, underscore, or hyphen")
    base = PurePosixPath(args.remote_root)
    if not base.is_absolute() or ".." in base.parts or not str(base).startswith("/data/") or not re.fullmatch(r"/[A-Za-z0-9_/-]+", str(base)):
        raise ValueError("remote root must be an absolute path underneath /data")
    return str(base / args.run_id)


def nested_ssh(args, command):
    return SSH + [args.gateway, shlex.join(SSH + [args.target, shlex.join(command)])]


def remote(args, command, timeout=120):
    return checked(nested_ssh(args, command), timeout=timeout)


def remote_helper(args, operation, container=False, timeout=120):
    root = run_root(args)
    command = ["python3", root + "/runner.py", operation, root]
    if container:
        command = ["docker", "exec", args.container] + command
    return remote(args, command, timeout=timeout)


def rsync_transport(args):
    return shlex.join(SSH + [args.gateway] + SSH)


def source_files(root, extra_excluded=()):
    excluded = EXCLUDED | set(extra_excluded)
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in dirs + sorted(names):
            path = Path(directory) / name
            if name in excluded or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"source snapshot rejects symlinks; materialize explicitly: {path}")
            if path.is_file():
                yield path


def candidate_source_files(worktree):
    worktree = Path(worktree).resolve()
    python_root = worktree / "python"
    if python_root.is_symlink():
        raise RuntimeError("candidate snapshot rejects directory symlink: python")
    for directory, dirs, names in os.walk(python_root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError("candidate snapshot rejects directory symlink: " + path.relative_to(worktree).as_posix())
        for name in sorted(names):
            original = Path(directory) / name
            if name in EXCLUDED or original.suffix in EXCLUDED_SUFFIXES:
                continue
            rel = original.relative_to(worktree).as_posix()
            if original.is_symlink():
                resolved = original.resolve(strict=True)
                if not resolved.is_relative_to(python_root):
                    raise RuntimeError("candidate link escapes Python source: " + rel)
                original = resolved
            yield rel, original


def workflow_options(args):
    kind = getattr(args, "profile_kind", None)
    workflow = getattr(args, "workflow", None) or ("profile" if kind == "ncu" else "candidate")
    kind = kind or ("nsys" if workflow in {"prepare", "final"} else "none")
    if getattr(args, "skip_nsys", False):
        if kind == "ncu":
            raise ValueError("--skip-nsys cannot be combined with NCU")
        kind = "none"
    if workflow in {"prepare", "final"} and kind != "nsys":
        raise ValueError("prepare/final require NSYS; use a separate --workflow profile run for NCU")
    if kind == "ncu" and workflow != "profile":
        raise ValueError("NCU is a separate --workflow profile run, never a timing run")
    if workflow == "profile" and kind == "none":
        raise ValueError("profile workflow requires --profile-kind nsys or ncu")
    if workflow == "final" and getattr(args, "smoke", False):
        raise ValueError("final workflow requires full measurements, not --smoke")
    return workflow, kind


def make_plan(root, task, args):
    task_dir = root + "/src/task"
    artifacts = root + "/artifacts"
    workloads = task_dir + "/workloads.json"
    baseline = root + "/src/baseline"
    candidate = root + "/src/candidate"
    workflow, kind = workflow_options(args)
    stages = []
    for side, source in (("baseline", baseline), ("candidate", candidate)):
        stages.append({"name": "preflight_" + side,
                       "argv": ["python3", "bench/preflight.py", "--source-root", source,
                                "--workloads", workloads, "--report", artifacts + "/preflight-" + side + ".json"],
                       "timeout": args.correctness_timeout})
    benchmark = ["python3", "bench/benchmark.py", "--baseline-root", baseline,
                 "--candidate-root", candidate, "--workloads", workloads]
    if workflow == "prepare":
        # Preserve the initial untouched-baseline READY evidence format.
        command = benchmark + ["--out", artifacts + "/benchmark"]
        if args.smoke:
            command.append("--smoke")
        stages.append({"name": "correctness_and_benchmark", "argv": command,
                       "timeout": args.benchmark_timeout})
    else:
        stages.append({"name": "correctness_gate", "argv": benchmark + [
            "--correctness-only", "--out", artifacts + "/correctness"],
            "timeout": args.correctness_timeout})
        if workflow != "profile":
            precision = ["python3", "precision/diagnose.py", "--baseline-root", baseline,
                         "--candidate-root", candidate, "--workloads", workloads,
                         "--out", artifacts + "/precision", "--scratch", root + "/scratch/precision",
                         "--expect-gpu-uuid", args.gpu]
            if not getattr(args, "no_precision_cache", False):
                precision += ["--baseline-cache", str(PurePosixPath(root).parent / ".precision-cache" / args.gpu)]
            stages.append({"name": "precision_diagnostic", "argv": precision,
                           "timeout": getattr(args, "precision_timeout", 3600), "non_gating": True})
            for name in (["benchmark", "benchmark_repeat"] if workflow == "final" else ["benchmark"]):
                command = benchmark + ["--correctness-report", artifacts + "/correctness/correctness.json",
                                       "--out", artifacts + "/" + name]
                if args.smoke:
                    command.append("--smoke")
                stages.append({"name": name, "argv": command, "timeout": args.benchmark_timeout})
    if kind == "ncu":
        stages.extend([
            {"name": "ncu_profile", "argv": ["ncu", "--set", getattr(args, "ncu_set", "basic"),
             "--clock-control", "none", "--replay-mode", "application", "--target-processes", "all",
             "--nvtx", "--nvtx-include", "kda_capture/", "--kernel-name-base", "demangled",
             "-k", "regex:" + args.kernel_regex, "--launch-count", "1", "-o", artifacts + "/candidate_ncu",
             "python3", "bench/profile_one.py", "--source-root", candidate, "--workloads", workloads,
             "--id", args.profile_id, "--iters", "1"], "timeout": args.profile_timeout},
            {"name": "ncu_export", "argv": ["ncu", "--import", artifacts + "/candidate_ncu.ncu-rep",
             "--page", "raw", "--csv"], "timeout": args.profile_timeout},
            {"name": "ncu_parse", "argv": ["python3", "scripts/parse_ncu.py", artifacts + "/candidate_ncu.ncu-rep",
             "--output", artifacts + "/ncu_metrics.json", "--expect-kernel", args.kernel_regex],
             "timeout": args.profile_timeout}])
    elif kind == "nsys":
        sides = [("candidate", candidate)] if workflow == "prepare" else [("baseline", baseline), ("candidate", candidate)]
        for side, source in sides:
            suffix = "" if workflow == "prepare" else "_" + side
            stages.extend([
                {"name": "nsys_profile" + suffix, "argv": ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                 "--cpuctxsw=none", "-o", artifacts + "/" + side + "_nsys", "python3", "bench/profile_one.py",
                 "--source-root", source, "--workloads", workloads, "--id", args.profile_id, "--iters", "5"],
                 "timeout": args.profile_timeout},
                {"name": "nsys_stats" + suffix, "argv": ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
                 artifacts + "/" + side + "_nsys.nsys-rep"], "timeout": args.profile_timeout}])
    if any(stage["timeout"] <= 0 for stage in stages):
        raise ValueError("stage timeouts must be positive")
    return {"cwd": task_dir, "workflow": workflow, "stages": stages}


def snapshot(args, destination):
    worktree = Path(args.worktree).resolve()
    task = relative_path(args.task)
    task_path = (Path(args.task_root).resolve() if args.task_root else worktree / task)
    if not task_path.is_dir():
        raise RuntimeError("task root does not exist: " + str(task_path))
    sys.path.insert(0, str(task_path / "bench"))
    from common import load_workloads
    workloads_file = Path(args.workloads).resolve() if args.workloads else task_path / "workloads.json"
    workloads = load_workloads(workloads_file)
    frozen = read_json(task_path / "baseline_manifest.json")
    if frozen["source_commit"] != BASE_COMMIT:
        raise RuntimeError("frozen baseline does not match the authorized source revision")
    workflow, kind = workflow_options(args)
    if kind == "ncu" and not args.kernel_regex:
        raise ValueError("NCU requires --kernel-regex from a verified NSYS target launch")
    if not args.profile_id:
        args.profile_id = next(row["id"] for row in workloads["cases"] if row["category"] == "deployment_grid" and row["total_tokens"] == 16384 and row["batch_size"] == 1)
    if args.profile_id not in {row["id"] for row in workloads["cases"]}:
        raise ValueError("profile workload does not exist")
    files = {}
    # Freeze the baseline from recorded bytes, including same-commit materialized links.
    for rel, expected in frozen["files_sha256"].items():
        original = task_path / ".baseline" / relative_path(rel)
        if original.is_symlink() or digest(original) != expected:
            raise RuntimeError("baseline hash mismatch: " + rel)
        target = destination / "src/baseline" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files["baseline/" + rel] = expected
    # The full Python tree ensures dependency changes travel with the candidate.
    for rel, original in candidate_source_files(worktree):
        target = destination / "src/candidate" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files["candidate/" + rel] = digest(target)
    baseline_files = {key.removeprefix("baseline/"): value for key, value in files.items() if key.startswith("baseline/")}
    candidate_files = {key.removeprefix("candidate/"): value for key, value in files.items() if key.startswith("candidate/")}
    if workflow == "prepare" and baseline_files != candidate_files:
        raise RuntimeError("prepare requires an untouched candidate matching the frozen baseline")
    if workflow in {"candidate", "final"} and not (task_path / "precision/diagnose.py").is_file():
        raise RuntimeError("candidate workflow requires precision/diagnose.py")
    for original in source_files(task_path, extra_excluded=("runtime", "profile", "results")):
        rel = original.relative_to(task_path).as_posix()
        target = destination / "src/task" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files["task/" + rel] = digest(target)
    shutil.copy2(workloads_file, destination / "src/task/workloads.json")
    files["task/workloads.json"] = digest(destination / "src/task/workloads.json")
    revision = checked(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    # Candidate source and task controls are intentionally separate in infra-loop.
    patch = checked(["git", "-C", str(worktree), "diff", "--binary", "HEAD", "--", "python"], strip_output=False)
    (destination / "source.diff").write_text(patch)
    if not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", args.gpu):
        raise ValueError("select an explicit GPU UUID")
    manifest = {"schema": 2, "run_id": args.run_id, "created_at": now(),
                "source_root": str(worktree), "source_revision": revision, "baseline_revision": BASE_COMMIT,
                "remote_original": "/data/sglang", "task": task, "gpu_uuid": args.gpu,
                "container": args.container, "target": args.target, "gateway": args.gateway,
                "idle_memory_mib": args.idle_memory_mib, "profile_kind": kind, "workflow": workflow,
                "git_status": checked(["git", "-C", str(worktree), "status", "--short"]),
                "files_sha256": files, "source_diff_sha256": digest(destination / "source.diff"),
                "workloads": workloads, "runner_sha256": digest(__file__),
                "scope": "preparation_smoke" if args.smoke else "synthetic_deployment_grid_not_live_serving_acceptance",
                "timing_metric": "CUDA events around one mutable public chunk_kda call; state/value restore excluded; see bench policy",
                "plan": make_plan(run_root(args), task, args)}
    write_json(destination / "manifest.json", manifest)
    shutil.copy2(__file__, destination / "runner.py")
    return manifest


def verify_source(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    actual_files = {str(path.relative_to(root / "src")) for path in source_files(root / "src")}
    expected_files = set(manifest["files_sha256"])
    if actual_files != expected_files:
        raise RuntimeError(f"source file set mismatch: extra={sorted(actual_files - expected_files)}, missing={sorted(expected_files - actual_files)}")
    for rel, expected in manifest["files_sha256"].items():
        path = root / "src" / relative_path(rel)
        if path.is_symlink() or digest(path) != expected:
            raise RuntimeError(f"source hash mismatch: {rel}")
    if digest(root / "source.diff") != manifest["source_diff_sha256"]:
        raise RuntimeError("source diff hash mismatch")
    if digest(root / "runner.py") != manifest["runner_sha256"]:
        raise RuntimeError("runner hash mismatch")
    return manifest


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def worker_matches(identity, root):
    if not identity or process_identity(identity["pid"]) != identity:
        return False
    try:
        command = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b"\0")
        return b"_worker" in command and os.fsencode(str(root)) in command
    except FileNotFoundError:
        return False


def gpu_state(timeout=60):
    deadline = time.monotonic() + timeout
    return {
        "gpus": checked(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=max(0.001, deadline - time.monotonic())),
        "compute_apps": checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], timeout=max(0.001, deadline - time.monotonic())),
    }


def idle_utilization(state, uuid, idle_memory_mib=64):
    rows = list(csv.reader(io.StringIO(state["gpus"]), skipinitialspace=True))
    gpu = next((row for row in rows if row[1] == uuid), None)
    if not gpu:
        raise RuntimeError(f"selected GPU not present: {uuid}")
    for app in csv.reader(io.StringIO(state["compute_apps"]), skipinitialspace=True):
        if app and app[0] == uuid:
            raise RuntimeError(f"selected GPU has a compute process: {app}")
    if float(gpu[3]) > idle_memory_mib:
        raise RuntimeError(f"selected GPU has allocated memory: {gpu}")
    return float(gpu[4])


def ensure_idle(state, uuid, idle_memory_mib=64):
    if idle_utilization(state, uuid, idle_memory_mib) > 0:
        raise RuntimeError(f"selected GPU utilization is nonzero: {uuid}")


def settle_gpu(uuid, samples, timeout=5.0, interval=0.5, idle_memory_mib=64):
    """Allow only utilization-only telemetry lag; retain every sample on failure."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"selected GPU utilization did not settle within {timeout:g}s: {uuid}")
        sample = gpu_state(timeout=remaining)
        samples.append({"captured_at": now(), **sample})
        if idle_utilization(sample, uuid, idle_memory_mib) == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))


def terminate_group(process, grace=5):
    # The session was created by this worker; never search by executable name.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before its children, so clean up the group even then.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def finalize(root, state):
    artifacts = root / "artifacts"
    state["finished_at"] = now()
    write_json(artifacts / "status.json", state)
    write_json(artifacts / "artifacts.sha256.json", artifact_hashes(artifacts))
    write_json(root / "finished.json", {"finished_at": now(), "exit_code": state.get("exit_code", 1)})


def execute_stage(stage, cwd, environment, artifacts, cancelled, lock_fd=None):
    record = {**stage, "started_at": now()}
    with (artifacts / (stage["name"] + ".log")).open("w") as logfile:
        process = subprocess.Popen(stage["argv"], cwd=cwd, env=environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(() if lock_fd is None else (lock_fd,)))
        try:
            record["pid"] = process.pid
            record["process_group"] = process.pid
            record["identity"] = process_identity(process.pid)
            write_json(artifacts / "active_stage.json", record)
            deadline = time.monotonic() + stage["timeout"]
            while process.poll() is None:
                if cancelled():
                    terminate_group(process)
                    record.update(state="cancelled", exit_code=130)
                    break
                if time.monotonic() >= deadline:
                    terminate_group(process)
                    record.update(state="timed_out", exit_code=124)
                    break
                time.sleep(0.25)
            else:
                record.update(state="completed" if process.returncode == 0 else "failed", exit_code=process.returncode)
            record["finished_at"] = now()
            write_json(artifacts / (stage["name"] + ".status.json"), record)
            write_json(artifacts / "active_stage.json", record)
        finally:
            # Clean up descendants after every outcome, including supervisor I/O failure.
            terminate_group(process)
    return record


def artifact_hashes(artifacts):
    result = {}
    for path in sorted(artifacts.rglob("*")):
        relative = path.relative_to(artifacts)
        # Reference tensors are temporary working data, removed by benchmark.py.
        # Ignore them consistently if a previous partial fetch retained a copy.
        if (len(relative.parts) > 2 and relative.parts[0] in REFERENCE_ARTIFACT_PARENTS
                and relative.parts[1].startswith("baseline-tensors-")):
            continue
        if path.is_file() and path.name != "artifacts.sha256.json":
            result[str(relative)] = digest(path)
    return result


def fetch_identity(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if digest(root / "manifest.json") != read_json(root / "ready.json")["manifest_sha256"]:
        raise RuntimeError("fetch manifest changed after upload verification")
    return {key: manifest[key] for key in ("run_id", "target", "gateway", "container")}


def resolve_fetch_identity(args, identity):
    for key in ("run_id", "target", "gateway"):
        if identity.get(key) != getattr(args, key):
            raise RuntimeError(f"remote fetch manifest belongs to a different {key}")
    container = identity.get("container")
    if not isinstance(container, str) or not container:
        raise RuntimeError("remote fetch manifest is missing its container")
    if args.container is not None and args.container != container:
        raise RuntimeError("remote fetch manifest belongs to a different container")
    args.container = container


def validate_fetch_destination(output, args):
    """Never mix evidence identities when refreshing a local download."""
    if not output.exists() or not any(output.iterdir()):
        return
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("nonempty fetch destination has no run manifest; choose a fresh output directory")
    previous = read_json(manifest_path)
    for key in ("run_id", "target", "gateway", "container"):
        if previous.get(key) != getattr(args, key):
            raise RuntimeError(f"fetch destination belongs to a different {key}; choose a fresh output directory")


def verify_fetched_artifacts(output):
    expected = read_json(output / "artifacts/artifacts.sha256.json")
    actual = artifact_hashes(output / "artifacts")
    if actual != expected:
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        mismatched = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
        raise RuntimeError(f"artifact verification failed: extra={extra}, missing={missing}, mismatched={mismatched}")
    if digest(output / "manifest.json") != read_json(output / "ready.json")["manifest_sha256"]:
        raise RuntimeError("fetched manifest hash mismatch")
    manifest = read_json(output / "manifest.json")
    for name, key in (("source.diff", "source_diff_sha256"), ("runner.py", "runner_sha256")):
        if digest(output / name) != manifest[key]:
            raise RuntimeError(f"fetched provenance hash mismatch: {name}")


def worker(root):
    root = Path(root)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    state = {"run_id": root.name, "state": "starting", "started_at": now(), "worker": process_identity(os.getpid()), "stages": []}
    cancelled = [False]
    signal.signal(signal.SIGTERM, lambda *_: cancelled.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: cancelled.__setitem__(0, True))
    write_json(artifacts / "status.json", state)
    lock = None
    try:
        manifest = verify_source(root)
        if (root / "cancel.requested").exists():
            state.update(state="cancelled", exit_code=130)
            return 130
        locks = LOCK_DIRECTORY
        locks.mkdir(exist_ok=True)
        lock = (locks / (manifest["gpu_uuid"] + ".lock")).open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state.update(state="gpu_busy", exit_code=75, error="cooperative GPU lock is already held")
            return 75
        state["lock_path"] = str(locks / (manifest["gpu_uuid"] + ".lock"))
        state["gpu_before"] = gpu_state()
        try:
            ensure_idle(state["gpu_before"], manifest["gpu_uuid"], manifest.get("idle_memory_mib", 64))
        except RuntimeError as exc:
            state.update(state="gpu_busy", exit_code=75, error=str(exc))
            return 75
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": manifest["gpu_uuid"], "KDA_RUN_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1", "TRITON_CACHE_DIR": str(root / "cache/triton"), "CUDA_CACHE_PATH": str(root / "cache/cuda"), "TORCH_EXTENSIONS_DIR": str(root / "cache/torch-extensions")})
        for key in ("TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
        environment.update({"SGLANG_CACHE_DIR": str(root / "cache/sglang"), "PYTHONUNBUFFERED": "1"})
        Path(environment["SGLANG_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        state["environment"] = {k: environment[k] for k in ("CUDA_VISIBLE_DEVICES", "KDA_RUN_ROOT", "PYTHONDONTWRITEBYTECODE", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR")}
        stages = manifest["plan"]["stages"]
        state.update(state="running", exit_code=None)
        write_json(artifacts / "status.json", state)
        for stage in stages:
            if cancelled[0] or (root / "cancel.requested").exists():
                state.update(state="cancelled", exit_code=130)
                break
            state["current_stage"] = stage["name"]
            write_json(artifacts / "status.json", state)
            record = execute_stage(stage, manifest["plan"]["cwd"], environment, artifacts, lambda: cancelled[0] or (root / "cancel.requested").exists(), lock.fileno())
            state["stages"].append(record)
            if record["exit_code"]:
                # Only precision is advisory. Cancellation and GPU cleanup remain hard stops.
                if stage["name"] == "precision_diagnostic" and stage.get("non_gating") and record["state"] != "cancelled":
                    state.setdefault("non_gating_failures", []).append({
                        "name": stage["name"], "state": record["state"], "exit_code": record["exit_code"],
                        "log": stage["name"] + ".log"})
                else:
                    state.update(state=record["state"], exit_code=record["exit_code"])
                    break
            samples = []
            state["gpu_after_" + stage["name"]] = {"samples": samples, "settle_timeout_seconds": 5.0}
            settle_gpu(manifest["gpu_uuid"], samples, idle_memory_mib=manifest.get("idle_memory_mib", 64))
        else:
            if cancelled[0] or (root / "cancel.requested").exists():
                state.update(state="cancelled", exit_code=130)
            else:
                state.update(state="completed", exit_code=0)
        verify_source(root)
    except Exception as exc:
        state.update(state="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            finalize(root, state)
        finally:
            if lock is not None:
                lock.close()
    return state.get("exit_code", 1)


def launch(root, manifest):
    container = json.loads(checked(["docker", "inspect", manifest["container"]]))[0]
    if not container["State"]["Running"]:
        raise RuntimeError("container is not running")
    if not any(m["Source"] == "/data" and m["Destination"] == "/data" and m["RW"] for m in container["Mounts"]):
        raise RuntimeError("container must mount /data read-write at /data")
    write_json(root / "artifacts/container.json", {"Id": container["Id"], "Image": container["Image"], "Mounts": container["Mounts"], "HostConfig": {k: container["HostConfig"].get(k) for k in ("IpcMode", "Privileged", "CapAdd", "DeviceRequests")}})
    # A timing-only candidate does not need either profiler installed.
    programs = {"python3"} | {stage["argv"][0] for stage in manifest["plan"]["stages"]
                              if stage["argv"][0] in {"nsys", "ncu"}}
    versions = {program: checked(["docker", "exec", manifest["container"], program, "--version"])
                for program in sorted(programs)}
    write_json(root / "artifacts/host.json", {"captured_at": now(), "uname": checked(["uname", "-a"]), "nvidia_smi": checked(["nvidia-smi"]), "topology": checked(["nvidia-smi", "topo", "-m"]), "tools": versions})
    write_json(root / "submission.json", {"started_at": now()})
    checked(["docker", "exec", "-d", manifest["container"], "python3", str(root / "runner.py"), "_worker", str(root)])


@contextmanager
def control_lock(root):
    """Serialize launch and pre-launch cancellation across host processes."""
    with (root / "control.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def launch_once(root):
    with control_lock(root):
        if (root / "cancel.requested").exists():
            raise RuntimeError("run was already cancelled; create a new run ID")
        manifest = verify_source(root)
        if digest(root / "manifest.json") != read_json(root / "ready.json")["manifest_sha256"]:
            raise RuntimeError("manifest changed after upload verification")
        with (root / "launch.json").open("x") as stream:
            json.dump({"requested_at": now(), "run_id": root.name}, stream)
        try:
            launch(root, manifest)
        except Exception as exc:
            if (root / "submission.json").exists():
                # A Docker client timeout does not prove the detached worker never started.
                write_json(root / "launch_error.json", {"at": now(), "error": str(exc), "detail": "submission uncertain; inspect status before any new run"})
            else:
                finalize(root, {"run_id": root.name, "state": "failed", "exit_code": 1, "error": f"launch failed: {exc}"})
            raise
        return {"run_id": root.name, "state": "submitted", "root": str(root)}


def host_operation(operation, root):
    root = Path(root)
    if operation == "_fetch_identity":
        return fetch_identity(root)
    if operation == "_verify":
        manifest = verify_source(root)
        (root / "artifacts").mkdir(exist_ok=True)
        write_json(root / "ready.json", {"verified_at": now(), "file_count": len(manifest["files_sha256"]), "manifest_sha256": digest(root / "manifest.json")})
        return read_json(root / "ready.json")
    if operation == "_launch":
        return launch_once(root)
    if operation == "_status":
        status_path = root / "artifacts/status.json"
        state = read_json(status_path) if status_path.exists() else {"run_id": root.name, "state": "submitted" if (root / "launch.json").exists() else "uploaded"}
        state["cancel_requested"] = (root / "cancel.requested").exists()
        state["artifacts_finalized"] = (root / "finished.json").exists()
        if (root / "launch_error.json").exists():
            state["launch_error"] = read_json(root / "launch_error.json")
        if state["state"] not in TERMINAL and state.get("worker"):
            manifest = read_json(root / "manifest.json")
            try:
                alive = json.loads(checked(["docker", "exec", manifest["container"], "python3", str(root / "runner.py"), "_probe", str(root)]))["alive"]
                if not alive:
                    state["persisted_state"] = state["state"]
                    state["state"] = "worker_missing"
            except subprocess.SubprocessError as exc:
                state["probe_error"] = str(exc)
        return state
    if operation == "_probe":
        state = read_json(root / "artifacts/status.json")
        return {"alive": worker_matches(state.get("worker"), root)}
    if operation == "_cancel":
        manifest = read_json(root / "manifest.json")
        with control_lock(root):
            write_json(root / "cancel.requested", {"requested_at": now()})
            if not (root / "launch.json").exists():
                state = {"run_id": root.name, "state": "cancelled", "exit_code": 130, "reason": "cancelled before launch"}
                (root / "artifacts").mkdir(exist_ok=True)
                finalize(root, state)
                return state
        return json.loads(checked(["docker", "exec", manifest["container"], "python3", str(root / "runner.py"), "_signal", str(root)]))
    if operation == "_signal":
        status_path = root / "artifacts/status.json"
        if not status_path.exists():
            return {"state": "cancel_requested_before_worker_start"}
        state = read_json(status_path)
        if state["state"] in TERMINAL:
            return state
        if not worker_matches(state.get("worker"), root):
            return {"state": "worker_missing", "detail": "No signal sent; recorded PID identity is not alive"}
        # A pidfd pins the process across the identity check and signal, preventing PID reuse.
        try:
            pidfd = os.pidfd_open(state["worker"]["pid"])
        except ProcessLookupError:
            return {"state": "worker_missing", "detail": "No signal sent; process already exited"}
        try:
            if not worker_matches(state.get("worker"), root):
                return {"state": "worker_missing", "detail": "No signal sent; PID identity changed"}
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        finally:
            os.close(pidfd)
        return {"state": "cancel_requested", "worker": state["worker"]}
    raise ValueError(operation)


def main():
    if len(sys.argv) == 3 and sys.argv[1].startswith("_"):
        if sys.argv[1] == "_worker":
            return worker(sys.argv[2])
        print(json.dumps(host_operation(sys.argv[1], sys.argv[2]), indent=2))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "sync", "run", "status", "cancel", "fetch"):
        command = sub.add_parser(action)
        command.add_argument("--gateway", default="maas")
        command.add_argument("--target", default="<gpu-node>")
        command.add_argument("--container", default=None if action == "fetch" else "kda_test",
                             help="fetch defaults to the verified run manifest's container; other actions default to kda_test")
        command.add_argument("--remote-root", default="<external-run-root>")
        if action != "preflight":
            command.add_argument("--run-id", required=True)
        if action == "sync":
            command.add_argument("--worktree", required=True)
            command.add_argument("--workloads", help="Resolved checkpoint-derived manifest; unresolved templates are rejected")
            command.add_argument("--task", default=DEFAULT_TASK)
            command.add_argument("--task-root", default=None, help="local task control root when it is outside the candidate worktree")
            command.add_argument("--gpu", default=DEFAULT_GPU)
            command.add_argument("--idle-memory-mib", type=int, default=64,
                                 help="documented idle driver memory ceiling; still requires no compute processes and zero utilization")
            command.add_argument("--workflow", choices=("candidate", "final", "prepare", "profile"),
                                 help="default: candidate; legacy --profile-kind ncu selects profile")
            command.add_argument("--profile-kind", choices=("none", "nsys", "ncu"), default=None,
                                 help="default: none for candidates, paired NSYS for final, NSYS for prepare")
            command.add_argument("--dry-run", action="store_true", help="validate/freeze snapshot and print plan locally; no SSH or GPU")
            command.add_argument("--no-precision-cache", action="store_true", help="collect fresh baseline and A/A diagnostic for all seeds")
            command.add_argument("--precision-timeout", type=int, default=3600)
            command.add_argument("--kernel-regex", default=None)
            command.add_argument("--ncu-set", choices=("basic", "full"), default="basic")
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--skip-nsys", action="store_true")
            command.add_argument("--profile-id", default=None)
            command.add_argument("--correctness-timeout", type=int, default=1800)
            command.add_argument("--benchmark-timeout", type=int, default=3600)
            command.add_argument("--profile-timeout", type=int, default=600)
        if action == "fetch":
            command.add_argument("--output", required=True)
            command.add_argument("--partial", action="store_true", help="fetch live logs, explicitly without final artifact verification")
    args = parser.parse_args()
    if args.action == "preflight":
        container = json.loads(remote(args, ["docker", "inspect", args.container]))[0]
        info = {"host": remote(args, ["hostname"]), "container": {k: container[k] for k in ("Id", "Image", "State", "Mounts")}, "gpus": remote(args, ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv"]), "compute_apps": remote(args, ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"]), "disk": remote(args, ["df", "-h", "/data"])}
        print(json.dumps(info, indent=2))
    elif args.action == "sync":
        workflow, kind = workflow_options(args)
        if kind == "ncu" and args.container == "kda_test":
            args.container = "kda_profile"
        root = run_root(args)
        with tempfile.TemporaryDirectory(prefix="kda-snapshot-") as temporary:
            manifest = snapshot(args, Path(temporary))
            verify_source(Path(temporary))
            if args.dry_run:
                print(json.dumps({"dry_run": True, "run_id": args.run_id, "workflow": workflow,
                                  "snapshot_files": len(manifest["files_sha256"]),
                                  "workloads_sha256": manifest["files_sha256"]["task/workloads.json"],
                                  "plan": manifest["plan"]}, indent=2))
                return 0
            remote(args, ["python3", "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.mkdir()", root])
            checked(["rsync", "-rlt", "-e", rsync_transport(args), temporary + "/", args.target + ":" + root + "/"], timeout=600)
            print(remote_helper(args, "_verify"))
    elif args.action in {"run", "status", "cancel"}:
        print(remote_helper(args, {"run": "_launch", "status": "_status", "cancel": "_cancel"}[args.action]))
    elif args.action == "fetch":
        state = json.loads(remote_helper(args, "_status"))
        if not args.partial and (state["state"] not in TERMINAL or not state["artifacts_finalized"]):
            raise RuntimeError("run artifacts are not finalized; use --partial for live logs")
        resolve_fetch_identity(args, json.loads(remote_helper(args, "_fetch_identity")))
        output = Path(args.output).resolve()
        validate_fetch_destination(output, args)
        output.mkdir(parents=True, exist_ok=True)
        root = run_root(args)
        sources = [args.target + ":" + root + "/" + name for name in ("artifacts", "manifest.json", "source.diff", "ready.json", "runner.py")]
        excludes = ["--exclude=/artifacts/" + parent + "/baseline-tensors-*/" for parent in REFERENCE_ARTIFACT_PARENTS]
        checked(["rsync", "-rlt", "--checksum", *excludes, "-e", rsync_transport(args)] + sources + [str(output) + "/"], timeout=600)
        if fetch_identity(output) != {key: getattr(args, key) for key in ("run_id", "target", "gateway", "container")}:
            raise RuntimeError("downloaded fetch manifest identity differs from the selected run")
        if not args.partial:
            verify_fetched_artifacts(output)
        print(json.dumps({"output": str(output), "state": state["state"], "verified": not args.partial}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
