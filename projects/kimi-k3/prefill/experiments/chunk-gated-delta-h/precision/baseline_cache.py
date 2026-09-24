"""Bounded, hash-checked baseline/A/A cache. No tensor libraries are imported."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import sha256, write_json
import metrics

SCHEMA = 1
REPORTS = {"baseline": "baseline.export.json", "baseline_repeat": "baseline_repeat.compare.json"}
# Exact names only: numerical/compiler options read by the frozen FLA source
# plus framework TF32 controls. Never snapshot secrets or per-run cache paths.
NUMERICAL_ENVIRONMENT = (
    "FLA_USE_FAST_OPS", "SGLANG_GDN_CHUNK_H_BV", "SGLANG_GDN_CHUNK_H_NUM_WARPS",
    "SGLANG_GDN_CHUNK_H_NUM_STAGES", "GDN_RECOMPUTE_SUPPRESS_LEVEL",
    "FLA_COMPILER_MODE", "FLA_CI_ENV", "FLA_CACHE_RESULTS", "FLA_USE_CUDA_GRAPH",
    "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_F32_DEFAULT",
)


def numerical_environment():
    return {name: os.environ.get(name) for name in NUMERICAL_ENVIRONMENT}


def runtime_identity(runtime):
    """Exclude source-root paths, which legitimately differ between A and B."""
    identity = {name: runtime[name] for name in
                ("python", "cuda", "device", "capability", "dependencies", "driver_version",
                 "numerical_environment")}
    identity["device_uuid"] = metrics.normalize_device_uuid(runtime["device_uuid"])
    if not identity["device_uuid"] or identity["device_uuid"] == "unavailable":
        raise ValueError("a real GPU UUID is required for baseline caching")
    return identity


def add_driver_version(runtime):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30)
    for line in result.stdout.splitlines():
        uuid, driver = (part.strip() for part in line.split(",", 1))
        if metrics.same_device_uuid(uuid, runtime["device_uuid"]):
            runtime["driver_version"] = driver
            return
    raise RuntimeError("nvidia-smi did not identify the active GPU driver")


def cache_identity(source, workloads, harness, precision, runtime):
    return {"schema_version": SCHEMA,
            "baseline_source_sha256": source["sglang_python_tree_sha256"],
            "baseline_entrypoint_sha256": source["entrypoint_sha256"],
            "workloads_sha256": workloads, "harness_sha256": harness,
            "precision_sha256": precision, "runtime": runtime_identity(runtime)}


@contextmanager
def open_cache(directory, identity):
    """Keep only one source/runtime identity and at most three seed entries.

    The advisory lock also protects standalone callers outside the GPU runner.
    Never follow a cache-entry symlink when validating or replacing its files.
    """
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = root / "entries"
        if entries.is_symlink():
            raise ValueError("cache entries directory must not be a symlink")
        try:
            old_identity = json.loads((root / "identity.json").read_text())
        except (OSError, ValueError):
            old_identity = None
        reset = old_identity != identity
        if reset:
            if entries.exists():
                shutil.rmtree(entries)
            write_json(root / "identity.json", identity)
        entries.mkdir(exist_ok=True)
        for pending in entries.glob("trial-*.pending"):
            if pending.is_symlink():
                raise ValueError("pending cache entry must not be a symlink")
            shutil.rmtree(pending)
        yield BaselineCache(entries, identity, reset)


class BaselineCache:
    def __init__(self, entries, identity, reset):
        self.entries, self.identity, self.reset = entries, identity, reset

    def seed_path(self, trial):
        if trial not in (0, 1, 2):
            raise ValueError("baseline cache supports the three declared trials only")
        return self.entries / f"trial-{trial:02d}"

    def load(self, trial, validate):
        """Return (entry, reports, miss reason); corrupt entries are never used."""
        entry = self.seed_path(trial)
        try:
            if entry.is_symlink():
                raise ValueError("cache entry must not be a symlink")
            manifest = json.loads((entry / "manifest.json").read_text())
            if manifest["identity"] != self.identity or manifest["trial"] != trial:
                raise ValueError("cache manifest identity or trial mismatch")
            reports = {}
            for role, name in REPORTS.items():
                path = entry / name
                if path.is_symlink() or sha256(path) != manifest["reports_sha256"][role]:
                    raise ValueError(f"cached {role} report hash mismatch")
                report = json.loads(path.read_text())
                validate(report)
                if report["role"] != role or report["trial"] != trial:
                    raise ValueError("cached worker role or trial mismatch")
                if report["mode"] != ("export" if role == "baseline" else "compare"):
                    raise ValueError("cached worker mode mismatch")
                if runtime_identity(report["runtime"]) != self.identity["runtime"]:
                    raise ValueError("cached worker device/runtime mismatch")
                if any(row.get("status") != "passed" for row in report["rows"]):
                    raise ValueError("cached worker contains failed rows")
                reports[role] = report
            baseline, repeat = reports["baseline"], reports["baseline_repeat"]
            if repeat["reference_report_sha256"] != manifest["reports_sha256"]["baseline"]:
                raise ValueError("cached A/A report references a different export")
            tensors = {}
            for position, (row, repeated) in enumerate(zip(baseline["rows"], repeat["rows"])):
                for name in ("reference", "oracle"):
                    if name == "oracle" and row["category"] != "correctness":
                        continue
                    filename = f"{position:04d}.pt" if name == "reference" else f"oracle-{position:04d}.pt"
                    path = entry / filename
                    if row[name + "_file"] != filename or path.is_symlink():
                        raise ValueError("cached tensor filename mismatch or symlink")
                    digest = sha256(path)
                    if digest != row[name + "_sha256"] or digest != repeated[name + "_sha256"]:
                        raise ValueError(f"cached {name} tensor hash mismatch")
                    tensors[filename] = digest
                if row["input_sha256"] != repeated["input_sha256"] or row["inputs"] != repeated["inputs"]:
                    raise ValueError("cached A/A inputs differ from baseline export")
            if manifest["tensors_sha256"] != tensors:
                raise ValueError("cached tensor manifest is incomplete or mismatched")
            return entry, reports, None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError) as error:
            return None, None, f"{type(error).__name__}: {error}"

    def publish(self, trial, scratch, export, repeat, validate):
        """Publish only complete, independently revalidated baseline evidence."""
        entry = self.seed_path(trial)
        pending = entry.with_name(entry.name + ".pending")
        for path in (pending, entry):
            if path.is_symlink():
                raise ValueError("cache entry must not be a symlink")
            if path.exists():
                shutil.rmtree(path)
        shutil.move(str(scratch), str(pending))
        for source, role in ((export, "baseline"), (repeat, "baseline_repeat")):
            shutil.copyfile(source, pending / REPORTS[role])
        write_json(pending / "manifest.json", {
            "identity": self.identity, "trial": trial,
            "reports_sha256": {role: sha256(pending / name) for role, name in REPORTS.items()},
            "tensors_sha256": {path.name: sha256(path) for path in sorted(pending.glob("*.pt"))}})
        pending.rename(entry)
        verified, _, reason = self.load(trial, validate)
        if verified is None:
            shutil.rmtree(entry)
            raise RuntimeError(f"new baseline cache failed validation: {reason}")
        return entry
