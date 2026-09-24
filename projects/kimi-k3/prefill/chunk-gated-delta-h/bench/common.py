"""Standard-library-only manifest, source-isolation, and reporting helpers."""

import ast
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

MODULE = "sglang.kernels.ops.attention.fla.kda"
ENTRYPOINT = Path("python/sglang/kernels/ops/attention/fla/kda.py")
ABI = {
    "q", "k", "v", "g", "beta", "initial_state", "initial_state_indices",
    "cu_seqlens", "A_log", "dt_bias", "lower_bound", "beta_is_raw",
    "use_qk_l2norm_in_kernel", "output_intermediate_states", "track_state",
    "track_chunk_idx",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_workloads(path, allow_unresolved=False):
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported workload schema_version")
    profile = document["model_profile"]
    if profile.get("status") != "RESOLVED":
        if not allow_unresolved or profile.get("status") != "UNRESOLVED":
            raise ValueError("Checkpoint model_profile is UNRESOLVED; configure workloads first")
    else:
        for key in ("num_heads", "local_num_heads", "head_k_dim", "head_v_dim", "attention_tp_size"):
            positive_int(profile.get(key), key)
        if profile["num_heads"] != profile["local_num_heads"] * profile["attention_tp_size"]:
            raise ValueError("TP head count mismatch")
        if profile["head_k_dim"] != profile["head_v_dim"] or profile["head_k_dim"] > 256:
            raise ValueError("Pinned K3 requires equal K/V dimensions <= 256")
        if profile.get("activation_dtype") not in ("bfloat16", "float16"):
            raise ValueError("activation_dtype must be bfloat16 or float16")
        if profile.get("state_dtype") != "bfloat16":
            raise ValueError("This deployment requires bfloat16 SSM state")
        bound = profile.get("gate_lower_bound")
        if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound) or bound >= 0):
            raise ValueError("gate_lower_bound must be null or a finite negative number")
    cases = document["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique ids")
    for case in cases:
        lengths = case["seq_lens"]
        if not lengths:
            raise ValueError(f"{case['id']}: empty seq_lens")
        for length in lengths:
            positive_int(length, "seq_len")
        if case["total_tokens"] != sum(lengths) or case["batch_size"] != len(lengths):
            raise ValueError(f"{case['id']}: packed token/batch count mismatch")
        if sum(lengths) > 16384 or len(lengths) > 128:
            raise ValueError(f"{case['id']}: exceeds deployment token/batch limits")
        for key, allowed in {
            "category": ("correctness", "deployment_grid"),
            "initial_state": ("zero", "random"),
            "state_layout": ("contiguous", "strided"),
            "index_mode": ("identity", "permuted", "padded"),
            "gate_mode": ("model", "standard", "safe"),
        }.items():
            if case.get(key) not in allowed:
                raise ValueError(f"{case['id']}: invalid {key}")
        for key in ("track_state", "output_intermediate_states"):
            if type(case.get(key)) is not bool:
                raise ValueError(f"{case['id']}: {key} must be boolean")
        splits = case.get("continuation_splits")
        if splits is not None:
            if not splits:
                raise ValueError("continuation_splits must be nonempty")
            for split in splits:
                positive_int(split, "continuation_split")
            if len(lengths) != 1 or sum(splits) != sum(lengths):
                raise ValueError(f"{case['id']}: invalid continuation splits")
            if case["category"] != "correctness":
                raise ValueError("Continuation cases belong in the bounded correctness suite")
        if case["category"] == "correctness" and sum(lengths) > 2048:
            raise ValueError("FP32 sequential oracle is bounded to 2048 tokens per case")
    if not any(c["category"] == "correctness" for c in cases):
        raise ValueError("Missing fixed correctness suite")
    if not any(c["category"] == "deployment_grid" for c in cases):
        raise ValueError("Missing deployment grid")
    return document


def source_info(source_root):
    root = Path(source_root).resolve(strict=True)
    entry = root / ENTRYPOINT
    tree = ast.parse(entry.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_kda")
    parameters = {argument.arg for argument in function.args.args + function.args.kwonlyargs}
    if ABI - parameters:
        raise ValueError(f"chunk_kda ABI missing {sorted(ABI - parameters)}")
    # Hash all SGLang Python source, including imported helpers, not just kda.py.
    files = sorted((root / "python/sglang").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256(path)}\n".encode())
    git_head = None
    toplevel = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    # An archived .baseline can sit inside the candidate worktree. Its parent
    # repository HEAD does not identify the archived source being measured.
    if toplevel.returncode == 0 and toplevel.stdout.strip() and Path(toplevel.stdout.strip()).resolve() == root:
        git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if git.returncode == 0:
            git_head = git.stdout.strip()
    return {
        "root": str(root), "git_head": git_head,
        "sglang_python_tree_sha256": digest.hexdigest(), "python_file_count": len(files),
        "entrypoint_sha256": sha256(entry), "abi_parameters": sorted(parameters),
    }


def worker_environment(source_root):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(source_root).resolve() / "python")
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def assert_origin(filename, expected):
    actual = Path(filename).resolve()
    expected = Path(expected).resolve()
    if actual != expected:
        raise RuntimeError(f"Source isolation violation: loaded {actual}, expected {expected}")
    return str(actual)


class SourceTreeFinder:
    """Resolve the complete sglang namespace without editable-install fallback."""

    def __init__(self, source_root):
        self.python_root = Path(source_root).resolve() / "python"
        self.package_root = self.python_root / "sglang"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang" and not fullname.startswith("sglang."):
            return None
        # Ignore package __path__ additions and other meta-path finders. Search
        # only the corresponding directory in the selected source tree.
        parent = self.python_root.joinpath(*fullname.split(".")[:-1])
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(parent)], target)
        if spec is None:
            # Raising, rather than returning None, prevents PEP 660 finders
            # from filling missing generated files such as sglang._version.
            raise ModuleNotFoundError(f"{fullname} is absent from selected source tree {self.package_root}", name=fullname)
        locations = list(spec.submodule_search_locations or [])
        if spec.origin is not None:
            locations.append(spec.origin)
        if not locations or any(not Path(location).resolve().is_relative_to(self.package_root) for location in locations):
            raise ImportError(f"Source isolation violation while resolving {fullname}: {locations}")
        return spec


def isolate_sglang_imports(source_root):
    # Cached modules bypass meta-path finders altogether. This harness requires
    # fresh workers; reject preloaded SGLang instead of silently reusing it.
    preloaded = [name for name in sys.modules if name == "sglang" or name.startswith("sglang.")]
    if preloaded:
        raise RuntimeError(f"Source isolation requires a fresh worker; SGLang already imported: {preloaded[0]}")
    finder = SourceTreeFinder(source_root)
    sys.meta_path.insert(0, finder)
    return finder


def load_kernel(source_root):
    """Called in a fresh worker only, after manifest validation."""
    root = Path(source_root).resolve()
    isolate_sglang_imports(root)
    sys.path.insert(0, str(root / "python"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU result was produced")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (10, 3):
        raise RuntimeError(f"GB300 compute capability (10, 3) is required; detected {capability}. No GPU result was produced")
    package = importlib.import_module("sglang")
    module = importlib.import_module(MODULE)
    origins = {
        "sglang": assert_origin(package.__file__, root / "python/sglang/__init__.py"),
        "chunk_kda": assert_origin(inspect.getsourcefile(module.chunk_kda), root / ENTRYPOINT),
    }
    for name, imported in tuple(sys.modules.items()):
        filename = getattr(imported, "__file__", None)
        if name.startswith("sglang.") and filename:
            if not Path(filename).resolve().is_relative_to(root / "python/sglang"):
                raise RuntimeError(f"Foreign imported dependency {name}: {filename}")
    if ABI - set(inspect.signature(module.chunk_kda).parameters):
        raise RuntimeError("Runtime chunk_kda ABI differs from static preflight")
    dependencies = {}
    for name in ("torch", "triton", "sgl_kernel", "flashinfer", "numpy"):
        imported = sys.modules.get(name)
        if imported is not None:
            dependencies[name] = {"file": getattr(imported, "__file__", None), "version": str(getattr(imported, "__version__", "unknown"))}
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    origins.update({"dependencies": dependencies, "python": sys.version, "cuda": torch.version.cuda,
                    "device": properties.name, "capability": list(capability),
                    "device_uuid": str(getattr(properties, "uuid", "unavailable"))})
    return torch, module.chunk_kda, origins


def selected_cases(document, category):
    return [case for case in document["cases"] if case["category"] == category]


def run_worker(script, source_root, arguments, report):
    command = [sys.executable, str(Path(__file__).with_name(script)), "--source-root", str(Path(source_root).resolve()), *arguments, "--report", str(Path(report).resolve())]
    try:
        result = subprocess.run(command, cwd=source_root, env=worker_environment(source_root), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as error:
        value = json.loads(Path(report).read_text()) if Path(report).exists() else {}
        value.update(status="failed", error="Worker exceeded the 1800 second limit", process_returncode=124)
        for name in ("stdout", "stderr"):
            content = getattr(error, name, None) or ""
            value[name + "_tail"] = (content.decode(errors="replace") if isinstance(content, bytes) else content)[-4000:]
        write_json(report, value)
        return value
    value = json.loads(Path(report).read_text()) if Path(report).exists() else {"status": "failed", "error": "Worker did not produce a report"}
    value["process_returncode"] = result.returncode
    if result.returncode != 0:
        value["status"] = "failed"
    value["stdout_tail"] = result.stdout[-4000:]
    value["stderr_tail"] = result.stderr[-4000:]
    write_json(report, value)
    return value
