#!/usr/bin/env python3
"""Resolve KDA shapes from checkpoint metadata; standard library only, no GPU."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


LENGTHS = (1, 63, 64, 65, 127, 128, 129, 512, 1024, 4096, 8192, 16384)
BATCHES = (1, 2, 8, 32, 128)
TASK_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def dtype_name(value, name):
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    if not isinstance(value, str):
        raise ValueError(f"{name} is unresolved; supply checkpoint dtype metadata")
    value = value.removeprefix("torch.")
    value = aliases.get(value, value)
    if value not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"unsupported {name}: {value!r}")
    return value


def resolve_profile(config, deployment, checkpoint_sha256, activation_dtype=None):
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a JSON object")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be a JSON object")
    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise ValueError("linear_attn_config is missing or invalid")
    prefill = deployment["prefill"]
    tp = positive_int(prefill.get("attention_tp_size"), "attention_tp_size")
    heads = positive_int(linear.get("num_heads"), "linear_attn_config.num_heads")
    k = positive_int(linear.get("head_dim"), "linear_attn_config.head_dim")
    v = positive_int(text.get("v_head_dim"), "v_head_dim")
    conv = positive_int(linear.get("short_conv_kernel_size"), "short_conv_kernel_size")
    if heads % tp:
        raise ValueError(f"num_heads={heads} is not divisible by attention_tp_size={tp}")
    if k > 256:
        raise ValueError("pinned Triton recurrent state kernel requires head_dim <= 256")
    if k != v:
        raise ValueError("v_head_dim != head_dim conflicts with the pinned square KimiLinear state pool")
    bound = linear.get("gate_lower_bound")
    if bound is not None and (
        isinstance(bound, bool) or not isinstance(bound, (int, float))
        or not math.isfinite(bound) or bound >= 0
    ):
        raise ValueError("gate_lower_bound must be null or a finite negative number")
    full_rank = linear.get("use_full_rank_gate", False)
    if type(full_rank) is not bool:
        raise ValueError("use_full_rank_gate must be a boolean")
    checkpoint_dtype = text.get("dtype", text.get("torch_dtype"))
    if checkpoint_dtype is None:
        checkpoint_dtype = config.get("dtype", config.get("torch_dtype"))
    configured_dtype = prefill.get("activation_dtype")
    chosen_dtype = activation_dtype or configured_dtype or checkpoint_dtype
    dtype = dtype_name(chosen_dtype, "activation_dtype")
    if dtype == "float32":
        raise ValueError("float32 checkpoint dtype does not establish the runtime activation dtype; pass --activation-dtype")
    state_dtype = dtype_name(prefill.get("state_dtype"), "state_dtype")
    return {
        "status": "RESOLVED",
        "resolution_scope": "checkpoint metadata plus supplied deployment; runtime backend/traffic unverified",
        "checkpoint_sha256": checkpoint_sha256,
        "config_scope": "text_config" if "text_config" in config else "root",
        "num_heads": heads,
        "local_num_heads": heads // tp,
        "head_k_dim": k,
        "head_v_dim": v,
        "attention_tp_size": tp,
        "activation_dtype": dtype,
        "activation_dtype_basis": "explicit_cli" if activation_dtype else ("deployment" if configured_dtype else "checkpoint_metadata"),
        "state_dtype": state_dtype,
        "gate_lower_bound": bound,
        "gate_lower_bound_basis": "checkpoint" if "gate_lower_bound" in linear else "source_default_null_after_checkpoint_read",
        "use_full_rank_gate": full_rank,
        "short_conv_kernel_size": conv,
        "a_log_dtype": "float32",
        "dt_bias_dtype": "float32",
        "beta_dtype": "float32",
        "beta_is_raw": False,
        "use_qk_l2norm_in_kernel": True,
        "state_axis_order": ["slot", "head", "value", "key"],
        "kernel_chunk_size": 64,
    }


def build_cases(deployment):
    prefill = deployment["prefill"]
    max_tokens = min(
        positive_int(prefill.get("chunked_prefill_size"), "chunked_prefill_size"),
        positive_int(prefill.get("max_prefill_tokens"), "max_prefill_tokens"),
    )
    max_batch = positive_int(prefill.get("max_running_requests"), "max_running_requests")
    cases = []

    def add(case_id, lens, category="deployment_grid", **variants):
        if len(lens) > max_batch or sum(lens) > max_tokens:
            return
        case = {
            "id": case_id, "category": category, "seq_lens": lens,
            "batch_size": len(lens), "total_tokens": sum(lens),
            "initial_state": "zero", "state_layout": "contiguous",
            "index_mode": "identity", "track_state": False,
            "output_intermediate_states": False,
            "continuation_splits": None, "gate_mode": "model",
        }
        case.update(variants)
        cases.append(case)

    for batch in BATCHES:
        for length in LENGTHS:
            add(f"uniform_b{batch}_s{length}", [length] * batch)
    add("ragged_boundary_b8", [1, 63, 64, 65, 127, 128, 129, 512])
    add("ragged_mixed_b2", [8192, 4096])
    add("resumed_chunk_s16384", [16384], initial_state="random")
    for batch in (32, 128):
        add(f"ragged_short_b{batch}", [LENGTHS[i % 7] for i in range(batch)])
    variants = [
        ("nonzero_state", {"initial_state": "random"}),
        ("permuted_slots", {"initial_state": "random", "index_mode": "permuted"}),
        ("strided_slots", {"initial_state": "random", "state_layout": "strided", "index_mode": "permuted"}),
        ("padded_state_index", {"initial_state": "random", "state_layout": "strided", "index_mode": "padded"}),
        ("fp32_track", {"initial_state": "random", "state_layout": "strided", "track_state": True, "output_intermediate_states": True}),
        ("standard_gate", {"gate_mode": "standard", "initial_state": "random"}),
        ("safe_gate", {"gate_mode": "safe", "initial_state": "random", "synthetic_gate_lower_bound": -5.0}),
    ]
    for name, settings in variants:
        add(name, [129, 257], category="correctness", **settings)
    add("zero_state_boundaries", [1, 63, 64, 65, 127, 128, 129], category="correctness")
    for splits in ([63, 1, 65], [64, 65], [1024, 1024]):
        add("continuation_" + "_".join(map(str, splits)), [sum(splits)],
            category="correctness", initial_state="random", continuation_splits=splits)
    return cases


def build_manifest(deployment, profile, deployment_sha256):
    return {
        "schema_version": 1,
        "source_commit": deployment["source_commit"],
        "provenance": "synthetic deployment-bounded grid; not captured production traffic",
        "deployment_sha256": deployment_sha256,
        "model_profile": profile,
        "deployment": deployment["prefill"],
        "required_runtime_confirmation": deployment["backend_resolution"]["required_runtime_evidence"],
        "cases": build_cases(deployment),
    }


def atomic_json(path, data):
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, default=TASK_ROOT / "deployment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-profile-output", type=Path)
    parser.add_argument("--activation-dtype", choices=("bfloat16", "float16"))
    args = parser.parse_args(argv)
    try:
        outputs = [args.output] + ([args.model_profile_output] if args.model_profile_output else [])
        inputs = {args.checkpoint_config.resolve(), args.deployment.resolve()}
        if any(path.resolve() in inputs for path in outputs):
            raise ValueError("output must not overwrite checkpoint or deployment input")
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError("workload output and model profile output must differ")
        checkpoint_raw = args.checkpoint_config.read_bytes()
        deployment_raw = args.deployment.read_bytes()
        deployment = json.loads(deployment_raw)
        if deployment.get("schema_version") != 1:
            raise ValueError("unsupported deployment schema_version")
        profile = resolve_profile(json.loads(checkpoint_raw), deployment,
                                  hashlib.sha256(checkpoint_raw).hexdigest(), args.activation_dtype)
        profile["checkpoint_config_path"] = str(args.checkpoint_config.resolve())
        manifest = build_manifest(deployment, profile, hashlib.sha256(deployment_raw).hexdigest())
        if args.model_profile_output:
            atomic_json(args.model_profile_output, profile)
        atomic_json(args.output, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"configuration error: {exc}\n")
    print(f"Resolved {len(manifest['cases'])} synthetic cases; H/rank={profile['local_num_heads']}, K={profile['head_k_dim']}, V={profile['head_v_dim']}. Runtime backend remains unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
