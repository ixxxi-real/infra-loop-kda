"""One source root per fresh process; accuracy only, no timing.

Reuses bench/common.py source isolation and bench/runtime.py input and
recurrence semantics unchanged. Nothing here measures performance.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import harness_hashes, load_kernel, load_workloads, sha256, source_info, write_json  # noqa: E402
from runtime import TOLERANCE, TRACK_TOLERANCE, Inputs, capture_case, recurrence, trial_seed  # noqa: E402

import metrics  # noqa: E402
from baseline_cache import add_driver_version, numerical_environment  # noqa: E402


def track_rows_for(case):
    """Rows the kernel is asked to write, mirroring runtime.Inputs exactly.

    Derived from the case alone so it is available for every case, not only the
    ones that build a fresh Inputs. Verified against the real Inputs when one
    exists. Rows left at the 17.0 sentinel are excluded from error statistics so
    untouched padding cannot dilute RMS.
    """
    if not case["track_state"]:
        return []
    indices = [(length - 1) // 64 for length in case["seq_lens"]]
    if case["batch_size"] > 1:
        indices[-1] = -1
    return [index for index, chunk in enumerate(indices) if chunk >= 0]


def harness_track_rows(inputs):
    """Rows the harness actually asks the kernel to write.

    runtime.Inputs builds track_indices_cpu unconditionally but sets track = None
    when the case disables tracking. With no scratch tensor there are no tracked
    rows, whatever the indices say. Track-enabled cases keep the valid indices.
    """
    if inputs.track is None:
        return []
    return [index for index, chunk in enumerate(inputs.track_indices_cpu) if chunk >= 0]


def inactive_rows(count, active):
    return [index for index in range(count) if index not in set(active)]


def fields(torch, snapshot, reference, case, active, track_rows, prefix, suffix, mixed=False):
    """Compare every returned field; state and track restricted to active rows."""
    rows = [metrics.tensor_metrics(torch, snapshot["output"], reference["output" + suffix],
                                  label=f"{prefix}.output", allow_mixed_dtype=mixed)]
    if active:
        rows.append(metrics.tensor_metrics(torch, snapshot["storage"], reference["storage" + suffix],
                                          label=f"{prefix}.state", rows=active,
                                          scope="active state slots", allow_mixed_dtype=mixed))
    if case["output_intermediate_states"] and "intermediates" + suffix in reference:
        rows.append(metrics.tensor_metrics(torch, snapshot["intermediates"], reference["intermediates" + suffix],
                                          label=f"{prefix}.intermediates", allow_mixed_dtype=mixed))
    if track_rows and reference.get("track" + suffix) is not None:
        rows.append(metrics.tensor_metrics(torch, snapshot["track"], reference["track" + suffix],
                                          label=f"{prefix}.track_fp32", rows=track_rows,
                                          scope="active FP32 tracked rows; sentinel 17.0 rows excluded",
                                          allow_mixed_dtype=mixed))
    return rows


def continuation_fields(torch, actual, reference, keys, active, prefix, mixed=False):
    """Split-aware comparison; state restricted to active slots, rest exact."""
    output_key, storage_key = keys
    rows = [metrics.tensor_metrics(torch, actual["cont_output"], reference[output_key],
                                  label=f"{prefix}.continuation_output", allow_mixed_dtype=mixed)]
    if active:
        rows.append(metrics.tensor_metrics(torch, actual["cont_storage"], reference[storage_key],
                                          label=f"{prefix}.continuation_state", rows=active,
                                          scope="active state slots", allow_mixed_dtype=mixed))
    return rows


def prepare(torch, inputs, role):
    """Isolate the changed preparation, before any public call.

    Clone immediately: the public call writes its operator result through the
    contiguous V buffer this returns.
    """
    l2norm = sys.modules["sglang.kernels.ops.attention.fla.l2norm"]
    fused = getattr(l2norm, "kda_prepare_qkv_fwd", None)
    if role == "candidate" and fused is not None:
        q, k, v = fused(inputs.q, inputs.k_tensor, inputs.v)
        path = "kda_prepare_qkv_fwd(q, k, v)"
    else:
        if role != "candidate" and fused is not None:
            raise RuntimeError("baseline source unexpectedly exposes kda_prepare_qkv_fwd")
        q = l2norm.l2norm_fwd(inputs.q.contiguous())
        k = l2norm.l2norm_fwd(inputs.k_tensor.contiguous())
        v = inputs.v.contiguous()
        path = "l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous()"
    torch.cuda.synchronize()
    values = {"prep_q": q.detach().cpu().clone(), "prep_k": k.detach().cpu().clone(),
              "prep_v": v.detach().cpu().clone()}
    del q, k, v
    return path, values


def prep_reference(torch, inputs, values, role):
    """Independent CPU FP32 normalization, then the activation-dtype boundary."""
    rows = []
    for name, source in (("prep_q", inputs.q), ("prep_k", inputs.k_tensor)):
        exact = source.detach().cpu().float()
        exact = exact / (exact.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        rows.append(metrics.tensor_metrics(torch, values[name].float(), exact,
                                          label=f"{name}.{role}_vs_fp32_reference",
                                          scope="FP32 normalization before activation-dtype cast"))
        rows.append(metrics.tensor_metrics(torch, values[name], exact.to(inputs.dtype),
                                          label=f"{name}.{role}_vs_reference_cast",
                                          scope="activation-dtype rounding of the FP32 reference"))
    rows.append(metrics.tensor_metrics(torch, values["prep_v"], inputs.original_v.detach().cpu(),
                                       label=f"prep_v.{role}_vs_input",
                                       scope="contiguous V must equal its input"))
    return rows


def continuation(torch, kernel, inputs, splits):
    """Split-aware recurrence, then the same splits through the public call."""
    expected = recurrence(inputs, splits)
    inputs.reset()
    outputs, offset = [], 0
    for length in splits:
        cu = torch.tensor([0, length], dtype=torch.int64, device="cuda")
        result = inputs.invoke(kernel, offset, offset + length, cu, track=False)
        outputs.append((result[0] if isinstance(result, tuple) else result).clone())
        offset += length
    torch.cuda.synchronize()
    return ({"cont_expected_output": expected["output"], "cont_expected_storage": expected["storage"]},
            {"cont_output": torch.cat(outputs, dim=1).detach().cpu(),
             "cont_storage": inputs.storage.detach().cpu()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--mode", choices=("export", "compare", "identity"), required=True)
    parser.add_argument("--role", choices=("baseline", "baseline_repeat", "candidate"), required=True)
    parser.add_argument("--reference-report")
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--expect-gpu-uuid")
    args = parser.parse_args()
    scratch = Path(args.scratch).resolve()
    report = {"status": "failed", "mode": args.mode, "role": args.role, "trial": args.trial,
              "rows": [], "harness_sha256": harness_hashes(), "precision_sha256": metrics.script_hashes()}
    try:
        document = load_workloads(args.workloads)
        profile = document["model_profile"]
        seed = trial_seed(args.trial)
        report.update(source=source_info(args.source_root), workloads_sha256=sha256(args.workloads),
                      trial_seed=seed, model_profile=profile,
                      tolerance_grid={"atol": metrics.ATOL, "rtol": metrics.RTOL},
                      original_acceptance={"end_to_end": dict(TOLERANCE), "track": dict(TRACK_TOLERANCE)})
        reference_rows = None
        if args.mode == "compare":
            reference = json.loads(Path(args.reference_report).read_text())
            if reference["status"] != "passed" or reference["workloads_sha256"] != report["workloads_sha256"]:
                raise ValueError("baseline reference report is not a passed matching-workload run")
            if reference["trial_seed"] != seed:
                raise ValueError("baseline reference belongs to a different trial seed")
            reference_rows = {row["id"]: row for row in reference["rows"]}
            report["baseline_source"] = reference["source"]
            report["reference_report_sha256"] = sha256(args.reference_report)
        numerical_options = numerical_environment()
        torch, kernel, origins = load_kernel(args.source_root)
        origins["numerical_environment"] = numerical_options
        report["runtime"] = origins
        add_driver_version(origins)
        if args.expect_gpu_uuid:
            if not metrics.same_device_uuid(origins["device_uuid"], args.expect_gpu_uuid):
                raise RuntimeError(f"device UUID {origins['device_uuid']} is not the locked {args.expect_gpu_uuid}")
            # Both spellings retained; the check compares identity, not spelling.
            report["device_uuid_checked"] = {"reported_by_torch": origins["device_uuid"],
                                             "locked_by_runner": args.expect_gpu_uuid}
        if args.mode == "identity":
            report["status"] = "passed"
            return 0
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        # The diagnostic reports a verdict about the unchanged acceptance gate, so
        # its copy of that gate must still be the harness's own values.
        if (metrics.ACCEPTANCE["atol"], metrics.ACCEPTANCE["rtol"], metrics.TOLERANCE_RELATIVE_RMS) != (
                TOLERANCE["atol"], TOLERANCE["rtol"], TOLERANCE["relative_rms"]):
            raise RuntimeError(f"acceptance tolerance drifted from the harness: {TOLERANCE}")
        if metrics.TRACK_ACCEPTANCE != dict(TRACK_TOLERANCE):
            raise RuntimeError(f"track tolerance drifted from the harness: {TRACK_TOLERANCE}")
        report["metrics_self_check"] = metrics.self_check(torch)
        scratch.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for position, case in enumerate(document["cases"]):
                oracle_file = scratch / f"oracle-{position:04d}.pt"
                reference_file = scratch / f"{position:04d}.pt"
                row = {"id": case["id"], "category": case["category"], "status": "failed", "comparisons": {}}
                try:
                    captured, snapshot = capture_case(torch, kernel, profile, case, seed=seed)
                    row.update({key: captured[key] for key in ("input_sha256", "inputs",
                                                               "inactive_storage_exact", "active_storage_indices")})
                    active, correctness = captured["active_storage_indices"], case["category"] == "correctness"
                    extras = {}
                    # Derived from the case, so tracked-row masking applies to every
                    # case, not only the ones that build a fresh Inputs below.
                    track_rows = track_rows_for(case)
                    row["active_track_rows"] = track_rows
                    if correctness:
                        inputs = Inputs(torch, profile, case, seed=seed)
                        if inputs.fingerprint() != row["input_sha256"]:
                            raise AssertionError("paired input fingerprint differs from the captured run")
                        if track_rows != harness_track_rows(inputs):
                            raise AssertionError(
                                f"derived tracked rows {track_rows} differ from the harness inputs "
                                f"{harness_track_rows(inputs)}")
                        row["prep_path"], prep_values = prepare(torch, inputs, args.role)
                        # Role-qualified: the A/A and A/B workers both produce this
                        # group, and merging them would mix the two kinds of evidence.
                        row["comparisons"][f"prep_vs_fp32_reference_{args.role}"] = prep_reference(
                            torch, inputs, prep_values, args.role)
                        inputs.reset()
                        computed = recurrence(inputs)
                        extras.update({name + "_oracle": value for name, value in computed.items() if value is not None})
                        extras.update(prep_values)
                        splits = case.get("continuation_splits")
                        if splits:
                            inputs.reset()
                            expected_split, actual_split = continuation(torch, kernel, inputs, splits)
                            extras.update(expected_split)
                            extras.update(actual_split)
                        del inputs
                    if args.mode == "export":
                        torch.save(snapshot, reference_file)
                        row.update(reference_file=reference_file.name, reference_sha256=sha256(reference_file))
                        if correctness:
                            torch.save(extras, oracle_file)
                            row.update(oracle_file=oracle_file.name, oracle_sha256=sha256(oracle_file))
                            row["comparisons"]["baseline_vs_oracle"] = fields(
                                torch, snapshot, extras, case, active, track_rows,
                                "baseline_vs_oracle", "_oracle", mixed=True)
                            if "cont_output" in extras:
                                row["comparisons"]["baseline_continuation_vs_oracle"] = continuation_fields(
                                    torch, extras, extras,
                                    ("cont_expected_output", "cont_expected_storage"), active,
                                    "baseline_continuation_vs_oracle", mixed=True)
                                row["continuation_inactive_exact"] = metrics.exactness(
                                    torch, extras["cont_storage"][inactive_rows(extras["cont_storage"].shape[0], active)],
                                    extras["cont_expected_storage"][inactive_rows(extras["cont_storage"].shape[0], active)],
                                    "continuation inactive storage and strided padding")
                    else:
                        baseline_row = reference_rows[case["id"]]
                        if row["input_sha256"] != baseline_row["input_sha256"] or row["inputs"] != baseline_row["inputs"]:
                            raise AssertionError("baseline/candidate inputs differ")
                        if sha256(reference_file) != baseline_row["reference_sha256"]:
                            raise AssertionError("baseline reference tensor hash mismatch")
                        expected = torch.load(reference_file, map_location="cpu", weights_only=True)
                        row["comparisons"][args.role + "_vs_baseline"] = fields(
                            torch, snapshot, {name + "_b": value for name, value in expected.items()},
                            case, active, track_rows, args.role + "_vs_baseline", "_b")
                        row["reference_sha256"] = baseline_row["reference_sha256"]
                        del expected
                        if correctness:
                            if sha256(oracle_file) != baseline_row["oracle_sha256"]:
                                raise AssertionError("shared oracle hash mismatch")
                            shared = torch.load(oracle_file, map_location="cpu", weights_only=True)
                            row["oracle_sha256"] = baseline_row["oracle_sha256"]
                            row["comparisons"][args.role + "_vs_oracle"] = fields(
                                torch, snapshot, shared, case, active, track_rows,
                                args.role + "_vs_oracle", "_oracle", mixed=True)
                            row["comparisons"][f"normalization_only_{args.role}_vs_baseline"] = [
                                metrics.tensor_metrics(torch, extras[name], shared[name],
                                                       label=f"{name}.{args.role}_vs_baseline",
                                                       scope="preparation output, before the public call")
                                for name in ("prep_q", "prep_k", "prep_v")]
                            # The shared oracle must be the one this side recomputed, or the
                            # two implementations are not being judged against the same reference.
                            if not all(metrics.byte_hash(torch, extras[name]) == metrics.byte_hash(torch, shared[name])
                                       for name in shared if name.endswith("_oracle")):
                                raise AssertionError("recomputed oracle differs from the shared oracle file")
                            row["oracle_recompute_identical"] = True
                            if "cont_output" in extras:
                                row["comparisons"][args.role + "_continuation_vs_oracle"] = continuation_fields(
                                    torch, extras, shared,
                                    ("cont_expected_output", "cont_expected_storage"), active,
                                    args.role + "_continuation_vs_oracle", mixed=True)
                                row["comparisons"][args.role + "_continuation_vs_baseline"] = continuation_fields(
                                    torch, extras, shared, ("cont_output", "cont_storage"), active,
                                    args.role + "_continuation_vs_baseline")
                                unwritten = inactive_rows(extras["cont_storage"].shape[0], active)
                                row["continuation_inactive_exact"] = metrics.exactness(
                                    torch, extras["cont_storage"][unwritten], shared["cont_storage"][unwritten],
                                    "continuation inactive storage and strided padding")
                            del shared
                    del snapshot
                    # A failed metric record must fail its row, not be summarized away.
                    records = [record for group in row["comparisons"].values() for record in group]
                    broken = [record["label"] for record in records if record.get("status") == "failed"]
                    if broken:
                        raise AssertionError(f"metric records failed: {broken}")
                    unchanged = [row[name] for name in ("continuation_inactive_exact",) if name in row]
                    if any(not record["exactly_unchanged"] for record in unchanged):
                        raise AssertionError("slots that must remain unchanged were written")
                    row["comparison_field_count"] = len(records)
                    del extras
                    row["status"] = "passed"
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                report["rows"].append(row)
                write_json(args.report, report)
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["rows"]) else "failed"
        if source_info(args.source_root)["sglang_python_tree_sha256"] != report["source"]["sglang_python_tree_sha256"]:
            raise RuntimeError("source changed during worker execution")
        if harness_hashes() != report["harness_sha256"] or metrics.script_hashes() != report["precision_sha256"]:
            raise RuntimeError("harness or precision scripts changed during worker execution")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        write_json(args.report, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
