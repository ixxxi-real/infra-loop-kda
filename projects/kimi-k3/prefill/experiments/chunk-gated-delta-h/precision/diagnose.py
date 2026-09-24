"""Accuracy-only A/A and A/B diagnostic driver. Measures no performance.

Per seed, in order, each in a fresh process so SGLang modules never mix:
  1. baseline  export   -> reference tensors + shared FP32 oracle
  2. baseline  compare  -> baseline_repeat_vs_baseline (A/A noise floor)
  3. candidate compare  -> candidate_vs_baseline (A/B)
With --baseline-cache, reuse hash-verified baseline tensors and A/A statistics
for the same baseline, harness, workloads, device and runtime. The cache keeps
only one identity and three seeds. Otherwise bulk tensors are deleted per seed.
"""

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from common import harness_hashes, load_workloads, selected_cases, sha256, source_info, worker_environment, write_json  # noqa: E402
from runtime import TOLERANCE, TRACK_TOLERANCE, trial_seed  # noqa: E402

import metrics  # noqa: E402
from baseline_cache import cache_identity, open_cache, runtime_identity  # noqa: E402

TRIALS = (0, 1, 2)
# Every group is role-qualified: the A/A and A/B workers each produce a
# preparation and continuation group, and merging them would mix the two kinds
# of evidence under one name.
GROUPS = ("baseline_vs_oracle", "baseline_repeat_vs_baseline", "baseline_repeat_vs_oracle",
          "candidate_vs_baseline", "candidate_vs_oracle",
          "normalization_only_baseline_repeat_vs_baseline", "normalization_only_candidate_vs_baseline",
          "prep_vs_fp32_reference_baseline", "prep_vs_fp32_reference_baseline_repeat",
          "prep_vs_fp32_reference_candidate",
          "baseline_continuation_vs_oracle",
          "baseline_repeat_continuation_vs_oracle", "baseline_repeat_continuation_vs_baseline",
          "candidate_continuation_vs_oracle", "candidate_continuation_vs_baseline")
REQUIRED = ("baseline_vs_oracle", "baseline_repeat_vs_baseline", "candidate_vs_baseline",
            "candidate_vs_oracle", "normalization_only_candidate_vs_baseline",
            "prep_vs_fp32_reference_baseline", "prep_vs_fp32_reference_candidate",
            "candidate_continuation_vs_oracle", "candidate_continuation_vs_baseline")
CSV_COLUMNS = ("trial", "trial_seed", "role", "comparison", "case_id", "category", "field", "scope",
               "actual_dtype", "expected_dtype", "dtype_match", "count", "bitwise_comparable",
               "bitwise_identical", "exact_equal_fraction", "max_abs", "mean_abs", "rmse",
               "reference_rms", "relative_rms", "p0.5", "p0.9", "p0.99", "p0.999",
               "acceptance_violations_grid", "acceptance_violations_direct",
               "acceptance_passes_direct", "violations_atol0_rtol0", "percentile_basis")


def run_worker(root, report, arguments, timeout):
    command = [sys.executable, "-B", str(Path(__file__).with_name("_worker.py")),
               "--source-root", str(Path(root).resolve()), *arguments, "--report", str(Path(report).resolve())]
    try:
        result = subprocess.run(command, cwd=root, env=worker_environment(root),
                                capture_output=True, text=True, timeout=timeout)
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout, stderr = (getattr(error, name, None) or b"" for name in ("stdout", "stderr"))
        stdout, stderr = (value.decode(errors="replace") if isinstance(value, bytes) else value
                          for value in (stdout, stderr))
    value = json.loads(Path(report).read_text()) if Path(report).exists() else {
        "status": "failed", "error": "worker produced no report"}
    value["process_returncode"] = returncode
    if returncode != 0:
        value["status"] = "failed"
        value.setdefault("error", f"worker exited {returncode}")
    value["stdout_tail"], value["stderr_tail"] = stdout[-4000:], stderr[-4000:]
    write_json(report, value)
    return value


def validate(report, expected_ids, expected_source, workloads_sha, harness, precision, trial, gpu_uuid):
    if report.get("status") != "passed" or report.get("process_returncode") != 0:
        raise RuntimeError(f"worker failed: {report.get('error', report.get('stderr_tail', 'see report'))}")
    if [row["id"] for row in report["rows"]] != expected_ids:
        raise RuntimeError("worker returned incomplete or reordered rows")
    if report["source"]["sglang_python_tree_sha256"] != expected_source["sglang_python_tree_sha256"]:
        raise RuntimeError("source tree changed after the run was frozen")
    if report["workloads_sha256"] != workloads_sha:
        raise RuntimeError("workloads changed during the run")
    if report.get("harness_sha256") != harness or report.get("precision_sha256") != precision:
        raise RuntimeError("harness or precision scripts changed during the run")
    if report.get("trial_seed") != trial_seed(trial):
        raise RuntimeError("worker used the wrong trial seed")
    if gpu_uuid and not metrics.same_device_uuid(report["runtime"]["device_uuid"], gpu_uuid):
        raise RuntimeError(f"worker ran on an unexpected device: {report['runtime']['device_uuid']} "
                           f"is not the locked {gpu_uuid}")


def blank():
    return {"field_count": 0, "case_ids": set(), "fields": set(),
            "bitwise_comparable_fields": 0, "bitwise_identical_fields": 0,
            "mixed_dtype_fields": 0, "fully_equal_value_fields": 0,
            "nonfinite_fields": 0, "max_abs": 0.0, "max_relative_rms": 0.0, "worst": None,
            "acceptance_violations_grid": 0, "acceptance_violations_direct": 0,
            "acceptance_failing_fields": 0, "acceptance_failures": [], "elements": 0,
            "violations": {metrics.key(a, r): 0 for r in metrics.RTOL for a in metrics.ATOL},
            "minimum_atol": {f"rtol={r:g}": 0.0 for r in metrics.RTOL}}


def accumulate(bucket, record, case_id, trial):
    if record.get("empty"):
        return
    bucket["field_count"] += 1
    bucket["case_ids"].add(case_id)
    bucket["fields"].add(record["label"].split(".")[-1])
    bucket["elements"] += record["count"]
    # Only same-dtype fields can carry a bitwise claim at all; count the
    # denominator too so a high identical count cannot read as universal.
    if record.get("bitwise_comparable"):
        bucket["bitwise_comparable_fields"] += 1
        if record.get("bitwise_identical"):
            bucket["bitwise_identical_fields"] += 1
    else:
        bucket["mixed_dtype_fields"] += 1
    if record.get("exact_equal_fraction") == 1.0:
        bucket["fully_equal_value_fields"] += 1
    if record.get("status") == "failed" or record.get("nonfinite_actual") or record.get("nonfinite_expected"):
        bucket["nonfinite_fields"] += 1
        return
    if record["max_abs"] > bucket["max_abs"]:
        bucket["max_abs"] = record["max_abs"]
        bucket["worst"] = {"case": case_id, "trial": trial, "field": record["label"],
                           "max_abs": record["max_abs"], "relative_rms": record["relative_rms"]}
    bucket["max_relative_rms"] = max(bucket["max_relative_rms"], record["relative_rms"])
    bucket["acceptance_violations_grid"] += record.get("acceptance_violations", 0)
    direct = record.get("original_acceptance_direct") or {}
    bucket["acceptance_violations_direct"] += direct.get("violations", 0)
    if direct.get("passes") is False:
        bucket["acceptance_failing_fields"] += 1
        bucket["acceptance_failures"].append({"case": case_id, "trial": trial, "field": record["label"],
                                              "violations": direct.get("violations"),
                                              "relative_rms": direct.get("relative_rms")})
    for name, value in record.get("violations", {}).items():
        bucket["violations"][name] += value["violations"]
    for name, value in record.get("minimum_atol", {}).items():
        bucket["minimum_atol"][name] = max(bucket["minimum_atol"][name], value)


def finish(bucket):
    bucket["case_ids"] = sorted(bucket["case_ids"])
    bucket["case_count"] = len(bucket["case_ids"])
    bucket["fields"] = sorted(bucket["fields"])
    bucket["violations_all_zero"] = all(value == 0 for value in bucket["violations"].values())
    bucket["exact_zero_row_violations"] = bucket["violations"][metrics.key(0.0, 0.0)]
    bucket["acceptance_passes_direct"] = bucket["acceptance_failing_fields"] == 0
    # Keep the summary compact; the CSV carries every field.
    bucket["acceptance_failures"] = bucket["acceptance_failures"][:10]
    return bucket


def csv_rows(writer, report, trial):
    for row in report["rows"]:
        for name, records in row.get("comparisons", {}).items():
            for record in records:
                if record.get("empty"):
                    continue
                direct = record.get("original_acceptance_direct") or {}
                writer.writerow({
                    "trial": trial, "trial_seed": report["trial_seed"], "role": report["role"],
                    "comparison": name,
                    "case_id": row["id"], "category": row["category"], "field": record["label"],
                    "scope": record.get("scope"), "actual_dtype": record.get("actual_dtype"),
                    "expected_dtype": record.get("expected_dtype"),
                    "dtype_match": record.get("dtype_match"), "count": record["count"],
                    "bitwise_comparable": record.get("bitwise_comparable"),
                    "bitwise_identical": record.get("bitwise_identical"),
                    "exact_equal_fraction": record.get("exact_equal_fraction"),
                    "max_abs": record.get("max_abs"), "mean_abs": record.get("mean_abs"),
                    "rmse": record.get("rmse"), "reference_rms": record.get("reference_rms"),
                    "relative_rms": record.get("relative_rms"),
                    **{f"p{value:g}": (record.get("percentiles") or {}).get(f"p{value:g}")
                       for value in metrics.PERCENTILES},
                    "acceptance_violations_grid": record.get("acceptance_violations"),
                    "acceptance_violations_direct": direct.get("violations"),
                    "acceptance_passes_direct": direct.get("passes"),
                    "violations_atol0_rtol0": (record.get("violations") or {}).get(
                        metrics.key(0.0, 0.0), {}).get("violations",
                                                       0 if record.get("violations_all_zero") else None),
                    "percentile_basis": record.get("percentile_basis")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--out", required=True, help="new report directory; must not already exist")
    parser.add_argument("--scratch", required=True, help="bulk reference tensors; outside artifacts, deleted per seed")
    parser.add_argument("--baseline-cache", help="optional persistent baseline/A/A cache; bounded to one identity and three seeds")
    parser.add_argument("--expect-gpu-uuid", default=None)
    parser.add_argument("--worker-timeout", type=int, default=1500)
    args = parser.parse_args()
    document = load_workloads(args.workloads)
    roots = {"baseline": str(Path(args.baseline_root).resolve()),
             "candidate": str(Path(args.candidate_root).resolve())}
    sources = {name: source_info(root) for name, root in roots.items()}
    workloads_sha, harness, precision = sha256(args.workloads), harness_hashes(), metrics.script_hashes()
    ids = [case["id"] for case in document["cases"]]
    correctness_ids = [case["id"] for case in selected_cases(document, "correctness")]
    out, scratch = Path(args.out).resolve(), Path(args.scratch).resolve()
    # Scratch is bulk tensor storage and is deleted; --out holds the only copy of
    # the worker reports and per-field.csv. If they overlap, cleanup destroys the
    # evidence and the summary written afterwards describes reports that no longer
    # exist — a "passed" run with nothing behind it, which erases the distinction
    # between complete evidence and missing data that workflow.md requires.
    if out.is_relative_to(scratch) or scratch.is_relative_to(out):
        parser.error("--out and --scratch must be separate, non-nested directories")
    # Recorded before any worker can create it: a pre-existing scratch directory
    # belongs to the caller, so this run may remove what it made inside it but must
    # not remove the directory itself.
    scratch_created_here = not scratch.exists()
    if args.baseline_cache:
        cache_path = Path(args.baseline_cache).resolve()
        if cache_path.is_relative_to(scratch) or scratch.is_relative_to(cache_path):
            parser.error("--baseline-cache and --scratch must be separate, non-nested directories")
        if cache_path.is_relative_to(out) or out.is_relative_to(cache_path):
            parser.error("--baseline-cache and --out must be separate, non-nested directories")
    out.mkdir(parents=True, exist_ok=False)
    # Everything this run writes under scratch goes in one directory it creates itself,
    # so cleanup removes what it made rather than whatever it can name. A caller's own
    # `trial-NN` directory is indistinguishable by name from ours, so a name-based
    # sweep would delete the caller's data.
    scratch.mkdir(parents=True, exist_ok=True)
    owned_scratch = Path(tempfile.mkdtemp(prefix="precision-", dir=scratch))
    summary = {
        "status": "failed", "kind": "accuracy_only_precision_diagnostic",
        "measures_performance": False, "sources": sources,
        "workloads_sha256": workloads_sha, "harness_sha256": harness, "precision_sha256": precision,
        "model_profile": document["model_profile"],
        "case_counts": {"total": len(ids), "correctness_oracle": len(correctness_ids),
                        "deployment_grid": len(ids) - len(correctness_ids)},
        "correctness_case_ids": correctness_ids, "all_case_ids": ids,
        "trials": list(TRIALS), "trial_seeds": {str(trial): trial_seed(trial) for trial in TRIALS},
        "tolerance_grid": {"atol": metrics.ATOL, "rtol": metrics.RTOL},
        "grid_cells": len(metrics.ATOL) * len(metrics.RTOL),
        "violation_predicate": "abs(actual - expected) <= atol + rtol * abs(expected); expected is the reference",
        "grid_arithmetic": metrics.GRID_ARITHMETIC,
        "original_acceptance": {"end_to_end": dict(TOLERANCE), "track": dict(TRACK_TOLERANCE),
                                "note": "unchanged acceptance tolerance, separate from this diagnostic grid; "
                                        "it is the loosest row of the grid",
                                "verdict_basis": "acceptance_violations_direct and acceptance_passes_direct are "
                                                 "computed directly in float32, not read off the float64 grid"},
        "near_zero_threshold": metrics.NEAR_ZERO, "percentiles": metrics.PERCENTILES,
        "comparison_semantics": {
            "baseline_vs_oracle": "original implementation against the independent CPU FP32 recurrence, correctness cases only",
            "baseline_repeat_vs_baseline": "original against itself in a fresh process, same seed; A/A noise floor",
            "candidate_vs_baseline": "final frozen kernel against the original, all cases; A/B",
            "candidate_vs_oracle": "final frozen kernel against the SAME shared oracle, correctness cases only",
            "baseline_repeat_vs_oracle": "the A/A repeat against the same shared oracle; should match baseline_vs_oracle",
            "normalization_only_candidate_vs_baseline": "preparation outputs before the public call, "
                                                        "candidate preparation against baseline l2norm_fwd + v.contiguous()",
            "normalization_only_baseline_repeat_vs_baseline": "the same preparation group for the A/A repeat",
            "prep_vs_fp32_reference_<role>": "that role's preparation against an independent CPU FP32 normalization, "
                                             "before and after the activation-dtype cast",
            "<role>_continuation_vs_oracle": "split-aware recurrence on the 3 continuation cases",
            "<role>_continuation_vs_baseline": "continuation outputs and final state against the baseline's own run"},
        "oracle": {"basis": "bench/runtime.recurrence, independent CPU FP32 delta recurrence",
                   "honesty": "respects the BF16 input normalization and state rounding boundaries of the public "
                              "call; it is NOT an ideal FP64 whole-model reference",
                   "scope": f"{len(correctness_ids)} bounded correctness cases, at most 2048 tokens each",
                   "shared": "one oracle file per case per seed, hash-verified and reused by both sides"},
        "limitations": [
            f"oracle comparison covers the {len(correctness_ids)} bounded correctness cases only; the "
            f"{len(ids) - len(correctness_ids)} deployment-grid cases are compared to the baseline, NOT to the oracle",
            "stricter-than-acceptance failures against the oracle are expected BF16 recurrence effects and are not "
            "by themselves a candidate regression; read candidate_vs_oracle beside baseline_vs_oracle",
            "A/B differences may include autotune nondeterminism; baseline_repeat_vs_baseline is the context for that",
            "evidence covers the frozen workload cases, seeds and device recorded here; it is not a global "
            "bitwise-equivalence proof over all shapes and call patterns",
            "no timing, profiling or registered-test evidence is produced by this run"],
        "trial_reports": [], "comparisons": {}}
    summary["baseline_cache"] = {"enabled": bool(args.baseline_cache), "trials": []}
    if args.baseline_cache:
        summary["baseline_cache"]["directory"] = str(cache_path)
        summary["limitations"].append("A cached A/A result describes the baseline repeat measured when the "
                                      "cache was built; disable baseline caching to measure a fresh noise floor")
    write_json(out / "summary.json", summary)
    buckets = {name: blank() for name in GROUPS}
    try:
        with ExitStack() as stack:
            cache, expected_runtime = None, None
            if args.baseline_cache:
                identity_arguments = ["--mode", "identity", "--role", "baseline", "--trial", "0",
                                      "--workloads", str(Path(args.workloads).resolve()),
                                      "--scratch", str(owned_scratch)]
                if args.expect_gpu_uuid:
                    identity_arguments += ["--expect-gpu-uuid", args.expect_gpu_uuid]
                probe = run_worker(roots["baseline"], out / "runtime-identity.json",
                                   identity_arguments, args.worker_timeout)
                validate(probe, [], sources["baseline"], workloads_sha, harness, precision,
                         0, args.expect_gpu_uuid)
                expected_runtime = runtime_identity(probe["runtime"])
                identity = cache_identity(sources["baseline"], workloads_sha, harness, precision, probe["runtime"])
                cache = stack.enter_context(open_cache(cache_path, identity))
                summary["baseline_cache"]["identity"] = identity
                summary["baseline_cache"]["identity_reset"] = cache.reset
            stream = stack.enter_context((out / "per-field.csv").open("w", newline=""))
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for trial in TRIALS:
                seed_scratch = owned_scratch / f"trial-{trial:02d}"
                try:
                    def validate_baseline(report):
                        validate(report, ids, sources["baseline"], workloads_sha, harness, precision,
                                 trial, args.expect_gpu_uuid)

                    cached_entry, cached_reports = None, None
                    if cache:
                        cached_entry, cached_reports, reason = cache.load(trial, validate_baseline)
                        summary["baseline_cache"]["trials"].append(
                            {"trial": trial, "status": "hit" if cached_reports else "miss",
                             "reason": reason})
                    tensor_directory = cached_entry or seed_scratch
                    common = ["--workloads", str(Path(args.workloads).resolve()), "--trial", str(trial)]
                    if args.expect_gpu_uuid:
                        common += ["--expect-gpu-uuid", args.expect_gpu_uuid]
                    export_report = out / f"trial-{trial:02d}.baseline.export.json"
                    plan = [("baseline", "baseline", ["--mode", "export", "--role", "baseline"], export_report),
                            ("baseline", "baseline_repeat",
                             ["--mode", "compare", "--role", "baseline_repeat",
                              "--reference-report", str(export_report)],
                             out / f"trial-{trial:02d}.baseline_repeat.compare.json"),
                            ("candidate", "candidate",
                             ["--mode", "compare", "--role", "candidate",
                              "--reference-report", str(export_report)],
                             out / f"trial-{trial:02d}.candidate.compare.json")]
                    for side, role, arguments, report_path in plan:
                        reused = cached_reports is not None and role in cached_reports
                        if reused:
                            report = cached_reports[role]
                            # Preserve exact bytes: the A/A report binds the export report's hash.
                            shutil.copyfile(cached_entry / ("baseline.export.json" if role == "baseline"
                                                          else "baseline_repeat.compare.json"), report_path)
                        else:
                            report = run_worker(roots[side], report_path,
                                                arguments + common + ["--scratch", str(tensor_directory)],
                                                args.worker_timeout)
                        validate(report, ids, sources[side], workloads_sha, harness, precision,
                                 trial, args.expect_gpu_uuid)
                        if expected_runtime is not None and runtime_identity(report["runtime"]) != expected_runtime:
                            raise RuntimeError("worker device/runtime changed after baseline cache identity probe")
                        summary["trial_reports"].append(
                            {"trial": trial, "role": role, "source_root_side": side,
                             "report": report_path.name, "device": report["runtime"]["device"],
                             "device_uuid": report["runtime"]["device_uuid"],
                             "reused_from_baseline_cache": reused,
                             "metrics_self_check": report["metrics_self_check"]})
                        summary.setdefault("runtime", report["runtime"])
                        csv_rows(writer, report, trial)
                        for row in report["rows"]:
                            for name, records in row.get("comparisons", {}).items():
                                if name not in buckets:
                                    raise RuntimeError(f"unexpected comparison group {name}")
                                for record in records:
                                    accumulate(buckets[name], record, row["id"], trial)
                        if cache and role == "baseline_repeat" and cached_reports is None:
                            tensor_directory = cache.publish(trial, seed_scratch, export_report,
                                                             report_path, validate_baseline)
                    stream.flush()
                finally:
                    shutil.rmtree(seed_scratch, ignore_errors=True)
        summary["comparisons"] = {name: finish(bucket) for name, bucket in buckets.items()
                                 if bucket["field_count"]}
        empty = [name for name in REQUIRED if name not in summary["comparisons"]]
        if empty:
            raise RuntimeError(f"required comparison groups produced no fields: {empty}")
        nonfinite = {name: bucket["nonfinite_fields"] for name, bucket in summary["comparisons"].items()
                     if bucket["nonfinite_fields"]}
        if nonfinite:
            raise RuntimeError(f"nonfinite values present: {nonfinite}")
        summary["status"] = "passed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["comparisons"] = {name: finish(bucket) for name, bucket in buckets.items()
                                 if bucket["field_count"]}
    finally:
        # The summary is written before any removal, so no cleanup path can destroy
        # evidence the summary then claims. Only `owned_scratch` is removed: this run
        # created it, so nothing in it belongs to the caller. `scratch` itself is
        # removed only when this run created that too, and only while empty, so a
        # caller's pre-existing scratch and its contents survive either outcome.
        write_json(out / "summary.json", summary)
        shutil.rmtree(owned_scratch, ignore_errors=True)
        if scratch_created_here and scratch.is_dir() and not any(scratch.iterdir()):
            scratch.rmdir()
    print(f"Precision diagnostic: {summary['status']}; report: {out / 'summary.json'}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
