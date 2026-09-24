"""Correctness-gated A/B public chunk_kda timing with interleaved isolated workers."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from common import harness_hashes, load_workloads, run_worker, selected_cases, sha256, source_info, write_json
from runtime import case_seed, summarize, trial_seed


CORRECTNESS_REPORTS = {
    "baseline.correctness.json": ("baseline", "correctness"),
    "candidate.correctness.json": ("candidate", "correctness"),
    "baseline.full-grid.json": ("baseline", "export"),
    "candidate.full-grid.json": ("candidate", "compare"),
}


def validate_report(report, expected_ids, expected_source, manifest_hash, expected_harness=None):
    if report.get("status") != "passed" or report.get("process_returncode") != 0:
        raise RuntimeError(f"Worker failed: {report.get('error', report.get('stderr_tail', 'see worker report'))}")
    if [row["id"] for row in report["rows"]] != expected_ids:
        raise RuntimeError("Worker returned incomplete or reordered rows")
    if any(row["status"] != "passed" for row in report["rows"]):
        raise RuntimeError("Worker row failed")
    if report["source"]["sglang_python_tree_sha256"] != expected_source["sglang_python_tree_sha256"]:
        raise RuntimeError("Source changed after the run was frozen")
    if report["workloads_sha256"] != manifest_hash:
        raise RuntimeError("Workloads changed during the run")
    if expected_harness is not None and report.get("harness_sha256") != expected_harness:
        raise RuntimeError("Harness changed during the run")


def reuse_correctness(path, out, sources, manifest_hash, frozen_harness, correctness_ids, ids):
    """Reuse a complete gate only for the same frozen source roots and harness."""
    path = Path(path).resolve(strict=True)
    gate = json.loads(path.read_text())
    expected = {
        "status": "passed", "sources": sources, "workloads_sha256": manifest_hash,
        "harness_sha256": frozen_harness, "oracle_case_count": len(correctness_ids),
        "oracle_implementations": ["baseline", "candidate"], "oracle_ids": correctness_ids,
        "baseline_comparison_case_count": len(ids), "baseline_comparison_ids": ids,
        "reports": list(CORRECTNESS_REPORTS),
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(f"Correctness report cannot be reused: {key} mismatch")
    hashes = gate.get("reports_sha256", {})
    if set(hashes) != set(CORRECTNESS_REPORTS):
        raise RuntimeError("Correctness report lacks the four worker report hashes")
    evidence = {}
    for filename, (name, mode) in CORRECTNESS_REPORTS.items():
        report_path = (path.parent / filename).resolve(strict=True)
        if report_path.parent != path.parent:
            raise RuntimeError("Correctness worker report escapes its report directory")
        content = report_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[filename]:
            raise RuntimeError(f"Correctness worker report hash mismatch: {filename}")
        report = json.loads(content)
        validate_report(report, correctness_ids if mode == "correctness" else ids,
                        sources[name], manifest_hash, frozen_harness)
        if report.get("mode") != mode or report.get("source") != sources[name]:
            raise RuntimeError(f"Correctness worker mode or source root mismatch: {filename}")
        if mode == "compare" and (
            report.get("baseline_source") != sources["baseline"]
            or report.get("reference_report_sha256") != hashes["baseline.full-grid.json"]
        ):
            raise RuntimeError("Candidate comparison does not reference the frozen baseline report")
        evidence[filename] = content
    # Keep timing artifacts self-contained, without trusting or rewriting any
    # report until all four have passed validation.
    for filename, content in evidence.items():
        (out / filename).write_bytes(content)
    write_json(out / "correctness.json", gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--out", required=True, help="New report directory; must not already exist")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--correctness-only", action="store_true", help="Run the complete oracle and full-grid gate without timing")
    phase.add_argument("--correctness-report", help="Reuse a passed correctness.json and its four worker reports from the same frozen snapshot")
    parser.add_argument("--smoke", action="store_true", help="All rows, one trial, one warmup and two samples; not a performance claim")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    document = load_workloads(args.workloads)
    if min(args.trials, args.warmup, args.samples) < 1:
        parser.error("trials, warmup and samples must be positive")
    trials, warmup, samples = (1, 1, 2) if args.smoke else (args.trials, args.warmup, args.samples)
    roots = {"baseline": str(Path(args.baseline_root).resolve()), "candidate": str(Path(args.candidate_root).resolve())}
    sources = {name: source_info(root) for name, root in roots.items()}
    manifest_hash = sha256(args.workloads)
    frozen_harness = harness_hashes()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {"status": "failed", "sources": sources, "workloads_sha256": manifest_hash,
               "harness_sha256": frozen_harness,
               "oracle_case_count": 0, "baseline_comparison_case_count": 0,
               "phase": "correctness" if args.correctness_only else "benchmark",
               "smoke": args.smoke, "trials": trials, "warmup": warmup, "samples_per_trial": samples,
               "measurement": {"cuda_event_ms": "one public chunk_kda invocation; includes internal allocations, copies, launches and possible CPU enqueue gaps; not isolated kernel time",
                               "synchronized_host_ms": "wall time including event recording and completion synchronization",
                               "excluded": "imports, JIT/autotune warmup, input allocation and value/state/track restores",
                               "call_abi": "TritonKDAKernel.extend argument mapping; preactivated FP32 sigmoid beta; packed QKV views"},
               "execution_order": [], "rows": []}
    write_json(out / "summary.json", summary)
    try:
        worker_args = ["--workloads", str(Path(args.workloads).resolve())]
        correctness_ids = [case["id"] for case in selected_cases(document, "correctness")]
        ids = [case["id"] for case in selected_cases(document, "deployment_grid")]
        if args.correctness_report:
            reuse_correctness(args.correctness_report, out, sources, manifest_hash,
                              frozen_harness, correctness_ids, ids)
            summary["reused_correctness_report"] = str(Path(args.correctness_report).resolve())
        else:
            for name, root in roots.items():
                report = run_worker("_worker.py", root, ["--mode", "correctness", *worker_args], out / f"{name}.correctness.json")
                validate_report(report, correctness_ids, sources[name], manifest_hash, frozen_harness)
            summary["oracle_case_count"] = len(correctness_ids)
            # Every timed row must first match the frozen baseline, including large
            # and highly batched shapes that exceed the independent recurrence budget.
            with tempfile.TemporaryDirectory(prefix="baseline-tensors-", dir=out) as references:
                baseline_report = out / "baseline.full-grid.json"
                report = run_worker("_worker.py", roots["baseline"],
                                    ["--mode", "export", "--reference-dir", references, *worker_args], baseline_report)
                validate_report(report, ids, sources["baseline"], manifest_hash, frozen_harness)
                report = run_worker("_worker.py", roots["candidate"],
                                    ["--mode", "compare", "--reference-dir", references,
                                     "--reference-report", str(baseline_report), *worker_args], out / "candidate.full-grid.json")
                summary["baseline_comparison_case_count"] = len(report.get("rows", []))
                validate_report(report, ids, sources["candidate"], manifest_hash, frozen_harness)
            write_json(out / "correctness.json", {
                "status": "passed", "oracle_case_count": len(correctness_ids),
                "oracle_implementations": ["baseline", "candidate"], "oracle_ids": correctness_ids,
                "baseline_comparison_case_count": len(ids), "baseline_comparison_ids": ids,
                "sources": sources, "workloads_sha256": manifest_hash, "harness_sha256": frozen_harness,
                "reports": list(CORRECTNESS_REPORTS),
                "reports_sha256": {name: sha256(out / name) for name in CORRECTNESS_REPORTS},
                "reference_tensors": "temporary files removed after validation; SHA256 retained in row reports",
            })
        summary["oracle_case_count"] = len(correctness_ids)
        summary["baseline_comparison_case_count"] = len(ids)
        if args.correctness_only:
            summary["status"] = "passed"
            write_json(out / "summary.json", summary)
            print(f"Correctness: passed; report: {out / 'correctness.json'}")
            return 0
        results = {name: {case_id: [] for case_id in ids} for name in roots}
        host_results = {name: {case_id: [] for case_id in ids} for name in roots}
        for trial in range(trials):
            for name in (("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")):
                report = run_worker("_worker.py", roots[name], ["--mode", "benchmark", "--trial", str(trial), "--warmup", str(warmup), "--samples", str(samples), *worker_args], out / f"trial-{trial:02d}.{name}.json")
                validate_report(report, ids, sources[name], manifest_hash, frozen_harness)
                if report.get("trial_seed") != trial_seed(trial):
                    raise RuntimeError("Worker used the wrong trial seed")
                summary["execution_order"].append({"trial": trial, "trial_seed": trial_seed(trial), "implementation": name})
                for row in report["rows"]:
                    if row["inputs"]["seed"] != case_seed(row["id"], trial_seed(trial)):
                        raise RuntimeError(f"Wrong input seed for {row['id']}")
                    for metric in ("cuda_event_ms", "synchronized_host_ms"):
                        if len(row[metric]) != samples or any(not math.isfinite(value) or value <= 0 for value in row[metric]):
                            raise RuntimeError(f"Invalid timing samples: {row['id']} {metric}")
                    results[name][row["id"]].extend(row["cuda_event_ms"])
                    host_results[name][row["id"]].extend(row["synchronized_host_ms"])
        for case_id in ids:
            row = {"id": case_id}
            for name in roots:
                row[name] = {"cuda_event_ms": summarize(results[name][case_id]), "synchronized_host_ms": summarize(host_results[name][case_id])}
            row["speedup"] = row["baseline"]["cuda_event_ms"]["median"] / row["candidate"]["cuda_event_ms"]["median"]
            summary["rows"].append(row)
        summary["geometric_mean_speedup"] = math.exp(sum(math.log(row["speedup"]) for row in summary["rows"]) / len(ids))
        summary["status"] = "passed"
    except Exception as error:
        summary["error"] = str(error)
        if not (out / "correctness.json").exists():
            write_json(out / "correctness.json", {
                "status": "failed", "error": str(error),
                "oracle_case_count": summary["oracle_case_count"],
                "baseline_comparison_case_count": summary["baseline_comparison_case_count"],
                "expected_oracle_case_count": len(selected_cases(document, "correctness")),
                "expected_baseline_comparison_case_count": len(selected_cases(document, "deployment_grid")),
            })
    write_json(out / "summary.json", summary)
    print(f"Benchmark: {summary['status']}; report: {out / 'summary.json'}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
