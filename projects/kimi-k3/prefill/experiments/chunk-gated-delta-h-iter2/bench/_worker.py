"""One source root per fresh Python process; invoked by the public CLIs."""

import argparse
import json
from pathlib import Path
import traceback

from common import harness_hashes, load_kernel, load_workloads, selected_cases, sha256, source_info, write_json
from runtime import TOLERANCE, capture_case, compare_snapshot, correctness_case, timed_case, trial_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", choices=("preflight", "correctness", "export", "compare", "benchmark"), required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--reference-report")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    report = {"status": "failed", "mode": args.mode, "rows": [],
              "oracle_case_count": 0, "baseline_comparison_case_count": 0}
    report["harness_sha256"] = harness_hashes()
    try:
        document = load_workloads(args.workloads)
        seed = trial_seed(args.trial)
        report.update({"source": source_info(args.source_root), "workloads_sha256": sha256(args.workloads),
                       "model_profile": document["model_profile"], "tolerance": TOLERANCE,
                       "trial": args.trial, "trial_seed": seed})
        reference_rows = None
        if args.mode in ("export", "compare"):
            if not args.reference_dir:
                raise ValueError("Reference directory is required")
            reference_dir = Path(args.reference_dir).resolve(strict=True)
            if args.mode == "compare":
                if not args.reference_report:
                    raise ValueError("Baseline reference report is required")
                reference = json.loads(Path(args.reference_report).read_text())
                expected_ids = [case["id"] for case in selected_cases(document, "deployment_grid")]
                if (reference["status"] != "passed" or reference["workloads_sha256"] != report["workloads_sha256"]
                        or [row["id"] for row in reference["rows"]] != expected_ids
                        or any(row["status"] != "passed" for row in reference["rows"])):
                    raise ValueError("Baseline reference coverage or workload provenance mismatch")
                reference_rows = {row["id"]: row for row in reference["rows"]}
                report["baseline_source"] = reference["source"]
                report["reference_report_sha256"] = sha256(args.reference_report)
        torch, kernel, origins = load_kernel(args.source_root)
        report["runtime"] = origins
        # Keep the independent CPU recurrence small and predictable on big servers.
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.mode == "preflight":
            report["status"] = "passed"
        else:
            if args.warmup < 1 or args.samples < 1:
                raise ValueError("warmup and samples must be positive")
            category = "correctness" if args.mode == "correctness" else "deployment_grid"
            cases = selected_cases(document, category)
            report["expected_ids"] = [case["id"] for case in cases]
            with torch.inference_mode():
                for position, case in enumerate(cases):
                    try:
                        if args.mode == "correctness":
                            row = correctness_case(torch, kernel, document["model_profile"], case)
                        elif args.mode in ("export", "compare"):
                            row, actual = capture_case(torch, kernel, document["model_profile"], case, seed=seed)
                            if args.mode == "export":
                                reference_file = reference_dir / f"{position:04d}.pt"
                                torch.save(actual, reference_file)
                                row.update(reference_file=reference_file.name, reference_sha256=sha256(reference_file))
                            else:
                                baseline_row = reference_rows[case["id"]]
                                reference_file = (reference_dir / baseline_row["reference_file"]).resolve(strict=True)
                                if not reference_file.is_relative_to(reference_dir):
                                    raise ValueError("Reference file escapes the run directory")
                                if sha256(reference_file) != baseline_row["reference_sha256"]:
                                    raise ValueError("Baseline reference tensor hash mismatch")
                                if row["input_sha256"] != baseline_row["input_sha256"] or row["inputs"] != baseline_row["inputs"]:
                                    raise ValueError("Baseline/candidate inputs differ")
                                expected = torch.load(reference_file, map_location="cpu", weights_only=True)
                                row["metrics"] = compare_snapshot(torch, actual, expected, row["active_storage_indices"])
                                row["reference_sha256"] = baseline_row["reference_sha256"]
                                del expected
                            del actual
                        else:
                            row = timed_case(torch, kernel, document["model_profile"], case, args.warmup, args.samples, seed=seed)
                    except Exception:
                        row = {"id": case["id"], "status": "failed", "error": traceback.format_exc()}
                    report["rows"].append(row)
                    report["oracle_case_count"] = len(report["rows"]) if args.mode == "correctness" else 0
                    report["baseline_comparison_case_count"] = len(report["rows"]) if args.mode == "compare" else 0
                    write_json(args.report, report)
            report["status"] = "passed" if all(row["status"] == "passed" for row in report["rows"]) else "failed"
        if source_info(args.source_root)["sglang_python_tree_sha256"] != report["source"]["sglang_python_tree_sha256"]:
            raise RuntimeError("Source changed during worker execution")
        if harness_hashes() != report["harness_sha256"]:
            raise RuntimeError("Harness changed during worker execution")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        write_json(args.report, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
