"""Run the fixed, bounded independent FP32 recurrence suite on one source root."""

import argparse
from pathlib import Path

from common import load_workloads, run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    load_workloads(args.workloads)  # Fail before even spawning a GPU worker.
    report = run_worker("_worker.py", args.source_root,
                        ["--mode", "correctness", "--workloads", str(Path(args.workloads).resolve())], args.report)
    print(f"Correctness: {report['status']}; report: {Path(args.report).resolve()}")
    return 0 if report["status"] == "passed" and report["process_returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
