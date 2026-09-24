"""Validate ABI/manifests statically, or inspect CUDA imports in a fresh worker."""

import argparse
import json
from pathlib import Path
import tempfile

from common import load_workloads, run_worker, sha256, source_info, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--report")
    parser.add_argument("--static-check", action="store_true", help="Standard library only; allow unresolved checkpoint metadata")
    args = parser.parse_args()
    try:
        document = load_workloads(args.workloads, allow_unresolved=args.static_check)
        report = {"status": "static_passed", "source": source_info(args.source_root),
                  "model_profile": document["model_profile"], "workloads_sha256": sha256(args.workloads),
                  "case_count": len(document["cases"]), "gpu_validation": "pending"}
        if not args.static_check:
            with tempfile.TemporaryDirectory(prefix="kda-preflight-") as temporary:
                report = run_worker("_worker.py", args.source_root,
                                    ["--mode", "preflight", "--workloads", str(Path(args.workloads).resolve())],
                                    Path(temporary) / "report.json")
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] in ("passed", "static_passed") else 1
    except (OSError, ValueError, KeyError, StopIteration) as error:
        report = {"status": "failed", "error": str(error)}
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
