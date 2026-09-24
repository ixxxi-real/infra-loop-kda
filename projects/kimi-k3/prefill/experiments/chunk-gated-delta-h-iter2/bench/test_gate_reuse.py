"""CPU-only checks for splitting the existing gate from timing."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark
import common
import runtime


class GateReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manifest = Path(__file__).resolve().parent.parent / "workloads.resolved.json"
        self.document = json.loads(self.manifest.read_text())
        self.sources = {
            name: {"root": str(self.root / name), "sglang_python_tree_sha256": name + "-frozen"}
            for name in ("baseline", "candidate")
        }
        self.harness = {"benchmark.py": "frozen"}
        self.calls = []
        self.fail_mode = None

    def worker(self, script, source_root, arguments, report_path):
        name = Path(source_root).name
        mode = arguments[arguments.index("--mode") + 1]
        self.calls.append((mode, name))
        trial = int(arguments[arguments.index("--trial") + 1]) if "--trial" in arguments else 0
        cases = common.selected_cases(self.document, "correctness" if mode == "correctness" else "deployment_grid")
        rows = [{"id": case["id"], "status": "passed"} for case in cases]
        if mode == "benchmark":
            for row in rows:
                row.update(cuda_event_ms=[1.0, 1.2], synchronized_host_ms=[1.5, 1.7],
                           inputs={"seed": runtime.case_seed(row["id"], runtime.trial_seed(trial))})
        if mode == self.fail_mode:
            rows[-1]["status"] = "failed"
        report = {
            "status": "passed", "mode": mode, "process_returncode": 0,
            "source": self.sources[name], "workloads_sha256": common.sha256(self.manifest),
            "harness_sha256": self.harness, "rows": rows, "trial_seed": runtime.trial_seed(trial),
        }
        if mode == "compare":
            report.update(baseline_source=self.sources["baseline"], reference_report_sha256=
                          common.sha256(arguments[arguments.index("--reference-report") + 1]))
        common.write_json(report_path, report)
        return report

    def run_cli(self, directory, *arguments):
        out = self.root / directory
        argv = ["benchmark.py", "--baseline-root", str(self.root / "baseline"),
                "--candidate-root", str(self.root / "candidate"), "--workloads", str(self.manifest),
                "--out", str(out), "--smoke", *map(str, arguments)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(benchmark, "load_workloads", return_value=self.document))
            stack.enter_context(patch.object(benchmark, "source_info", side_effect=lambda root: self.sources[Path(root).name]))
            stack.enter_context(patch.object(benchmark, "harness_hashes", return_value=self.harness))
            stack.enter_context(patch.object(benchmark, "run_worker", side_effect=self.worker))
            stack.enter_context(patch("builtins.print"))
            status = benchmark.main()
        return status, out

    def create_gate(self, directory="gate"):
        status, out = self.run_cli(directory, "--correctness-only")
        self.assertEqual(status, 0)
        gate_path = out / "correctness.json"
        gate = json.loads(gate_path.read_text())
        self.assertEqual(gate["oracle_case_count"], 11)
        self.assertEqual(gate["baseline_comparison_case_count"], 51)
        self.assertEqual(set(gate["reports_sha256"]), set(benchmark.CORRECTNESS_REPORTS))
        return gate_path, gate

    def test_split_runs_gate_once_and_copies_evidence_for_timing(self):
        gate_path, gate = self.create_gate()
        self.assertEqual(self.calls, [("correctness", "baseline"), ("correctness", "candidate"),
                                     ("export", "baseline"), ("compare", "candidate")])
        summary = json.loads(gate_path.with_name("summary.json").read_text())
        self.assertEqual(summary["phase"], "correctness")
        self.assertEqual(summary["rows"], [])
        self.calls.clear()
        status, out = self.run_cli("timing", "--correctness-report", gate_path)
        self.assertEqual(status, 0)
        self.assertEqual(self.calls, [("benchmark", "baseline"), ("benchmark", "candidate")])
        self.assertEqual(json.loads((out / "correctness.json").read_text()), gate)
        for filename in benchmark.CORRECTNESS_REPORTS:
            self.assertEqual((out / filename).read_bytes(), gate_path.with_name(filename).read_bytes())
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(len(summary["rows"]), 51)
        self.assertEqual(summary["reused_correctness_report"], str(gate_path))

    def test_failed_gate_never_starts_timing(self):
        self.fail_mode = "compare"
        status, out = self.run_cli("failed", "--correctness-only")
        self.assertEqual(status, 1)
        self.assertEqual(json.loads((out / "correctness.json").read_text())["status"], "failed")
        self.assertFalse(any(mode == "benchmark" for mode, _ in self.calls))

    def test_invalid_gate_or_worker_evidence_cannot_start_timing(self):
        mutations = {
            "gate failed": lambda gate, report: gate.update(status="failed"),
            "source changed": lambda gate, report: gate["sources"]["candidate"].update(sglang_python_tree_sha256="changed"),
            "other root": lambda gate, report: gate["sources"]["candidate"].update(root="/other/snapshot"),
            "workload changed": lambda gate, report: gate.update(workloads_sha256="changed"),
            "harness changed": lambda gate, report: gate.update(harness_sha256={}),
            "partial gate": lambda gate, report: gate["baseline_comparison_ids"].pop(),
            "incorrect count": lambda gate, report: gate.update(oracle_case_count=10),
            "missing hashes": lambda gate, report: gate.pop("reports_sha256"),
            "partial worker": lambda gate, report: report["rows"].pop(),
            "failed worker": lambda gate, report: report.update(status="failed"),
            "failed process": lambda gate, report: report.update(process_returncode=1),
            "failed row": lambda gate, report: report["rows"][-1].update(status="failed"),
            "wrong mode": lambda gate, report: report.update(mode="benchmark"),
            "worker root changed": lambda gate, report: report["source"].update(root="/other/snapshot"),
            "worker source changed": lambda gate, report: report["source"].update(sglang_python_tree_sha256="changed"),
            "worker workload changed": lambda gate, report: report.update(workloads_sha256="changed"),
            "worker harness changed": lambda gate, report: report.update(harness_sha256={}),
            "wrong baseline": lambda gate, report: report["baseline_source"].update(sglang_python_tree_sha256="changed"),
            "wrong reference": lambda gate, report: report.update(reference_report_sha256="changed"),
        }
        for index, (label, mutate) in enumerate(mutations.items()):
            with self.subTest(label=label):
                gate_path, gate = self.create_gate(f"gate-{index}")
                worker_path = gate_path.with_name("candidate.full-grid.json")
                report = json.loads(worker_path.read_text())
                mutate(gate, report)
                common.write_json(worker_path, report)
                if "reports_sha256" in gate:
                    gate["reports_sha256"][worker_path.name] = common.sha256(worker_path)
                common.write_json(gate_path, gate)
                self.calls.clear()
                status, out = self.run_cli(f"timing-{index}", "--correctness-report", gate_path)
                self.assertEqual(status, 1)
                self.assertEqual(self.calls, [])
                self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "failed")

    def test_missing_or_modified_files_cannot_start_timing(self):
        for index, mutation in enumerate(("missing gate", "missing worker", "modified worker")):
            with self.subTest(mutation=mutation):
                gate_path, _ = self.create_gate(f"gate-{index}")
                worker_path = gate_path.with_name("baseline.correctness.json")
                if mutation == "missing gate":
                    gate_path.unlink()
                elif mutation == "missing worker":
                    worker_path.unlink()
                else:
                    worker_path.write_text(worker_path.read_text() + "\n")
                self.calls.clear()
                status, _ = self.run_cli(f"timing-{index}", "--correctness-report", gate_path)
                self.assertEqual(status, 1)
                self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
