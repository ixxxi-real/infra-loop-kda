"""CPU-only regression tests for source isolation and trustworthy measurements."""

import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import benchmark
import common
import runtime


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "workloads.resolved.json"


class StaticTests(unittest.TestCase):
    def test_entrypoints_are_lazy_and_unresolved_execution_fails_before_import(self):
        guard = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'sglang'}:
            raise AssertionError('Forbidden GPU import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
script = sys.argv[2]
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name='__main__')
"""
        for name in ("preflight.py", "correctness.py", "benchmark.py", "profile_one.py", "_worker.py"):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, "-B", "-c", guard, str(HERE), str(HERE / name), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            unresolved = json.loads(TEMPLATE.read_text())
            unresolved["model_profile"]["status"] = "UNRESOLVED"
            manifest = Path(temporary) / "unresolved.json"
            manifest.write_text(json.dumps(unresolved))
            result = subprocess.run([sys.executable, "-B", "-c", guard, str(HERE), str(HERE / "correctness.py"), "--source-root", str(HERE.parents[2]), "--workloads", str(manifest), "--report", str(Path(temporary) / "report.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("UNRESOLVED", result.stderr)
            self.assertNotIn("Forbidden GPU import", result.stderr)

    def test_manifest_rejects_dangerous_or_inconsistent_shapes(self):
        document = json.loads(TEMPLATE.read_text())
        document["model_profile"].update(status="RESOLVED", num_heads=64, local_num_heads=8,
                                          head_k_dim=128, head_v_dim=128, attention_tp_size=8,
                                          activation_dtype="bfloat16", state_dtype="bfloat16", gate_lower_bound=-5.0)
        mutations = (
            lambda d: d["model_profile"].update(local_num_heads=7),
            lambda d: d["model_profile"].update(state_dtype="float32"),
            lambda d: d["cases"][0].update(total_tokens=99999),
            lambda d: d["cases"][0].update(seq_lens=[0]),
            lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
            lambda d: d["cases"][0].update(continuation_splits=[-1, 2]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workloads.json"
            path.write_text(json.dumps(document))
            common.load_workloads(path)
            for mutate in mutations:
                invalid = copy.deepcopy(document)
                mutate(invalid)
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    common.load_workloads(path)

    def test_incomplete_failed_or_changed_worker_cannot_publish_results(self):
        source = {"sglang_python_tree_sha256": "frozen"}
        report = {"status": "passed", "process_returncode": 0, "source": source,
                  "workloads_sha256": "manifest", "rows": [{"id": "one", "status": "passed"}]}
        benchmark.validate_report(report, ["one"], source, "manifest")
        for mutate in (
            lambda r: r.update(rows=[]),
            lambda r: r.update(process_returncode=1),
            lambda r: r.update(source={"sglang_python_tree_sha256": "changed"}),
            lambda r: r.update(workloads_sha256="changed"),
            lambda r: r["rows"][0].update(status="failed"),
        ):
            invalid = copy.deepcopy(report)
            mutate(invalid)
            with self.assertRaises(RuntimeError):
                benchmark.validate_report(invalid, ["one"], source, "manifest")

    def test_finite_negative_gate_policy_matches_configurator(self):
        document = json.loads(TEMPLATE.read_text())
        document["model_profile"].update(status="RESOLVED", num_heads=64, local_num_heads=8,
                                          head_k_dim=128, head_v_dim=128, attention_tp_size=8,
                                          activation_dtype="bfloat16", state_dtype="bfloat16")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workloads.json"
            for bound in (None, -0.25, -5.0, -9.0):
                document["model_profile"]["gate_lower_bound"] = bound
                path.write_text(json.dumps(document))
                common.load_workloads(path)
            for bound in (True, 0, 1, float("nan"), float("inf"), -float("inf")):
                document["model_profile"]["gate_lower_bound"] = bound
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    common.load_workloads(path)

    def test_full_grid_gate_precedes_timing_and_paired_trials_vary_seed(self):
        document = json.loads(TEMPLATE.read_text())
        sources = {"sglang_python_tree_sha256": "frozen"}
        for fail_comparison in (True, False):
            calls = []

            def worker(script, source_root, arguments, report_path):
                mode = arguments[arguments.index("--mode") + 1]
                trial = int(arguments[arguments.index("--trial") + 1]) if "--trial" in arguments else 0
                calls.append((mode, Path(source_root).name, trial))
                cases = common.selected_cases(document, "correctness" if mode == "correctness" else "deployment_grid")
                rows = [{"id": case["id"], "status": "passed"} for case in cases]
                if mode == "benchmark":
                    for row in rows:
                        row.update(cuda_event_ms=[1.0, 1.2], synchronized_host_ms=[1.5, 1.7],
                                   inputs={"seed": runtime.case_seed(row["id"], runtime.trial_seed(trial))})
                if mode == "compare" and fail_comparison:
                    rows[-1]["status"] = "failed"
                report = {"status": "passed", "process_returncode": 0, "source": sources,
                          "workloads_sha256": "manifest", "harness_sha256": {"harness": "frozen"},
                          "rows": rows, "trial_seed": runtime.trial_seed(trial)}
                common.write_json(report_path, report)
                return report

            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary) / "report"
                argv = ["benchmark.py", "--baseline-root", str(Path(temporary) / "baseline"),
                        "--candidate-root", str(Path(temporary) / "candidate"),
                        "--workloads", str(TEMPLATE), "--out", str(out),
                        "--trials", "2", "--samples", "2", "--warmup", "1"]
                with patch.object(sys, "argv", argv), patch.object(benchmark, "load_workloads", return_value=document), \
                        patch.object(benchmark, "source_info", return_value=sources), \
                        patch.object(benchmark, "sha256", return_value="manifest"), \
                        patch.object(benchmark, "harness_hashes", return_value={"harness": "frozen"}), \
                        patch.object(benchmark, "run_worker", side_effect=worker):
                    self.assertEqual(benchmark.main(), 1 if fail_comparison else 0)
                self.assertEqual([call[0] for call in calls[:4]], ["correctness", "correctness", "export", "compare"])
                correctness = json.loads((out / "correctness.json").read_text())
                self.assertEqual(correctness["oracle_case_count"], 11)
                self.assertEqual(correctness["baseline_comparison_case_count"], 51)
                self.assertFalse(any(out.glob("baseline-tensors-*")))
                if fail_comparison:
                    self.assertEqual(len(calls), 4)
                    self.assertEqual(correctness["status"], "failed")
                else:
                    self.assertEqual(calls[4:], [("benchmark", "baseline", 0), ("benchmark", "candidate", 0),
                                                ("benchmark", "candidate", 1), ("benchmark", "baseline", 1)])
                    summary = json.loads((out / "summary.json").read_text())
                    seeds = [call["trial_seed"] for call in summary["execution_order"]]
                    self.assertEqual(seeds[0], seeds[1])
                    self.assertEqual(seeds[2], seeds[3])
                    self.assertNotEqual(seeds[0], seeds[2])
                    self.assertEqual(len(summary["rows"]), 51)

    def test_worker_timeout_cannot_leave_partial_report_passing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            common.write_json(path, {"status": "passed", "rows": [{"id": "one", "status": "passed"}]})
            timeout = subprocess.TimeoutExpired("worker", 1800, output=b"partial stdout", stderr=b"partial stderr")
            with patch.object(common.subprocess, "run", side_effect=timeout):
                report = common.run_worker("_worker.py", temporary, [], path)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["process_returncode"], 124)
            self.assertEqual(len(report["rows"]), 1)
            self.assertEqual(report["stderr_tail"], "partial stderr")

    def test_timing_summary_includes_population_dispersion(self):
        result = runtime.summarize([1.0, 3.0, 5.0])
        self.assertEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["std"], (8 / 3) ** 0.5)
        self.assertEqual(runtime.summarize([3.0])["std"], 0.0)

    def test_each_timing_sample_restores_state_before_start_event(self):
        operations = []

        class FakeInputs:
            fresh = False

            def __init__(self, *args, **kwargs):
                pass

            def reset(self):
                operations.append("restore")
                self.fresh = True

            def invoke(self, kernel):
                if not self.fresh:
                    raise AssertionError("Stateful call replayed without reset")
                self.fresh = False
                operations.append("invoke")

            def describe(self):
                return {}

        class Event:
            count = 0

            def __init__(self, **kwargs):
                self.label = "start" if Event.count == 0 else "finish"
                Event.count += 1

            def record(self):
                operations.append(self.label)

            def synchronize(self):
                operations.append("event_sync")

            def elapsed_time(self, other):
                return 0.25

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event, synchronize=lambda: operations.append("sync")))
        with patch.object(runtime, "Inputs", FakeInputs):
            row = runtime.timed_case(torch, None, {}, {"id": "case"}, warmup=2, samples=3)
        self.assertEqual(operations[:6], ["restore", "invoke", "sync"] * 2)
        self.assertEqual(operations[6:9], ["start", "finish", "event_sync"])
        self.assertEqual(operations[9:], ["restore", "sync", "start", "invoke", "finish", "event_sync"] * 3)
        self.assertEqual(len(row["cuda_event_ms"]), 3)

    def test_foreign_installed_sglang_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Source isolation violation"):
            common.assert_origin("/tmp/site-packages/sglang/__init__.py", "/tmp/requested/python/sglang/__init__.py")


if __name__ == "__main__":
    unittest.main()
