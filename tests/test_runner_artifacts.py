"""Artifact completeness checked against **real** harness output.

``tests/test_flow.py`` exercises the runner against fixture harnesses that this
project wrote, so they can only prove the checker agrees with its own
assumptions. These tests instead drive the genuine
``projects/kimi-k3/prefill/bench/benchmark.py`` -- with its worker subprocess mocked, as
the task's own ``test_gate_reuse.py`` does -- and assert the checker against the
report schema the real harness actually emits.

That distinction matters because the real schema has two shapes a naive checker
gets wrong:

* ``correctness.json`` records ``reports_sha256`` as a filename -> sha256
  manifest. The worker reports are its **keys**, so a checker that only inspects
  string *values* never sees them, and deleting one goes unnoticed.
* ``summary.json`` records ``reused_correctness_report`` as an **absolute** path
  into another run's directory. It is provenance, not a local artifact, and
  existence-checking it reports a complete run as incomplete once the bundle is
  fetched or moved.

CPU only. No GPU, no model, no network: the worker is replaced by a function
that writes schema-shaped reports.
"""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from tests import fixtures
from tools.k3 import runner, util

BENCH_DIR = fixtures.REAL_ROOT / "tasks" / "prefill-kda" / "bench"
WORKLOADS = fixtures.REAL_ROOT / "tasks" / "prefill-kda" / "workloads.resolved.json"


def _import_harness():
    """Import the real harness modules, as the task's own tests do."""
    if not (BENCH_DIR / "benchmark.py").is_file():
        raise unittest.SkipTest("prefill bench harness is not present")
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    import benchmark  # noqa: E402
    import common  # noqa: E402
    import runtime  # noqa: E402

    return benchmark, common, runtime


class RealHarnessReportTests(fixtures.TempDirCase):
    """Produce genuine harness output, then check the runner's verdict on it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark, cls.common, cls.runtime = _import_harness()
        if not WORKLOADS.is_file():
            raise unittest.SkipTest("resolved workloads are not present")

    def setUp(self) -> None:
        super().setUp()
        self.document = json.loads(WORKLOADS.read_text(encoding="utf-8"))
        self.sources = {
            name: {
                "root": str(self.base / name),
                "sglang_python_tree_sha256": name + "-frozen",
            }
            for name in ("baseline", "candidate")
        }
        self.harness = {"benchmark.py": "frozen"}
        self.calls: List[tuple] = []

    # --- harness driving -------------------------------------------------

    def worker(self, script, source_root, arguments, report_path):
        """Stand in for the worker subprocess, emitting schema-shaped reports."""
        name = Path(source_root).name
        mode = arguments[arguments.index("--mode") + 1]
        self.calls.append((mode, name))
        trial = (
            int(arguments[arguments.index("--trial") + 1])
            if "--trial" in arguments
            else 0
        )
        category = "correctness" if mode == "correctness" else "deployment_grid"
        cases = self.common.selected_cases(self.document, category)
        rows = [{"id": case["id"], "status": "passed"} for case in cases]
        if mode == "benchmark":
            for row in rows:
                row.update(
                    cuda_event_ms=[1.0, 1.2],
                    synchronized_host_ms=[1.5, 1.7],
                    inputs={
                        "seed": self.runtime.case_seed(
                            row["id"], self.runtime.trial_seed(trial)
                        )
                    },
                )
        report = {
            "status": "passed",
            "mode": mode,
            "process_returncode": 0,
            "source": self.sources[name],
            "workloads_sha256": self.common.sha256(WORKLOADS),
            "harness_sha256": self.harness,
            "rows": rows,
            "trial_seed": self.runtime.trial_seed(trial),
        }
        if mode == "compare":
            report.update(
                baseline_source=self.sources["baseline"],
                reference_report_sha256=self.common.sha256(
                    arguments[arguments.index("--reference-report") + 1]
                ),
            )
        self.common.write_json(report_path, report)
        return report

    def run_harness(self, directory: str, *arguments) -> Path:
        out = self.base / directory
        argv = [
            "benchmark.py",
            "--baseline-root",
            str(self.base / "baseline"),
            "--candidate-root",
            str(self.base / "candidate"),
            "--workloads",
            str(WORKLOADS),
            "--out",
            str(out),
            "--smoke",
            *map(str, arguments),
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(
                patch.object(
                    self.benchmark, "load_workloads", return_value=self.document
                )
            )
            stack.enter_context(
                patch.object(
                    self.benchmark,
                    "source_info",
                    side_effect=lambda root: self.sources[Path(root).name],
                )
            )
            stack.enter_context(
                patch.object(
                    self.benchmark, "harness_hashes", return_value=self.harness
                )
            )
            stack.enter_context(
                patch.object(self.benchmark, "run_worker", side_effect=self.worker)
            )
            stack.enter_context(patch("builtins.print"))
            status = self.benchmark.main()
        self.assertEqual(status, 0, "the real harness did not succeed")
        return out

    def manifest_for(self, out: Path) -> Dict[str, Any]:
        """A run manifest declaring this directory as the expected artifact."""
        return {
            "run_id": "real-harness",
            "expected_artifacts": [str(out)],
            "output_dir": str(out),
        }

    def report_for(self, out: Path) -> Dict[str, Any]:
        return runner.artifact_report(self.manifest_for(out))

    # --- baseline: the real output must read as complete ------------------

    def test_real_gate_output_is_complete_evidence(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        gate = json.loads((out / "correctness.json").read_text(encoding="utf-8"))
        # Confirm we are testing the real schema, not an assumption about it.
        self.assertIn("reports_sha256", gate)
        self.assertEqual(len(gate["reports_sha256"]), 4)

        report = self.report_for(out)
        self.assertTrue(
            report["complete"],
            "real passing harness output was not recognised as complete: %s"
            % (report["incomplete"] + report["missing"]),
        )
        summary = report["summaries"][str(out)]
        # All four worker reports were derived from the manifest and verified.
        self.assertGreaterEqual(summary["verified_hashes"], 4)
        self.assertEqual(summary["hash_mismatches"], [])

    # --- the defect: a deleted worker report must be caught --------------

    def test_deleting_a_manifest_named_worker_report_is_caught(self) -> None:
        """The reports are manifest *keys*; a value-only walk misses them."""
        out = self.run_harness("gate", "--correctness-only")
        self.assertTrue(self.report_for(out)["complete"])

        victim = out / "candidate.full-grid.json"
        self.assertTrue(victim.is_file(), "expected worker report is absent")
        victim.unlink()

        report = self.report_for(out)
        self.assertFalse(
            report["complete"],
            "deleting a worker report named in reports_sha256 was not detected",
        )
        self.assertTrue(
            any("candidate.full-grid.json" in item for item in report["incomplete"]),
            report["incomplete"],
        )

    def test_every_manifest_named_report_is_individually_required(self) -> None:
        out_base = self.run_harness("gate", "--correctness-only")
        gate = json.loads(
            (out_base / "correctness.json").read_text(encoding="utf-8")
        )
        names = sorted(gate["reports_sha256"])
        self.assertEqual(len(names), 4)

        for index, name in enumerate(names):
            out = self.run_harness("gate-%d" % index, "--correctness-only")
            (out / name).unlink()
            with self.subTest(deleted=name):
                report = self.report_for(out)
                self.assertFalse(
                    report["complete"],
                    "deleting %s was not detected" % name,
                )

    def test_modified_worker_report_fails_the_recorded_hash(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        victim = out / "baseline.correctness.json"
        victim.write_text(
            victim.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        report = self.report_for(out)
        self.assertFalse(
            report["complete"],
            "a modified worker report passed the completeness check",
        )
        summary = report["summaries"][str(out)]
        self.assertTrue(summary["hash_mismatches"], summary)

    def test_emptied_worker_report_is_caught(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        (out / "candidate.correctness.json").write_text("", encoding="utf-8")
        self.assertFalse(self.report_for(out)["complete"])

    # --- the other defect: provenance must not be existence-checked ------

    def test_absolute_provenance_path_is_not_treated_as_a_local_artifact(self) -> None:
        """``reused_correctness_report`` points outside this directory."""
        gate_out = self.run_harness("gate", "--correctness-only")
        gate_path = gate_out / "correctness.json"
        timing_out = self.run_harness(
            "timing", "--correctness-report", str(gate_path)
        )

        summary = json.loads(
            (timing_out / "summary.json").read_text(encoding="utf-8")
        )
        # Confirm the real harness really does record an absolute path here.
        self.assertTrue(
            Path(summary["reused_correctness_report"]).is_absolute(),
            "expected an absolute provenance path in the real summary",
        )

        report = self.report_for(timing_out)
        self.assertTrue(
            report["complete"],
            "real timing output was reported incomplete: %s"
            % (report["incomplete"] + report["missing"]),
        )
        detail = report["summaries"][str(timing_out)]
        self.assertIn("reused_correctness_report", detail["provenance"])
        self.assertNotIn(
            summary["reused_correctness_report"],
            detail["referenced_missing"],
        )

    def test_provenance_stays_uncheckable_after_the_source_run_is_removed(self) -> None:
        """Mirrors a fetched bundle: the referenced run is not on this machine."""
        import shutil

        gate_out = self.run_harness("gate", "--correctness-only")
        timing_out = self.run_harness(
            "timing", "--correctness-report", str(gate_out / "correctness.json")
        )
        self.assertTrue(self.report_for(timing_out)["complete"])

        # The gate run disappears, exactly as it would after a transfer.
        shutil.rmtree(gate_out)

        report = self.report_for(timing_out)
        self.assertTrue(
            report["complete"],
            "a complete run was reported incomplete because a provenance path "
            "no longer resolves: %s" % (report["incomplete"] + report["missing"]),
        )

    def test_timing_output_requires_its_own_copied_reports(self) -> None:
        """Reuse copies the worker reports; deleting a copy must still fail."""
        gate_out = self.run_harness("gate", "--correctness-only")
        timing_out = self.run_harness(
            "timing", "--correctness-report", str(gate_out / "correctness.json")
        )
        (timing_out / "baseline.full-grid.json").unlink()
        self.assertFalse(
            self.report_for(timing_out)["complete"],
            "a deleted copied report in the timing run was not detected",
        )

    # --- failed runs must never read as complete -------------------------

    def test_failed_summary_is_not_complete(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        summary_path = out / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self.assertFalse(self.report_for(out)["complete"])

    def test_failed_gate_is_not_complete(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        gate_path = out / "correctness.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["status"] = "failed"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        report = self.report_for(out)
        self.assertFalse(
            report["complete"], "a failed correctness gate read as complete"
        )

    def test_zero_case_count_in_the_real_gate_is_not_complete(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        gate_path = out / "correctness.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["oracle_case_count"] = 0
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        self.assertFalse(self.report_for(out)["complete"])

    # --- schema-required files, independent of what the summary names -----
    #
    # Collecting only what a summary *references* is not enough. These two cases
    # were reported as still passing after the sidecar fix: a directory can
    # contain nothing but summary.json and be called complete, because nothing
    # was ever required of it.

    def test_summary_alone_is_not_complete_evidence(self) -> None:
        """Deleting everything except summary.json must not read as complete."""
        out = self.run_harness("summary-only", "--correctness-only")
        self.assertTrue(self.report_for(out)["complete"])

        survivors = []
        for item in sorted(out.iterdir()):
            if item.name == "summary.json":
                survivors.append(item.name)
                continue
            if item.is_file():
                item.unlink()
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            ["summary.json"],
            "the fixture did not reduce the directory to summary.json alone",
        )

        report = self.report_for(out)
        self.assertFalse(
            report["complete"],
            "a directory holding only summary.json was reported as complete "
            "evidence: %s" % report,
        )
        detail = report["summaries"][str(out)]
        self.assertEqual(detail["schema"], "benchmark")
        self.assertIn(
            "correctness.json",
            " ".join(detail["referenced_missing"]),
            "correctness.json was not required by the benchmark schema: %s"
            % detail,
        )

    def test_missing_timing_trial_reports_are_caught(self) -> None:
        """Per-trial worker reports are required, derived from execution_order."""
        out = self.run_harness("trial-missing")
        self.assertTrue(
            self.report_for(out)["complete"],
            "a full timing run was not recognised as complete",
        )

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        order = summary.get("execution_order") or []
        self.assertTrue(order, "the real summary recorded no execution_order")

        expected = sorted(
            "trial-%02d.%s.json" % (entry["trial"], entry["implementation"])
            for entry in order
        )
        present = sorted(p.name for p in out.glob("trial-*.json"))
        self.assertTrue(
            set(expected).issubset(set(present)),
            "derived trial names %s are not the files the harness wrote %s"
            % (expected, present),
        )

        victims = ["trial-00.baseline.json", "trial-00.candidate.json"]
        removed = []
        for name in victims:
            target = out / name
            if target.is_file():
                target.unlink()
                removed.append(name)
        self.assertTrue(
            removed,
            "neither %s exists; the trial naming assumption is wrong. Files: %s"
            % (victims, present),
        )

        report = self.report_for(out)
        self.assertFalse(
            report["complete"],
            "deleting trial reports %s was not detected: %s" % (removed, report),
        )
        missing = " ".join(report["summaries"][str(out)]["referenced_missing"])
        for name in removed:
            self.assertIn(name, missing, report["summaries"][str(out)])

    def test_each_executed_trial_report_is_individually_required(self) -> None:
        base = self.run_harness("trial-base")
        summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
        names = sorted(
            "trial-%02d.%s.json" % (entry["trial"], entry["implementation"])
            for entry in (summary.get("execution_order") or [])
        )
        self.assertTrue(names)
        for index, name in enumerate(names):
            out = self.run_harness("trial-each-%d" % index)
            (out / name).unlink()
            with self.subTest(deleted=name):
                self.assertFalse(
                    self.report_for(out)["complete"],
                    "deleting %s was not detected" % name,
                )

    def test_missing_summary_is_not_complete(self) -> None:
        out = self.run_harness("gate", "--correctness-only")
        (out / "summary.json").unlink()
        report = self.report_for(out)
        self.assertFalse(report["complete"])
        self.assertTrue(
            any("summary.json is absent" in item for item in report["incomplete"]),
            report["incomplete"],
        )

    def test_real_gate_records_the_contract_case_counts(self) -> None:
        """Documents the counts the contract gate actually executed."""
        out = self.run_harness("gate", "--correctness-only")
        gate = json.loads((out / "correctness.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["oracle_case_count"], 11)
        self.assertEqual(gate["baseline_comparison_case_count"], 51)
        summary = self.report_for(out)["summaries"][str(out)]
        self.assertEqual(summary["case_counts"]["oracle_case_count"], 11)
        self.assertEqual(
            summary["case_counts"]["baseline_comparison_case_count"], 51
        )


if __name__ == "__main__":
    unittest.main()
