"""CPU-only cache corruption/identity checks and driver reuse integration."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import baseline_cache
import diagnose
from common import sha256, write_json
from runtime import trial_seed


RUNTIME = {"python": "python-fixture", "cuda": "13", "device": "GB300", "capability": [10, 3],
           "device_uuid": "GPU-1234", "driver_version": "580", "dependencies": {"torch": {"version": "2.8"}},
           "numerical_environment": baseline_cache.numerical_environment()}
SOURCE = {"sglang_python_tree_sha256": "tree", "entrypoint_sha256": "entry"}
HARNESSES = {"bench.py": "bench"}
PRECISION = {"diagnose.py": "precision"}
DOCUMENT = {"model_profile": {}, "cases": [{"id": "case", "category": "correctness"}]}
GROUPS = {"baseline": ["baseline_vs_oracle", "prep_vs_fp32_reference_baseline"],
          "baseline_repeat": ["baseline_repeat_vs_baseline"],
          "candidate": ["candidate_vs_baseline", "candidate_vs_oracle",
                        "normalization_only_candidate_vs_baseline", "prep_vs_fp32_reference_candidate",
                        "candidate_continuation_vs_oracle", "candidate_continuation_vs_baseline"]}


def fake_worker(root, report_path, arguments, timeout):
    args = dict(zip(arguments[::2], arguments[1::2]))
    role, mode, trial = args["--role"], args["--mode"], int(args["--trial"])
    report = {"status": "passed", "process_returncode": 0, "source": SOURCE,
              "workloads_sha256": "workloads", "harness_sha256": HARNESSES,
              "precision_sha256": PRECISION, "trial_seed": trial_seed(trial), "trial": trial,
              "role": role, "mode": mode,
              "runtime": dict(RUNTIME, numerical_environment=baseline_cache.numerical_environment()),
              "metrics_self_check": "passed", "rows": []}
    if mode != "identity":
        scratch = Path(args["--scratch"])
        scratch.mkdir(parents=True, exist_ok=True)
        row = {"id": "case", "category": "correctness", "status": "passed", "input_sha256": "input",
               "inputs": {"seed": trial_seed(trial)}, "comparisons": {}}
        for kind, filename in (("reference", "0000.pt"), ("oracle", "oracle-0000.pt")):
            if mode == "export":
                (scratch / filename).write_bytes(f"tensor {trial} {kind}".encode())
                row[kind + "_file"] = filename
            row[kind + "_sha256"] = sha256(scratch / filename)
        record = {"label": "output", "count": 1, "max_abs": 0, "relative_rms": 0,
                  "original_acceptance_direct": {"violations": 0, "passes": True}}
        row["comparisons"] = {group: [dict(record)] for group in GROUPS[role]}
        report["rows"].append(row)
        if mode == "compare":
            report["reference_report_sha256"] = sha256(args["--reference-report"])
    write_json(report_path, report)
    return report


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = baseline_cache.cache_identity(SOURCE, "workloads", HARNESSES, PRECISION, RUNTIME)

    def validate(self, report):
        diagnose.validate(report, ["case"], SOURCE, "workloads", HARNESSES, PRECISION, 0, "1234")

    def populate(self, cache):
        scratch = self.root / "scratch"
        export, repeat = self.root / "export.json", self.root / "repeat.json"
        common = ["--trial", "0", "--scratch", str(scratch)]
        fake_worker(None, export, ["--role", "baseline", "--mode", "export"] + common, 1)
        fake_worker(None, repeat, ["--role", "baseline_repeat", "--mode", "compare",
                                  "--reference-report", str(export)] + common, 1)
        return cache.publish(0, scratch, export, repeat, self.validate)

    def test_hit_and_missing_seed(self):
        with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
            entry = self.populate(cache)
            loaded, reports, reason = cache.load(0, self.validate)
            self.assertEqual(loaded, entry)
            self.assertEqual(set(reports), {"baseline", "baseline_repeat"})
            self.assertIsNone(reason)
            self.assertIsNone(cache.load(1, self.validate)[0])
        with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
            self.assertFalse(cache.reset)
            self.assertIsNotNone(cache.load(0, self.validate)[0])

    def test_report_tensor_and_partial_corruption_are_misses(self):
        for filename in ("baseline.export.json", "baseline_repeat.compare.json", "0000.pt", "oracle-0000.pt", "manifest.json"):
            with self.subTest(filename=filename):
                with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
                    entry = self.populate(cache)
                    path = entry / filename
                    original = path.read_bytes()
                    path.write_bytes(original + b"corrupt")
                    self.assertIsNone(cache.load(0, self.validate)[0])
                    path.unlink()
                    self.assertIsNone(cache.load(0, self.validate)[0])

    def test_changed_identity_discards_previous_entry(self):
        for section in ("baseline_source_sha256", "workloads_sha256", "harness_sha256", "precision_sha256", "runtime"):
            with self.subTest(section=section):
                with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
                    self.populate(cache)
                changed = copy.deepcopy(self.identity)
                changed[section] = "changed"
                with baseline_cache.open_cache(self.root / "cache", changed) as cache:
                    self.assertTrue(cache.reset)
                    self.assertIsNone(cache.load(0, self.validate)[0])

    def test_numerical_environment_drift_invalidates_cache(self):
        for name in baseline_cache.NUMERICAL_ENVIRONMENT:
            with self.subTest(variable=name):
                with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
                    self.populate(cache)
                with patch.dict(os.environ, {name: "changed"}):
                    runtime = dict(RUNTIME, numerical_environment=baseline_cache.numerical_environment())
                changed = baseline_cache.cache_identity(SOURCE, "workloads", HARNESSES, PRECISION, runtime)
                with baseline_cache.open_cache(self.root / "cache", changed) as cache:
                    self.assertTrue(cache.reset)
                    self.assertIsNone(cache.load(0, self.validate)[0])

    def test_unrelated_environment_and_cache_paths_are_excluded(self):
        original = baseline_cache.numerical_environment()
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/different/run", "UNRELATED_SECRET": "private"}):
            self.assertEqual(original, baseline_cache.numerical_environment())
        self.assertNotIn("UNRELATED_SECRET", original)
        self.assertNotIn("TRITON_CACHE_DIR", original)

    def test_rehashed_wrong_runtime_or_report_is_rejected(self):
        for mutation in (lambda report: report["runtime"].update(driver_version="different"),
                         lambda report: report["runtime"].update(device_uuid="GPU-other"),
                         lambda report: report.update(trial_seed=999),
                         lambda report: report.update(reference_report_sha256="wrong"),
                         lambda report: report["rows"][0].update(status="failed")):
            with baseline_cache.open_cache(self.root / "cache", self.identity) as cache:
                entry = self.populate(cache)
                path = entry / baseline_cache.REPORTS["baseline_repeat"]
                report = json.loads(path.read_text())
                mutation(report)
                write_json(path, report)
                manifest = json.loads((entry / "manifest.json").read_text())
                manifest["reports_sha256"]["baseline_repeat"] = sha256(path)
                write_json(entry / "manifest.json", manifest)
                self.assertIsNone(cache.load(0, self.validate)[0])

    def test_uuid_spelling_is_normalized_but_runtime_changes_are_not(self):
        alternate = dict(RUNTIME, device_uuid="1234")
        self.assertEqual(baseline_cache.runtime_identity(RUNTIME), baseline_cache.runtime_identity(alternate))
        alternate["cuda"] = "14"
        self.assertNotEqual(baseline_cache.runtime_identity(RUNTIME), baseline_cache.runtime_identity(alternate))

    def test_driver_runs_every_candidate_seed_and_reuses_only_baseline(self):
        calls = []
        def worker(*args):
            calls.append(args[2])
            return fake_worker(*args)
        def run(number, runner=worker, use_cache=True):
            args = ["diagnose.py", "--baseline-root", str(self.root), "--candidate-root", str(self.root),
                    "--workloads", str(self.root / "workloads.json"), "--out", str(self.root / f"out-{number}"),
                    "--scratch", str(self.root / "scratch")]
            if use_cache:
                args += ["--baseline-cache", str(self.root / "cache")]
            with patch("sys.argv", args), patch.object(diagnose, "load_workloads", return_value=DOCUMENT), \
                 patch.object(diagnose, "source_info", return_value=SOURCE), \
                 patch.object(diagnose, "sha256", return_value="workloads"), \
                 patch.object(diagnose, "harness_hashes", return_value=HARNESSES), \
                 patch.object(diagnose.metrics, "script_hashes", return_value=PRECISION), \
                 patch.object(diagnose, "run_worker", side_effect=runner):
                result = diagnose.main()
            return result, json.loads((self.root / f"out-{number}" / "summary.json").read_text())
        result, cold = run(1)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 10)  # Identity probe + 3 workers * 3 seeds.
        calls.clear()
        result, warm = run(2)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 4)  # Identity probe + candidate at all 3 seeds.
        self.assertEqual([item["status"] for item in warm["baseline_cache"]["trials"]], ["hit"] * 3)
        self.assertEqual(cold["comparisons"], warm["comparisons"])
        candidate_calls = [dict(zip(args[::2], args[1::2])) for args in calls][1:]
        self.assertEqual([item["--trial"] for item in candidate_calls], ["0", "1", "2"])
        self.assertTrue(all(item["--role"] == "candidate" for item in candidate_calls))
        calls.clear()
        result, uncached = run("uncached", use_cache=False)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 9)
        self.assertFalse(uncached["baseline_cache"]["enabled"])
        self.assertEqual(cold["comparisons"], uncached["comparisons"])
        def changed_worker(*args):
            report = fake_worker(*args)
            if report["role"] == "candidate":
                report["runtime"] = dict(RUNTIME, driver_version="changed")
            return report
        result, failed = run(3, changed_worker)
        self.assertEqual(result, 1)
        self.assertIn("device/runtime changed", failed["error"])
        (self.root / "cache" / "entries" / "trial-01" / "0000.pt").write_bytes(b"corrupt")
        calls.clear()
        result, repaired = run(4)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 6)  # Only the corrupt seed rebuilds baseline + A/A.
        self.assertEqual([item["status"] for item in repaired["baseline_cache"]["trials"]],
                         ["hit", "miss", "hit"])
        self.assertEqual(cold["comparisons"], repaired["comparisons"])


class ScratchOwnershipTests(unittest.TestCase):
    """Cleanup must never reach the report directory or a caller's own scratch data.

    `--scratch` is bulk tensor storage that gets deleted; `--out` holds the only copy
    of the worker reports and per-field.csv. An overlapping configuration made cleanup
    destroy the evidence and then write a `passed` summary describing reports that no
    longer existed, which is exactly the complete-versus-missing-data distinction
    workflow.md requires the diagnostic to preserve.

    These drive the real `main()`, because the defect lived in argument handling and the
    `finally` block rather than in any single validated function.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_json(self.root / "workloads.json", DOCUMENT)

    def run_driver(self, out, scratch):
        arguments = ["diagnose.py", "--baseline-root", str(self.root), "--candidate-root", str(self.root),
                     "--workloads", str(self.root / "workloads.json"),
                     "--out", str(out), "--scratch", str(scratch)]
        with patch("sys.argv", arguments), patch.object(diagnose, "load_workloads", return_value=DOCUMENT), \
             patch.object(diagnose, "source_info", return_value=SOURCE), \
             patch.object(diagnose, "sha256", return_value="workloads"), \
             patch.object(diagnose, "harness_hashes", return_value=HARNESSES), \
             patch.object(diagnose.metrics, "script_hashes", return_value=PRECISION), \
             patch.object(diagnose, "run_worker", side_effect=fake_worker):
            return diagnose.main()

    def test_overlapping_out_and_scratch_are_refused_before_anything_runs(self):
        for name, out, scratch in (
                ("out inside scratch", self.root / "run/reports", self.root / "run"),
                ("scratch inside out", self.root / "reports", self.root / "reports/scratch"),
                ("identical paths", self.root / "same", self.root / "same"),
                # Distinct spellings of the same directory must not slip through: the
                # check compares resolved paths, not the strings the caller supplied.
                ("same directory via dot segments", self.root / "run/./reports", self.root / "run"),
        ):
            with self.subTest(name), self.assertRaises(SystemExit) as raised:
                self.run_driver(out, scratch)
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(Path(out).exists(), "refusal must happen before the report directory is created")

    def test_disjoint_run_keeps_every_artifact_and_removes_its_own_scratch(self):
        out, scratch = self.root / "out", self.root / "scratch"
        self.assertEqual(self.run_driver(out, scratch), 0)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "passed")
        # Every report the summary claims must still be on disk beside it.
        self.assertTrue((out / "per-field.csv").exists())
        for entry in summary["trial_reports"]:
            self.assertTrue((out / entry["report"]).exists(), entry["report"])
        self.assertEqual(len([name for name in os.listdir(out) if name.startswith("trial-")]),
                         len(summary["trial_reports"]))
        self.assertFalse(scratch.exists(), "scratch created by this run should be removed")

    def test_pre_existing_scratch_keeps_caller_data_and_the_directory(self):
        out, scratch = self.root / "out", self.root / "scratch"
        scratch.mkdir()
        keep = scratch / "caller-data.bin"
        keep.write_bytes(b"caller owns this")
        self.assertEqual(self.run_driver(out, scratch), 0)
        self.assertEqual(keep.read_bytes(), b"caller owns this")
        self.assertTrue(scratch.is_dir(), "a scratch directory this run did not create must survive")
        self.assertEqual([path.name for path in scratch.iterdir()], ["caller-data.bin"])

    def caller_owned_tree(self, scratch):
        """A caller's own tree whose names collide with the ones this run uses.

        `trial-00` is exactly what the per-seed directories are called, so a cleanup
        that removes paths it can *name* rather than paths it *created* destroys this.
        """
        scratch.mkdir(parents=True)
        (scratch / "trial-00").mkdir()
        (scratch / "trial-00" / "sentinel").write_bytes(b"caller trial data")
        (scratch / "trial-07").mkdir()
        (scratch / "trial-07" / "nested").mkdir()
        (scratch / "trial-07" / "nested" / "deep.bin").write_bytes(b"deep")
        (scratch / "loose.bin").write_bytes(b"loose")
        return {
            scratch / "trial-00" / "sentinel": b"caller trial data",
            scratch / "trial-07" / "nested" / "deep.bin": b"deep",
            scratch / "loose.bin": b"loose",
        }

    def assert_caller_tree_intact(self, scratch, expected):
        self.assertTrue(scratch.is_dir())
        for path, contents in expected.items():
            self.assertTrue(path.exists(), f"{path} was destroyed")
            self.assertEqual(path.read_bytes(), contents, f"{path} was modified")

    def test_name_colliding_caller_directories_survive_a_successful_run(self):
        out, scratch = self.root / "out", self.root / "scratch"
        expected = self.caller_owned_tree(scratch)
        self.assertEqual(self.run_driver(out, scratch), 0)
        self.assert_caller_tree_intact(scratch, expected)
        # This run's own scratch is gone, so success still cleans up after itself.
        self.assertEqual(sorted(path.name for path in scratch.iterdir()),
                         ["loose.bin", "trial-00", "trial-07"])

    def test_name_colliding_caller_directories_survive_a_worker_failure(self):
        out, scratch = self.root / "out", self.root / "scratch"
        expected = self.caller_owned_tree(scratch)

        def broken_worker(*arguments):
            # Raise from inside the run rather than mutating a report field: the
            # runtime-identity check is reached only with --baseline-cache, and what
            # this test needs is any failure that leaves cleanup to the finally block.
            report = fake_worker(*arguments)
            if report["role"] == "candidate":
                raise RuntimeError("worker exploded mid-run")
            return report

        argv = ["diagnose.py", "--baseline-root", str(self.root), "--candidate-root", str(self.root),
                "--workloads", str(self.root / "workloads.json"),
                "--out", str(out), "--scratch", str(scratch)]
        with patch("sys.argv", argv), patch.object(diagnose, "load_workloads", return_value=DOCUMENT), \
             patch.object(diagnose, "source_info", return_value=SOURCE), \
             patch.object(diagnose, "sha256", return_value="workloads"), \
             patch.object(diagnose, "harness_hashes", return_value=HARNESSES), \
             patch.object(diagnose.metrics, "script_hashes", return_value=PRECISION), \
             patch.object(diagnose, "run_worker", side_effect=broken_worker):
            self.assertEqual(diagnose.main(), 1)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("worker exploded mid-run", summary["error"])
        self.assert_caller_tree_intact(scratch, expected)
        self.assertEqual(sorted(path.name for path in scratch.iterdir()),
                         ["loose.bin", "trial-00", "trial-07"])


if __name__ == "__main__":
    unittest.main()
