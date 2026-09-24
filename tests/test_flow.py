"""End-to-end flow over the existing interfaces.

Covers the connected path a user actually walks:

``task-create`` -> ``validate`` -> ``workspace prepare`` -> ``doctor`` ->
``agent plan`` -> ``run plan/start/status/cancel/fetch`` -> ``export bundle``

plus the regressions raised in review: run-id traversal and collision, baseline
identity binding, artifact completeness, fetch destination remapping, and
supervisor descendant cancellation.

Nothing here contacts a GPU, a model, an API, SSH or the network. ``claude`` and
``codex`` are mock shell scripts, the measurement harnesses are fixtures, and
every file lives under a temporary directory. A fixture run is explicitly a
fixture: the tests assert that the tooling refuses to call it acceptance.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests import fixtures
from tools.k3 import config as config_mod
from tools.k3 import (
    agent,
    export,
    gitq,
    lifecycle,
    runner,
    taskfactory,
    toolchain,
    util,
    workspace,
)


class FlowCase(fixtures.TempDirCase):
    """A prepared project with a registered, startable task."""

    task_id = "decode-kda"

    def setUp(self) -> None:
        super().setUp()
        self.source = fixtures.make_source_repo(self.base)
        self.project = fixtures.make_project(self.base, source=self.source)
        self.root = self.project["root"]
        self.config_path = self.project["config_path"]
        self.mockbin = fixtures.make_mock_bin(self.base)

        self.task_dir = fixtures.add_task(
            self.project,
            self.task_id,
            base_commit=self.source["base_commit"],
            status=lifecycle.READY,
            workload_status=lifecycle.RESOLVED,
        )
        self.config = self.project["config"]

        # Mock agent CLIs come first on PATH; the real ones are never invoked.
        self._original_path = os.environ.get("PATH", "")
        os.environ["PATH"] = "%s%s%s" % (
            self.mockbin,
            os.pathsep,
            self._original_path,
        )
        self.addCleanup(self._restore_path)

    def _restore_path(self) -> None:
        os.environ["PATH"] = self._original_path

    def reload_config(self) -> Dict[str, Any]:
        self.config = config_mod.load_json(self.config_path)
        return self.config

    def prepare_workspace(self) -> Dict[str, Any]:
        return workspace.prepare(
            root=self.root,
            config_path=self.config_path,
            config=self.config,
            task_id=self.task_id,
        )

    def write_task(self, **fields: Any) -> None:
        record = config_mod.load_json(self.task_dir / "task.json")
        record.update(fields)
        util.write_json(self.task_dir / "task.json", record)


# ------------------------------------------------------------- full sequence


class ConnectedFlowTests(FlowCase):
    def test_validate_accepts_the_registered_task(self) -> None:
        errors = config_mod.validate(self.config_path, self.config, root=self.root)
        self.assertEqual(errors, [])

    def test_scaffold_then_validate_then_prepare(self) -> None:
        """A scaffolded task validates, but stays unstartable until resolved."""
        created = taskfactory.scaffold(
            root=self.root,
            config_path=self.config_path,
            config=self.config,
            task_id="integration-kda",
            kind="integration",
            base_commit=self.source["base_commit"],
            objective="Validate the serving path for the accepted kernel.",
            validation_command="python3 bench/correctness.py --help",
            performance_command=None,
            register=True,
        )
        self.assertTrue(created["registered"])
        self.assertFalse(created["startable"])

        config = self.reload_config()
        self.assertEqual(
            config_mod.validate(self.config_path, config, root=self.root), []
        )

        record = config_mod.load_json(
            self.root / created["task_path"] / "task.json"
        )
        with self.assertRaises(util.ToolError):
            lifecycle.require_startable(record, "integration-kda")

    def test_agent_plan_is_ready_and_spawns_nothing(self) -> None:
        self.prepare_workspace()
        plan = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertTrue(
            plan["ready"], "not ready: %s" % plan["not_ready_reasons"]
        )
        self.assertFalse(plan["would_spawn"])

        argv = plan["invocation"]["argv"]
        self.assertEqual(argv[0], "claude")
        self.assertIn("--plugin-dir", argv)
        self.assertIn("/humanize:start-rlcr-loop", plan["invocation"]["prompt"])
        # The configured writer effort and model reach the argv.
        self.assertIn("--model", argv)
        self.assertIn("opus", argv)
        self.assertIn("--effort", argv)
        self.assertIn("max", argv)
        # Identity is recorded and hashable.
        for key in ("toolchain_sha256", "contract_sha256", "plan_sha256"):
            self.assertEqual(len(plan["identity"][key]), 64)

    def test_agent_plan_is_reproducible(self) -> None:
        self.prepare_workspace()
        first = agent.plan(self.root, self.config_path, self.config, self.task_id)
        second = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertEqual(
            first["identity"]["plan_sha256"], second["identity"]["plan_sha256"]
        )

    def test_dry_run_and_start_plan_identities_agree(self) -> None:
        """The documented property: a dry run and the real start share an identity.

        ``start`` chooses a session id and passes ``--session-id <uuid>``, which a
        dry run does not have. Removing only the *value* would leave a dangling
        ``--session-id`` flag in one argv and not the other, so the hashes would
        differ and the documented guarantee would be false.
        """
        self.prepare_workspace()
        dry = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertNotIn("--session-id", dry["invocation"]["argv"])

        session_id = util.new_uuid()
        materialized = agent.plan(
            self.root,
            self.config_path,
            self.config,
            self.task_id,
            session_id=session_id,
        )
        # The real invocation carries the flag and its value.
        argv = materialized["invocation"]["argv"]
        self.assertIn("--session-id", argv)
        self.assertIn(session_id, argv)
        self.assertEqual(argv[argv.index("--session-id") + 1], session_id)

        # The identity hash is nevertheless the same.
        self.assertEqual(
            dry["identity"]["plan_sha256"],
            materialized["identity"]["plan_sha256"],
            "the dry-run and start plan identities disagree, so the documented "
            "reproducibility property does not hold",
        )

    def test_two_starts_with_different_session_ids_share_one_identity(self) -> None:
        self.prepare_workspace()
        first = agent.plan(
            self.root,
            self.config_path,
            self.config,
            self.task_id,
            session_id=util.new_uuid(),
        )
        second = agent.plan(
            self.root,
            self.config_path,
            self.config,
            self.task_id,
            session_id=util.new_uuid(),
        )
        self.assertNotEqual(
            first["invocation"]["argv"], second["invocation"]["argv"]
        )
        self.assertEqual(
            first["identity"]["plan_sha256"], second["identity"]["plan_sha256"]
        )

    def test_strip_session_id_removes_the_flag_and_its_value(self) -> None:
        argv = [
            "claude",
            "-p",
            "prompt",
            "--session-id",
            "abc-123",
            "--model",
            "opus",
        ]
        self.assertEqual(
            agent.strip_session_id(argv),
            ["claude", "-p", "prompt", "--model", "opus"],
        )
        # Idempotent, and a no-op when the flag is absent.
        stripped = agent.strip_session_id(argv)
        self.assertEqual(agent.strip_session_id(stripped), stripped)

    def test_agent_plan_records_no_secret_values(self) -> None:
        self.prepare_workspace()
        os.environ["K3_FAKE_API_TOKEN"] = "super-secret"
        self.addCleanup(os.environ.pop, "K3_FAKE_API_TOKEN", None)
        plan = agent.plan(self.root, self.config_path, self.config, self.task_id)
        rendered = json.dumps(plan)
        self.assertNotIn("super-secret", rendered)

    def test_agent_plan_without_a_workspace_is_not_ready(self) -> None:
        plan = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertFalse(plan["ready"])
        self.assertTrue(
            any("workspace" in reason for reason in plan["not_ready_reasons"]),
            plan["not_ready_reasons"],
        )

    def test_archive_mode_workspace_is_refused_for_the_loop(self) -> None:
        """The loop needs commit history, so a tree-only workspace is refused."""
        workspace.prepare(
            root=self.root,
            config_path=self.config_path,
            config=self.config,
            task_id=self.task_id,
            mode=workspace.MODE_ARCHIVE,
        )
        plan = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertFalse(plan["ready"])
        self.assertTrue(
            any("mode clone" in reason for reason in plan["not_ready_reasons"]),
            plan["not_ready_reasons"],
        )

    def test_accepted_task_cannot_start_a_loop(self) -> None:
        self.prepare_workspace()
        self.write_task(status=lifecycle.ACCEPTED, optimization_status="accepted")
        with self.assertRaises(util.ToolError) as caught:
            agent.start(self.root, self.config_path, self.reload_config(), self.task_id)
        message = str(caught.exception)
        self.assertIn("immutable", message)
        self.assertIsNone(agent._load_session(self.task_dir))

    def test_scaffolded_task_cannot_start_a_loop(self) -> None:
        self.prepare_workspace()
        self.write_task(status=lifecycle.SCAFFOLDED)
        with self.assertRaises(util.ToolError):
            agent.start(self.root, self.config_path, self.reload_config(), self.task_id)

    def test_placeholder_workload_cannot_start_a_loop(self) -> None:
        self.prepare_workspace()
        self.write_task(workload_status=lifecycle.PLACEHOLDER)
        with self.assertRaises(util.ToolError):
            agent.start(self.root, self.config_path, self.reload_config(), self.task_id)

    def test_unsupported_reviewer_effort_blocks_without_the_overlay(self) -> None:
        """A non-upstream effort is refused, never silently downgraded.

        The pinned Humanize release accepts only ``xhigh|high|medium|low``. With
        the overlay disabled, asking for ``ultra`` must block with an actionable
        reason rather than quietly running at a lower effort.
        """
        self.prepare_workspace()
        config = json.loads(json.dumps(self.config))
        config["no_commit_overlay"]["enabled"] = False
        config["workflow"]["reviewer"]["effort"] = "ultra"
        plan = agent.plan(self.root, self.config_path, config, self.task_id)
        self.assertFalse(plan["ready"])
        self.assertTrue(
            any("NOT downgraded" in reason for reason in plan["not_ready_reasons"]),
            plan["not_ready_reasons"],
        )

    def test_supported_reviewer_effort_reaches_the_prompt(self) -> None:
        """The configured effort is passed through to the loop command."""
        self.prepare_workspace()
        plan = agent.plan(self.root, self.config_path, self.config, self.task_id)
        self.assertTrue(plan["ready"], plan["not_ready_reasons"])
        self.assertIn("gpt-6-astra:xhigh", plan["invocation"]["prompt"])

    def test_reviewer_codex_home_and_flags_are_frozen_into_plan(self) -> None:
        """The Humanize child cannot fall back to the default Codex profile."""
        self.prepare_workspace()
        config = json.loads(json.dumps(self.config))
        codex_home = self.base / "codex-bak"
        codex_home.mkdir()
        config["workflow"]["reviewer"].update(
            {
                "codex_home": str(codex_home),
                "bypass_sandbox": True,
                "disable_apps": True,
            }
        )
        plan = agent.plan(self.root, self.config_path, config, self.task_id)
        self.assertTrue(plan["ready"], plan["not_ready_reasons"])
        env = plan["invocation"]["env"]
        self.assertEqual(env["CODEX_HOME"], str(codex_home.resolve()))
        self.assertEqual(env["HUMANIZE_CODEX_BYPASS_SANDBOX"], "true")
        self.assertEqual(env["KDA_REVIEWER_MODEL"], "gpt-6-astra")
        self.assertEqual(env["KDA_REVIEWER_EFFORT"], "xhigh")
        self.assertEqual(env["KDA_REVIEWER_DISABLE_APPS"], "true")

    def test_reviewer_shim_contains_forced_codex_bak_flags(self) -> None:
        """The generated literal ``codex`` command is an isolated launcher."""
        reviewer = {
            "model": "gpt-6-astra",
            "effort": "ultra",
            "codex_home": str((self.base / "codex-bak").resolve()),
            "bypass_sandbox": True,
            "disable_apps": True,
        }
        (self.base / "codex-bak").mkdir()
        shim = agent.build_shim_dir(
            self.task_dir,
            "reviewer-shim",
            {
                "codex": str(self.mockbin / "codex"),
            },
            reviewer=reviewer,
        )
        launcher = shim / "codex"
        self.assertTrue(launcher.is_file())
        content = launcher.read_text(encoding="utf-8")
        for marker in (
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            "gpt-6-astra",
            'model_reasoning_effort=\"ultra\"',
            "--disable apps",
        ):
            self.assertIn(marker, content)
        self.assertEqual(
            agent.verify_shim(shim, {
                "codex": str(self.mockbin / "codex"),
            }, reviewer=reviewer),
            [],
        )

    def test_agent_status_reports_no_session_before_start(self) -> None:
        self.prepare_workspace()
        status = agent.status(self.root, self.config, self.task_id)
        self.assertEqual(status["outcome"], "no-session")
        self.assertFalse(status["accepted"])

    def test_stop_refuses_a_wrong_or_recycled_pid(self) -> None:
        self.prepare_workspace()
        agent.agent_dir(self.task_dir).mkdir(parents=True, exist_ok=True)
        util.write_json(
            agent.session_path(self.task_dir),
            {
                "uuid": "fixture",
                "task_id": self.task_id,
                "state": agent.SESSION_RUNNING,
                "pid": os.getpid(),
                # A recorded identity that cannot match this process.
                "process_identity": {
                    "lstart": "Mon Jan  1 00:00:00 1990",
                    "command": "some-other-process",
                },
            },
        )
        with self.assertRaises(util.ToolError) as caught:
            agent.stop(self.root, self.config, self.task_id)
        self.assertIn("refusing to signal", str(caught.exception))


# --------------------------------------------------------------- run adapter


class RunnerFlowTests(FlowCase):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_record = self.prepare_workspace()

    def plan(self, operation: str = "correctness", **kwargs) -> Dict[str, Any]:
        return runner.plan(
            self.root, self.config_path, self.config, self.task_id, operation, **kwargs
        )

    def test_plan_freezes_distinct_roots_and_identity(self) -> None:
        record = self.plan()
        self.assertTrue(record["ready"], record["blockers"])
        self.assertTrue(record["roots"]["distinct"])
        self.assertNotEqual(record["roots"]["baseline"], record["roots"]["candidate"])
        self.assertFalse(record["invocation"]["shell"])
        self.assertFalse(record["would_spawn"])
        self.assertFalse(record["gpu"]["performed_by_this_cli"])
        self.assertEqual(len(record["identity"]["run_sha256"]), 64)

    def test_identity_changes_when_the_candidate_changes(self) -> None:
        first = self.plan()
        candidate = Path(self.workspace_record["workspace"]["absolute_path"])
        (candidate / "python" / "kernels" / "kda.py").write_text(
            "def chunk_kda(q, k, v):\n    return v, k, q\n", encoding="utf-8"
        )
        second = self.plan()
        self.assertNotEqual(
            first["identity"]["inputs_sha256"],
            second["identity"]["inputs_sha256"],
            "a candidate edit left the run identity unchanged",
        )

    def test_identical_trees_warn_that_ab_would_show_no_difference(self) -> None:
        record = self.plan()
        self.assertFalse(record["inputs"]["candidate_differs_from_baseline"])
        self.assertTrue(
            any("byte-identical" in warning for warning in record["warnings"]),
            record["warnings"],
        )

    # --- run id -----------------------------------------------------------

    def test_traversing_run_ids_are_refused(self) -> None:
        for bad in ("../escape", "a/b", "/abs", "..", ".", "", "x" * 200):
            with self.subTest(run_id=bad):
                with self.assertRaises(util.ToolError):
                    self.plan(run_id=bad)

    def test_duplicate_run_id_is_refused(self) -> None:
        first = self.plan(run_id="fixed-run")
        self.assertEqual(first["run_id"], "fixed-run")
        with self.assertRaises(util.ToolError) as caught:
            self.plan(run_id="fixed-run")
        self.assertIn("already exists", str(caught.exception))
        # The first run's record survived intact.
        again = config_mod.load_json(runner.record_path(self.task_dir, "fixed-run"))
        self.assertEqual(
            again["identity"]["run_sha256"], first["identity"]["run_sha256"]
        )

    def test_status_of_an_unknown_run_does_not_create_state(self) -> None:
        runs = self.task_dir / "runtime" / "runs"
        if runs.exists():
            for child in runs.iterdir():
                self.assertTrue(child.is_dir())
        with self.assertRaises(util.ToolError):
            runner.status(self.root, self.config, self.task_id, run_id="absent-run")
        self.assertFalse((runs / "absent-run").exists())

    # --- baseline binding -------------------------------------------------

    def test_mutated_baseline_blocks_the_plan(self) -> None:
        baseline = Path(self.workspace_record["baseline"]["absolute_path"])
        (baseline / "injected.txt").write_text("tampered\n", encoding="utf-8")
        record = self.plan()
        self.assertFalse(record["ready"])
        self.assertTrue(
            any("baseline tree has changed" in item for item in record["blockers"]),
            record["blockers"],
        )

    def test_edited_task_base_commit_blocks_the_plan(self) -> None:
        """Same baseline bytes, different claimed provenance."""
        self.write_task(base_commit=self.source["later_commit"])
        self.write_task(
            source={
                "repository": "https://example.invalid/fixture/sglang.git",
                "branch": "main",
                "commit": self.source["later_commit"],
            }
        )
        record = runner.plan(
            self.root,
            self.config_path,
            self.config,
            self.task_id,
            "correctness",
        )
        self.assertFalse(record["ready"])
        self.assertTrue(
            any("prepared from" in item for item in record["blockers"]),
            record["blockers"],
        )

    def test_baseline_integrity_is_recorded_in_the_plan(self) -> None:
        record = self.plan()
        integrity = record["baseline_integrity"]
        self.assertTrue(integrity["checked"])
        self.assertTrue(integrity["matches_prepared"])
        self.assertTrue(integrity["identity_bound"])
        self.assertEqual(integrity["recorded_task_id"], self.task_id)

    # --- local execution --------------------------------------------------

    def run_to_completion(
        self, operation: str = "correctness", env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Start a run and poll until it genuinely reaches a terminal state.

        ``RUN_UNKNOWN`` is deliberately **not** treated as terminal. It means
        "the outcome could not be determined *yet*", so returning on it would
        report ``exit_code: None`` for a run that is simply still in flight --
        and would let the test finish while the child is still writing into the
        temporary directory.
        """
        for key, value in (env or {}).items():
            os.environ[key] = value
            self.addCleanup(os.environ.pop, key, None)
        started = runner.start(
            self.root, self.config_path, self.config, self.task_id, operation
        )
        terminal = (
            runner.RUN_EXITED,
            runner.RUN_SIGNALLED,
            runner.RUN_CANCELLED,
        )
        deadline = time.time() + 90
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            last = runner.status(
                self.root, self.config, self.task_id, started["run_id"]
            )
            if last["execution"]["state"] in terminal:
                return last
            time.sleep(0.2)
        self.fail(
            "run did not reach a terminal state within the timeout; last state=%s "
            "detail=%s"
            % (
                (last.get("execution") or {}).get("state"),
                (last.get("execution") or {}).get("detail"),
            )
        )

    def test_successful_run_records_execution_and_evidence_separately(self) -> None:
        status = self.run_to_completion()
        self.assertEqual(status["execution"]["exit_code"], 0)
        self.assertTrue(status["execution"]["ran_to_completion"])
        self.assertTrue(status["evidence"]["complete"])
        # Neither is acceptance.
        self.assertFalse(status["accepted"])
        self.assertFalse(status["gates_evaluated"])
        self.assertFalse(status["gpu_performed"])

    def test_zero_exit_with_no_artifact_is_not_complete(self) -> None:
        status = self.run_to_completion(env={"K3_FIXTURE_NO_REPORT": "1"})
        self.assertEqual(status["execution"]["exit_code"], 0)
        self.assertTrue(status["execution"]["ran_to_completion"])
        self.assertFalse(
            status["evidence"]["complete"],
            "a zero exit code with no artifact was reported as complete evidence",
        )
        self.assertTrue(status["evidence"]["artifacts"]["missing"])

    def test_failed_status_in_a_report_is_not_complete(self) -> None:
        status = self.run_to_completion(env={"K3_FIXTURE_STATUS": "failed"})
        self.assertFalse(status["evidence"]["complete"])
        self.assertTrue(
            any("status='failed'" in item or "failed" in item
                for item in status["evidence"]["artifacts"]["incomplete"]),
            status["evidence"]["artifacts"]["incomplete"],
        )

    def test_nonzero_exit_is_reported_as_such(self) -> None:
        status = self.run_to_completion(env={"K3_FIXTURE_EXIT": "3"})
        self.assertEqual(status["execution"]["exit_code"], 3)
        self.assertFalse(status["execution"]["ran_to_completion"])

    def test_directory_artifact_requires_a_passing_summary(self) -> None:
        status = self.run_to_completion("benchmark")
        self.assertTrue(status["evidence"]["complete"], status["evidence"])

        no_summary = self.run_to_completion(
            "benchmark", env={"K3_FIXTURE_NO_SUMMARY": "1"}
        )
        self.assertFalse(
            no_summary["evidence"]["complete"],
            "a report directory without summary.json was treated as complete",
        )

    def test_directory_artifact_requires_referenced_reports(self) -> None:
        status = self.run_to_completion(
            "benchmark", env={"K3_FIXTURE_DROP_REPORT": "1"}
        )
        self.assertFalse(
            status["evidence"]["complete"],
            "a summary naming an absent report was treated as complete",
        )

    def test_zero_case_count_is_never_a_pass(self) -> None:
        status = self.run_to_completion("benchmark", env={"K3_FIXTURE_CASES": "0"})
        self.assertFalse(
            status["evidence"]["complete"],
            "a gate that executed no cases was treated as complete evidence",
        )

    def test_fetch_records_an_inventory_for_the_local_adapter(self) -> None:
        started = self.run_to_completion()
        result = runner.fetch(self.root, self.config, self.task_id)
        self.assertIsNotNone(result["location"])
        inventory = config_mod.load_json(
            runner.run_dir(self.task_dir, result["run_id"]) / "fetch-inventory.json"
        )
        self.assertEqual(inventory["run_id"], result["run_id"])
        self.assertTrue(inventory["evidence_complete"])

    def test_fetch_destination_outside_the_root_is_refused(self) -> None:
        self.run_to_completion()
        with self.assertRaises(util.ToolError):
            runner.fetch(
                self.root,
                self.config,
                self.task_id,
                destination=str(self.base / "outside"),
            )

    # --- site boundary ----------------------------------------------------

    def test_site_adapter_without_a_wrapper_states_the_boundary(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["runner"]["adapter"] = "site-wrapper"
        record = runner.plan(
            self.root, self.config_path, config, self.task_id, "correctness"
        )
        self.assertFalse(record["ready"])
        self.assertIsNotNone(record["site_boundary"])
        self.assertIn("does not implement SSH", record["site_boundary"])
        self.assertIn("resource locking", record["site_boundary"])

    def test_site_wrapper_must_be_an_argv_list(self) -> None:
        with self.assertRaises(util.ToolError):
            runner.SiteWrapperAdapter("ssh host run.sh")

    def test_run_is_managed_by_the_adapter_it_started_with(self) -> None:
        started = self.run_to_completion()
        config = json.loads(json.dumps(self.config))
        config["runner"]["adapter"] = "site-wrapper"
        config["runner"]["wrapper"] = ["true"]
        status = runner.status(
            self.root, config, self.task_id, started["run_id"]
        )
        self.assertEqual(status["adapter"], runner.ADAPTER_LOCAL)
        self.assertIsNotNone(status["adapter_drift"])
        self.assertIn("started by", status["adapter_drift"])


# ------------------------------------------------------------- supervisor


class SupervisorTests(fixtures.TempDirCase):
    """Cancellation must reach the whole process tree, and only it."""

    def setUp(self) -> None:
        super().setUp()
        self.supervisor = (
            fixtures.REAL_ROOT / "tools" / "k3" / "supervise.py"
        )
        self.status_path = self.base / "status.json"

    def _parent_script(self) -> Path:
        """A parent that spawns a grandchild worker, mirroring the harnesses."""
        script = self.base / "parent.py"
        script.write_text(
            "import os, subprocess, sys, time\n"
            "child = subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(300)']\n"
            ")\n"
            "open(sys.argv[1], 'w').write(str(child.pid))\n"
            "time.sleep(300)\n",
            encoding="utf-8",
        )
        return script

    def test_cancel_terminates_child_and_grandchild(self) -> None:
        pid_file = self.base / "grandchild.pid"
        parent = self._parent_script()
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.supervisor),
                str(self.status_path),
                sys.executable,
                str(parent),
                str(pid_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.addCleanup(self._cleanup, process)

        # Wait for the grandchild to exist.
        deadline = time.time() + 30
        grandchild: Optional[int] = None
        while time.time() < deadline:
            if pid_file.is_file():
                text = pid_file.read_text(encoding="utf-8").strip()
                if text:
                    grandchild = int(text)
                    break
            time.sleep(0.1)
        self.assertIsNotNone(grandchild, "grandchild worker never started")
        self.assertTrue(util.pid_alive(grandchild))

        # An unrelated process must survive: cancellation is scoped.
        sentinel = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._cleanup, sentinel)

        os.kill(process.pid, signal.SIGTERM)

        deadline = time.time() + 40
        while time.time() < deadline:
            if not util.pid_alive(grandchild):
                break
            time.sleep(0.2)
        self.assertFalse(
            util.pid_alive(grandchild),
            "the grandchild worker outlived cancellation; on a GPU node it "
            "would still hold the device",
        )
        self.assertTrue(
            util.pid_alive(sentinel.pid),
            "cancellation killed an unrelated process",
        )

        process.wait(timeout=30)
        status = config_mod.load_json(self.status_path)
        self.assertTrue(status["cancelled"])
        self.assertEqual(status["state"], "cancelled")
        self.assertFalse(status["exited_normally"])
        self.assertTrue(status["descendants_cleared"], status["descendants_detail"])

    def test_normal_exit_records_a_durable_zero(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.supervisor),
                str(self.status_path),
                sys.executable,
                "-c",
                "print('done')",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=60)
        status = config_mod.load_json(self.status_path)
        self.assertEqual(status["state"], "exited")
        self.assertEqual(status["exit_code"], 0)
        self.assertTrue(status["exited_normally"])
        self.assertFalse(status["cancelled"])

    def test_nonzero_exit_is_durably_recorded(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.supervisor),
                str(self.status_path),
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=60)
        status = config_mod.load_json(self.status_path)
        self.assertEqual(status["exit_code"], 7)
        self.assertTrue(status["exited_normally"])

    def test_missing_command_is_recorded_not_silent(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.supervisor),
                str(self.status_path),
                str(self.base / "no-such-binary"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=60)
        status = config_mod.load_json(self.status_path)
        self.assertEqual(status["state"], "failed-to-spawn")
        self.assertFalse(status["exited_normally"])

    def _cleanup(self, process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass


# ----------------------------------------------------------------- export


class ExportFlowTests(FlowCase):
    def setUp(self) -> None:
        super().setUp()
        self.prepare_workspace()

    def bundle(self, **kwargs) -> Dict[str, Any]:
        return export.bundle(
            self.root, self.config_path, self.config, self.task_id, **kwargs
        )

    def test_bundle_diffs_against_the_staged_index_without_a_commit(self) -> None:
        # Stage a file, then modify it: the classic unborn-HEAD situation.
        tracked = self.root / "tracked.txt"
        tracked.write_text("staged content\n", encoding="utf-8")
        fixtures.git(["add", "tracked.txt"], cwd=self.root)
        tracked.write_text("working tree content\n", encoding="utf-8")
        staged_before = gitq.staged_listing(self.root)

        manifest = self.bundle(label="first")
        local = manifest["local_changes"]
        self.assertEqual(local["base"], "staged-index")
        self.assertFalse(local["head_exists_at_export"])
        self.assertFalse(local["diff_is_empty"])
        self.assertGreater(local["staged_entry_count"], 0)

        # The index was not touched, and no commit was created.
        self.assertEqual(gitq.staged_listing(self.root), staged_before)
        self.assertFalse(gitq.head_exists(self.root))
        self.assertEqual(manifest["git_writes"]["commits_created"], 0)
        self.assertFalse(manifest["git_writes"]["index_modified"])

    def test_bundle_captures_binary_changes(self) -> None:
        blob = self.root / "data.bin"
        blob.write_bytes(bytes(range(256)))
        fixtures.git(["add", "data.bin"], cwd=self.root)
        blob.write_bytes(bytes(reversed(range(256))))

        manifest = self.bundle(label="binary")
        diff = (
            self.root
            / str(manifest["bundle"])
            / str(manifest["local_changes"]["diff_file"])
        ).read_text(encoding="utf-8", errors="replace")
        self.assertIn("GIT binary patch", diff)
        self.assertGreater(manifest["local_changes"]["binary_patch_sections"], 0)

    def test_bundle_lists_untracked_files_separately(self) -> None:
        (self.root / "untracked-note.md").write_text("note\n", encoding="utf-8")
        manifest = self.bundle(label="untracked")
        listing = (
            self.root
            / str(manifest["bundle"])
            / str(manifest["local_changes"]["untracked_file_list"])
        ).read_text(encoding="utf-8")
        self.assertIn("untracked-note.md", listing)

    def test_bundle_excludes_itself_from_its_own_diff(self) -> None:
        manifest = self.bundle(label="self")
        listing = (
            self.root
            / str(manifest["bundle"])
            / str(manifest["local_changes"]["untracked_file_list"])
        ).read_text(encoding="utf-8")
        self.assertNotIn(str(manifest["bundle"]), listing)

    def test_bundle_reports_no_gpu_and_no_review(self) -> None:
        manifest = self.bundle(label="claims")
        self.assertFalse(manifest["gpu"]["performed"])
        self.assertFalse(manifest["review"]["formal_review_performed"])
        self.assertTrue(
            manifest["publication_safety"]["contains_local_absolute_paths"]
        )

    def test_bundle_makes_missing_proof_explicit(self) -> None:
        manifest = self.bundle(label="missing")
        # The fixture task has no reports/ directory and no evidence manifest.
        self.assertFalse(manifest["complete"])
        self.assertTrue(manifest["missing_proof"])
        inventory = (
            self.root / str(manifest["bundle"]) / "INVENTORY.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Missing proof", inventory)
        self.assertIn("GPU performed   : no", inventory)

    def test_bundle_copies_a_patch_only_when_its_hash_matches(self) -> None:
        deliverables = self.task_dir / "deliverables"
        deliverables.mkdir(parents=True, exist_ok=True)
        patch = deliverables / "accepted-kernel.patch"
        patch.write_text("--- a/x\n+++ b/x\n", encoding="utf-8")

        self.write_task(
            patch_file="deliverables/accepted-kernel.patch",
            patch_sha256=util.sha256_file(patch),
        )
        manifest = self.bundle(label="withpatch")
        self.assertTrue(manifest["patch"]["copied"])
        self.assertTrue(manifest["patch"]["hash_verified"])

        # Now corrupt the patch: it must be refused, not silently copied.
        patch.write_text("--- a/x\n+++ b/x\n+tampered\n", encoding="utf-8")
        manifest = self.bundle(label="tampered")
        self.assertFalse(manifest["patch"]["copied"])
        self.assertFalse(manifest["patch"]["hash_verified"])
        self.assertTrue(
            any("hash mismatch" in item for item in manifest["missing_proof"]),
            manifest["missing_proof"],
        )

    def test_each_bundle_is_a_fresh_directory(self) -> None:
        first = self.bundle(label="fixed")
        with self.assertRaises(util.ToolError) as caught:
            self.bundle(label="fixed")
        self.assertIn("already exists", str(caught.exception))
        self.assertTrue((self.root / str(first["bundle"])).is_dir())

    def test_bundle_records_source_identity_and_availability(self) -> None:
        manifest = self.bundle(label="identity")
        identity = manifest["source_identity"]
        self.assertEqual(identity["task_base_commit"], self.source["base_commit"])
        self.assertTrue(identity["task_base_available_locally"])
        self.assertEqual(len(manifest["toolchain"]["sha256"]), 64)

    def test_unavailable_base_is_explicit_missing_proof(self) -> None:
        self.write_task(base_commit="f" * 40)
        self.write_task(
            source={
                "repository": "https://example.invalid/fixture/sglang.git",
                "branch": "main",
                "commit": "f" * 40,
            }
        )
        manifest = self.bundle(label="nobase")
        self.assertTrue(
            any("NOT available" in item for item in manifest["missing_proof"]),
            manifest["missing_proof"],
        )


if __name__ == "__main__":
    unittest.main()
