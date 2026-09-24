"""Unit coverage for the control-plane modules.

Every test runs against synthetic fixtures under a temporary directory. No GPU,
SSH, model, API or network is involved, and no Git write ever touches this
repository or any external checkout.

The emphasis is on the negative invariants: a missing dependency must not
resolve to a parent repository, a traversing identifier must be refused, a
modified baseline must not pass as the prepared one, and partial output must not
read as complete evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests import fixtures
from tools.k3 import config as config_mod
from tools.k3 import (
    agent,
    export,
    gitq,
    lifecycle,
    paths,
    runner,
    taskfactory,
    toolchain,
    util,
    workspace,
)


# --------------------------------------------------------------------- paths


class PathSafetyTests(fixtures.TempDirCase):
    def test_root_discovery_requires_every_marker(self) -> None:
        incomplete = self.base / "incomplete"
        (incomplete / "tools").mkdir(parents=True)
        (incomplete / "tools" / "k3ctl.py").write_text("x\n", encoding="utf-8")
        self.assertFalse(paths.looks_like_root(incomplete))

    def test_explicit_root_must_be_a_real_checkout(self) -> None:
        with self.assertRaises(paths.PathError):
            paths.find_root(env={"K3_PROJECT_ROOT": str(self.base)})

    def test_root_is_found_from_a_foreign_working_directory(self) -> None:
        project = fixtures.make_project(self.base)
        foreign = self.base / "elsewhere"
        foreign.mkdir()
        found = paths.find_root(
            start=foreign, env={"K3_PROJECT_ROOT": str(project["root"])}
        )
        self.assertEqual(found, project["root"].resolve())

    def test_traversal_and_absolute_paths_are_refused(self) -> None:
        for candidate in ("../escape", "/etc/passwd", "a/../../b", ""):
            with self.subTest(path=candidate):
                with self.assertRaises(paths.PathError):
                    paths.safe_join(self.base, candidate)

    def test_symlinked_component_is_refused(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        inside = self.base / "inside"
        inside.mkdir()
        (inside / "link").symlink_to(outside)
        with self.assertRaises(paths.PathError):
            paths.safe_join(inside, "link/file.txt")


# --------------------------------------------------------------------- util


class TreeHashTests(fixtures.TempDirCase):
    """Symlinks must be visible to hashing, not silently skipped."""

    def test_retargeted_symlink_changes_the_digest(self) -> None:
        tree = self.base / "tree"
        tree.mkdir()
        (tree / "a.txt").write_text("alpha\n", encoding="utf-8")
        (tree / "b.txt").write_text("beta\n", encoding="utf-8")
        link = tree / "link"
        link.symlink_to("a.txt")

        before = util.tree_digest(util.tree_hashes(tree))
        link.unlink()
        link.symlink_to("b.txt")
        after = util.tree_digest(util.tree_hashes(tree))

        self.assertNotEqual(
            before,
            after,
            "a retargeted symlink left the tree digest unchanged, so a changed "
            "source symlink would be invisible to baseline verification",
        )

    def test_symlink_is_recorded_not_skipped(self) -> None:
        tree = self.base / "tree"
        tree.mkdir()
        (tree / "target.txt").write_text("x\n", encoding="utf-8")
        (tree / "link").symlink_to("target.txt")
        hashes = util.tree_hashes(tree)
        self.assertIn("link", hashes)
        self.assertIn("target.txt", hashes)

    def test_symlink_digest_does_not_collide_with_file_content(self) -> None:
        """A link to "x" must not hash like a file containing "x"."""
        left = self.base / "left"
        left.mkdir()
        (left / "entry").symlink_to("x")

        right = self.base / "right"
        right.mkdir()
        (right / "entry").write_text("x", encoding="utf-8")

        self.assertNotEqual(
            util.tree_hashes(left)["entry"], util.tree_hashes(right)["entry"]
        )

    def test_relative_in_tree_symlinks_are_supported(self) -> None:
        """Real source trees contain them; SGLang tracks three."""
        tree = self.base / "tree"
        (tree / "nested").mkdir(parents=True)
        (tree / "nested" / "real.txt").write_text("content\n", encoding="utf-8")
        (tree / "alias").symlink_to("nested/real.txt")
        hashes = util.tree_hashes(tree)
        self.assertEqual(len(hashes), 2)
        self.assertIn("alias", hashes)


class ProcessIdentityTests(unittest.TestCase):
    def test_incomplete_identity_never_matches(self) -> None:
        for recorded in (None, {}, {"lstart": ""}, {"command": "x"}):
            with self.subTest(recorded=recorded):
                matched, detail = util.process_matches(os.getpid(), recorded)
                self.assertFalse(matched, detail)

    def test_live_process_matches_its_own_identity(self) -> None:
        identity = util.process_identity(os.getpid())
        self.assertIsNotNone(identity)
        matched, detail = util.process_matches(os.getpid(), identity)
        self.assertTrue(matched, detail)


# ------------------------------------------------------------------ doctor


class DoctorTests(fixtures.TempDirCase):
    def test_uninitialized_dependency_is_missing_not_inherited(self) -> None:
        """An empty external dir must not resolve to the parent repository."""
        project = fixtures.make_project(self.base, with_submodules=False)
        report = toolchain.inspect(
            project["root"], project["config_path"], project["config"]
        )
        names = {item["name"]: item for item in report["dependencies"]}
        for dependency in ("dependency-kda", "dependency-humanize"):
            with self.subTest(dependency=dependency):
                self.assertEqual(names[dependency]["status"], toolchain.MISSING)
        self.assertEqual(report["status"], "failed")

    def test_initialized_dependencies_report_ok(self) -> None:
        project = fixtures.make_project(self.base, with_submodules=True)
        report = toolchain.inspect(
            project["root"], project["config_path"], project["config"]
        )
        names = {item["name"]: item for item in report["dependencies"]}
        self.assertEqual(names["dependency-kda"]["status"], toolchain.OK)
        self.assertEqual(names["dependency-humanize"]["status"], toolchain.OK)

    def test_wrong_pin_is_incompatible(self) -> None:
        project = fixtures.make_project(self.base, with_submodules=True)
        config = dict(project["config"])
        dependencies = json.loads(json.dumps(config["dependencies"]))
        dependencies["kda"]["commit"] = "b" * 40
        config["dependencies"] = dependencies
        report = toolchain.inspect(project["root"], project["config_path"], config)
        names = {item["name"]: item for item in report["dependencies"]}
        self.assertEqual(names["dependency-kda"]["status"], toolchain.INCOMPATIBLE)
        self.assertIn("dependency-kda", report["incompatible"])

    def test_dirty_dependency_is_reported_separately_from_the_pin(self) -> None:
        project = fixtures.make_project(self.base, with_submodules=True)
        (project["root"] / "external" / "kda" / "dirty.txt").write_text(
            "local edit\n", encoding="utf-8"
        )
        report = toolchain.inspect(
            project["root"], project["config_path"], project["config"]
        )
        names = {item["name"]: item for item in report["dependencies"]}
        # The pin itself is still correct.
        self.assertEqual(names["dependency-kda"]["status"], toolchain.OK)
        # Dirtiness is a separate, non-fatal observation.
        states = {item["name"] for item in report["dependency_state"]}
        self.assertIn("dependency-kda-worktree", states)
        self.assertEqual(report["status"], "degraded")

    def test_missing_task_base_is_reported_as_unavailable(self) -> None:
        source = fixtures.make_source_repo(self.base)
        project = fixtures.make_project(self.base, source=source)
        fixtures.add_task(project, "decode-kda", base_commit="c" * 40)
        report = toolchain.inspect(
            project["root"],
            project["config_path"],
            project["config"],
            task_id="decode-kda",
        )
        availability = report["source_availability"]
        self.assertFalse(availability["task_base_available"])
        detail = " ".join(item["detail"] for item in availability["checks"])
        self.assertIn("not a substitute", detail)

    def test_identity_is_stable_and_hashable(self) -> None:
        project = fixtures.make_project(self.base, with_submodules=True)
        report = toolchain.inspect(
            project["root"], project["config_path"], project["config"]
        )
        first = toolchain.identity(report)
        second = toolchain.identity(report)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(len(first["sha256"]), 64)


# ------------------------------------------------------------- taskfactory


class TaskFactoryTests(fixtures.TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.source = fixtures.make_source_repo(self.base)
        self.project = fixtures.make_project(self.base, source=self.source)

    def scaffold(self, **overrides):
        kwargs = dict(
            root=self.project["root"],
            config_path=self.project["config_path"],
            config=self.project["config"],
            task_id="decode-kda",
            kind="decode",
            base_commit=self.source["base_commit"],
            objective="Reduce decode latency without changing gate semantics.",
            validation_command="python3 bench/correctness.py --help",
            performance_command="python3 bench/benchmark.py --help",
        )
        kwargs.update(overrides)
        return taskfactory.scaffold(**kwargs)

    def test_fresh_scaffold_is_unstartable(self) -> None:
        result = self.scaffold()
        self.assertFalse(result["startable"])
        record = config_mod.load_json(
            self.project["root"] / result["task_path"] / "task.json"
        )
        self.assertEqual(record["status"], lifecycle.SCAFFOLDED)
        blockers = lifecycle.start_blockers(record)
        self.assertTrue(blockers)
        with self.assertRaises(util.ToolError):
            lifecycle.require_startable(record, "decode-kda")

    def test_scaffold_never_contains_acceptance_fields(self) -> None:
        result = self.scaffold()
        record = config_mod.load_json(
            self.project["root"] / result["task_path"] / "task.json"
        )
        for field in taskfactory.FORBIDDEN_SCAFFOLD_FIELDS:
            self.assertNotIn(field, record)

    def test_scaffold_references_and_hashes_the_kda_prompt(self) -> None:
        result = self.scaffold()
        reference = result["kda_reference"]
        self.assertTrue(reference["prompt_available"])
        self.assertEqual(len(reference["prompt_sha256"]), 64)
        prompt = (
            self.project["root"] / result["task_path"] / "prompt.md"
        ).read_text(encoding="utf-8")
        self.assertIn(reference["prompt_sha256"], prompt)
        self.assertIn("Kernel Design Agents", prompt)

    def test_invalid_ids_and_commits_are_refused_before_any_write(self) -> None:
        for bad_id in ("../escape", "Has Space", "UPPER", "a", "x/y"):
            with self.subTest(task_id=bad_id):
                with self.assertRaises(util.ToolError):
                    self.scaffold(task_id=bad_id)
        with self.assertRaises(util.ToolError):
            self.scaffold(base_commit="abc123")
        # Nothing was created by any refused call.
        created = sorted(p.name for p in (self.project["root"] / "tasks").iterdir())
        self.assertEqual(created, [])

    def test_duplicate_task_id_is_refused(self) -> None:
        self.scaffold()
        config = dict(self.project["config"])
        config["tasks"] = list(config.get("tasks") or []) + [
            {"id": "decode-kda", "kind": "decode", "path": "tasks/decode-kda"}
        ]
        with self.assertRaises(util.ToolError):
            self.scaffold(config=config)

    def test_existing_path_is_never_overwritten(self) -> None:
        self.scaffold()
        with self.assertRaises(util.ToolError):
            self.scaffold()

    def test_registering_into_a_tracked_example_is_refused(self) -> None:
        example = self.project["root"] / "config" / "project.example.json"
        example.write_text(
            json.dumps(self.project["config"], indent=2), encoding="utf-8"
        )
        with self.assertRaises(util.ToolError):
            self.scaffold(config_path=example, register=True)

    def test_dry_run_writes_nothing(self) -> None:
        result = self.scaffold(dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse((self.project["root"] / result["task_path"]).exists())


# --------------------------------------------------------------- workspace


class WorkspaceTests(fixtures.TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.source = fixtures.make_source_repo(self.base)
        self.project = fixtures.make_project(self.base, source=self.source)
        self.task_dir = fixtures.add_task(
            self.project, "decode-kda", base_commit=self.source["base_commit"]
        )

    def prepare(self, **overrides):
        kwargs = dict(
            root=self.project["root"],
            config_path=self.project["config_path"],
            config=self.project["config"],
            task_id="decode-kda",
        )
        kwargs.update(overrides)
        return workspace.prepare(**kwargs)

    def test_prepare_materialises_the_exact_base(self) -> None:
        record = self.prepare()
        self.assertEqual(record["source"]["base_commit"], self.source["base_commit"])
        clone = Path(record["workspace"]["absolute_path"])
        self.assertEqual(gitq.head_commit(clone), self.source["base_commit"])
        self.assertFalse(record["is_ready_verdict"])

    def test_baseline_and_candidate_are_distinct_but_identical(self) -> None:
        record = self.prepare()
        baseline = Path(record["baseline"]["absolute_path"])
        candidate = Path(record["workspace"]["absolute_path"])
        self.assertNotEqual(baseline.resolve(), candidate.resolve())
        self.assertTrue(record["base_tree"]["baseline_matches_candidate"])

    def test_clone_and_archive_modes_agree_on_the_tree(self) -> None:
        clone_record = self.prepare()
        clone_digest = clone_record["base_tree"]["digest"]
        # The baseline is materialised by git archive, the candidate by clone.
        self.assertEqual(clone_record["baseline"]["digest"], clone_digest)

    def test_missing_base_commit_is_a_source_availability_error(self) -> None:
        fixtures.add_task(self.project, "other-task", base_commit="d" * 40)
        with self.assertRaises(workspace.SourceUnavailable) as caught:
            self.prepare(task_id="other-task", config=self.project["config"])
        message = str(caught.exception)
        self.assertIn("source availability error", message)
        self.assertIn("not a substitute", message)

    def test_public_pin_is_never_substituted(self) -> None:
        """A task base that differs from the pin must not fall back to it."""
        config = json.loads(json.dumps(self.project["config"]))
        config["source"]["commit"] = self.source["later_commit"]
        fixtures.add_task(self.project, "pinned-task", base_commit="e" * 40)
        merged = json.loads(json.dumps(self.project["config"]))
        merged["source"]["commit"] = self.source["later_commit"]
        with self.assertRaises(workspace.SourceUnavailable):
            self.prepare(task_id="pinned-task", config=merged)

    def test_existing_workspace_is_never_replaced(self) -> None:
        self.prepare()
        with self.assertRaises(util.ToolError) as caught:
            self.prepare()
        self.assertIn("never deletes", str(caught.exception))

    def test_dirty_source_is_refused_without_the_flag(self) -> None:
        (self.source["path"] / "uncommitted.txt").write_text("x\n", encoding="utf-8")
        with self.assertRaises(util.ToolError) as caught:
            self.prepare()
        self.assertIn("provenance", str(caught.exception))
        record = self.prepare(allow_dirty_source=True)
        self.assertTrue(record["source"]["state"]["dirty"])

    def test_symlinked_destination_is_refused(self) -> None:
        runtime = self.task_dir / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        elsewhere = self.base / "elsewhere"
        elsewhere.mkdir()
        (runtime / "workspace").symlink_to(elsewhere)
        with self.assertRaises(util.ToolError):
            self.prepare()

    def test_control_dir_is_locally_excluded_so_no_commit_is_needed(self) -> None:
        record = self.prepare()
        clone = Path(record["workspace"]["absolute_path"])
        self.assertTrue(record["workspace"]["control_dir_excluded_locally"])
        self.assertFalse(gitq.is_dirty(clone))
        self.assertFalse(
            gitq.is_tracked(clone, "%s/plan.md" % workspace.CONTROL_DIR)
        )

    def test_review_base_ref_exists_in_the_owned_clone_only(self) -> None:
        record = self.prepare()
        clone = Path(record["workspace"]["absolute_path"])
        self.assertTrue(gitq.has_commit(clone, workspace.REVIEW_BASE_BRANCH))
        # The source repository is untouched.
        rc, out, _ = util.git(
            ["branch", "--list", workspace.REVIEW_BASE_BRANCH],
            cwd=self.source["path"],
        )
        self.assertEqual(out.strip(), "")

    def test_plan_and_snapshot_are_independent_files(self) -> None:
        record = self.prepare()
        control = Path(record["workspace"]["absolute_path"]) / workspace.CONTROL_DIR
        plan = control / "plan.md"
        snapshot = control / "plan-snapshot.md"
        self.assertTrue(plan.is_file() and snapshot.is_file())
        self.assertFalse(plan.samefile(snapshot))
        self.assertEqual(util.sha256_file(plan), util.sha256_file(snapshot))


if __name__ == "__main__":
    unittest.main()
