"""Regression tests for the no-commit Humanize compatibility overlay.

Two layers are tested:

1. **Guard behaviour** -- the overlay's helper functions are sourced into a real
   Bash shell and exercised directly. These tests use temporary files and mock
   inputs; no model, API, GPU, SSH or network is involved.
2. **Anchored edits** -- the overlay is built against the genuinely pinned
   upstream release and every modified file is syntax-checked. These tests skip
   when ``external/humanize`` is not initialized, so a fresh checkout without
   submodules still passes the suite.

The invariants that matter most here are negative ones: the overlay must fail
*closed* when its configuration is incomplete, and it must not block a review
merely because reviewer prose happens to quote an error string.
"""

from __future__ import annotations

import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from tests import fixtures
from tools.k3 import overlay, toolchain, util

REAL_ROOT = fixtures.REAL_ROOT


def _bash() -> str:
    check = toolchain.bash_check()
    if check["status"] != toolchain.OK:
        raise unittest.SkipTest("bash >= 4 is unavailable: %s" % check["detail"])
    return str(check["path"])


class GuardBehaviourTests(fixtures.TempDirCase):
    """Exercise the overlay guard functions in a real Bash shell."""

    def setUp(self) -> None:
        super().setUp()
        self.bash = _bash()
        self.guard = self.base / "k3-no-commit.sh"
        self.guard.write_text(overlay.GUARD_SOURCE, encoding="utf-8")

        self.plan = self.base / "plan.md"
        self.plan.write_text("frozen plan content\n", encoding="utf-8")
        self.snapshot = self.base / "plan-snapshot.md"
        self.snapshot.write_text("frozen plan content\n", encoding="utf-8")
        self.digest = util.sha256_file(self.plan)

    def run_guard(self, script: str, env: dict = None) -> subprocess.CompletedProcess:
        """Source the guard and run *script*, returning the completed process."""
        body = 'source "%s"\n%s' % (self.guard, script)
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "HOME": str(self.base),
        }
        environment.update(env or {})
        return subprocess.run(
            [self.bash, "-c", body],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    def valid_env(self) -> dict:
        return {
            "K3_NO_COMMIT_REVIEW": "true",
            "K3_PLAN_FILE": str(self.plan),
            "K3_PLAN_SHA256": self.digest,
            "K3_PLAN_SNAPSHOT": str(self.snapshot),
            "K3_PLAN_SNAPSHOT_SHA256": self.digest,
        }

    # ------------------------------------------------------------- opt-in

    def test_guard_is_inert_unless_explicitly_enabled(self) -> None:
        result = self.run_guard('k3_no_commit_enabled && echo ON || echo OFF')
        self.assertEqual(result.stdout.decode().strip(), "OFF")

    def test_disabled_preserves_upstream_status_filter(self) -> None:
        """With the policy off, the upstream exclusion pattern is unchanged."""
        script = (
            "k3_filter_git_status ' M src/a.py\n"
            "?? .humanize/state' '^\\?\\? \\.humanize[-/]'"
        )
        result = self.run_guard(script)
        self.assertEqual(result.stdout.decode().strip(), "M src/a.py")

    # ------------------------------------------------- the central invariant

    def test_enabled_with_valid_config_disables_only_the_commit_gate(self) -> None:
        result = self.run_guard(
            "k3_filter_git_status ' M src/a.py' '^\\?\\? \\.humanize[-/]'",
            env=self.valid_env(),
        )
        self.assertEqual(result.stdout.decode().strip(), "")

    def test_enabled_with_invalid_config_fails_closed(self) -> None:
        """A misconfigured policy must leave the upstream commit gate in force.

        This is the most important negative test in the suite: if the freeze
        configuration is incomplete, the dirty-tree gate must NOT be disabled,
        because the review would otherwise run with no anchor at all.
        """
        for dropped in (
            "K3_PLAN_FILE",
            "K3_PLAN_SHA256",
            "K3_PLAN_SNAPSHOT",
            "K3_PLAN_SNAPSHOT_SHA256",
        ):
            env = self.valid_env()
            del env[dropped]
            with self.subTest(missing=dropped):
                result = self.run_guard(
                    "k3_filter_git_status ' M src/a.py' '^\\?\\? \\.humanize[-/]'",
                    env=env,
                )
                self.assertEqual(
                    result.stdout.decode().strip(),
                    "M src/a.py",
                    "commit gate was disabled despite missing %s" % dropped,
                )

    def test_snapshot_must_be_an_independent_file(self) -> None:
        alias = self.base / "alias.md"
        alias.symlink_to(self.plan)
        for snapshot in (self.plan, alias):
            env = self.valid_env()
            env["K3_PLAN_SNAPSHOT"] = str(snapshot)
            with self.subTest(snapshot=snapshot.name):
                result = self.run_guard(
                    "k3_no_commit_config_violation >/dev/null "
                    "&& echo ACCEPTED || echo REJECTED",
                    env=env,
                )
                self.assertEqual(result.stdout.decode().strip(), "REJECTED")

    def test_digests_must_be_hex_and_must_agree(self) -> None:
        cases = {
            "non-hex": {"K3_PLAN_SHA256": "nothex"},
            "wrong-length": {"K3_PLAN_SHA256": "abc123"},
            "mismatched": {"K3_PLAN_SNAPSHOT_SHA256": "b" * 64},
        }
        for label, override in cases.items():
            env = self.valid_env()
            env.update(override)
            with self.subTest(case=label):
                result = self.run_guard(
                    "k3_no_commit_config_violation >/dev/null "
                    "&& echo ACCEPTED || echo REJECTED",
                    env=env,
                )
                self.assertEqual(result.stdout.decode().strip(), "REJECTED")

    # ------------------------------------------------------------ plan freeze

    def test_clean_plan_passes_and_drift_blocks(self) -> None:
        env = self.valid_env()
        result = self.run_guard(
            "k3_plan_freeze_violation >/dev/null && echo CLEAN || echo BLOCKED",
            env=env,
        )
        self.assertEqual(result.stdout.decode().strip(), "CLEAN")

        self.plan.write_text("tampered content\n", encoding="utf-8")
        result = self.run_guard(
            "k3_plan_freeze_violation >/dev/null && echo CLEAN || echo BLOCKED",
            env=env,
        )
        self.assertEqual(result.stdout.decode().strip(), "BLOCKED")

    def test_snapshot_drift_blocks_independently_of_the_live_plan(self) -> None:
        self.snapshot.write_text("tampered snapshot\n", encoding="utf-8")
        result = self.run_guard(
            "k3_plan_freeze_violation >/dev/null && echo CLEAN || echo BLOCKED",
            env=self.valid_env(),
        )
        self.assertEqual(result.stdout.decode().strip(), "BLOCKED")

    def test_write_block_covers_plan_snapshot_and_alias_only(self) -> None:
        alias = self.base / "alias.md"
        alias.symlink_to(self.plan)
        unrelated = self.base / "kernel.py"
        unrelated.write_text("x = 1\n", encoding="utf-8")

        expectations = {
            self.plan: "BLOCKED",
            self.snapshot: "BLOCKED",
            alias: "BLOCKED",
            unrelated: "ALLOWED",
        }
        for target, expected in expectations.items():
            with self.subTest(target=target.name):
                result = self.run_guard(
                    'k3_plan_write_blocked "%s" && echo BLOCKED || echo ALLOWED'
                    % target,
                    env=self.valid_env(),
                )
                self.assertEqual(result.stdout.decode().strip(), expected)

    # --------------------------------------------------- inconclusive review

    def test_standalone_cancellation_result_blocks_finalize(self) -> None:
        for text in (
            "Review was interrupted.",
            "Request interrupted by user",
            "Cancelled.",
        ):
            log = self.base / "review.log"
            log.write_text(text + "\n", encoding="utf-8")
            with self.subTest(result=text):
                out = self.run_guard(
                    'k3_review_inconclusive_reason "%s"' % log
                ).stdout.decode()
                self.assertTrue(
                    out.strip(), "a standalone %r result was treated as a pass" % text
                )

    def test_reviewer_prose_is_not_mistaken_for_a_cancellation(self) -> None:
        """A review that *discusses* an error string is data, not a verdict.

        Blocking here would make the guard untrustworthy: a reviewer quoting a
        test fixture, or explaining a 403 handler, must still be able to pass a
        clean review.
        """
        cases = {
            "quoted-fixture": (
                'The test asserts "Review was interrupted." is handled.\n'
                "No issues found.\n"
            ),
            "403-in-prose": (
                "The 403 handler correctly retries on rate limit errors.\n"
                "Looks good overall.\n"
            ),
            "error-word-midline": (
                "This error: message is built correctly by the handler.\n"
                "Nothing blocking.\n"
            ),
        }
        for label, text in cases.items():
            log = self.base / "review.log"
            log.write_text(text, encoding="utf-8")
            with self.subTest(case=label):
                out = self.run_guard(
                    'k3_review_inconclusive_reason "%s"' % log
                ).stdout.decode()
                self.assertEqual(
                    out.strip(), "", "false block on reviewer prose (%s)" % label
                )

    def test_final_cli_error_line_blocks_finalize(self) -> None:
        log = self.base / "review.log"
        log.write_text("reviewing...\nerror: stream disconnected\n", encoding="utf-8")
        out = self.run_guard(
            'k3_review_inconclusive_reason "%s"' % log
        ).stdout.decode()
        self.assertTrue(out.strip())

    def test_missing_log_leaves_the_guard_inert(self) -> None:
        """With no log there is nothing to judge, so the guard says nothing."""
        self.assertEqual(
            self.run_guard('k3_review_inconclusive_reason "%s/absent.log"' % self.base)
            .stdout.decode()
            .strip(),
            "",
        )

    def test_logs_without_substantive_content_are_inconclusive(self) -> None:
        """A review with no real output is not a credible pass.

        Upstream hard-errors only on a *zero-byte* log (``test -s``). A log
        holding just a newline or spaces clears that check, matches no
        ``[P0-9]`` marker, and is therefore promoted to a clean pass -- which is
        the gap this guard closes.

        The zero-byte case is included for defence in depth. In the real hook
        chain upstream's own gate fires first and this guard is never reached
        there; ``tests/test_hook_chain.py`` exercises that ordering.
        """
        cases = {
            "zero-byte": "",
            "single-newline": "\n",
            "whitespace-only": "   \n\t\n  \n",
            "punctuation-only": "---\n...\n",
        }
        for label, text in cases.items():
            log = self.base / "review.log"
            log.write_text(text, encoding="utf-8")
            with self.subTest(case=label):
                out = self.run_guard(
                    'k3_review_inconclusive_reason "%s"' % log
                ).stdout.decode()
                self.assertTrue(
                    out.strip(),
                    "a review log with no substantive content (%s) was treated "
                    "as a pass" % label,
                )

    def test_minimal_real_verdict_is_not_blocked(self) -> None:
        """The bar is the lowest possible one, so real verdicts always clear it."""
        for text in ("No issues found.", "LGTM", "ok\n"):
            log = self.base / "review.log"
            log.write_text(text + "\n", encoding="utf-8")
            with self.subTest(verdict=text):
                out = self.run_guard(
                    'k3_review_inconclusive_reason "%s"' % log
                ).stdout.decode()
                self.assertEqual(
                    out.strip(), "", "a real verdict %r was blocked" % text
                )

    # ------------------------------------------------------------- effort

    def test_configured_review_effort_is_preserved_not_downgraded(self) -> None:
        result = self.run_guard('k3_review_effort ultra')
        self.assertEqual(result.stdout.decode().strip(), "ultra")

    def test_absent_review_effort_falls_back_to_the_upstream_default(self) -> None:
        result = self.run_guard("k3_review_effort ''")
        self.assertEqual(result.stdout.decode().strip(), "high")


class AnchoredEditTests(unittest.TestCase):
    """Build the overlay against the genuinely pinned upstream release."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = REAL_ROOT / "external" / "humanize"
        if not (cls.upstream / ".claude-plugin" / "plugin.json").is_file():
            raise unittest.SkipTest(
                "external/humanize is not initialized; skipping overlay build"
            )
        from tools.k3 import config as config_mod

        cls.config_path, cls.config = config_mod.resolve_config(
            "config/project.example.json", root=REAL_ROOT
        )
        cls.record = overlay.build(REAL_ROOT, cls.config_path, cls.config)

    def test_every_anchored_edit_applied_exactly_once(self) -> None:
        """An anchor that no longer matches must fail the build, not be skipped."""
        applied = self.record["edits_applied"]
        expected = len(overlay._edits())
        self.assertEqual(
            len(applied),
            expected,
            "expected %d anchored edits, recorded %d" % (expected, len(applied)),
        )

    def test_overlay_is_derived_from_the_pinned_commit(self) -> None:
        pinned = self.config["dependencies"]["humanize"]["commit"]
        self.assertEqual(self.record["upstream"]["commit"], pinned)

    def test_every_modified_file_is_valid_shell(self) -> None:
        """Anchored replacement that produced broken shell would be silent."""
        bash = _bash()
        plugin = REAL_ROOT / str(self.record["overlay"]["plugin_dir"])
        targets = sorted(self.record["changed_files"]) + sorted(
            self.record["added_files"]
        )
        self.assertTrue(targets)
        for relative in targets:
            with self.subTest(file=relative):
                completed = subprocess.run(
                    [bash, "-n", str(plugin / relative)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    "syntax error in %s: %s"
                    % (relative, completed.stderr.decode("utf-8", "replace")),
                )

    def test_upstream_checkout_is_not_modified(self) -> None:
        """The overlay is a copy; the pinned submodule must be untouched."""
        current = util.tree_digest(
            util.tree_hashes(self.upstream, skip=(".git", "__pycache__"))
        )
        self.assertEqual(current, self.record["upstream"]["tree_digest"])

    def test_build_is_content_addressed_and_idempotent(self) -> None:
        """A rebuild reuses the verified build instead of replacing it."""
        again = overlay.build(REAL_ROOT, self.config_path, self.config)
        self.assertTrue(again["reused_existing_build"])
        self.assertEqual(again["fingerprint"], self.record["fingerprint"])

    def test_verify_reports_tampering(self) -> None:
        result = overlay.verify(REAL_ROOT, self.config)
        self.assertEqual(result["status"], "ok")

        plugin = REAL_ROOT / str(self.record["overlay"]["plugin_dir"])
        guard = plugin / overlay.GUARD_RELPATH
        original = guard.read_text(encoding="utf-8")
        try:
            guard.write_text(original + "\n# tampered\n", encoding="utf-8")
            tampered = overlay.verify(REAL_ROOT, self.config)
            self.assertEqual(tampered["status"], "failed")
            self.assertTrue(tampered["problems"])
        finally:
            guard.write_text(original, encoding="utf-8")

    def test_patch_records_the_full_diff(self) -> None:
        patch = REAL_ROOT / str(self.record["patch"]["path"])
        self.assertTrue(patch.is_file())
        self.assertEqual(util.sha256_file(patch), self.record["patch"]["sha256"])
        text = patch.read_text(encoding="utf-8")
        # Every changed file must appear in the auditable diff.
        for relative in self.record["changed_files"]:
            self.assertIn(relative, text)

    def test_overlay_explicitly_disabled_resolves_pinned_upstream(self) -> None:
        # The project example enables this overlay because it is required for
        # the configured ``ultra`` reviewer effort.  The resolver must still
        # honour an explicit opt-out for callers that need pristine upstream.
        config = deepcopy(self.config)
        config["no_commit_overlay"]["enabled"] = False
        plugin, provenance = overlay.resolve_plugin_dir(REAL_ROOT, config)
        self.assertFalse(provenance["no_commit_overlay"])
        self.assertEqual(plugin.resolve(), self.upstream.resolve())


if __name__ == "__main__":
    unittest.main()
