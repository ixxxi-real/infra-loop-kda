"""End-to-end tests of the real Humanize hook chain under the overlay.

These tests execute the genuinely overlaid ``loop-codex-stop-hook.sh`` against a
synthetic loop directory, with a mock ``codex`` on ``PATH``. They assert the
hook's real JSON decisions and the real state-file transitions that
``end_loop`` performs.

Nothing here contacts a model, an API, a GPU, SSH or the network. The ``codex``
binary is a shell script whose output is driven by environment variables, and
every file lives under a temporary directory. A fixture loop is explicitly a
fixture: reaching ``complete-state.md`` here is evidence that the *hook chain*
behaves correctly, never evidence about any kernel.

The invariants under test are mostly negative. A failed, empty, interrupted or
finding-bearing review must **not** reach a terminal completion, and the
overlay must not weaken any of those gates while it disables the dirty-tree
commit requirement.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests import fixtures
from tools.k3 import config as config_mod
from tools.k3 import overlay, toolchain, util

REAL_ROOT = fixtures.REAL_ROOT
LOOP_TIMESTAMP = "2026-09-21_10-00-00"
SESSION_ID = "fixture-session-0001"

#: Terminal state files written by ``end_loop``.
TERMINAL_FILES = tuple(
    "%s-state.md" % reason
    for reason in ("complete", "cancel", "maxiter", "stop", "unexpected")
)

MOCK_CODEX = r'''#!/usr/bin/env bash
# Mock Codex CLI for hook-chain tests. Contacts nothing.
set -euo pipefail

mode="exec"
for arg in "$@"; do
  case "$arg" in
    review) mode="review" ;;
    exec) mode="exec" ;;
  esac
done

if [[ "${1:-}" == "--help" ]]; then
  echo "--disable"
  exit 0
fi

record="${K3_MOCK_RECORD:-/dev/null}"
{
  printf '%s |' "$mode"
  for arg in "$@"; do printf ' %s' "$arg"; done
  printf '\n'
} >> "$record"

if [[ "$mode" == "review" ]]; then
  printf '%s\n' "${K3_MOCK_REVIEW_OUTPUT-No issues found.}"
  exit "${K3_MOCK_REVIEW_EXIT:-0}"
fi

# Summary review (codex exec).
#
# Upstream requires a "Mainline Progress Verdict:" line and reads the LAST line
# as the completion marker. A mock that emits only "COMPLETE" is unrealistic:
# the hook blocks at the verdict gate and the review phase is never reached,
# which would make every downstream assertion pass vacuously.
printf '%s\n' "${K3_MOCK_EXEC_VERDICT-Mainline Progress Verdict: ADVANCED}"
printf '%s\n' "${K3_MOCK_EXEC_OUTPUT-COMPLETE}"
exit "${K3_MOCK_EXEC_EXIT:-0}"
'''

SUMMARY = """\
# Round 0 Summary

## What Was Implemented

A fixture change, for hook-chain testing only.

## Files Changed

- kernel.py

## Validation

Fixture harness only; no GPU run, no real measurement.

## Remaining Items

None.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: fixture summary
"""

GOAL_TRACKER = """\
# Goal Tracker

## IMMUTABLE SECTION

### Ultimate Goal

Exercise the hook chain in a fixture workspace.

### Acceptance Criteria

- AC-1: The hook chain reaches its decisions through the real gates.
- AC-2: No fixture run is treated as real evidence.

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| [mainline] Exercise the hook chain | AC-1 | in_progress | coding | claude | fixture |

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
"""

ROUND_CONTRACT = """\
# Round 0 Contract

- Mainline Objective: exercise the hook chain in a fixture workspace.
- Target ACs: AC-1
- Blocking Side Issues In Scope: none
- Queued Side Issues Out of Scope: none
- Success Criteria: the hook reaches a decision through the real gates.
"""

PLAN = """\
# Fixture plan

## Goal

Exercise the real hook chain against a synthetic loop.

## Acceptance criteria

- AC-1: The hook chain runs its real gates.
- AC-2: Fixture output is never treated as real evidence.

## Steps

1. Drive the stop hook.
2. Assert the decisions.
"""


def _require_upstream() -> Path:
    upstream = REAL_ROOT / "external" / "humanize"
    if not (upstream / "hooks" / "loop-codex-stop-hook.sh").is_file():
        raise unittest.SkipTest(
            "external/humanize is not initialized; skipping hook-chain tests"
        )
    return upstream


def _require_jq() -> str:
    located = util.which("jq")
    if located is None:
        raise unittest.SkipTest("jq is required by the Humanize hooks")
    return located


def _require_bash() -> str:
    check = toolchain.bash_check()
    if check["status"] != toolchain.OK:
        raise unittest.SkipTest("bash >= 4 is unavailable: %s" % check["detail"])
    return str(check["path"])


class HookChainCase(fixtures.TempDirCase):
    """Drives the overlaid stop hook against a synthetic loop."""

    @classmethod
    def setUpClass(cls) -> None:
        _require_upstream()
        cls.bash = _require_bash()
        cls.jq = _require_jq()
        config_path, config = config_mod.resolve_config(
            "config/project.example.json", root=REAL_ROOT
        )
        record = overlay.build(REAL_ROOT, config_path, config)
        cls.plugin = REAL_ROOT / str(record["overlay"]["plugin_dir"])
        cls.stop_hook = cls.plugin / "hooks" / "loop-codex-stop-hook.sh"
        cls.prompt_hook = cls.plugin / "hooks" / "loop-plan-file-validator.sh"
        cls.write_hook = cls.plugin / "hooks" / "loop-write-validator.sh"

    def setUp(self) -> None:
        super().setUp()
        self.repo = self.base / "workspace"
        (self.repo / ".kda-task").mkdir(parents=True)
        fixtures.git(["init", "--quiet", "."], cwd=self.repo)
        (self.repo / "kernel.py").write_text("x = 1\n", encoding="utf-8")
        fixtures.git(["add", "-A"], cwd=self.repo)
        fixtures.git(["commit", "--quiet", "-m", "fixture base"], cwd=self.repo)
        self.base_commit = fixtures.git(
            ["rev-parse", "HEAD"], cwd=self.repo
        ).strip()
        fixtures.git(
            ["branch", "--force", "kda-review-base", self.base_commit], cwd=self.repo
        )

        # The plan and its independent snapshot live in the locally-excluded
        # control directory, so no commit is needed for either.
        git_dir = self.repo / ".git"
        (git_dir / "info").mkdir(parents=True, exist_ok=True)
        (git_dir / "info" / "exclude").write_text("/.kda-task/\n", encoding="utf-8")
        self.plan = self.repo / ".kda-task" / "plan.md"
        self.plan.write_text(PLAN, encoding="utf-8")
        self.snapshot = self.repo / ".kda-task" / "plan-snapshot.md"
        self.snapshot.write_text(PLAN, encoding="utf-8")
        self.plan_digest = util.sha256_file(self.plan)

        self.loop = self.repo / ".humanize" / "rlcr" / LOOP_TIMESTAMP
        self.loop.mkdir(parents=True)
        (self.loop / "plan.md").write_text(PLAN, encoding="utf-8")
        (self.loop / "goal-tracker.md").write_text(GOAL_TRACKER, encoding="utf-8")
        (self.loop / "round-0-contract.md").write_text(
            ROUND_CONTRACT, encoding="utf-8"
        )
        (self.loop / "round-0-summary.md").write_text(SUMMARY, encoding="utf-8")
        (self.repo / ".humanize" / "bitlesson.md").write_text(
            "# BitLesson\n\nNo lessons yet.\n", encoding="utf-8"
        )
        self.state = self.loop / "state.md"
        self.state.write_text(self.state_text(), encoding="utf-8")

        self.mockbin = self.base / "mockbin"
        self.mockbin.mkdir()
        codex = self.mockbin / "codex"
        codex.write_text(MOCK_CODEX, encoding="utf-8")
        codex.chmod(0o755)
        self.record = self.base / "codex-calls.log"
        self.cache = self.base / "cache"
        self.cache.mkdir()

    def state_text(self, **overrides: Any) -> str:
        fields: Dict[str, Any] = {
            "current_round": 0,
            "max_iterations": 42,
            "codex_model": "gpt-6-astra",
            "codex_effort": "ultra",
            "codex_timeout": 120,
            "push_every_round": "false",
            "full_review_round": 5,
            "plan_file": ".kda-task/plan.md",
            "plan_tracked": "false",
            "start_branch": "main",
            "base_branch": "kda-review-base",
            "base_commit": getattr(self, "base_commit", "0" * 40),
            "review_started": "false",
            "ask_codex_question": "false",
            "session_id": SESSION_ID,
            "agent_teams": "false",
            "privacy_mode": "true",
            "bitlesson_required": "true",
            "bitlesson_file": ".humanize/bitlesson.md",
            "bitlesson_allow_empty_none": "true",
            "mainline_stall_count": 0,
            "last_mainline_verdict": "unknown",
            "drift_status": "normal",
            "started_at": "2026-09-21T10:00:00Z",
        }
        fields.update(overrides)
        lines = ["---"]
        lines += ["%s: %s" % (key, value) for key, value in fields.items()]
        lines += ["---", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------- invocation

    def hook_env(self, no_commit: bool = True, **extra: str) -> Dict[str, str]:
        env = {
            "PATH": "%s:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
            % self.mockbin,
            "HOME": str(self.base),
            "CLAUDE_PROJECT_DIR": str(self.repo),
            "XDG_CACHE_HOME": str(self.cache),
            "K3_MOCK_RECORD": str(self.record),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
        if no_commit:
            env.update(
                {
                    "K3_NO_COMMIT_REVIEW": "true",
                    "K3_PLAN_FILE": str(self.plan),
                    "K3_PLAN_SHA256": self.plan_digest,
                    "K3_PLAN_SNAPSHOT": str(self.snapshot),
                    "K3_PLAN_SNAPSHOT_SHA256": self.plan_digest,
                }
            )
        env.update(extra)
        return env

    def run_stop_hook(
        self, env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        payload = json.dumps(
            {
                "session_id": SESSION_ID,
                "transcript_path": str(self.base / "transcript.jsonl"),
                "stop_hook_active": False,
            }
        )
        completed = subprocess.run(
            [self.bash, str(self.stop_hook)],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=env if env is not None else self.hook_env(),
            timeout=180,
        )
        stdout = completed.stdout.decode("utf-8", "replace").strip()
        result: Dict[str, Any] = {
            "rc": completed.returncode,
            "stdout": stdout,
            "stderr": completed.stderr.decode("utf-8", "replace"),
            "decision": None,
            "reason": "",
        }
        if stdout:
            try:
                parsed = json.loads(stdout)
            except ValueError:
                # Some paths emit human-readable text rather than JSON.
                return result
            result["decision"] = parsed.get("decision")
            result["reason"] = parsed.get("reason") or ""
            result["systemMessage"] = parsed.get("systemMessage") or ""
        return result

    def codex_calls(self) -> List[str]:
        if not self.record.is_file():
            return []
        return [
            line.strip()
            for line in self.record.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def codex_modes(self) -> List[str]:
        return [line.split("|", 1)[0].strip() for line in self.codex_calls()]

    def terminal_state_files(self) -> List[str]:
        return [name for name in TERMINAL_FILES if (self.loop / name).is_file()]

    def dirty_the_tree(self) -> None:
        (self.repo / "kernel.py").write_text("x = 2  # candidate edit\n", encoding="utf-8")

    def assert_not_terminal(self, message: str) -> None:
        self.assertEqual(
            self.terminal_state_files(),
            [],
            "%s -- loop reached a terminal state: %s"
            % (message, self.terminal_state_files()),
        )

    def assert_no_finalize(self, message: str) -> None:
        self.assertFalse(
            (self.loop / "finalize-state.md").is_file(),
            "%s -- loop entered the finalize phase" % message,
        )

    def state_file_text(self) -> str:
        """Concatenate whichever state files currently exist."""
        text = ""
        for item in sorted(self.loop.iterdir()):
            if item.name.endswith("state.md") and item.is_file():
                text += item.read_text(encoding="utf-8")
        return text

    def assert_code_review_ran(self) -> List[str]:
        """Require that ``codex review`` actually ran.

        Assertions of the form "X did not finalize the loop" are trivially true
        when no review ever ran. Every such test calls this first, so fixture
        drift that stalls the loop earlier fails loudly instead of turning a
        whole group into vacuous passes.
        """
        calls = [line for line in self.codex_calls() if line.startswith("review")]
        self.assertTrue(
            calls,
            "codex review never ran, so this assertion would be vacuous.\n"
            "codex calls: %s\nloop files: %s"
            % (self.codex_calls(), sorted(i.name for i in self.loop.iterdir())),
        )
        return calls

    def prepare_round_artifacts(self) -> None:
        """Write the round-0 artifacts a real writer would produce."""
        self.dirty_the_tree()
        (self.loop / "goal-tracker.md").write_text(GOAL_TRACKER, encoding="utf-8")
        (self.loop / "round-0-contract.md").write_text(
            ROUND_CONTRACT, encoding="utf-8"
        )
        (self.loop / "round-0-summary.md").write_text(SUMMARY, encoding="utf-8")

    def run_one_cycle(self, **env: str) -> Dict[str, Any]:
        """Run exactly one stop-hook invocation and require it to reach review.

        A single invocation performs the summary review and, once that reports
        COMPLETE, falls through to the code review in the *same* pass. Driving
        the loop across two invocations would mean the first one consumed the
        clean-review path, so a mock configured for the second invocation would
        never be exercised -- and an assertion like "this outcome did not
        finalize" would hold only because the loop was stuck earlier.

        Asserting the exact call sequence keeps that impossible: the review
        under test is the one that ran.
        """
        self.prepare_round_artifacts()
        result = self.run_stop_hook(env=self.hook_env(**env))
        self.assertEqual(
            self.codex_modes(),
            ["exec", "review"],
            "expected exactly one summary review then one code review in a "
            "single invocation.\ncalls: %s\ndecision=%s reason=%s\n"
            "loop files: %s\nstderr tail: %s"
            % (
                self.codex_calls(),
                result.get("decision"),
                (result.get("reason") or "")[:300],
                sorted(item.name for item in self.loop.iterdir()),
                result.get("stderr", "")[-800:],
            ),
        )
        return result


class NoCommitPolicyTests(HookChainCase):
    """The overlay disables the commit gate, and nothing else."""

    def test_dirty_tree_blocks_by_default_without_the_overlay(self) -> None:
        """Upstream behaviour must be intact when the policy is off."""
        self.dirty_the_tree()
        result = self.run_stop_hook(env=self.hook_env(no_commit=False))
        self.assertEqual(result["decision"], "block")
        self.assertIn("commit", result["reason"].lower())
        # The commit gate fires before any review is attempted.
        self.assertEqual(self.codex_modes(), [])
        self.assert_no_finalize("dirty tree without the overlay")

    def test_dirty_tree_proceeds_to_review_under_the_no_commit_policy(self) -> None:
        """The whole point: uncommitted work reaches a real review.

        The assertion targets the *gate* specifically. A plain substring search
        for "commit" is wrong here: downstream prompts legitimately discuss
        committing, so matching anywhere in the reason text conflates the gate
        firing with ordinary prompt prose.
        """
        self.dirty_the_tree()
        result = self.run_stop_hook()
        self.assertNotIn(
            "git not clean",
            result["reason"].lower(),
            "the dirty-tree commit gate still fired under the no-commit policy",
        )
        self.assertNotIn(
            "please commit all changes",
            result["reason"].lower(),
            "the dirty-tree commit gate still fired under the no-commit policy",
        )
        # A real summary review ran.
        self.assertIn("exec", self.codex_modes())

    def test_finalize_instructions_forbid_committing(self) -> None:
        """Disabling the gate is not enough if the prompt still says to commit.

        Upstream's finalize prompt instructs the writer to commit *and* to
        refactor after the review. Under the no-commit policy the commit would
        be exactly the one the gate no longer catches, and the refactor would
        change reviewed source. Both must be replaced.

        The finalize prompt is delivered in the block JSON ``reason``, not
        written to a ``*-prompt.md`` file: the files matching that glob are the
        *reviewer* audit records.
        """
        result = self.run_one_cycle()
        self.assertTrue(
            (self.loop / "finalize-state.md").is_file(),
            "a clean review did not enter the finalize phase",
        )

        reason = (result["reason"] or "").lower()
        self.assertTrue(reason.strip(), "the finalize phase delivered no prompt")
        self.assertIn(
            "do not create a git commit",
            reason,
            "the finalize instructions do not forbid committing",
        )
        self.assertIn(
            "do not refactor",
            reason,
            "the finalize instructions do not forbid post-review refactoring",
        )

    def test_review_findings_prompt_carries_the_no_commit_note(self) -> None:
        """The writer prompt on the findings path must carry the prohibition."""
        self.run_one_cycle(
            K3_MOCK_REVIEW_OUTPUT="[P1] Unchecked index in kernel.py:12"
        )

        # The writer prompt for the next round is a file; reviewer audit files
        # are excluded because they are not writer-facing.
        writer_prompts = "\n".join(
            item.read_text(encoding="utf-8")
            for item in sorted(self.loop.iterdir())
            if item.is_file()
            and item.name.endswith("-prompt.md")
            and "review-prompt" not in item.name
        )
        self.assertTrue(writer_prompts.strip(), "no writer prompt was written")
        self.assertIn(
            "do not create a git commit",
            writer_prompts.lower(),
            "the writer prompt does not carry the no-commit prohibition",
        )

    def test_prompts_keep_upstream_commit_wording_when_policy_is_off(self) -> None:
        """With the policy off, upstream prompt wording must be unchanged."""
        env = self.hook_env(no_commit=False)
        # Commit the candidate edit so the upstream gate is satisfied and the
        # loop can proceed to write prompts.
        self.dirty_the_tree()
        fixtures.git(["add", "-A"], cwd=self.repo)
        fixtures.git(["commit", "--quiet", "-m", "candidate work"], cwd=self.repo)
        self.run_stop_hook(env=env)
        self.run_stop_hook(env=env)
        prompts = "\n".join(
            item.read_text(encoding="utf-8")
            for item in sorted(self.loop.iterdir())
            if item.is_file() and item.name.endswith("-prompt.md")
        )
        if prompts.strip():
            self.assertNotIn(
                "do not create a git commit",
                prompts.lower(),
                "the no-commit prohibition leaked into a default-mode prompt",
            )

    def test_misconfigured_policy_falls_back_to_the_commit_gate(self) -> None:
        """Fail closed: an unanchored review must not be allowed to proceed."""
        self.dirty_the_tree()
        env = self.hook_env()
        del env["K3_PLAN_SNAPSHOT_SHA256"]
        result = self.run_stop_hook(env=env)
        self.assertEqual(result["decision"], "block")
        self.assert_no_finalize("misconfigured no-commit policy")

    def test_review_uses_the_uncommitted_target(self) -> None:
        """The review must see the actual uncommitted work.

        There is no conditional here on purpose: if ``codex review`` never ran,
        that is a failure, not a reason to skip the assertion.
        """
        self.run_one_cycle()
        review_calls = self.assert_code_review_ran()
        self.assertTrue(
            any("--uncommitted" in line for line in review_calls),
            "codex review was not pointed at the uncommitted work: %s"
            % review_calls,
        )
        self.assertFalse(
            any("--base" in line for line in review_calls),
            "codex review still used a --base range under the no-commit policy: %s"
            % review_calls,
        )


class PlanFreezeTests(HookChainCase):
    """The frozen plan is enforced at the real gates."""

    def test_plan_drift_blocks_at_the_stop_gate(self) -> None:
        self.dirty_the_tree()
        self.plan.write_text(PLAN + "\n## Sneaky new goal\n", encoding="utf-8")
        result = self.run_stop_hook()
        self.assertEqual(result["decision"], "block")
        self.assertIn("plan", result["reason"].lower())
        # The freeze gate fires before any review is attempted.
        self.assertEqual(self.codex_modes(), [])
        self.assert_not_terminal("plan drift")

    def test_snapshot_drift_blocks_even_when_the_live_plan_is_intact(self) -> None:
        self.dirty_the_tree()
        self.snapshot.write_text(PLAN + "\n## Divergent snapshot\n", encoding="utf-8")
        result = self.run_stop_hook()
        self.assertEqual(result["decision"], "block")
        self.assert_not_terminal("snapshot drift")

    def test_write_validator_blocks_editing_the_frozen_plan(self) -> None:
        payload = json.dumps(
            {
                "session_id": SESSION_ID,
                "tool_name": "Write",
                "tool_input": {"file_path": str(self.plan), "content": "new"},
            }
        )
        completed = subprocess.run(
            [self.bash, str(self.write_hook)],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(),
            timeout=60,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "read-only", completed.stderr.decode("utf-8", "replace").lower()
        )

    def test_write_validator_allows_editing_source(self) -> None:
        payload = json.dumps(
            {
                "session_id": SESSION_ID,
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(self.repo / "kernel.py"),
                    "content": "x = 3\n",
                },
            }
        )
        completed = subprocess.run(
            [self.bash, str(self.write_hook)],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(),
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "editing source was blocked: %s"
            % completed.stderr.decode("utf-8", "replace"),
        )


class ReviewOutcomeTests(HookChainCase):
    """Every unhappy review outcome must stay non-terminal."""

    def test_failed_summary_review_never_runs_code_review(self) -> None:
        self.prepare_round_artifacts()
        self.run_stop_hook(env=self.hook_env(K3_MOCK_EXEC_EXIT="2"))
        self.assertEqual(
            self.codex_modes(),
            ["exec"],
            "code review ran despite a failed summary review",
        )
        self.assert_no_finalize("failed summary review")
        self.assert_not_terminal("failed summary review")

    def test_summary_review_without_complete_marker_blocks(self) -> None:
        self.prepare_round_artifacts()
        result = self.run_stop_hook(
            env=self.hook_env(
                K3_MOCK_EXEC_OUTPUT="Needs more work on error handling."
            )
        )
        self.assertEqual(result["decision"], "block")
        self.assertEqual(
            self.codex_modes(),
            ["exec"],
            "code review ran despite no COMPLETE marker",
        )
        self.assert_no_finalize("summary review did not report COMPLETE")
        self.assert_not_terminal("summary review did not report COMPLETE")

    def test_review_findings_do_not_finalize(self) -> None:
        self.run_one_cycle(
            K3_MOCK_REVIEW_OUTPUT="[P1] Unchecked index in kernel.py:12"
        )
        self.assert_no_finalize("review reported [P1] findings")
        self.assertEqual(
            self.terminal_state_files(),
            [],
            "a review with [P1] findings reached a terminal state",
        )

    def test_failed_review_command_does_not_finalize(self) -> None:
        self.run_one_cycle(K3_MOCK_REVIEW_EXIT="3")
        self.assert_no_finalize("review command failed")
        self.assertNotIn(
            "complete-state.md",
            self.terminal_state_files(),
            "a failed review command completed the loop",
        )

    def test_empty_review_output_does_not_finalize(self) -> None:
        self.run_one_cycle(K3_MOCK_REVIEW_OUTPUT="")
        self.assert_no_finalize("review produced no output")
        self.assertNotIn(
            "complete-state.md",
            self.terminal_state_files(),
            "an empty review completed the loop",
        )

    def test_interrupted_review_does_not_finalize(self) -> None:
        """The overlay's own added gate: a cancellation is not a clean pass."""
        self.run_one_cycle(K3_MOCK_REVIEW_OUTPUT="Review was interrupted.")
        self.assert_no_finalize("review was interrupted")
        self.assertNotIn(
            "complete-state.md",
            self.terminal_state_files(),
            "an interrupted review completed the loop",
        )

    def test_reviewer_prose_quoting_an_error_still_passes(self) -> None:
        """A clean review that discusses an error string must not be blocked.

        The positive control for the interruption guard: a review that merely
        quotes the marker must still reach finalize, otherwise the guard would
        be blocking legitimate clean reviews.
        """
        self.run_one_cycle(
            K3_MOCK_REVIEW_OUTPUT=(
                'The test asserts "Review was interrupted." is handled.\n'
                "No issues found."
            )
        )
        self.assertTrue(
            (self.loop / "finalize-state.md").is_file(),
            "a clean review that quotes the marker was wrongly blocked; loop "
            "files: %s" % sorted(item.name for item in self.loop.iterdir()),
        )

    def test_clean_review_reaches_finalize(self) -> None:
        """Baseline positive control for the whole group."""
        self.run_one_cycle()
        self.assertTrue(
            (self.loop / "finalize-state.md").is_file(),
            "a clean review did not reach finalize; loop files: %s"
            % sorted(item.name for item in self.loop.iterdir()),
        )


class FullLifecycleTests(HookChainCase):
    """One path from the real setup script to a real terminal completion.

    Every other test in this file starts from a hand-written ``state.md``, which
    proves the gates behave but not that the loop can be *created* under the
    no-commit policy. This test runs the genuine ``setup-rlcr-loop.sh``, binds
    the session through the genuine PostToolUse hook, and drives the genuine
    stop hook through all three phases until ``end_loop`` writes
    ``complete-state.md``.

    A fixture completion is evidence about the hook chain only. It is not
    evidence about any kernel, and no measurement occurs.
    """

    def setUp(self) -> None:
        super().setUp()
        # This test creates the loop itself, so remove the pre-built one.
        import shutil

        shutil.rmtree(self.repo / ".humanize", ignore_errors=True)
        self.setup_script = self.plugin / "scripts" / "setup-rlcr-loop.sh"
        self.post_bash_hook = self.plugin / "hooks" / "loop-post-bash-hook.sh"

    def run_setup(self, *extra: str) -> subprocess.CompletedProcess:
        argv = [
            self.bash,
            str(self.setup_script),
            ".kda-task/plan.md",
            "--codex-model",
            "gpt-6-astra:ultra",
            "--base-branch",
            "kda-review-base",
            # Skip the methodology-analysis phase so the finalize phase ends at
            # complete-state.md directly.
            "--privacy",
            "--max",
            "5",
            *extra,
        ]
        return subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(),
            timeout=180,
        )

    def bind_session(self) -> None:
        """Run the real PostToolUse hook that records session_id in state.md."""
        payload = json.dumps(
            {
                "session_id": SESSION_ID,
                "tool_name": "Bash",
                "tool_input": {"command": str(self.setup_script)},
            }
        )
        completed = subprocess.run(
            [self.bash, str(self.post_bash_hook)],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(),
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "post-bash session binding failed: %s"
            % completed.stderr.decode("utf-8", "replace"),
        )

    def discover_loop(self) -> Path:
        base = self.repo / ".humanize" / "rlcr"
        self.assertTrue(base.is_dir(), "setup did not create the loop directory")
        candidates = sorted(item for item in base.iterdir() if item.is_dir())
        self.assertTrue(candidates, "setup created no loop directory")
        return candidates[-1]

    def test_setup_creates_a_loop_and_binds_the_session(self) -> None:
        completed = self.run_setup()
        self.assertEqual(
            completed.returncode,
            0,
            "setup-rlcr-loop.sh failed: %s"
            % completed.stderr.decode("utf-8", "replace")[-1500:],
        )
        self.loop = self.discover_loop()
        state = self.loop / "state.md"
        self.assertTrue(state.is_file(), "setup wrote no state.md")

        # The configured non-upstream effort survived the real setup script.
        text = state.read_text(encoding="utf-8")
        self.assertIn("codex_effort: ultra", text)
        self.assertIn("base_branch: kda-review-base", text)

        # session_id starts empty and is filled by the real PostToolUse hook.
        self.assertIn("session_id:", text)
        self.bind_session()
        self.assertIn(
            "session_id: %s" % SESSION_ID,
            state.read_text(encoding="utf-8"),
        )

    def test_unpatched_setup_rejects_the_configured_effort(self) -> None:
        """The overlay is load-bearing at setup time too, not only in the hook."""
        upstream_setup = (
            REAL_ROOT / "external" / "humanize" / "scripts" / "setup-rlcr-loop.sh"
        )
        completed = subprocess.run(
            [
                self.bash,
                str(upstream_setup),
                ".kda-task/plan.md",
                "--codex-model",
                "gpt-6-astra:ultra",
                "--base-branch",
                "kda-review-base",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(no_commit=False),
            timeout=120,
        )
        combined = (completed.stdout + completed.stderr).decode("utf-8", "replace")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Invalid codex effort", combined)

    def test_full_lifecycle_reaches_completion_with_no_commit(self) -> None:
        """setup -> summary review -> code review -> finalize -> complete.

        The decisive assertion is the last one: the loop completes while the
        candidate work is still uncommitted, and the repository still has
        exactly the one commit it started with.
        """
        completed = self.run_setup()
        self.assertEqual(
            completed.returncode,
            0,
            "setup failed: %s" % completed.stderr.decode("utf-8", "replace")[-1500:],
        )
        self.loop = self.discover_loop()
        self.bind_session()

        # Claude does its work: edits source, writes the round artifacts.
        self.dirty_the_tree()
        (self.loop / "goal-tracker.md").write_text(GOAL_TRACKER, encoding="utf-8")
        (self.loop / "round-0-contract.md").write_text(
            ROUND_CONTRACT, encoding="utf-8"
        )
        (self.loop / "round-0-summary.md").write_text(SUMMARY, encoding="utf-8")

        # Phase 1: the summary review runs and reports COMPLETE.
        first = self.run_stop_hook()
        self.assertIn(
            "exec",
            self.codex_modes(),
            "summary review never ran: %s" % first["stderr"][-1200:],
        )

        # Phase 2: the code review runs against the uncommitted work.
        self.run_stop_hook()
        review_calls = self.assert_code_review_ran()
        self.assertTrue(
            any("--uncommitted" in line for line in review_calls),
            "code review did not target the uncommitted work: %s" % review_calls,
        )

        # Phase 3: finalize. The hook expects finalize-summary.md.
        finalize_state = self.loop / "finalize-state.md"
        self.assertTrue(
            finalize_state.is_file(),
            "a clean review did not enter the finalize phase; loop files: %s"
            % sorted(item.name for item in self.loop.iterdir()),
        )
        (self.loop / "finalize-summary.md").write_text(SUMMARY, encoding="utf-8")
        self.run_stop_hook()

        # The loop reached a real terminal completion, written by end_loop.
        self.assertTrue(
            (self.loop / "complete-state.md").is_file(),
            "the loop did not complete; loop files: %s"
            % sorted(item.name for item in self.loop.iterdir()),
        )

        # The whole point of the overlay: completion with nothing committed.
        log = fixtures.git(["log", "--oneline"], cwd=self.repo).strip().splitlines()
        self.assertEqual(
            len(log),
            1,
            "the loop created a commit despite the no-commit policy: %s" % log,
        )
        dirty = fixtures.git(
            ["status", "--porcelain"], cwd=self.repo
        ).strip()
        self.assertIn(
            "kernel.py",
            dirty,
            "the candidate edit is no longer uncommitted in the working tree",
        )

    def test_lifecycle_does_not_complete_when_review_finds_issues(self) -> None:
        """The same path, but a real finding must prevent completion."""
        self.assertEqual(self.run_setup().returncode, 0)
        self.loop = self.discover_loop()
        self.bind_session()
        self.dirty_the_tree()
        (self.loop / "goal-tracker.md").write_text(GOAL_TRACKER, encoding="utf-8")
        (self.loop / "round-0-contract.md").write_text(
            ROUND_CONTRACT, encoding="utf-8"
        )
        (self.loop / "round-0-summary.md").write_text(SUMMARY, encoding="utf-8")

        self.run_stop_hook()
        self.run_stop_hook(
            env=self.hook_env(
                K3_MOCK_REVIEW_OUTPUT="[P0] Out-of-bounds write in kernel.py:7"
            )
        )
        self.assert_code_review_ran()
        self.assertFalse(
            (self.loop / "complete-state.md").is_file(),
            "a [P0] finding still completed the loop",
        )


class ConfiguredEffortTests(HookChainCase):
    """A non-upstream effort must reach the reviewer, not be downgraded."""

    def test_configured_effort_is_accepted_and_passed_through(self) -> None:
        self.dirty_the_tree()
        self.run_stop_hook()
        calls = self.codex_calls()
        self.assertTrue(calls, "codex was never invoked")
        self.assertTrue(
            any("model_reasoning_effort=ultra" in line for line in calls),
            "the configured effort did not reach the reviewer: %s" % calls,
        )

    def test_unpatched_upstream_rejects_the_same_effort(self) -> None:
        """Confirms the overlay is load-bearing, not decorative."""
        upstream_hook = (
            REAL_ROOT / "external" / "humanize" / "hooks" / "loop-codex-stop-hook.sh"
        )
        self.dirty_the_tree()
        payload = json.dumps(
            {"session_id": SESSION_ID, "stop_hook_active": False}
        )
        completed = subprocess.run(
            [self.bash, str(upstream_hook)],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo),
            env=self.hook_env(no_commit=False),
            timeout=120,
        )
        combined = (completed.stdout + completed.stderr).decode("utf-8", "replace")
        self.assertIn(
            "Invalid codex effort",
            combined,
            "pinned upstream unexpectedly accepted a non-upstream effort; the "
            "overlay's effort widening may no longer be needed",
        )


if __name__ == "__main__":
    unittest.main()
