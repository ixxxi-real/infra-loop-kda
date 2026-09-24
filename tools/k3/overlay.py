#!/usr/bin/env python3
"""Project-owned Humanize compatibility overlay for a no-commit user policy.

The pinned Humanize release requires a clean working tree and a commit per
round, and its effort parser only recognises ``xhigh|high|medium|low``. This
project's policy is that the agent does not commit. Rather than editing the
submodule, mutating anything global, or silently downgrading the configured
reviewer effort, the overlay copies the pinned plugin into an **ignored**
workspace and applies a small, audited set of anchored edits.

What the overlay changes, and nothing more:

1. Widens the effort parser so a non-upstream effort (for example ``ultra``)
   is accepted instead of rejected or silently downgraded. Three sites.
2. Stops the *code-review* phase from hardcoding ``high``, which would
   otherwise downgrade the configured reviewer effort even when state carries
   a higher one.
3. Neutralises **only** the dirty-tree commit requirement, and only while
   ``K3_NO_COMMIT_REVIEW=true``.
4. Points ``codex review`` at ``--uncommitted`` instead of ``--base <ref>``
   when running under the no-commit policy, so the review sees the actual
   uncommitted work.
5. Adds a plan-freeze guard: the live plan and an independent snapshot are both
   re-hashed at the prompt and stop gates, and any drift blocks.

What the overlay deliberately does **not** touch: review failure handling,
empty-output handling, severity/`[P0-9]` parsing, phase transitions, summary
and contract validation, BitLesson checks, branch checks, or the finalize and
complete transitions. No verdict or loop state is ever written here -- the
original hook remains the only component that decides and records outcomes.

Every edit is anchored to exact upstream text and must match exactly once. If
upstream content shifts, the build fails closed and reports the precise gap
instead of producing a silently different overlay.
"""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as config_mod
from . import paths, util

#: Effort values accepted after the overlay widens the parser.
WIDENED_EFFORT_PATTERN = "^(ultra|xhigh|high|medium|low)$"
WIDENED_EFFORT_MESSAGE = "ultra, xhigh, high, medium, low"

GUARD_RELPATH = "hooks/lib/k3-no-commit.sh"

GUARD_SOURCE = r'''#!/usr/bin/env bash
#
# Project-owned compatibility helpers for infra-loop-kda.
#
# Added by 'k3ctl overlay build'. This file is NOT part of upstream Humanize.
# It is opt-in: every behavioural change is gated on K3_NO_COMMIT_REVIEW=true,
# so with the variable unset the plugin behaves exactly like the pinned
# upstream release.
#
# This helper never writes loop state, never records a verdict, and never
# decides whether a review passed.

# True only when the no-commit policy is explicitly enabled.
k3_no_commit_enabled() {
    [[ "${K3_NO_COMMIT_REVIEW:-false}" == "true" ]]
}

# Validate the frozen-plan configuration required by the no-commit policy.
#
# Prints a violation and returns 1 when the policy is enabled but the plan
# freeze is not fully, correctly configured. This is mandatory: with the
# dirty-tree commit requirement disabled, an unfrozen plan would mean a review
# with no anchor at all. Fail closed.
#
# Requirements:
#   * all four variables present;
#   * both digests exactly 64 lowercase hex characters;
#   * live plan and snapshot are DIFFERENT files (not the same path, not a
#     symlink or hardlink to each other) -- the snapshot must be independent;
#   * both digests identical, because the snapshot is an independent copy of
#     the same authorized plan content.
k3_no_commit_config_violation() {
    k3_no_commit_enabled || return 0

    local plan="${K3_PLAN_FILE:-}"
    local plan_sha="${K3_PLAN_SHA256:-}"
    local snapshot="${K3_PLAN_SNAPSHOT:-}"
    local snapshot_sha="${K3_PLAN_SNAPSHOT_SHA256:-}"
    local problems=""

    [[ -n "$plan" ]]         || problems="${problems}- K3_PLAN_FILE is not set"$'\n'
    [[ -n "$plan_sha" ]]     || problems="${problems}- K3_PLAN_SHA256 is not set"$'\n'
    [[ -n "$snapshot" ]]     || problems="${problems}- K3_PLAN_SNAPSHOT is not set"$'\n'
    [[ -n "$snapshot_sha" ]] || problems="${problems}- K3_PLAN_SNAPSHOT_SHA256 is not set"$'\n'

    if [[ -n "$plan_sha" && ! "$plan_sha" =~ ^[0-9a-f]{64}$ ]]; then
        problems="${problems}- K3_PLAN_SHA256 is not a 64-character lowercase hex digest"$'\n'
    fi
    if [[ -n "$snapshot_sha" && ! "$snapshot_sha" =~ ^[0-9a-f]{64}$ ]]; then
        problems="${problems}- K3_PLAN_SNAPSHOT_SHA256 is not a 64-character lowercase hex digest"$'\n'
    fi

    if [[ -n "$plan" && -n "$snapshot" ]]; then
        # -ef is true for the same file, including via symlink or hardlink.
        if [[ "$plan" == "$snapshot" ]] || [[ -e "$plan" && -e "$snapshot" && "$plan" -ef "$snapshot" ]]; then
            problems="${problems}- the plan snapshot is the same file as the live plan; an independent snapshot is required"$'\n'
        else
            local resolved_plan resolved_snapshot
            resolved_plan="$(k3_resolve_path "$plan")"
            resolved_snapshot="$(k3_resolve_path "$snapshot")"
            if [[ -n "$resolved_plan" && "$resolved_plan" == "$resolved_snapshot" ]]; then
                problems="${problems}- the plan snapshot resolves to the live plan; an independent snapshot is required"$'\n'
            fi
        fi
    fi

    if [[ -n "$plan_sha" && -n "$snapshot_sha" && "$plan_sha" != "$snapshot_sha" ]]; then
        problems="${problems}- the live plan and snapshot digests differ; the snapshot must be an independent copy of the same authorized plan"$'\n'
    fi

    if [[ -n "$problems" ]]; then
        printf '# No-Commit Policy Misconfigured\n\n'
        printf 'K3_NO_COMMIT_REVIEW=true disables the dirty-tree commit requirement,\n'
        printf 'so a fully frozen plan is mandatory. Detected:\n\n'
        printf '%s\n' "$problems"
        printf 'Refusing to run an unanchored review.\n'
        return 1
    fi
    return 0
}

# Portable SHA-256 of a file. Empty output means "could not hash".
k3_sha256() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" 2>/dev/null | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" 2>/dev/null | awk '{print $1}'
    fi
}

# Filter a cached 'git status --porcelain' block.
#
# Under the no-commit policy this returns empty, which disables the dirty-tree
# commit requirement and NOTHING else. With the policy off it applies the
# upstream exclusion pattern unchanged.
k3_filter_git_status() {
    local status_text="$1"
    local pattern="$2"
    # Fail closed: the commit gate is only disabled when the no-commit policy is
    # enabled AND its frozen-plan configuration is valid. A misconfigured policy
    # leaves the upstream gate fully in force.
    if k3_no_commit_enabled && k3_no_commit_config_violation >/dev/null; then
        return 0
    fi
    printf '%s' "$status_text" | grep -vE "$pattern" || true
}

# Instruction text for "what to do with your work before exiting".
#
# Disabling the dirty-tree commit gate is not enough on its own: upstream's
# prompts still tell the writer to commit, which directly contradicts the
# no-commit policy and would produce a commit the gate no longer catches. Under
# the policy this returns an explicit prohibition instead; with the policy off
# it returns the upstream text unchanged.
k3_commit_instruction() {
    local upstream_text="$1"
    if k3_no_commit_enabled && k3_no_commit_config_violation >/dev/null; then
        printf '%s' 'Do NOT create a git commit, and do not stage anything. This project uses a no-commit policy: leave your work uncommitted in the working tree so the reviewer reads the actual uncommitted diff. Preserve the reviewed source and evidence exactly as reviewed, and do not modify the frozen plan.'
    else
        printf '%s' "$upstream_text"
    fi
}

# Writer-facing note appended to a *writer* prompt under the no-commit policy.
#
# Printed only when the policy is enabled and correctly configured; otherwise it
# prints nothing and the upstream prompt is delivered verbatim.
#
# This is deliberately additive and writer-only. Reviewer prompts are never
# touched: weakening a review instruction would be a worse failure than the bug
# this fixes.
k3_no_commit_prompt_note() {
    k3_no_commit_enabled || return 0
    k3_no_commit_config_violation >/dev/null || return 0
    cat <<'K3_NOTE_EOF'

## Source Control Policy (project override)

This project uses a **no-commit policy**. It replaces any instruction above
that tells you to commit.

- Do NOT create a git commit. Do not run `git commit` or `git add`, and do not
  stage anything.
- Leave your work uncommitted in the working tree: the reviewer reads the
  actual uncommitted diff.
- Preserve the reviewed source and evidence exactly as reviewed.
- Do NOT modify the frozen implementation plan or its snapshot.

Everything else in this prompt still applies, including writing your summary
and addressing every review finding.
K3_NOTE_EOF
}

# Replacement trailing instructions for the finalize phase.
#
# The finalize prompt asks for post-review refactoring *and* a commit. Both are
# wrong here: refactoring after the review would change reviewed source, and
# committing defeats the policy. Prints nothing when the policy is off.
k3_finalize_prompt_override() {
    k3_no_commit_enabled || return 0
    k3_no_commit_config_violation >/dev/null || return 0
    local finalize_summary_file="$1"
    cat <<K3_FINALIZE_EOF

---

## Finalize Instructions (project override)

These instructions replace the "Before Exiting" and simplification sections
above.

1. Do NOT refactor, simplify, or otherwise change the reviewed source now.
   The review applied to the source exactly as it stands; changing it here
   would invalidate that review.
2. Do NOT create a git commit and do NOT stage anything. Leave the work
   uncommitted in the working tree.
3. Do NOT modify the frozen implementation plan or its snapshot.
4. Preserve the reviewed source and all evidence exactly as reviewed.
5. Complete all \`[mainline]\` and \`[blocking]\` tasks. \`[queued]\` tasks may
   remain only if documented as non-blocking follow-up work.
6. Write your finalize summary to: **${finalize_summary_file}**

Your summary should record what was implemented, which files changed, what was
validated, and any remaining items.
K3_FINALIZE_EOF
}

# Effort actually used for the code-review phase.
#
# Upstream hardcodes "high" here, which downgrades a higher configured effort.
# Preserve whatever the loop state carries, falling back to upstream's value.
k3_review_effort() {
    local configured="${1:-}"
    if [[ -n "$configured" ]]; then
        printf '%s' "$configured"
    else
        printf '%s' "high"
    fi
}

# Resolve a path to its canonical form, following symlinks including the leaf.
# Symlink aliases must resolve to the same path as their target so an alias
# cannot be used to bypass a write block.
k3_resolve_path() {
    local path="$1"
    [[ -n "$path" ]] || return 0
    if command -v realpath >/dev/null 2>&1; then
        realpath "$path" 2>/dev/null || printf '%s' "$path"
        return 0
    fi
    local dir base
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    ( cd "$dir" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$base" ) || printf '%s' "$path"
}

# Return 0 when a Write/Edit target is the frozen plan or its snapshot.
#
# Inert when no plan paths are configured. Compares both verbatim and
# fully-resolved forms so a symlink alias is caught too.
k3_plan_write_blocked() {
    local candidate="$1"
    [[ -n "$candidate" ]] || return 1
    local plan="${K3_PLAN_FILE:-}"
    local snapshot="${K3_PLAN_SNAPSHOT:-}"
    if [[ -z "$plan" && -z "$snapshot" ]]; then
        return 1
    fi
    local resolved target resolved_target
    resolved="$(k3_resolve_path "$candidate")"
    for target in "$plan" "$snapshot"; do
        [[ -n "$target" ]] || continue
        if [[ "$candidate" == "$target" ]]; then
            return 0
        fi
        resolved_target="$(k3_resolve_path "$target")"
        if [[ -n "$resolved" && "$resolved" == "$resolved_target" ]]; then
            return 0
        fi
    done
    return 1
}

# Print a reason when a code review ended in a cancellation/interruption
# result rather than a clean pass.
#
# Upstream treats "no [P0-9] marker found" as a pass. The narrow gap this
# closes is: the CLI exits zero, produces output, but that output is a
# standalone cancellation notice. Ordinary provider failures (403, rate
# limits, transport errors) already fail the non-zero-exit gate in
# run_codex_code_review, or the empty-stdout gate in detect_review_issues;
# they are deliberately NOT re-matched here.
#
# Matching rules, chosen so reviewer prose is never mistaken for a verdict:
#   * only the final response lines are examined, not arbitrary earlier
#     tool output;
#   * a cancellation marker must be the ENTIRE line. A review that quotes
#     "Review was interrupted." inside code, a test fixture or a sentence is
#     data, not a result;
#   * an explicit CLI error is recognised only when the last non-empty line
#     is itself the error.
#
# This guard only ever ADDS a block; it never turns a failure into a pass.
k3_review_inconclusive_reason() {
    local log_file="$1"
    [[ -f "$log_file" ]] || return 0

    # A review whose entire output carries no substantive content is not a
    # credible "no issues found" verdict.
    #
    # Upstream hard-errors only on a *zero-byte* log (test -s). A log holding
    # just a newline, spaces, or punctuation passes that check, matches no
    # [P0-9] marker, and is therefore promoted to a clean pass. The bar here is
    # deliberately the lowest possible one -- at least one alphanumeric
    # character somewhere in the log -- so a genuine verdict such as
    # "No issues found." always clears it and no real review is ever blocked.
    if ! grep -q '[[:alnum:]]' "$log_file" 2>/dev/null; then
        printf '%s' 'The code review produced no substantive output (no alphanumeric content at all), so its result is inconclusive. A review counts as a pass only when it actually reported a verdict.'
        return 0
    fi

    local tail_text
    tail_text="$(tail -n 5 "$log_file" 2>/dev/null | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//' || true)"
    [[ -n "$tail_text" ]] || return 0

    # Standalone cancellation/interruption result line (whole-line match).
    if printf '%s\n' "$tail_text" | grep -qE '^(Review was interrupted\.?|Request interrupted( by user)?\.?|Interrupted\.?|Aborted\.?|Cancell?ed\.?|Review cancell?ed\.?)$'; then
        printf '%s' 'The code review ended with a standalone cancellation/interruption result instead of a completed review. No [P0-9] findings counts as a pass only when the review actually finished.'
        return 0
    fi

    # Explicit final CLI error: the last non-empty line is the error itself.
    local last_line
    last_line="$(printf '%s\n' "$tail_text" | grep -v '^$' | tail -n 1)"
    if [[ -n "$last_line" ]] && printf '%s' "$last_line" | grep -qE '^(([Ee]rror|ERROR|[Ff]atal|FATAL)([: ]|$)|stream error([: ]|$)|codex: (error|fatal)([: ]|$))'; then
        printf '%s' 'The code review ended with an explicit CLI error as its final output instead of a completed review. No [P0-9] findings counts as a pass only when the review actually finished.'
        return 0
    fi
    return 0
}

# Print the plan-freeze violation text and return 1 when the live plan or its
# independent snapshot no longer matches its frozen SHA-256.
#
# Both paths and both hashes are supplied by the caller through the
# environment, so this stays generic: no task, plan path or scope rule is
# hardcoded. When no hashes are configured the guard is inert.
k3_plan_freeze_violation() {
    # Under the no-commit policy the freeze configuration is mandatory, so
    # check it first. This makes the single guard call site cover both.
    local config_violation
    if ! config_violation="$(k3_no_commit_config_violation)"; then
        printf '%s' "$config_violation"
        return 1
    fi

    local plan="${K3_PLAN_FILE:-}"
    local plan_sha="${K3_PLAN_SHA256:-}"
    local snapshot="${K3_PLAN_SNAPSHOT:-}"
    local snapshot_sha="${K3_PLAN_SNAPSHOT_SHA256:-}"
    local violations=""
    local actual=""

    if [[ -n "$plan" && -n "$plan_sha" ]]; then
        if [[ ! -f "$plan" ]]; then
            violations="${violations}- Live plan is missing: $plan"$'\n'
        else
            actual="$(k3_sha256 "$plan")"
            if [[ -z "$actual" ]]; then
                violations="${violations}- Cannot hash the live plan: $plan"$'\n'
            elif [[ "$actual" != "$plan_sha" ]]; then
                violations="${violations}- Live plan changed: $plan"$'\n'
                violations="${violations}  expected $plan_sha"$'\n'
                violations="${violations}  actual   $actual"$'\n'
            fi
        fi
    fi

    if [[ -n "$snapshot" && -n "$snapshot_sha" ]]; then
        if [[ ! -f "$snapshot" ]]; then
            violations="${violations}- Plan snapshot is missing: $snapshot"$'\n'
        else
            actual="$(k3_sha256 "$snapshot")"
            if [[ -z "$actual" ]]; then
                violations="${violations}- Cannot hash the plan snapshot: $snapshot"$'\n'
            elif [[ "$actual" != "$snapshot_sha" ]]; then
                violations="${violations}- Plan snapshot changed: $snapshot"$'\n'
                violations="${violations}  expected $snapshot_sha"$'\n'
                violations="${violations}  actual   $actual"$'\n'
            fi
        fi
    fi

    if [[ -n "$violations" ]]; then
        printf '# Plan Freeze Violation\n\n'
        printf 'The implementation plan is frozen for this loop. Detected:\n\n'
        printf '%s\n' "$violations"
        printf 'Restore the plan to its frozen content before continuing.\n'
        printf 'The goal and acceptance criteria must not drift mid-loop.\n'
        return 1
    fi
    return 0
}
'''


class Edit:
    """A single anchored replacement in one overlay file."""

    def __init__(self, relpath: str, anchor: str, replacement: str, note: str):
        self.relpath = relpath
        self.anchor = anchor
        self.replacement = replacement
        self.note = note

    def apply(self, text: str) -> str:
        count = text.count(self.anchor)
        if count != 1:
            raise util.ToolError(
                "overlay anchor did not match exactly once in %s (found %d).\n"
                "Edit: %s\n"
                "The pinned upstream text has changed; the overlay is refusing to "
                "guess. Re-derive this anchor against the pinned commit."
                % (self.relpath, count, self.note)
            )
        return text.replace(self.anchor, self.replacement, 1)


def _edits() -> List[Edit]:
    """The complete, ordered overlay edit set."""
    edits: List[Edit] = []

    # ---- 1. widen the effort parser (three independent sites) -------------
    edits.append(
        Edit(
            "hooks/lib/loop-common.sh",
            'if [[ -n "$_cfg_codex_effort" && ! "$_cfg_codex_effort" =~ '
            "^(xhigh|high|medium|low)$ ]]; then\n"
            '    echo "Warning: Invalid codex_effort in merged config: '
            '$_cfg_codex_effort" >&2\n'
            '    echo "  Must be one of: xhigh, high, medium, low" >&2',
            'if [[ -n "$_cfg_codex_effort" && ! "$_cfg_codex_effort" =~ '
            "%s ]]; then\n" % WIDENED_EFFORT_PATTERN
            + '    echo "Warning: Invalid codex_effort in merged config: '
            '$_cfg_codex_effort" >&2\n'
            '    echo "  Must be one of: %s" >&2' % WIDENED_EFFORT_MESSAGE,
            "widen merged-config effort parser to accept a non-upstream effort",
        )
    )
    edits.append(
        Edit(
            "scripts/setup-rlcr-loop.sh",
            'if [[ ! "$CODEX_EFFORT" =~ ^(xhigh|high|medium|low)$ ]]; then\n'
            '    echo "Error: Invalid codex effort: $CODEX_EFFORT" >&2\n'
            '    echo "  Must be one of: xhigh, high, medium, low" >&2',
            'if [[ ! "$CODEX_EFFORT" =~ %s ]]; then\n' % WIDENED_EFFORT_PATTERN
            + '    echo "Error: Invalid codex effort: $CODEX_EFFORT" >&2\n'
            '    echo "  Must be one of: %s" >&2' % WIDENED_EFFORT_MESSAGE,
            "widen setup effort parser so a configured effort is not rejected",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            'if [[ ! "$CODEX_EXEC_EFFORT" =~ ^(xhigh|high|medium|low)$ ]]; then\n'
            '    echo "Error: Invalid codex effort in state file: '
            '$CODEX_EXEC_EFFORT" >&2\n'
            '    echo "  Must be one of: xhigh, high, medium, low" >&2',
            'if [[ ! "$CODEX_EXEC_EFFORT" =~ %s ]]; then\n' % WIDENED_EFFORT_PATTERN
            + '    echo "Error: Invalid codex effort in state file: '
            '$CODEX_EXEC_EFFORT" >&2\n'
            '    echo "  Must be one of: %s" >&2' % WIDENED_EFFORT_MESSAGE,
            "widen stop-hook state effort parser",
        )
    )

    # ---- 2. stop the review phase from downgrading the configured effort --
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            'CODEX_REVIEW_MODEL="$CODEX_EXEC_MODEL"\nCODEX_REVIEW_EFFORT="high"',
            'CODEX_REVIEW_MODEL="$CODEX_EXEC_MODEL"\n'
            "# Overlay: upstream hardcodes \"high\" here, which silently downgrades a\n"
            "# higher configured effort for the code-review phase. Preserve the\n"
            "# effort the loop state actually carries.\n"
            'CODEX_REVIEW_EFFORT="$(k3_review_effort "$CODEX_EXEC_EFFORT")"',
            "preserve the configured reviewer effort in the code-review phase",
        )
    )

    # ---- 3. neutralise only the dirty-tree commit requirement -------------
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            "\n            HUMANIZE_UNTRACKED_PATTERN='^\\?\\? \\.humanize[-/]'\n"
            '            GIT_STATUS_FOR_BLOCK=$(echo "$GIT_STATUS_CACHED" | '
            'grep -vE "$HUMANIZE_UNTRACKED_PATTERN" || true)',
            "\n            HUMANIZE_UNTRACKED_PATTERN='^\\?\\? \\.humanize[-/]'\n"
            "            # Overlay: under the no-commit policy this yields empty, which\n"
            "            # disables the dirty-tree commit requirement and nothing else.\n"
            '            GIT_STATUS_FOR_BLOCK=$(k3_filter_git_status '
            '"$GIT_STATUS_CACHED" "$HUMANIZE_UNTRACKED_PATTERN")',
            "no-commit gate: post-methodology-analysis dirty-tree block",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            "\n    HUMANIZE_UNTRACKED_PATTERN='^\\?\\? \\.humanize[-/]'\n"
            '    GIT_STATUS_FOR_BLOCK=$(echo "$GIT_STATUS_CACHED" | '
            'grep -vE "$HUMANIZE_UNTRACKED_PATTERN" || true)',
            "\n    HUMANIZE_UNTRACKED_PATTERN='^\\?\\? \\.humanize[-/]'\n"
            "    # Overlay: under the no-commit policy this yields empty, which\n"
            "    # disables the dirty-tree commit requirement and nothing else.\n"
            '    GIT_STATUS_FOR_BLOCK=$(k3_filter_git_status '
            '"$GIT_STATUS_CACHED" "$HUMANIZE_UNTRACKED_PATTERN")',
            "no-commit gate: main pre-exit dirty-tree block",
        )
    )

    # ---- 4. review the uncommitted diff instead of a base range -----------
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            '    local review_base="${BASE_COMMIT:-$BASE_BRANCH}"\n'
            '    local review_base_type="branch"\n'
            '    if [[ -n "$BASE_COMMIT" ]]; then\n'
            '        review_base_type="commit"\n'
            "    fi",
            '    local review_base="${BASE_COMMIT:-$BASE_BRANCH}"\n'
            '    local review_base_type="branch"\n'
            '    if [[ -n "$BASE_COMMIT" ]]; then\n'
            '        review_base_type="commit"\n'
            "    fi\n"
            "    # Overlay: with no commits to diff, review the uncommitted work\n"
            "    # directly. Failure, empty-output and severity handling below are\n"
            "    # untouched.\n"
            "    local k3_review_target=()\n"
            "    if k3_no_commit_enabled; then\n"
            "        k3_review_target=(--uncommitted)\n"
            '        review_base_type="uncommitted"\n'
            "    else\n"
            '        k3_review_target=(--base "$review_base")\n'
            "    fi",
            "build the review target array (uncommitted vs base range)",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            '        echo "codex review ${CODEX_DISABLE_HOOKS_ARGS[*]} --base '
            '$review_base ${CODEX_REVIEW_ARGS[*]}"',
            '        echo "codex review ${CODEX_DISABLE_HOOKS_ARGS[*]} '
            '${k3_review_target[*]} ${CODEX_REVIEW_ARGS[*]}"',
            "record the actual review argv in the audit command file",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            '    (cd "$PROJECT_ROOT" && run_with_timeout "$CODEX_TIMEOUT" codex '
            'review "${CODEX_DISABLE_HOOKS_ARGS[@]}" --base "$review_base" '
            '"${CODEX_REVIEW_ARGS[@]}") \\',
            '    (cd "$PROJECT_ROOT" && run_with_timeout "$CODEX_TIMEOUT" codex '
            'review "${CODEX_DISABLE_HOOKS_ARGS[@]}" "${k3_review_target[@]}" '
            '"${CODEX_REVIEW_ARGS[@]}") \\',
            "invoke codex review against the overlay-selected target",
        )
    )

    # ---- 5. plan freeze guard, every phase --------------------------------
    # Placed immediately after state parsing so it also covers the review and
    # finalize phases, which the later commit-gate region can skip entirely.
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            "# Validate critical fields were actually present (not just defaulted)",
            "# Overlay: the plan is frozen for this loop. Any drift in the live plan\n"
            "# or its independent snapshot blocks here, in every phase (normal,\n"
            "# review, finalize). This ADDS a gate; it removes none.\n"
            "if ! K3_FREEZE_REASON=$(k3_plan_freeze_violation); then\n"
            "    jq -n \\\n"
            '        --arg reason "$K3_FREEZE_REASON" \\\n'
            '        --arg msg "Loop: Blocked - the frozen plan changed" \\\n'
            "        '{\n"
            '            "decision": "block",\n'
            '            "reason": $reason,\n'
            '            "systemMessage": $msg\n'
            "        }'\n"
            "    exit 0\n"
            "fi\n"
            "\n"
            "# Validate critical fields were actually present (not just defaulted)",
            "plan freeze guard covering normal, review and finalize phases",
        )
    )

    # ---- 6. an inconclusive review is never a pass ------------------------
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            "    else\n"
            "        # No issues found (exit code 1) - proceed to finalize\n"
            '        echo "Code review passed with no issues. Proceeding to '
            'finalize phase." >&2\n'
            '        enter_finalize_phase "" "$success_msg"\n'
            "    fi",
            "    else\n"
            "        # Overlay: no [P0-9] findings counts as a pass only when the\n"
            "        # review actually completed. An interrupted review, a transport\n"
            "        # error, or a 401/403/429 rejection also produces no marker and\n"
            "        # would otherwise be promoted to a pass.\n"
            "        K3_INCONCLUSIVE=$(k3_review_inconclusive_reason "
            '"$CODEX_REVIEW_LOG_FILE")\n'
            '        if [[ -n "$K3_INCONCLUSIVE" ]]; then\n'
            '            block_review_failure "$round" "$K3_INCONCLUSIVE" '
            '"${CODEX_REVIEW_EXIT_CODE:-N/A}"\n'
            "        fi\n"
            "        # No issues found (exit code 1) - proceed to finalize\n"
            '        echo "Code review passed with no issues. Proceeding to '
            'finalize phase." >&2\n'
            '        enter_finalize_phase "" "$success_msg"\n'
            "    fi",
            "block finalize when the review was inconclusive rather than clean",
        )
    )

    # ---- 7. the frozen plan is read-only for Write and Edit ---------------
    #
    # Built with explicit token replacement rather than %-formatting: the shell
    # body contains its own format specifiers, and %-formatting the whole
    # literal would try to fill those too.
    validator_guard = "\n".join(
        [
            'FILE_PATH=$(echo "$HOOK_INPUT" | jq -r \'.tool_input.file_path // ""\')',
            "",
            "# Overlay: the frozen plan and its independent snapshot are",
            "# read-only for the whole loop, including via symlink aliases.",
            'if k3_plan_write_blocked "$FILE_PATH"; then',
            '    echo "# Frozen Plan Is Read-Only" >&2',
            '    echo "" >&2',
            '    echo "You tried to __ACTION__ $FILE_PATH, which is the frozen" >&2',
            '    echo "implementation plan (or its independent snapshot) for this loop." >&2',
            '    echo "" >&2',
            '    echo "The plan defines the goal and acceptance criteria and must not" >&2',
            '    echo "change mid-loop. Record new information in the round summary or" >&2',
            '    echo "the goal tracker mutable section instead." >&2',
            "    exit 2",
            "fi",
        ]
    )
    for relpath, action in (
        ("hooks/loop-write-validator.sh", "write to"),
        ("hooks/loop-edit-validator.sh", "edit"),
    ):
        edits.append(
            Edit(
                relpath,
                'FILE_PATH=$(echo "$HOOK_INPUT" | jq -r '
                "'.tool_input.file_path // \"\"')",
                validator_guard.replace("__ACTION__", action),
                "block Write/Edit against the frozen plan in %s"
                % Path(relpath).name,
            )
        )
    edits.append(
        Edit(
            "hooks/loop-plan-file-validator.sh",
            'PLAN_FILE="$STATE_PLAN_FILE"',
            'PLAN_FILE="$STATE_PLAN_FILE"\n'
            "\n"
            "# Overlay: re-check the frozen plan and its independent snapshot on every\n"
            "# prompt, so a change made through any tool is caught at the next gate.\n"
            "if ! K3_FREEZE_REASON=$(k3_plan_freeze_violation); then\n"
            "    jq -n \\\n"
            '        --arg reason "$K3_FREEZE_REASON" \\\n'
            "        '{\n"
            '            "decision": "block",\n'
            '            "reason": $reason\n'
            "        }'\n"
            "    exit 0\n"
            "fi",
            "plan freeze guard at the prompt gate",
        )
    )

    # ---- 8. stop instructing the writer to commit -------------------------
    #
    # Disabling the dirty-tree gate is necessary but not sufficient: upstream's
    # prompts still tell the writer to commit. Left alone, the writer would
    # create a commit the gate no longer catches, silently defeating the policy.
    #
    # These edits apply *after* rendering, so they cover the template path and
    # the shell-fallback path with one edit each. They are writer-only and
    # additive: no reviewer prompt or review instruction is touched, because
    # weakening a review would be a worse failure than the bug being fixed.
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            "    fi\n"
            "\n"
            "    jq -n \\\n"
            '        --arg reason "$finalize_prompt" \\',
            "    fi\n"
            "\n"
            "    # Overlay: append the project finalize instructions, which replace\n"
            "    # the post-review refactoring and commit steps above. Refactoring\n"
            "    # after the review would change reviewed source. Inert when the\n"
            "    # no-commit policy is off.\n"
            '    finalize_prompt="${finalize_prompt}$(k3_finalize_prompt_override '
            '"$finalize_summary_file")"\n'
            "\n"
            "    jq -n \\\n"
            '        --arg reason "$finalize_prompt" \\',
            "finalize prompt: replace refactor/commit steps under the policy",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            '    append_task_tag_routing_note "$next_prompt_file"',
            "    # Overlay: writer-only no-commit note, appended after rendering so\n"
            "    # it covers both the template and fallback paths.\n"
            '    k3_no_commit_prompt_note >> "$next_prompt_file"\n'
            '    append_task_tag_routing_note "$next_prompt_file"',
            "review-phase prompt: append the no-commit note",
        )
    )
    edits.append(
        Edit(
            "hooks/loop-codex-stop-hook.sh",
            'load_and_render_safe "$TEMPLATE_DIR" "claude/next-round-footer.md" '
            '"$FOOTER_FALLBACK" \\\n'
            '    "NEXT_SUMMARY_FILE=$NEXT_SUMMARY_FILE" >> "$NEXT_PROMPT_FILE"\n'
            'append_task_tag_routing_note "$NEXT_PROMPT_FILE"',
            'load_and_render_safe "$TEMPLATE_DIR" "claude/next-round-footer.md" '
            '"$FOOTER_FALLBACK" \\\n'
            '    "NEXT_SUMMARY_FILE=$NEXT_SUMMARY_FILE" >> "$NEXT_PROMPT_FILE"\n'
            "# Overlay: writer-only no-commit note, appended after rendering so it\n"
            "# covers both the template and fallback paths.\n"
            'k3_no_commit_prompt_note >> "$NEXT_PROMPT_FILE"\n'
            'append_task_tag_routing_note "$NEXT_PROMPT_FILE"',
            "next-round footer: append the no-commit note",
        )
    )

    # ---- 9. make the helpers available to every hook ----------------------
    edits.append(
        Edit(
            "hooks/lib/loop-common.sh",
            "unset _cfg_codex_model _cfg_codex_effort _cfg_agent_teams",
            "unset _cfg_codex_model _cfg_codex_effort _cfg_agent_teams\n"
            "\n"
            "# Overlay: project-owned no-commit compatibility helpers. Inert unless\n"
            "# K3_NO_COMMIT_REVIEW=true (plan-freeze checks activate only when the\n"
            "# caller supplies plan hashes).\n"
            '_k3_no_commit_lib="$(dirname "${BASH_SOURCE[0]:-$0}")/k3-no-commit.sh"\n'
            'if [[ -f "$_k3_no_commit_lib" ]]; then\n'
            '    source "$_k3_no_commit_lib"\n'
            "fi\n"
            "unset _k3_no_commit_lib",
            "source the overlay helper library from loop-common.sh",
        )
    )
    return edits


def workspace_dir(root: Path, config: Dict[str, Any]) -> Path:
    settings = config_mod.overlay_settings(config)
    return paths.safe_join(root, str(settings["workspace"]))


def _edit_set_digest() -> str:
    """Digest of the overlay edit set, so a changed adapter changes the build."""
    payload = [
        {
            "relpath": edit.relpath,
            "anchor": edit.anchor,
            "replacement": edit.replacement,
            "note": edit.note,
        }
        for edit in _edits()
    ]
    payload.append({"added": GUARD_RELPATH, "source": GUARD_SOURCE})
    return util.sha256_json(payload)


def fingerprint(upstream_digest: str, edit_digest: str) -> str:
    return util.sha256_text("%s\0%s" % (upstream_digest, edit_digest))[:16]


def build_dir(root: Path, config: Dict[str, Any], fp: str) -> Path:
    return workspace_dir(root, config) / ("build-%s" % fp)


def plugin_dir_for(build_path: Path) -> Path:
    return build_path / "plugin"


def record_file_for(build_path: Path) -> Path:
    return build_path / "overlay.json"


def pointer_path(root: Path, config: Dict[str, Any]) -> Path:
    return workspace_dir(root, config) / "current.json"


def _upstream_dir(root: Path, config: Dict[str, Any]) -> Path:
    settings = config_mod.workflow_settings(config)
    relative = (settings.get("humanize") or {}).get("plugin_dir") or "external/humanize"
    target = paths.safe_join(root, str(relative))
    if not target.is_dir():
        raise util.ToolError(
            "pinned Humanize plugin is not present at %s; run: "
            "git submodule update --init --recursive" % relative
        )
    return target


def _copy_plugin(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        str(source),
        str(destination),
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def build(
    root: Path,
    config_path: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the overlay into a content-addressed directory and record provenance.

    The build directory name is derived from the upstream tree digest and the
    adapter's own edit set, so:

    * rebuilding with identical inputs reuses the existing verified build and
      changes nothing on disk;
    * any change to upstream or to the adapter produces a *new* directory,
      leaving earlier builds -- possibly loaded by a running session --
      untouched.

    Nothing is ever deleted. There is deliberately no force/replace option.
    """
    upstream = _upstream_dir(root, config)
    workspace = workspace_dir(root, config)
    relative_workspace = paths.relative_to_root(root, workspace)

    # The overlay is runtime output: it must live in an ignored location.
    from . import gitq

    if not gitq.is_ignored(root, relative_workspace.rstrip("/") + "/"):
        raise util.ToolError(
            "overlay workspace %s is not ignored by Git.\n"
            "Add it to .gitignore before building: the overlay is runtime output "
            "and must never be tracked." % relative_workspace
        )

    upstream_hashes = util.tree_hashes(upstream, skip=(".git", "__pycache__"))
    upstream_digest = util.tree_digest(upstream_hashes)
    edit_digest = _edit_set_digest()
    fp = fingerprint(upstream_digest, edit_digest)

    target = build_dir(root, config, fp)
    if target.exists():
        existing = verify(root, config, fingerprint_override=fp)
        if existing["status"] == "ok":
            _write_pointer(root, config, fp, target)
            record = config_mod.load_json(record_file_for(target))
            record["reused_existing_build"] = True
            return record
        raise util.ToolError(
            "an overlay build already exists at %s but does not verify:\n  - %s\n"
            "It is not replaced automatically because a running session may be "
            "using it. Remove that directory deliberately, then rebuild."
            % (
                paths.relative_to_root(root, target),
                "\n  - ".join(existing["problems"]),
            )
        )

    # Build into a sibling temporary directory, then move it into place, so an
    # interrupted build never leaves a half-written plugin at the real path.
    staging = target.with_name(target.name + ".partial-%d" % os.getpid())
    if staging.exists():
        shutil.rmtree(str(staging))
    staging.mkdir(parents=True)

    try:
        plugin = plugin_dir_for(staging)
        _copy_plugin(upstream, plugin)

        guard = plugin / GUARD_RELPATH
        guard.parent.mkdir(parents=True, exist_ok=True)
        util.write_text(guard, GUARD_SOURCE)

        changed: Dict[str, Dict[str, str]] = {}
        diff_chunks: List[str] = []
        applied: List[str] = []

        grouped: Dict[str, List[Edit]] = {}
        for edit in _edits():
            grouped.setdefault(edit.relpath, []).append(edit)

        for relpath, group in grouped.items():
            edited = plugin / relpath
            if not edited.is_file():
                raise util.ToolError(
                    "overlay target is missing from the pinned plugin: %s" % relpath
                )
            original = edited.read_text(encoding="utf-8")
            text = original
            for edit in group:
                text = edit.apply(text)
                applied.append("%s: %s" % (relpath, edit.note))
            if text != original:
                util.write_text(edited, text)
                changed[relpath] = {
                    "upstream_sha256": upstream_hashes.get(relpath, ""),
                    "overlay_sha256": util.sha256_text(text),
                }
                diff_chunks.append(
                    "".join(
                        difflib.unified_diff(
                            original.splitlines(keepends=True),
                            text.splitlines(keepends=True),
                            fromfile="upstream/%s" % relpath,
                            tofile="overlay/%s" % relpath,
                            n=3,
                        )
                    )
                )

        guard_text = guard.read_text(encoding="utf-8")
        diff_chunks.append(
            "".join(
                difflib.unified_diff(
                    [],
                    guard_text.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile="overlay/%s" % GUARD_RELPATH,
                    n=3,
                )
            )
        )

        patch_text = "".join(diff_chunks)
        patch_path = staging / "overlay.patch"
        util.write_text(patch_path, patch_text)

        overlay_hashes = util.tree_hashes(plugin, skip=(".git", "__pycache__"))
        settings = config_mod.overlay_settings(config)

        record = {
            "schema": "k3ctl/overlay/2",
            "generated_at": util.utcnow(),
            "config": str(config_path),
            "fingerprint": fp,
            "enabled_in_config": settings["enabled"],
            "opt_in_env": {"K3_NO_COMMIT_REVIEW": "true"},
            "upstream": {
                "path": paths.relative_to_root(root, upstream),
                "commit": _upstream_commit(root, upstream),
                "file_count": len(upstream_hashes),
                "tree_digest": upstream_digest,
            },
            "edit_set_digest": edit_digest,
            "overlay": {
                "build_dir": paths.relative_to_root(root, target),
                "plugin_dir": "%s/plugin" % paths.relative_to_root(root, target),
                "file_count": len(overlay_hashes),
                "tree_digest": util.tree_digest(overlay_hashes),
            },
            "added_files": {GUARD_RELPATH: util.sha256_text(guard_text)},
            "changed_files": changed,
            "edits_applied": applied,
            "patch": {
                "path": "%s/overlay.patch" % paths.relative_to_root(root, target),
                "sha256": util.sha256_text(patch_text),
                "bytes": len(patch_text.encode("utf-8")),
            },
            "preserved": [
                "review failure and empty-output handling",
                "severity and [P0-9] parsing",
                "summary, contract and BitLesson validation",
                "phase transitions, finalize and complete",
                "plan tracking and branch consistency checks",
            ],
            "not_modified": [
                paths.relative_to_root(root, upstream),
                "any global or user-home installation",
                "this repository's Git index",
            ],
            "note": (
                "The adapter never writes loop state or a review verdict. The "
                "original Humanize hook remains the only component that decides "
                "outcomes and performs finalize/complete transitions."
            ),
        }
        util.write_json(record_file_for(staging), record)
        os.replace(str(staging), str(target))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise

    _write_pointer(root, config, fp, target)
    record["reused_existing_build"] = False
    return record


def _write_pointer(
    root: Path, config: Dict[str, Any], fp: str, target: Path
) -> None:
    util.write_json(
        pointer_path(root, config),
        {
            "schema": "k3ctl/overlay-pointer/1",
            "updated_at": util.utcnow(),
            "fingerprint": fp,
            "build_dir": paths.relative_to_root(root, target),
            "plugin_dir": "%s/plugin" % paths.relative_to_root(root, target),
        },
    )


def current_fingerprint(root: Path, config: Dict[str, Any]) -> Optional[str]:
    target = pointer_path(root, config)
    if not target.is_file():
        return None
    try:
        pointer = config_mod.load_json(target)
    except config_mod.ValidationError:
        return None
    value = pointer.get("fingerprint")
    return str(value) if value else None


def _upstream_commit(root: Path, upstream: Path) -> Optional[str]:
    from . import gitq

    if gitq.is_own_root(upstream):
        return gitq.head_commit(upstream)
    return gitq.recorded_gitlinks(root).get(paths.relative_to_root(root, upstream))


def verify(
    root: Path,
    config: Dict[str, Any],
    fingerprint_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-hash a built overlay and compare it with its provenance record."""
    fp = fingerprint_override or current_fingerprint(root, config)
    if not fp:
        raise util.ToolError(
            "no overlay built; run: k3ctl overlay build --config <config>"
        )
    target = build_dir(root, config, fp)
    record_path = record_file_for(target)
    problems: List[str] = []

    if not record_path.is_file():
        return {
            "status": "failed",
            "fingerprint": fp,
            "problems": ["overlay record is missing: %s" % record_path],
            "build_dir": paths.relative_to_root(root, target),
        }
    record = config_mod.load_json(record_path)
    plugin = plugin_dir_for(target)

    if not plugin.is_dir():
        return {
            "status": "failed",
            "fingerprint": fp,
            "problems": ["overlay plugin directory is missing: %s" % plugin],
            "build_dir": paths.relative_to_root(root, target),
        }

    overlay_hashes = util.tree_hashes(plugin, skip=(".git", "__pycache__"))
    current_digest = util.tree_digest(overlay_hashes)
    recorded_digest = (record.get("overlay") or {}).get("tree_digest")
    if recorded_digest and current_digest != recorded_digest:
        problems.append(
            "overlay tree digest changed since it was built "
            "(recorded %s, current %s)" % (recorded_digest[:12], current_digest[:12])
        )

    for relpath, hashes in (record.get("changed_files") or {}).items():
        actual = overlay_hashes.get(relpath)
        if actual is None:
            problems.append("changed file is missing from the overlay: %s" % relpath)
        elif actual != hashes.get("overlay_sha256"):
            problems.append("changed file no longer matches its record: %s" % relpath)

    for relpath, digest in (record.get("added_files") or {}).items():
        actual = overlay_hashes.get(relpath)
        if actual is None:
            problems.append("added file is missing from the overlay: %s" % relpath)
        elif actual != digest:
            problems.append("added file no longer matches its record: %s" % relpath)

    # Upstream must still be the pinned tree the overlay was derived from.
    try:
        upstream = _upstream_dir(root, config)
    except util.ToolError as exc:
        problems.append(str(exc))
    else:
        upstream_digest = util.tree_digest(
            util.tree_hashes(upstream, skip=(".git", "__pycache__"))
        )
        recorded_upstream = (record.get("upstream") or {}).get("tree_digest")
        if recorded_upstream and upstream_digest != recorded_upstream:
            problems.append(
                "pinned upstream tree changed since the overlay was built "
                "(recorded %s, current %s); rebuild the overlay"
                % (recorded_upstream[:12], upstream_digest[:12])
            )

    # The adapter's own edit set must still be the one that produced this build.
    recorded_edits = record.get("edit_set_digest")
    if recorded_edits and recorded_edits != _edit_set_digest():
        problems.append(
            "the overlay adapter changed since this build was produced; "
            "rebuild to get a new build directory"
        )

    patch_info = record.get("patch") or {}
    if patch_info.get("path"):
        patch_path = root / str(patch_info["path"])
        if not patch_path.is_file():
            problems.append("overlay patch is missing: %s" % patch_info["path"])
        elif util.sha256_file(patch_path) != patch_info.get("sha256"):
            problems.append("overlay patch no longer matches its recorded hash")

    return {
        "status": "ok" if not problems else "failed",
        "fingerprint": fp,
        "problems": problems,
        "build_dir": paths.relative_to_root(root, target),
        "overlay_path": paths.relative_to_root(root, plugin),
        "record": paths.relative_to_root(root, record_path),
        "upstream_commit": (record.get("upstream") or {}).get("commit"),
        "changed_file_count": len(record.get("changed_files") or {}),
        "added_file_count": len(record.get("added_files") or {}),
        "edits_applied": record.get("edits_applied") or [],
    }


def resolve_plugin_dir(
    root: Path, config: Dict[str, Any]
) -> Tuple[Path, Dict[str, Any]]:
    """Return the plugin directory the agent should load, plus provenance.

    When the overlay is enabled it must verify cleanly; otherwise the pinned
    upstream plugin is used unchanged.
    """
    settings = config_mod.overlay_settings(config)
    if not settings["enabled"]:
        upstream = _upstream_dir(root, config)
        return upstream, {
            "source": "pinned-upstream",
            "path": paths.relative_to_root(root, upstream),
            "no_commit_overlay": False,
        }
    result = verify(root, config)
    if result["status"] != "ok":
        raise util.ToolError(
            "no-commit overlay is enabled but does not verify:\n  - %s\n"
            "Rebuild it with 'k3ctl overlay build', or disable "
            "no_commit_overlay.enabled." % "\n  - ".join(result["problems"])
        )
    return plugin_dir_for(build_dir(root, config, result["fingerprint"])), {
        "source": "no-commit-overlay",
        "path": result["overlay_path"],
        "no_commit_overlay": True,
        "fingerprint": result["fingerprint"],
        "upstream_commit": result["upstream_commit"],
        "changed_files": result["changed_file_count"],
        "added_files": result["added_file_count"],
    }
