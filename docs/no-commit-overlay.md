# The no-commit compatibility overlay

This project's policy is that the agent does not commit. The pinned Humanize
release requires a clean working tree and a commit per round. Those two facts
conflict, and the overlay is how the conflict is resolved without editing the
submodule, mutating anything global, or silently degrading a review.

## What it is

`k3ctl overlay build` copies the **pinned** Humanize plugin into an ignored
directory and applies a small, fully recorded set of anchored edits:

```text
external/humanize                      (pinned, never modified)
        │  copy
        ▼
external/.overlay/humanize-no-commit/build-<fingerprint>/
├── plugin/          the overlaid copy, loaded via --plugin-dir
├── overlay.patch    the complete diff against the pinned release
└── overlay.json     provenance: upstream commit, per-file hashes, edit list
```

The build directory name is derived from the upstream tree digest **and** the
adapter's own edit set. So rebuilding with identical inputs reuses the existing
verified build and changes nothing on disk, while any change to either produces
a *new* directory — leaving a build that a running session may be using
untouched. Nothing is ever deleted, and there is deliberately no `--force`.

## What it changes, and nothing more

| # | Change | Why |
| --- | --- | --- |
| 1 | Widens the effort parser at three sites | A configured effort such as `ultra` is otherwise rejected outright. It is accepted, never silently downgraded. |
| 2 | Stops the code-review phase hardcoding `high` | Upstream sets `CODEX_REVIEW_EFFORT="high"` unconditionally, which downgrades a higher configured effort. |
| 3 | Neutralises **only** the dirty-tree commit requirement | The policy's actual purpose. Two sites. |
| 4 | Points `codex review` at `--uncommitted` | With no commits to diff, the review must see the real uncommitted work. |
| 5 | Adds a plan-freeze guard at every phase | The plan and an independent snapshot are re-hashed at the prompt, write and stop gates. |
| 6 | Blocks finalize on an inconclusive review | A cancelled or contentless review produces no `[P0-9]` marker and would otherwise be promoted to a pass. |
| 7 | Makes the frozen plan read-only for Write/Edit | Including through symlink aliases. |
| 8 | Replaces writer instructions that say "commit" | Disabling the gate is not enough if the prompt still tells the writer to commit. |

## What it deliberately does not touch

- Review failure handling, empty-output handling, `[P0-9]` severity parsing.
- Summary, contract, BitLesson, branch and phase-transition validation.
- `enter_finalize_phase`, `end_loop`, and every terminal state decision.
- **Any reviewer prompt or review instruction.** Weakening a review would be a
  worse failure than the bug being fixed. Edit 8 is writer-only and additive.

The adapter never writes loop state and never records a verdict. The original
hook remains the only component that decides outcomes.

## Fail-closed, and opt-in

Every change is gated on `K3_NO_COMMIT_REVIEW=true`. With the variable unset the
plugin behaves exactly like the pinned upstream release.

When the policy *is* enabled, its frozen-plan configuration becomes mandatory.
The overlay refuses to disable the commit gate unless all of the following hold:

- `K3_PLAN_FILE`, `K3_PLAN_SHA256`, `K3_PLAN_SNAPSHOT` and
  `K3_PLAN_SNAPSHOT_SHA256` are all set;
- both digests are 64-character lowercase hex;
- the snapshot is a genuinely **different file** from the live plan — not the
  same path, not a symlink or hardlink to it;
- both digests are identical, because the snapshot is an independent copy of the
  same authorized plan.

If any of those fails, the upstream dirty-tree gate stays fully in force. An
unanchored review is never allowed to proceed. This is the single most important
property of the overlay, and it is tested directly.

## Anchored edits fail closed too

Each edit is anchored to exact upstream text and must match **exactly once**. If
upstream content shifts, the build fails with the precise anchor that no longer
matches rather than producing a silently different overlay.

This caught a real mistake during development: four edits were applied to
shell-fallback strings that are dead code whenever the corresponding template
file exists. The build succeeded and tests passed while behaviour was unchanged
— exactly the "argv exists, therefore integrated" trap. Those edits were
replaced with post-render ones that cover both the template and fallback paths.

## Verifying a build

```bash
make overlay-build
make overlay-verify
```

`verify` re-hashes the built tree and compares it against the recorded
provenance, and additionally confirms:

- the pinned upstream tree still matches the digest the overlay was derived from;
- the adapter's own edit set still matches;
- every changed and added file still matches its recorded hash;
- `overlay.patch` still matches its recorded hash.

`agent plan` and `agent start` refuse to run with the overlay enabled unless it
verifies cleanly.

## Enabling it

```json
{
  "no_commit_overlay": {
    "enabled": true,
    "workspace": "external/.overlay/humanize-no-commit"
  }
}
```

Then `agent start` loads the overlaid plugin through `--plugin-dir` and sets the
plan-freeze environment from the prepared workspace's control directory, where
`plan.md` and `plan-snapshot.md` are written as independent files with identical
content.

## The flow it enables

```text
setup-rlcr-loop.sh          clean tree + plan untracked  (satisfied by the
                            locally-excluded .kda-task/ control directory)
        ▼
Claude edits source         tree becomes dirty; no commit is made
        ▼
summary review              codex exec, real gates, Mainline Progress Verdict
        ▼
code review                 codex review --uncommitted
        ▼
finalize                    project instructions: no commit, no post-review
                            refactor, preserve reviewed source and evidence
        ▼
complete-state.md           written by upstream's own end_loop
```

An end-to-end test drives exactly this path against the real pinned hooks and
asserts the repository still has its original single commit, with the candidate
edit still uncommitted in the working tree.

## Scope of the evidence

The hook-chain tests use a synthetic loop, a fixture source repository, and mock
`codex`/`claude` executables. They establish that the **gates and the plumbing**
behave correctly, including that unhappy outcomes stay non-terminal. They
establish nothing about model behaviour, and no GPU work is involved.

One upstream behaviour is asserted directly as a guard against the overlay
becoming unnecessary or wrong: the *unpatched* pinned release is run and must
still reject the configured effort. If upstream ever accepts it natively, that
test fails and the corresponding edit should be removed.
