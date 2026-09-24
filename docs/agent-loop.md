# The agent loop

Claude is the sole writer. Codex reviews. The loop itself is upstream Humanize's
`/humanize:start-rlcr-loop` — this project does not reimplement it, does not
call the Humanize setup script directly, and never writes loop state or a review
verdict of its own.

## What the adapter actually does

`k3ctl agent` is an adapter around a real invocation. Its whole job is to make
that invocation reproducible, verifiable and safe to stop.

```text
k3ctl agent start
  │
  ├── lifecycle gate ────────── refuses scaffolded / accepted / closed tasks
  ├── exclusive task lock ───── one session per task
  ├── readiness check ───────── every blocker reported, nothing spawned yet
  ├── PATH shim ─────────────── the verified codex/bash/jq are the ones that run
  ├── persist session ───────── uuid, argv, cwd, identity, logs, BEFORE launch
  │
  └── spawn: claude -p "/humanize:start-rlcr-loop .kda-task/plan.md …"
                    --plugin-dir <pinned or overlay plugin>
                    --session-id <uuid chosen here>
        │
        └── Claude invokes the official command itself, which runs the real
            setup script, which creates the real loop state.
```

The distinction matters: the adapter launches *Claude*, and Claude runs the
*official* command. Nothing here fabricates a loop directory, a round, a state
file or a verdict.

## `agent plan` — reproducible dry run

`plan` never spawns a process. It freezes and records:

- the exact `argv`, `cwd` and declared non-secret environment;
- the toolchain identity hash, the contract identity hash, and a plan identity
  hash over all of it;
- which plugin directory would be loaded, and whether it is the pinned upstream
  or the audited overlay;
- **every reason the task is not ready**, as a list.

Two `plan` calls on an unchanged configuration produce the same
`identity.plan_sha256`. The session id is deliberately excluded from that hash,
so a dry run and the real start of the same configuration share one identity.

Readiness is a conjunction, and each part fails loudly:

| Requirement | Why |
| --- | --- |
| Task lifecycle permits a start | A scaffolded or accepted task must not run |
| Workload explicitly `resolved` | A placeholder workload cannot drive a loop |
| Workspace prepared in `clone` mode | The loop needs commit history |
| Workspace tree clean, plan untracked | Upstream's setup script requires both |
| Plan file present in the control dir | The loop needs a plan input |
| Task base commit available locally | Otherwise provenance is unprovable |
| `doctor` reports no errors | Missing dependencies are not worked around |
| Reviewer effort accepted by the plugin | It is never silently downgraded |
| No session already running | One writer at a time |

## Why the session id is chosen up front

Humanize binds a loop to a Claude session through the `session_id:` field in its
state file. This adapter generates the session UUID *before* launch and passes
it with `--session-id`, so:

- `agent status` binds to the correct loop directly, instead of guessing that
  the newest one belongs to this session;
- `agent resume` does not have to scrape a session id back out of a stream log;
- two loops in one workspace cannot be confused for each other.

## `agent status` — evidence, not a verdict

`status` reports three independent things and refuses to collapse them:

1. **Process state**, verified against the recorded start time and command. A
   recycled pid is reported as a mismatch, not as the original process.
2. **The plugin's own loop state** — the real `state.md` frontmatter, the phase,
   and whether `end_loop` has written a terminal
   `{complete,cancel,maxiter,stop,unexpected}-state.md`.
3. **Claude's real stream**, summarised from the `stream-json` log.

`accepted` is always `false` here. Acceptance requires the plugin's own review
verdict plus human review; a zero exit code is not acceptance, and neither is a
`complete-state.md` on its own.

## `agent resume` — the same loop, never a new one

Resume rebuilds its invocation from the values **frozen at start** — writer
settings, plugin path, toolchain paths — not from whatever the configuration
says now. Configuration drift is reported rather than silently followed.

It refuses to run when:

- the recorded process is still alive (stop it first);
- there is no loop state to resume;
- the loop already reached a terminal state. A completed or cancelled loop is
  never restarted automatically. Review its result, or create a new task.

## `agent stop` — only the process it started

`stop` signals a pid only when the recorded start time **and** command still
match. An incomplete identity record never matches anything, so a partially
written record cannot authorise signalling an unrelated pid. Unrelated
processes and services are never touched.

Stopping is not a verdict: no loop state and no review outcome is written.

## Isolation: where the loop actually runs

The loop runs inside the task's prepared workspace, not in this repository:

```text
projects/<project>/<stage>/runtime/
├── workspace/            the candidate clone at the exact base commit
│   ├── .kda-task/        task control files, locally excluded from Git
│   │   ├── plan.md               the live plan
│   │   ├── plan-snapshot.md      an independent frozen copy
│   │   ├── contract.md, task.json, prompt.md
│   │   └── harness/              snapshotted measurement scripts
│   └── .humanize/rlcr/   the plugin's own loop state
├── baseline/             the immutable baseline tree at the same commit
└── agent/                plan.json, session.json, task.lock, logs/
```

`.kda-task/` is excluded through the clone's `.git/info/exclude`, which is
local-only. That is what lets the plan be untracked *and* the tree clean at
once — satisfying upstream's two start conditions with no commit and no index
write anywhere.

## The PATH shim

Humanize's hooks invoke the literal names `codex`, `jq` and `bash`. Recording a
verified toolchain proves nothing if the hooks resolve a different binary at
runtime, so `start` builds a small directory of shims and prepends it to
`PATH`. The `codex` shim is a generated reviewer launcher when isolation is
configured. It sets the configured `CODEX_HOME` and prepends the exact review
flags, while `jq` and `bash` remain symlinks to the verified binaries. A shell
alias such as `codex-bak` is not relied upon because aliases are not inherited
by the subprocesses spawned by Humanize.

## What is tested, and what is not

Tested against the genuinely pinned Humanize release, with mock model CLIs:

- the real hook chain from `setup-rlcr-loop.sh` through summary review, code
  review and finalize, to a real `complete-state.md`, with **zero commits**;
- that a review with findings, a failed review, an empty review, an inconclusive
  review and a drifted plan each fail to complete the loop;
- that the frozen plan is read-only at the prompt, write and stop gates.

Not exercised: any real model call, any GPU work, any site runner. The mock
CLIs prove the *plumbing and the gates*, not model behaviour.
