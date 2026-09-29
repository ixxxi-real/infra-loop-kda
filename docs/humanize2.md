# Humanize2 block

Infra Loop-KDA now carries two independent Humanize integrations:

| block | runtime | state owner | entrypoint |
| --- | --- | --- | --- |
| `humanize` | PolyArch Humanize1 Claude plugin | `.humanize/rlcr` | `k3ctl agent ...` |
| `humanize2` | `humanfia/humanize` (`hmz`) | Humanize2's own flow state | `k3ctl humanize2 ...` |

The old block is unchanged.  A Humanize2 run uses the same prepared task clone,
but it does not load the Claude plugin, reuse RLCR state, or silently downgrade
the configured agents.  The default `flows/infra_loop_kda_flame_chase` wrapper
delegates to the pinned `external/flowverse` `flame_chase_agent_cleanup`
implementation. It gives the
isolated Codex-bak profile the first coding turn, an independent Claude session
the review/iteration turn, and Codex performs periodic cleanup. The flow creates
a fresh session for every coding turn; no reviewer inherits the writer's
conversation. The wrapper snapshots and restores `.kda-task` around every
cleaner epoch, so candidate ledgers, benchmark/failure records and profiler
evidence survive cleanup.

The public configuration selects `humanize2` as the default KDA backend.
Therefore `k3ctl agent plan/start/status/resume/stop` and the corresponding
`make agent-*` targets route to Humanize2. The explicit `k3ctl humanize2 ...`
commands remain available when the backend should be visible in a script. Set
`workflow.humanize_backend` to `humanize` only for a legacy Humanize1 run.

## Setup

Humanize2 and its flowverse are pinned submodules:

```sh
git submodule update --init --recursive external/humanize2
git submodule update --init --recursive external/flowverse
uv tool install --editable external/humanize2
hmz --help
```

`hmz` drives already-authenticated coding CLIs.  Infra Loop-KDA does not store
provider tokens.  The example configuration isolates Codex through
`~/.codex-bak`; Humanize2 invokes the normal `codex` binary with this
`CODEX_HOME`, which is the reliable process-level equivalent of the
`codex-bak` shell alias. The default first chaser and cleaner use
`codex/gpt-6-astra:ultra`; the second chaser is `claude/claude-opus-5:max` so
the review has a different model family. Change that path only in a local,
ignored configuration when a different profile is intentional.

The reviewer is advisory: it records a pass/reject reason but cannot promote a
candidate. Promotion requires the immutable-baseline paired benchmark gate,
repeated measurements, correctness and precision evidence, and a final
independent Claude or human review of the frozen evidence bundle.

## Run gates

Prepare the workspace and review the frozen command first:

```sh
python3 tools/k3ctl.py workspace prepare --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 plan --config config/project.local.json --task prefill-kda
```

The plan is read-only and records its identity under
`projects/.../runtime/humanize2/plan.json`.  It refuses missing `hmz`, an
unprepared or dirty clone, unsafe `work_paths`, an unresolved workload, and an
incomplete task contract.  Start is explicit:

```sh
python3 tools/k3ctl.py humanize2 start --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 status --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 stop --config config/project.local.json --task prefill-kda
```

The adapter records only the command, non-secret environment, PID identity and
log path.  Humanize2 remains responsible for agent sessions, trace output,
resume semantics, and flow verdicts.  To continue a stopped flow, use
`humanize2 start --resume`; do not start a second flow in the same workspace.

The cleanup flows rewrite the prepared clone's Git history into an `epoch N`
commit.  This is why the block never targets `external/sglang` directly and
why `plan` is a required review point before `start`.
