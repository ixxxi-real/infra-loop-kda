---
name: humanize2
version: 0.1.0
description: Humanize2 flow integration for Infra Loop-KDA
---

# Humanize2 in Infra Loop-KDA

Humanize2 is a separate `hmz` runtime.  It is not the Claude plugin used by
the existing `agent` commands and it must not share the old `.humanize/rlcr`
state directory.  The control plane prepares the task workspace first, then
the Humanize2 block runs an explicit flow inside that workspace.

The default flow is the project-local protected wrapper
`flows/infra_loop_kda_flame_chase`, which delegates to the pinned
`external/flowverse/flows/flame_chase_agent_cleanup`: the isolated Codex-bak
profile takes the first coding turn, an independent Claude session reviews and
iterates, while a Codex cleaner periodically removes scratch output and records
the next action. The first chaser and cleaner use `codex/gpt-6-astra:ultra` and
the second chaser uses `claude/claude-opus-5:max`. Every coding turn is a fresh
session. The wrapper snapshots and restores `.kda-task` around every cleaner
epoch, where candidate, benchmark, failure and profiler evidence must be stored.
The cleaner can rewrite history, so run this only in
the prepared task clone, never in `external/sglang` or the project checkout.

The configured Codex home is `~/.codex-bak`. It is passed as `CODEX_HOME` to
the Codex child process rather than relying on a shell alias. It is a profile
path, not a credential: credentials remain owned by the Codex CLI and are
never written to project JSON. Every backend must already be authenticated in
its own CLI.

## Control-plane commands

```sh
python3 tools/k3ctl.py humanize2 plan --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 start --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 status --config config/project.local.json --task prefill-kda
python3 tools/k3ctl.py humanize2 stop --config config/project.local.json --task prefill-kda
```

`plan` is read-only and is the required review point.  `start` refuses when
the workspace, `hmz` executable, or task contract is not ready.  The adapter
records the command and process identity under the task's `runtime/humanize2`
directory; Humanize2 remains the source of truth for flow state and traces.

Do not pass API keys in configuration or prompts. Do not treat the reviewer
answer as a benchmark verdict. Do not run the cleanup flow against a dirty
source checkout. Use a fresh workspace when changing `work_paths` or switching
between Humanize1 and Humanize2.
