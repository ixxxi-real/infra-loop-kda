# Configuration

`project.example.json` is the only configuration intended for public Git. Copy it to `project.local.json` for a real machine. The latter is ignored by `.gitignore`.

Keep these values local:

- model checkpoint paths and hashes when they identify private artifacts;
- gateway URLs, node names, GPU UUIDs and container names;
- registry credentials, tokens and private image references;
- local evidence roots and artifact-store credentials.

The active Humanize2 block intentionally selects the isolated `~/.codex-bak`
Codex home for the first chaser and cleaner. They are configured as
`codex/gpt-6-astra:ultra`; the second chaser is an independent
`claude/claude-opus-5:max` reviewer. The path is exported as `CODEX_HOME`; a
shell alias is not required. The legacy Humanize1 adapter keeps its Claude-host and
Codex-reviewer launch flags (`--dangerously-bypass-approvals-and-sandbox`,
`gpt-6-astra`, `model_reasoning_effort="ultra"`, and `--disable apps`) through
its private PATH shim.

The validator accepts placeholders in the example file. The scaffold pins a public SGLang baseline; if your task uses another baseline, replace it with a complete 40-character Git SHA. A local file must not retain angle-bracket placeholders before a real run.

The public `dependencies` and `skills` sections record the pinned KDA,
Humanize1, Humanize2 and flowverse workflow libraries plus the paths used by the skill
installer. Humanize1 remains available through `k3ctl agent`; Humanize2 is an
independent `k3ctl humanize2` block and uses `~/.codex-bak` for its Codex
profile by default. The public `workflow.humanize_backend` is set to
`humanize2`, so the normal KDA `agent` entrypoints use the new runtime. Set it
to `humanize` only for a legacy plugin run. The `runtime` section remains
machine-specific and must stay in the ignored local override.
