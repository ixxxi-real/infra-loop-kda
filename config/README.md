# Configuration

`project.example.json` is the only configuration intended for public Git. Copy it to `project.local.json` for a real machine. The latter is ignored by `.gitignore`.

Keep these values local:

- model checkpoint paths and hashes when they identify private artifacts;
- gateway URLs, node names, GPU UUIDs and container names;
- registry credentials, tokens and private image references;
- local evidence roots and artifact-store credentials.

The reviewer settings in the public example intentionally select the isolated
`~/.codex-bak` Codex home. The agent adapter also forces the reviewer launch
flags (`--dangerously-bypass-approvals-and-sandbox`, `gpt-6-astra`,
`model_reasoning_effort="ultra"`, and `--disable apps`) through its private
PATH shim; a shell alias is not required.

The validator accepts placeholders in the example file. The scaffold pins a public SGLang baseline; if your task uses another baseline, replace it with a complete 40-character Git SHA. A local file must not retain angle-bracket placeholders before a real run.

The public `dependencies` and `skills` sections record the pinned KDA and
Humanize workflow libraries plus the paths used by the skill installer. The
`runtime` section remains machine-specific and must stay in the ignored local
override.
