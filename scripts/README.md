# Project scripts

These scripts are intentionally small entry points around the pinned
submodules:

- `doctor.sh` checks required commands and initialized submodules.
- `install-skills.sh` links the project, Humanize2, KernelWiki, and NCU skills
  into `${CODEX_HOME:-~/.codex-bak}/skills`. Humanize1 remains a Claude plugin
  and is not installed globally.

For a disposable installation, set both `CODEX_HOME` and `XDG_CONFIG_HOME` so
Humanize's hook/config writes stay inside a temporary directory.

The scripts do not download model checkpoints, start a gateway, or write raw
benchmark data into Git. Those operations belong to a machine-specific runner.
