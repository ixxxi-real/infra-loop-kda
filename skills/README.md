# Skills used by this project

The project keeps agent instructions separate from runtime code.
`scripts/install-skills.sh` links the following pinned sources into a skills
directory:

| Linked name | Source | Role |
| --- | --- | --- |
| `kernel-optimization` | `skills/kernel-optimization` | Project contract, evidence and source-compatibility rules |
| `kernelwiki` | `external/kda/skills/KernelWiki` | Kernel design references and query tools |
| `ncu-report-skill` | `external/kda/skills/ncu-report-skill` | Nsight Compute collection and diagnosis |

Kernel Design Agents (`external/kda`) and Humanize (`external/humanize`) remain
upstream submodules. This repository does not copy or silently fork their skill
implementations, so updating a skill is an explicit submodule update that can be
reviewed in Git.

## The installer writes nothing by default

Its destination is a user's home directory, so an accidental invocation — or a
CI run — must not modify it. Run with no arguments, the installer only reports
what it *would* do:

```bash
make init          # fetch the pinned submodules
make skills-check  # report only; writes nothing
```

To apply the links:

```bash
make skills-install SKILLS_ARGS="--apply"
```

To try it against a throwaway directory first:

```bash
make skills-install SKILLS_ARGS="--apply --home $(pwd)/.codex-test"
```

Other options:

| Option | Effect |
| --- | --- |
| `--apply` | Actually create the symlinks. |
| `--home DIR` | Treat `DIR` as the Codex home; links go to `DIR/skills`. |
| `--target DIR` | Use `DIR` directly as the skills directory. |
| `--with-humanize-installer` | Additionally run the upstream Humanize installer. |

An existing symlink is replaced. An existing **non**-symlink is never touched:
the installer refuses and tells you to move it aside deliberately.

## Humanize is not installed globally

The upstream Humanize installer writes into the Codex home and installs a native
Codex stop hook. This project does not need it: the loop is driven through
Claude's `--plugin-dir` pointing at the pinned plugin (or the audited no-commit
overlay), which requires no global installation at all.

So that installer never runs implicitly. Request it explicitly if you want it:

```bash
make skills-install SKILLS_ARGS="--apply --with-humanize-installer"
```

Prefer an isolated `CODEX_HOME` when trying that, since it installs a stop hook:

```bash
make skills-install \
  SKILLS_ARGS="--apply --with-humanize-installer --home $(pwd)/.codex-test"
```

See [`docs/agent-loop.md`](../docs/agent-loop.md) for how the loop is driven, and
[`docs/no-commit-overlay.md`](../docs/no-commit-overlay.md) for the overlay.
