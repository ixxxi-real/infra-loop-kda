# Infra Loop KDA

Infra Loop KDA is a reproducible engineering control plane for automated
GPU-kernel optimization. It combines KDA's evidence loop, Humanize's
plan/review workflow, and project-owned correctness, precision and benchmark
gates.

Kimi K3 is the first model project in the repository. It is organized by
serving stage:

- [Kimi K3 prefill](projects/kimi-k3/prefill/) — accepted `chunk-kda` kernel
  patch, validation harnesses and provenance.
- [Kimi K3 decode](projects/kimi-k3/decode/) — the next optimization stage,
  currently scaffolded without a fabricated result.

The repository is a control plane, not a fork of SGLang. The source runtime is
pinned through `external/sglang`; raw runs, logs, profiler files and agent
transcripts stay outside Git.

## Start with the optimization contract

The two documents that govern new work are:

1. [Kernel optimization skill](skills/kernel-optimization/SKILL.md)
2. [Kernel optimization handbook](docs/kernel-optimization-handbook.md)

They define how an agent chooses a strategy from source and profiler evidence,
how KDA gates the loop, and how a candidate becomes an accepted patch.

For the current code and diff, start at
[projects/kimi-k3/prefill/kernels](projects/kimi-k3/prefill/kernels/). Historical
reports are intentionally under
[projects/kimi-k3/prefill/archive](projects/kimi-k3/prefill/archive/).

## Quick start

```bash
git clone --recurse-submodules https://github.com/ixxxi-real/infra-loop-kda.git
cd infra-loop-kda

make doctor
make setup
source .venv/bin/activate

make status
make task-list
```

For offline control-plane work:

```bash
python3 tools/k3ctl.py validate --config config/project.example.json
make test
make task-test
```

`make setup` initializes the pinned SGLang, Kernel Design Agents and Humanize
submodules, creates a virtual environment and runs the dependency-free checks.
It does not install CUDA, PyTorch, SGLang or a checkpoint.

The URL contains `ixxxi-real` because this checkout has not been assigned a
GitHub owner yet. When publishing, replace it once with the organization or
user that owns the repository. After a normal clone, the equivalent submodule
step is:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The bootstrap is deliberately split into two layers. `make setup` builds the
CPU-only control plane and runs its tests. GPU dependencies remain machine
specific and are installed in the image or environment used by a task. A
fresh checkout can therefore validate its project metadata on a laptop before
any CUDA machine is involved.

For a release or a pull request, run the same clean-tree checks used by CI:

```bash
make verify
make package
```

`make package` writes a wheel and source archive to `dist/`. The package only
contains the `k3ctl` control-plane command; project manifests, task contracts
and submodule pins remain in the repository so a clone is the reproducible
unit of work.

The complete walkthrough, including a no-submodule offline check, GPU task
setup, troubleshooting, and publishing instructions is in
[docs/getting-started.md](docs/getting-started.md).

## Repository map

| Path | Role |
| --- | --- |
| `projects/<project>/<stage>/kernels/` | Current kernel code references and patches |
| `projects/<project>/<stage>/design/` | Contract, plan and source provenance |
| `projects/<project>/<stage>/bench/` | Correctness and timing harness |
| `projects/<project>/<stage>/precision/` | Numerical validation |
| `projects/<project>/<stage>/archive/` | Historical reports and retired candidates |
| `skills/kernel-optimization/` | Primary optimization skill |
| `docs/` | Handbooks and control-plane architecture |
| `evidence/manifests/` | Small tracked summaries and checksums |
| `.infra/` | Ignored logs, runs, sessions and caches |
| `external/sglang` | Pinned source runtime (Git submodule) |
| `external/kda` | Kernel Design Agents workflow (Git submodule) |
| `external/humanize` | Humanize loop plugin (Git submodule) |
| `tools/k3ctl.py` | Canonical control-plane CLI |

The old `tasks/` directory is now only a compatibility area for synthetic
fixtures and generic tooling defaults. New model work belongs below
`projects/`.

## KDA and Humanize boundaries

KDA supplies the basic flow, constraints and evidence gates. The optimization
agent chooses concrete candidates using source inspection, profiler data and
candidate lineage. Humanize supplies the writer/reviewer loop. A parameter
sweep is an experiment; it is not a substitute for a source-based strategy.

The configured workflow uses Claude as writer and Codex as reviewer. The
reviewer is isolated from the default Codex profile: the project exports
`CODEX_HOME=~/.codex-bak`, and the agent's PATH shim launches every nested
review through the real Codex binary with:

```text
--dangerously-bypass-approvals-and-sandbox
--model gpt-6-astra
-c 'model_reasoning_effort="ultra"'
--disable apps
```

The Humanize plugin is loaded through the project's verified no-commit overlay,
which widens its effort parser and preserves `ultra` through the review phase.
This is explicit configuration, not a shell alias, so it also applies when an
agent is launched by another terminal or process.

## Local configuration

Copy the public template before using a real machine:

```bash
cp config/project.example.json config/project.local.json
make validate CONFIG=config/project.local.json
```

The local file is ignored. Keep checkpoints, gateway addresses, tokens, GPU
identifiers and private host paths there or in external evidence storage.

## Useful commands

```bash
make validate
make status
make task-list
make test
make task-test
make skills-check
make skills-install SKILLS_ARGS="--apply --home $(pwd)/.codex-test"
```

To scaffold a future task, register it explicitly below the relevant project
directory:

```bash
k3ctl task-create \
  --tasks-dir projects/kimi-k3/decode \
  --id decode-kda \
  --kind decode \
  --base-commit <exact-40-character-commit> \
  --objective "…" \
  --validation-command "…" \
  --performance-command "…"
```

See [docs/lifecycle.md](docs/lifecycle.md), [docs/adding-a-task.md](docs/adding-a-task.md), [docs/runner-interface.md](docs/runner-interface.md), [docs/security.md](docs/security.md) and [docs/runtime-layout.md](docs/runtime-layout.md) for the detailed contracts.
