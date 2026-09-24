# Getting started from a fresh clone

This repository is the control plane for reproducible GPU-kernel optimization.
It owns task contracts, workload definitions, validation harnesses, evidence
manifests and the agent workflow. The SGLang source and the KDA/Humanize tools
are pinned Git submodules; they are not copied into this repository.

## 1. Install the host tools

The CPU-only control plane needs:

- Git 2.30 or newer;
- Python 3.9 or newer;
- a POSIX shell. Agent workflows require Bash 4 or newer and `jq`;
- network access for the first submodule checkout and Python build backend.

CUDA, PyTorch, Triton, Nsight Compute and model checkpoints are only needed
when running a GPU task. They are intentionally not installed by the control
plane bootstrap.

## 2. Clone the complete repository

After this project is published, replace `<OWNER>` with the GitHub owner:

```bash
git clone --recurse-submodules https://github.com/<OWNER>/infra-loop-kda.git
cd infra-loop-kda
```

If the repository was cloned without `--recurse-submodules`, initialize the
pinned dependencies explicitly:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The checked-in gitlinks select exact commits. Do not replace them with a
floating branch when collecting evidence.

## 3. Bootstrap and verify the control plane

```bash
make setup
source .venv/bin/activate
make verify
```

`make setup` is idempotent. It creates `.venv`, installs this repository in
editable mode, validates the public example configuration and runs the CPU
test suite. `make verify` repeats the repository, metadata, test and workload
example checks. Neither command launches Claude, Codex, a GPU job or a model.

Useful read-only commands are:

```bash
make doctor
make status
make task-list
k3ctl --help
```

The public example intentionally contains placeholders for private checkpoint,
gateway and GPU information. Copy it before configuring a real machine:

```bash
cp config/project.example.json config/project.local.json
# edit only the local file; it is ignored by Git
make validate CONFIG=config/project.local.json
```

## 4. Build the distributable control-plane package

```bash
make package
ls dist/
```

This produces a wheel and source archive. The wheel installs the `k3ctl`
command and its Python modules. The repository checkout is still required for
task manifests, pinned submodules and project evidence.

To test the wheel from outside the repository:

```bash
make install-check
```

## 5. Prepare a GPU task

Use a host or container that already provides a compatible CUDA/PyTorch/Triton
stack. Fill in the private runtime fields in `config/project.local.json`, then
inspect the task before launching an agent:

```bash
make doctor-agent CONFIG=config/project.local.json TASK=prefill-kda
make workspace-prepare CONFIG=config/project.local.json TASK=prefill-kda
make workspace-status CONFIG=config/project.local.json TASK=prefill-kda
make agent-plan CONFIG=config/project.local.json TASK=prefill-kda
```

The plan command is dry-run only. When the source, workload and runtime are
ready, start the configured Humanize/KDA loop through the project CLI. Keep
raw profiler output and sessions in the ignored runtime/evidence locations;
commit only redacted manifests, source patches and conclusions.

## 6. Add a new task

Create a task under the relevant `projects/<model>/<stage>/` directory. A task
must include a contract, source trace, model profile, frozen workload, baseline
and candidate validation commands, and an evidence manifest. Follow:

- [docs/adding-a-task.md](adding-a-task.md)
- [docs/lifecycle.md](lifecycle.md)
- [docs/reproducibility.md](reproducibility.md)
- [docs/security.md](security.md)

Register the task in a local configuration first, validate it, then add the
public task metadata to the repository. Private runtime paths, checkpoint
locations, credentials, GPU UUIDs and raw profiler databases must never enter
Git.

## 7. Publish to GitHub

Create an empty GitHub repository named `infra-loop-kda`, set its default branch
to `main`, and add the remote locally:

```bash
git remote add origin https://github.com/<OWNER>/infra-loop-kda.git
git add -A
git diff --cached --check
git commit -m "Initial engineering baseline"
git push -u origin main
```

Before pushing, confirm that the staged tree contains the three gitlinks under
`external/`, the `.github/workflows/ci.yml` workflow, `config/project.example.json`
and no ignored runtime directory. GitHub Actions runs the CPU control-plane
matrix on every push and pull request; the GPU workflow remains an explicit
operator action because it needs private hardware and credentials.

## Troubleshooting

**`make setup` cannot fetch setuptools.** The editable install uses PEP 517 and
needs `setuptools>=64`. Configure a package index or preinstall the backend,
then rerun `make install`.

**The doctor reports missing submodules.** Run the recursive submodule update
from step 2. The control-plane tests can run without submodules, but agent
workflow and overlay tests cannot.

**`k3ctl` cannot find the repository from another directory.** Set
`K3_PROJECT_ROOT=/absolute/path/to/infra-loop-kda` or pass an explicit config
path. The CLI refuses to adopt an unrelated directory as the project root.

**A GPU task has no CUDA tools.** This is an environment problem, not a control
plane build failure. Install or select the task's pinned runtime image and
rerun `make doctor-agent` before preparing a workspace.
