# Toolchain and reproducible setup

The repository has three layers so a new contributor can start without
guessing which parts are machine-specific:

1. **Control plane** — Python 3.9+, the root `tools/k3ctl.py`, metadata,
   reports, and CPU-only unit tests. `make setup` installs this layer in a
   local virtual environment without external Python dependencies.
2. **Workflow layer** — the `external/kda` and `external/humanize` submodules.
   KDA supplies KernelWiki and NCU skills recursively; Humanize supplies the
   planning/review loop and its Codex hook installer.
3. **GPU runtime layer** — `external/sglang`, CUDA, PyTorch, model checkpoints,
   containers, and a target GPU. This layer is deliberately not pinned to one
   host image because it must match the deployment machine.

## Fresh clone

```bash
git clone <your-github-url> infra-loop-kda
cd infra-loop-kda
make setup
```

`make setup` runs, in order, recursive submodule initialization, virtualenv
creation, editable installation of the root package, metadata validation, and
the CPU/control-plane tests. It does not contact a model registry or start a
server.

Use `make doctor` to diagnose missing commands or uninitialized submodules.
Use `make init` alone when the virtualenv is managed by another environment
manager.

## Skills

The skill installer writes nothing by default. Its destination is a user's home
directory, so an accidental invocation must not modify it:

```bash
make skills-check                            # report only; writes nothing
make skills-install SKILLS_ARGS="--apply"    # create the symlinks
```

For a safe trial against a throwaway directory:

```bash
make skills-install SKILLS_ARGS="--apply --home $(pwd)/.codex-test"
```

The upstream Humanize installer — which writes into the Codex home and installs
a native Codex stop hook — never runs implicitly. This project loads the pinned
plugin through Claude's `--plugin-dir`, so no global installation is required.
Request it explicitly with `--with-humanize-installer` if you want it anyway.

The project skill is tracked at `skills/kernel-optimization/SKILL.md`; upstream skills
remain in their submodules. See [`skills/README.md`](../skills/README.md).

## GPU task setup

The correct CUDA/PyTorch/SGLang installation depends on the target GPU and
serving image. After installing that runtime according to the deployment
environment, copy the public config and fill only local values:

```bash
cp config/project.example.json config/project.local.json
make validate CONFIG=config/project.local.json
make status CONFIG=config/project.local.json
```

The task runner must record the exact source commit, workload input, runtime
image, correctness result, timing result, and raw-artifact checksums in an
evidence manifest. Raw checkpoints, gateway addresses, tokens, GPU UUIDs and
profiler databases are ignored by Git and should remain in external storage.

## Updating upstream tools

Update one dependency at a time, inspect its license and changelog, then stage
the new gitlink:

```bash
git -C external/kda fetch --tags origin
git -C external/kda checkout <reviewed-commit>
git add external/kda
```

Run `make validate`, `make test`, and `make task-test` after every update. A
new SGLang commit requires task-level correctness and performance validation;
the accepted prefill patch is not automatically portable across source bases.
