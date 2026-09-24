# Infra Loop KDA architecture

Infra Loop KDA is organized around three boundaries:

- **Project**: a model or serving family, such as `kimi-k3`.
- **Stage**: a serving phase, such as `prefill` or `decode`.
- **Kernel**: a concrete source entrypoint and its patch, checks and evidence.

The control plane in `tools/` operates on stage packages. It does not vendor
the SGLang runtime. `external/sglang` is a pinned source dependency, while
the stage package records the exact task base and the patch to apply.

Current layout:

- `projects/kimi-k3/prefill/kernels/` is the prominent code and diff area.
- `projects/kimi-k3/prefill/design/` contains the contract and provenance.
- `projects/kimi-k3/prefill/bench/` and `precision/` contain executable gates.
- `projects/kimi-k3/prefill/archive/` contains historical reports.
- `.infra/` is ignored runtime state for logs, runs, sessions and caches.
- `skills/kernel-optimization/SKILL.md` and
  `docs/kernel-optimization-handbook.md` define the optimization method.

A new project should add a directory below `projects/`; a new serving stage
should add a sibling below that project; a new kernel should add a directory
below the stage's `kernels/`. Do not create a second top-level project for
each experiment or parameter sweep.
