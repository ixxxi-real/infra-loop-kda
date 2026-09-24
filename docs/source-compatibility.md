# Source compatibility

The repository contains two deliberately separate source identities:

- `external/sglang` is the public submodule baseline recorded in `config/project.example.json`.
- `projects/kimi-k3/prefill` records the exact historical source commit used by the accepted patch: `8eea3c25a3eaa3c850dbdb0ede2bba7de8f03e93`.

They must not be conflated. The accepted patch is a source diff whose correctness evidence belongs to the task base. A dry `git apply --check` may succeed on another nearby commit while its helper ABI or caller semantics still differ. Before applying it in a fresh clone:

1. Make the task base commit reachable from the configured remote, or reconstruct it from a reviewed patch series.
2. Create an isolated source worktree at that commit.
3. Run `git apply --check projects/kimi-k3/prefill/kernels/chunk-kda/accepted-kernel.patch`.
4. Apply it, run the task's CPU checks and source-import checks, then perform GPU correctness and integration validation.
5. If you port the patch to another base, create a new patch manifest and do not overwrite the historical one.

This rule prevents a clean-looking patch from being applied to a source tree whose helper ABI or caller semantics have changed.

For this migration, `git apply --check` was verified against the exact task base and also happens to apply syntactically to the current public submodule commit. The latter is only a mechanical check; no correctness, timing or serving evidence is transferred across the two commits.
