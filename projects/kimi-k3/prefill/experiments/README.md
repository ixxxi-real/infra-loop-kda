# Prefill experiments

This directory contains optimization tasks that are still being evaluated.
They are useful evidence and reusable workload designs, but none of them is
the current accepted prefill deliverable.

| Directory | Role | Status |
| --- | --- | --- |
| `chunk-gated-delta-h/` | H-state kernel search | Unresolved; candidates rejected by the full-workload gate |
| `chunk-gated-delta-h-iter2/` | Follow-up H-state search | Unresolved; inherits the H-state workload |
| `prefill-kda-iter2/` | Full prefill follow-up | Unresolved; separate from the accepted patch |
| `inter-solve-fused/` | Inter-solve workload artifact | Diagnostic workload for the corresponding task under `tasks/` |

Each experiment owns its task contract, source trace, workload, candidate
lineage, and experiment-specific notes. Local workspaces, copied upstream
trees, agent sessions, profiler output, and raw evidence remain in ignored
runtime directories.

The accepted result is kept one level above in `kernels/chunk-kda/`. Do not
copy an experiment patch into the accepted path without passing the full
promotion gates and updating the root task manifest.
