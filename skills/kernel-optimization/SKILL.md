---
name: kernel-optimization
description: Operating contract for reproducible GPU kernel optimization across model projects.
---

# Kernel optimization skill

Use this skill for every kernel change under `projects/<project>/<stage>/`.
The repository is an evidence-driven control plane: a candidate becomes a
deliverable only after source identity, correctness, precision and performance
gates are all recorded.

## Read before editing

1. Read the stage `README.md`, `design/contract.md`,
   `design/source-trace.md`, and `task.json`.
2. Locate the exact source checkout and commit named by `task.json`.
   Never substitute the current submodule head for an unavailable task base.
3. Read the kernel's current patch and the latest accepted result before
   proposing a new candidate.

## Optimization loop

1. Establish a baseline on the target workload and record the environment.
2. Use KDA's basic flow to form a hypothesis from source and profiler evidence.
   The agent may choose the concrete candidate, but the KDA contract supplies
   the gates, lineage and rejection discipline.
3. Change one structural idea or one parameter family at a time.
4. Run correctness and precision before spending GPU time on a sweep.
5. Compare paired measurements on the same source commit and workload. Repeat
   unstable wins; keep a rejected candidate with its reason.
6. Promote only a patch whose source, workload, checksums and evidence are
   reproducible. Keep serving integration as a separate status.

## Profiling rules

Use KernelWiki for prior art and the NCU report skill for profiler collection
and diagnosis. Keep raw profiles, logs, transcripts, checkpoints and machine
identifiers in the ignored runtime area or external evidence storage. Track a
small manifest, checksums and interpretation in the project.

A parameter sweep is evidence, not a strategy by itself. Prefer candidates
that change a bottleneck shown by profiling: data movement, register pressure,
occupancy, synchronization, instruction mix or launch geometry.

## Acceptance gates

- A non-zero correctness case count passes with the contract's tolerances.
- Precision checks cover the declared seeds and layouts.
- Performance is measured on the declared workload with paired repetitions.
- The exact source commit and patch hash are recorded.
- Rejected or unstable candidates remain visible in the candidate lineage.
- Human review is complete before `accepted`; serving integration is validated
  separately.

Never put checkpoints, gateway URLs, tokens, GPU UUIDs or private host paths
in tracked files. Use ignored `config/*.local.json` and external evidence.
