# Infra Loop KDA kernel optimization handbook

This handbook describes how the repository turns a kernel idea into a
repeatable deliverable. It is the primary guide for both agent-driven and
manual optimization.

## Repository shape

```
infra-loop-kda/
├── projects/
│   └── kimi-k3/
│       ├── prefill/
│       │   ├── kernels/          # source patches and kernel-level deliverables
│       │   ├── design/           # contract, plan and source provenance
│       │   ├── bench/            # correctness and timing harnesses
│       │   ├── precision/        # numerical validation
│       │   └── archive/          # reports and historical records
│       └── decode/                # the next stage; no fake result is added
├── skills/kernel-optimization/   # the operating skill
├── docs/                         # architecture and workflow handbooks
├── evidence/manifests/           # small tracked summaries and checksums
└── .infra/                       # ignored runs, logs, sessions and caches
```

Code and diffs are deliberately in `projects/*/*/kernels`. Execution history
is kept in `archive/` or `.infra/` so it does not compete with the current
kernel deliverable.

## What a candidate must answer

Every candidate has one sentence for each question:

- Which source function and commit changed?
- Which workload row or model shape exercises it?
- Which profiler observation motivated the change?
- What correctness and precision checks protect behavior?
- What paired timing result determines promotion or rejection?

If the motivation cannot be tied to source or measurement, it is a hypothesis
to test, not an optimization conclusion.

## Strategy selection

KDA defines the loop, safety gates and evidence contract. The optimizer agent
selects the concrete strategy from three inputs:

1. source inspection (launch geometry, memory access, fusion boundaries and
   dataflow);
2. profiler evidence (registers, occupancy, tensor activity, memory and stalls);
3. lineage (already accepted and rejected candidates).

Use parameter tuning only when the bottleneck is parameter-sensitive. When a
family is exhausted or unstable, move to a structural hypothesis: reduce live
state, change tiling or staging, fuse a producer/consumer, remove a redundant
normalization, or split a kernel. A slower or unstable result is a useful
rejection record.

## Promotion order

```
source identity -> correctness -> precision -> paired benchmark
              -> review -> accepted patch -> serving integration
```

The accepted patch is never copied into the runtime source silently. Apply it
in an isolated source worktree, verify the patch checksum, rerun the task gates,
then record integration separately.

## Runtime hygiene

Raw NCU files, benchmark logs, agent transcripts and temporary workspaces belong
outside Git. The control plane stores only manifests, checksums and a concise
interpretation. This makes a checkout readable: a reviewer sees the current
kernel and its acceptance evidence first, with history available when needed.

## Kimi K3 map

Kimi K3 is one project inside Infra Loop KDA. Its current stage is
`projects/kimi-k3/prefill`; its next stage is
`projects/kimi-k3/decode`. The prefill package contains the accepted
`chunk-kda` patch plus the validation harnesses. Decode is intentionally
scaffolded and has no claimed benchmark result.
