# Kimi K3 prefill

This is the accepted prefill stage package. The current optimization target is
the chunk-kda kernel.

## Start with the deliverable

- kernels/chunk-kda/accepted-kernel.patch
- kernels/chunk-kda/patch-manifest.json
- kernels/chunk-kda/README.md

The source-only patch applies to the exact source commit recorded in
task.json. The runtime source itself remains in the pinned external/sglang
checkout.

## Directory hierarchy

The root of this stage is intentionally small. Its contents have different
roles:

| Path | Importance | Purpose |
| --- | --- | --- |
| `task.json`, `design/` | Primary | Accepted stage contract, source trace and plan |
| `kernels/` | Primary | The only promoted source patch and its manifest |
| `bench/`, `precision/`, `scripts/` | Shared | Reusable correctness, precision and profiling harnesses |
| `model-profile.json`, `workloads*.json`, `*-example.json` | Shared | Public Kimi K3 and deployment inputs |
| `experiments/` | Secondary | Unaccepted task lineages and diagnostic workloads |
| `archive/` | Historical | Retired reports and final historical conclusions |
| `runtime/` | Local only | Agent sessions, runs and external evidence; ignored by Git |

Start with the root files and `kernels/chunk-kda/`. Enter `experiments/` only
when reproducing or extending an unaccepted optimization attempt.

## Gates and provenance

- design/contract.md
- design/source-trace.md
- design/plan.md
- bench/
- precision/
- archive/reports/

Raw benchmark logs and profiler captures are external artifacts. The package
tracks their manifest and interpretation, not machine-specific output.
