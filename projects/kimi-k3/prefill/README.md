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

## Gates and provenance

- design/contract.md
- design/source-trace.md
- design/plan.md
- bench/
- precision/
- archive/reports/

Raw benchmark logs and profiler captures are external artifacts. The package
tracks their manifest and interpretation, not machine-specific output.
