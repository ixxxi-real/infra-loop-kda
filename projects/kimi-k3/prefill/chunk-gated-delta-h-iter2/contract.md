# Task contract: chunk-gated-delta-h-iter2

## Objective

Optimize `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on GB300 Kimi K3 prefill by reducing measured state-kernel register/dataflow cost while preserving recurrence state semantics.

## Fixed interface and scope

- Base commit: `9ac2710bd37622f38edb078cc753244a3c38c334`.
- First candidate may change only `chunk_delta_h.py`.
- Local H=12, K=V=128, BT=64, BF16 activation/state, full-rank gate, lower bound -5.
- Preserve `chunk_kda` ABI, gate variants, state pool layout, slot/index stride and permutation semantics, -1/padding behavior, continuation rounding, `SAVE_NEW_VALUE`, `TRACK_STATE`, aliases and fallback paths.
- Keep Triton as the implementation language. CUDA is deferred until a measured Triton dataflow candidate is proven insufficient.
- Every timing replay restores mutated state, tracked outputs and aliased value buffers.

## Workloads

`workloads.resolved.json` is the immutable 62-case acceptance manifest.
`workloads.state-dataflow-focused-v1.json` is a 33-case diagnostic manifest. It
contains inherited cases plus explicitly added boundary/state probes; it may
guide profiling and candidate selection, but cannot replace the full manifest
or alter its denominator.

## Gates

1. Full correctness: non-zero bounded case count, 11 oracle cases and 51
   deployment cases, with no failures.
2. Precision: declared three seeds and all state/layout/continuation obligations
   pass without nonfinite values or inactive-slot mutation.
3. Performance: two independent paired five-trial runs on the full workload,
   geometric mean >=1.03x and no stable row regression >5%.
4. Evidence records exact source commit, workload hashes, device/toolchain,
   imported module paths and runner IDs.

A focused-only speedup is diagnostic and cannot be promoted. Rejected candidates
remain recorded with their reason; the workspace is restored to base before the
next candidate. Human review is required before acceptance.
