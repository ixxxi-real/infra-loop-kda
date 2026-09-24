# Plan input: inter-solve-fused-kernel

## Goal

Optimize `chunk_kda_fwd_kernel_inter_solve_fused` for Kimi K3 prefill on GB300
without changing the chunk_kda ABI, safe-gate behavior, fusion flags, intermediate
outputs, or state semantics.

## Fixed target

- Base commit: `9ac2710bd37622f38edb078cc753244a3c38c334`.
- Allowed source: `python/sglang/kernels/ops/attention/fla/chunk_intra.py` only.
- Kimi K3 rank-local shape: `H=12`, `K=V=128`, `BT=64`, `BC=16`, BF16 inputs.
- Kernel grid: `(NT, B*H)`; `NT` is uniform `ceil(T/64)` or the varlen chunk-index count.
- The production checkpoint uses the safe gate (`lower_bound=-5`), so baseline
  evidence must exercise `USE_SAFE_GATE=True`.
- `FUSE_RECOMPUTE` and `FUSE_DIAGONAL` are enabled by `kda.py` when
  packed `NT_total*H <= 256`; v1 explicitly covers both regimes.

## Workload and acceptance

Use `workloads.resolved.json` as the only acceptance workload. It contains 11
correctness and 36 deployment-grid cases with fusion-regime annotations. The
workload is frozen and its hash is `915b967f3b9cd30945cfc8218101dbb226f7ba1fb144216e3408869712d08c9d`.

Correctness and precision must pass before timing. Promotion requires two
independent five-trial paired full-workload runs with geomean >= 1.03 and no
stable row regression >5%, plus NCU/NSYS evidence and human review. A failed
candidate is recorded and the workspace is reset before the next candidate.

## Candidate directions

1. Profile baseline by fusion regime and separate inter-solve kernel time from
   the public chunk_kda call.
2. Test one dataflow/layout change in `chunk_intra.py`, preserving output
   layouts and state-free intermediate semantics.
3. Test interior/tail specialization only if NCU shows boundary checks or tail
   predication is material on exact BT=64 rows.
4. Test fusion-flag or launch-geometry changes only when the baseline profile
   shows that the selected regime is wrong for a specific grid bucket. Do not
   enable multi-config autotune without state/output provenance.

Do not copy H-state candidate evidence; this is a separate kernel and workload.
