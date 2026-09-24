# Task contract: inter-solve-fused-kernel

## Objective

Optimize `chunk_kda_fwd_kernel_inter_solve_fused` in
`python/sglang/kernels/ops/attention/fla/chunk_intra.py` for Kimi K3 prefill on
GB300 while preserving the `chunk_kda` ABI and all inter/solve outputs.

## Kernel semantics

The kernel computes inter-subchunk `Aqk`/`Akk` blocks and triangular solve in one
pass. Depending on `FUSE_RECOMPUTE` and `FUSE_DIAGONAL`, it may also compute
packed `w`, `u`, and `kg` outputs or diagonal blocks. It uses `BC=16`, autotuned
`BK`, `NT=ceil(T/64)` for uniform inputs, and chunk indices for varlen inputs.
The Kimi K3 path uses the safe gate with lower bound `-5`.

## Correctness requirements

Run the 11 oracle cases and all 36 deployment cases in
`workloads.resolved.json`. Cover both fusion regimes, 63/64/65 and 127/128/129
boundaries, varlen/ragged input, safe gate, random initial state and tracked
intermediate outputs. Preserve `Aqk`, `Akk`, `Akkd`, `w_out`, `u_out`, `kg_out`,
all aliases, dtype/rounding behavior and public `chunk_kda` arguments.

## Constraints

- Only `python/sglang/kernels/ops/attention/fla/chunk_intra.py` may change in
  the first candidate.
- Keep the full workload frozen; diagnostic subsets cannot replace it.
- One candidate direction per round. Reset to base before the next candidate.
- Correctness/precision failures or a performance miss are recorded as a
  rejection and followed by the next candidate while budget remains.
- Do not claim acceptance from an isolated inter-solve kernel speedup.

## Promotion

Two independent paired full-workload runs must each reach at least 3% geomean
speedup with no stable row regression above 5%, alongside correctness,
precision, exact source/workload hashes and human review.
