# KDA prefill optimization round — 2026-09-28

## Scope and evidence

The target remains the complete `chunk_kda` prefill operator. A warmed NSYS
`kda_capture` query on `uniform_b1_s16384` showed the H-state update
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` as the dominant stage: about
1.485 ms across five captures, versus about 0.230 ms for the output kernel.
The earlier all-launch `cuda_gpu_kern_sum` totals were not used as hotspot
evidence because they include autotune/cache activity.

The task scope was extended to `chunk_delta_h.py` to match that measured
pipeline stage. The extension is recorded in `task.json` and `contract.md`.

## Candidates

| Candidate | Correctness | Precision | 51-case geomean | Result |
|---|---:|---:|---:|---|
| H `BV=64`, warps=4 | pass | pass | 0.9964x | rejected; long 16K rows about 0.918x |
| H `BV=32`, warps=8 | pass | pass | 0.9712x | rejected; broad batched/short regressions |
| H `BV=32`, warps=2 | pass | pass | 0.9752x | rejected; no end-to-end gain |
| H `BV=16`, warps=4 | pass | pass | 1.0180x | rejected; long rows improve, full grid remains below gate |
| H explicit BV dispatch | pass | pass | 1.0230x | rejected; long rows improve, full grid remains below gate |
| O transpose-free H view | pass | pass | 0.9951x | rejected; full grid regresses and shows no stable structural gain |

Each result is a complete candidate workflow on GPU2, with five paired trials
and the frozen workload SHA
`3452bc547ce0439296444409b9cc651fd5d18ff6e27c66c1f0e59fad27a254c5`.
The H source is restored to its original `BV=32`, `num_warps=4`,
`num_stages=2` configuration. No candidate is promoted and no serving image
was built or deployed.

## Next decision

The tile/warp family does not meet the 1.03x gate. The explicit BV dispatch
candidate confirmed that the long-prefill benefit is real (`1.065x` on
`uniform_b1_s16384`) but is diluted by short and batched shapes; its complete
51-case geomean was only `1.0230x`, so it was also rejected. NCU has supplied
register, shared-memory, occupancy, memory-throughput and tensor-issue metrics.
Further work should target one structural issue from those metrics, or a
serving-specific dispatch contract with a denominator that matches production;
blind tile/warp sweeps are stopped. The base source remains unchanged.

## Deep structural candidate — H/O fused recurrent loop

This candidate added an opt-in path for batch-1, untracked long prefill. It
computed the O result inside the H recurrence while the current H tile and
`v_new` value were live, with the intent of removing the intermediate H and
`v_new` global-memory round trips and the separate O launch. Batched, ragged,
tracked, and intermediate-state calls retained the reference fallback.

The corrected run passed all 62 correctness cases and the precision diagnostic
with zero candidate-vs-baseline acceptance violations (maximum relative RMS
`0.000679`; candidate-vs-oracle maximum `0.003374`). It nevertheless regressed
the performance target: the five-trial, 51-case geometric mean was `0.9859x`.
The long single-sequence rows were `0.6849x` at 4K, `0.5502x` at 8K, and
`0.4664x` at 16K; `resumed_chunk_s16384` was `0.4679x`. The fused work is
duplicated across the H kernel's V tiles and increases live values/register
pressure, so the saved stores do not compensate for the extra in-loop work.
The source was restored to the base commit and no image or serving integration
was made. Evidence is retained under
`runtime/remote-evidence/fused-h-o-candidate-r3-gpu2-20260928/`.

## Deep structural candidate — O-stage transpose-free H view

The candidate changed only `chunk_gla_fwd_kernel_o` in `kda.py`. The state is
still physically stored as `[V, K]`, but the block pointer presents the same
bytes as a logical `[K, V]` tile so the MMA consumes it directly instead of
calling `tl.trans(b_h)` for every K tile. H-stage code, state writeback, gate
semantics, ABI, and workload were unchanged.

The isolated GPU2 run completed all 62 cases (11 oracle/correctness and 51
deployment cases). Correctness passed. Precision passed with zero acceptance
violations; candidate-vs-baseline was exactly equal in the tracked fields, and
candidate-vs-oracle stayed within the existing `atol=0.003, rtol=0.03` gate.
The five paired benchmark trial geomeans were `1.0233x`, `0.9949x`, `1.0290x`,
`0.9797x`, and `0.9479x`; the aggregate 51-case geomean was `0.9951x`.
The result is therefore rejected and the source was restored to the base
commit. Evidence is retained under
`runtime/remote-evidence/full-o-transpose-candidate-gpu2-20260928/`.
