# Kimi K3 prefill KDA contract

## Scope

The task optimizes the public `chunk_kda` prefill path in SGLang. The accepted candidate is based on source commit `8eea3c25a3eaa3c850dbdb0ede2bba7de8f03e93` and changes only:

- `python/sglang/kernels/ops/attention/fla/kda.py`
- `python/sglang/kernels/ops/attention/fla/l2norm.py`

The public signature, gate semantics, cache write-back, FP32 tracking and return aliases remain unchanged. The patch fuses Q/K/V preparation for strided packed QKV views and keeps the original path for unsupported layouts.

## Correctness gate

The frozen workload contains 11 bounded FP32-oracle cases and 51 deployment-bounded baseline/candidate cases. It covers chunk boundaries, ragged batches, continuation, random and zero state, slot permutations, strided state, padding indices, intermediate state output and FP32 tracking.

Acceptance thresholds are `atol=0.003`, `rtol=0.03`, `relative_rms=0.03`, plus the separate FP32 tracking constraint. The accepted evidence recorded 11/11 oracle cases and 51/51 grid cases, with no candidate-vs-baseline violations.

## Precision diagnostic

Three fixed seeds and 62 cases were run as an advisory diagnostic. It is evidence about the frozen inputs and device, not a proof over all layouts. The accepted run recorded 378/378 comparable fields bitwise identical for candidate-vs-baseline and for the A/A control.

## Timing gate

Two independent five-trial measurements used paired CUDA-event timing, five warmups and 30 samples per row. The geometric mean speed ratios were 1.1615484367 and 1.1750340327. No row showed a pooled regression; the smallest row was a noisy single-token case near the 1.02 threshold.

## Integration boundary

The optimization result is accepted, but the patch is `not_applied` in this public control repository. TP8 serving, real caller dispatch, breakable CUDA graphs, prefix cache and production traffic remain separate integration work. Do not describe this task as serving-ready until those checks have their own evidence.
