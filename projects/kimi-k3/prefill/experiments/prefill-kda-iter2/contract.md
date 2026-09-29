# Task contract: prefill-kda-iter2

This is a fresh optimization task descended from the accepted `prefill-kda`
result. It does not reopen or modify that immutable task.

## Objective

Continue optimizing the Kimi K3 TP8 prefill `chunk_kda` path on GB300. The
candidate must improve the current source baseline while preserving the public
`chunk_kda` ABI, Q/K/V layout handling, gate math, state-pool indexing and
write-back, FP32 tracking, return aliases and all fallback paths.

## Target and allowed files

The primary entrypoint is
`python/sglang/kernels/ops/attention/fla/kda.py::chunk_kda`. The measured
pipeline includes its `chunk_intra`, `chunk_delta_h` (H state update), and
output stages. `l2norm.py` and `chunk_delta_h.py` may be changed only when
they are part of one measured candidate. No other source file may be edited.
The baseline is commit `9ac2710bd37622f38edb078cc753244a3c38c334`.

The copied workload matrix has 62 bounded cases covering chunk boundaries,
ragged batches, continuation state, zero/random state, slot permutations,
strided state pools, padding indices, intermediate states and FP32 tracking.
It is input shape coverage, not inherited acceptance evidence; every case must
run again on this task's exact base and candidate.

## Correctness and precision

Run `bench/correctness.py` with a non-zero bounded case count and compare both
to the frozen baseline and the FP32 oracle using the inherited tolerances
`atol=0.003`, `rtol=0.03`, `relative_rms=0.03`, plus the FP32 tracking constraint.
Repeat the fixed-seed precision diagnostic and verify state restoration before
each timed replay. A zero-case or baseline-only run is not a pass.

## Performance gate

Run two independent paired CUDA-event benchmark sets on the same GB300 workload
and exact source commit, with warmups and repeated trials. Promote only when the
geometric-mean speedup is at least 3% and no stable workload row regresses more
than 5%. Record launch configuration, effective backend, source revision and
workload checksum. Use NCU/NSYS to explain the limiter before choosing a
structural candidate; do not promote a blind parameter sweep.

The baseline tree is immutable for every candidate. Each benchmark invocation
must interleave baseline and candidate trials with the same workload seed,
restore state before every timed replay, and retain at least two independent
paired runs. A smoke run is diagnostic only and cannot satisfy this gate. The
reviewer may return only a pass/reject rationale; the benchmark harness, not the
reviewer, decides whether the performance gate passed.

## Constraints

- Keep the implementation in Triton/PyTorch code already used by this path;
  defer a CUDA rewrite until a measured Triton candidate is shown insufficient.
- One candidate at a time; rejected candidates stay in `candidates.jsonl` with
  their measured reason.
- Keep `candidates.jsonl`, `benchmark.csv`, rejected-candidate reasons, failed-run
  logs and raw NCU/NSYS/profiler artifacts under `.kda-task/evidence/` (or another
  path below `.kda-task`). Humanize2 cleanup is forbidden from deleting or
  rewriting these records.
- Preserve TP8 prefill dispatch, ragged handling, cache/state semantics and all
  unsupported-layout fallbacks.
- Do not use Triton autotune configurations that replay a kernel which mutates
  state unless the harness restores every state/output buffer safely.
- Human review is required before `accepted`; serving integration is a separate
  state and is not implied by kernel acceptance.
- The final candidate requires an independent Claude or human review over the
  frozen evidence bundle after all automated gates pass.

## Commands

Validation:
```
python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>
```

Evaluation:
```
python3 bench/benchmark.py --baseline-root <baseline> --candidate-root <candidate> --workloads <workloads> --out <out>
```
