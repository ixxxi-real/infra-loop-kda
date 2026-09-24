# Plan input: chunk-gated-delta-h

This file is the human-authored input to planning. It is not the executable
plan and it is not loop state.

## Goal

Reduce GB300 Kimi K3 prefill latency for chunk_gated_delta_rule_fwd_kernel_h_blockdim64 by lowering register pressure and increasing useful state-kernel parallelism without changing recurrence state semantics.

## Acceptance criteria

- AC-1: The validation command `python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>` passes with a bounded,
  non-zero case count.
- AC-2: Performance evidence is recorded against base commit
  `9ac2710bd37622f38edb078cc753244a3c38c334` with the resolved workload file.
- AC-3: The change stays within the allowed source files listed in `task.json`.

## Iteration policy

- Keep `workloads.resolved.json` frozen as the sole full-workload acceptance
  manifest. Do not change its cases or denominator during candidate search.
- A candidate that passes correctness and precision but misses the full-workload
  performance threshold is a normal rejected candidate: record it in
  `candidates.jsonl`, restore the workspace to the base commit, and continue
  with the next single candidate while the round budget remains.
- Do not stop for human review merely because AC-2 failed. Stop for review only
  after a candidate meets all gates, after the configured round budget is
  exhausted, or when an external blocker/contract ambiguity requires a human
  decision.
- Every candidate round uses the same source commit, workload hash, replay
  restoration policy and full-workload denominator, so results remain paired
  and comparable.

## Baseline and validation path

The baseline is commit `9ac2710bd37622f38edb078cc753244a3c38c334` with
`BV=32`, `num_warps=4`, `num_stages=2`. The kernel grid is
`(ceil(V/BV), N*H)` and each CTA carries a `[BV, 64]` FP32 state tile for every
64-key slice. The recurrence loops over chunks in order and writes the final
state back to `initial_state` when requested.

The earlier Kimi K3 campaign is lineage evidence, not acceptance evidence for
this new task: BV64/warps8, BV32/warps8, fixed BV16 variants, dynamic BV16 and
state/output fusion were rejected or unstable. Existing correctness and
precision harnesses cover the relevant state invariants, but a new resolved
workload must prove that the target H kernel actually executes on the target
source checkout.

## Risks and unknowns

- The current trace reports 137–181 registers/thread and about 2% occupancy,
  but the occupancy estimate is workload/specialization dependent. Fresh NCU
  data is required to distinguish register allocation from spills, memory
  stalls, synchronization and low wave count.
- The state recurrence is sequential across chunks; reducing live state can
  add shared/global traffic or an extra reduction/launch.
- `initial_state` is mutated in place. Benchmark replay without restoration is
  invalid.
- The historical synthetic grid is not production serving evidence. The real
  GB300 8K prefill shape and the Kimi K3 caller must be recorded separately.

## Candidate directions

1. **Baseline profile and shape-dispatch control (low risk):** collect paired
   NSYS/NCU for the real GB300 shape, then re-test `BV=16` only for a tightly
   defined long-single-sequence bucket. Treat the earlier unstable BV16 result
   as a rejection until a fresh pair is stable.
2. **Register-lifetime reduction (primary structural candidate):** refactor the
   four statically declared K slices (`b_h1..b_h4`) into shorter-lived K-slice
   regions or a controlled loop, measuring register count, spills and tensor
   activity. Preserve the per-chunk recurrence and exact state write-back.
3. **K-slice/state staging experiment (higher risk):** stage a K slice in shared
   memory or an explicit temporary and reduce live FP32 state. Accept only if
   extra synchronization and memory traffic are lower than the register
   savings.
4. **Chunk-tiling/state spill experiment (highest risk):** split the recurrence
   into a producer/state-update pair only if NCU shows the single CTA is
   irreducibly wave-starved. This may add launches and global state traffic and
   is a later candidate, not the first implementation.

Do not combine these directions in one candidate. Keep the existing state /
output fusion candidate rejected because its extra q/g/A reads and register
footprint caused a material regression.

## First concrete steps

1. Resolve a workload with exact `H=12,K=V=128,BT=64`, real 8K prefill,
   boundary/ragged/state cases and a state-restoring replay harness.
2. Prepare the exact base commit and verify the real caller reaches
   `chunk_gated_delta_rule_fwd_h`.
3. Run baseline correctness, precision, paired benchmark, NSYS and NCU before
   editing code.
4. Implement only the register-lifetime candidate in a fresh workspace.
5. Run correctness and precision first, then two independent paired benchmark
   runs; record the candidate and rejection reason in lineage files.
