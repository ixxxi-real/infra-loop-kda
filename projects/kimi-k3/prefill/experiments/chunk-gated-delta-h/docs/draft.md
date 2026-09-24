# Optimization plan draft: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

## Baseline

The new task is separate from the accepted `prefill-kda` QKV-preparation task.
Its source base is `9ac2710bd37622f38edb078cc753244a3c38c334`, and its only
initially allowed source file is
`python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`.

The target is the Kimi K3 prefill H-state kernel with local `H=12`, `K=V=128`,
chunk size `BT=64`, BF16 inputs/state and the model gate variants. The current
launch defaults are `BV=32`, four warps and two stages. The grid is
`(ceil(V/BV), N*H)`; the CTA owns a `[BV,64]` FP32 state tile per K slice and
serially advances the chunk recurrence.

## Evidence and motivation

The GB300 trace reports this kernel as the largest KDA compute kernel: about
18–20.5 ms across the captured prefill and roughly 2% reported occupancy, with
137–181 registers/thread and 45,056 bytes of shared memory. A controlled NCU
run reported register-limited occupancy and very low tensor-pipe activity.
These observations motivate a register/dataflow investigation, but they do
not yet prove whether the limiting factor is allocation, spills, memory stalls,
or wave count; the first run must collect fresh NCU counters on the exact target
shape.

## Existing lineage

The previous campaign rejected BV/warp/stage-only variants: BV64/warps8,
BV32/warps8, fixed BV16/warps2, fixed BV16/warps8 and dynamic BV16 were either
slower, not materially faster, or unstable across the required measurement
pair. State/output fusion passed correctness but regressed end-to-end because
additional q/g/A reads and the larger fused register footprint outweighed the
saved intermediate traffic. The baseline remains BV32/warps4/stages2.

The kernel mutates `initial_state` in place. Multi-config Triton autotune is
unsafe because timing replays corrupt the state pool and restoring the pool can
exceed memory. Every benchmark replay must restore the state and tracking
buffers before invocation.

## Ranked hypotheses

1. **Register-lifetime reduction (primary):** shorten the live range of the
   `b_h1..b_h4` FP32 state tiles by processing K slices through a controlled
   lifetime boundary. Measure registers, local-memory spills, tensor activity,
   and recurrence time. This is the first structural candidate.
2. **Shape-dispatched BV16 control:** only retest BV16 for the real long,
   single-sequence bucket after the profile is frozen. This is a control for
   whether the production trace differs from the rejected synthetic grid; it
   is not a promotion candidate until two paired runs are stable.
3. **Explicit K-slice staging:** stage one K slice in shared memory or a
   temporary buffer to reduce registers. Account for barriers, extra traffic
   and cross-slice accumulation before implementing.
4. **State spill/split:** split state update into multiple launches only if the
   profile shows a single CTA cannot achieve useful waves after the first
   candidate. This is highest risk because it adds synchronization and state
   traffic.

## Workload and gates

The task needs a resolved workload rather than copying the accepted QKV task's
workload. It must include the real GB300 8K prefill shape plus boundary lengths,
batch variation, ragged sequences, continuation, zero/random state, strided and
padded state pools, slot permutations, `SAVE_NEW_VALUE` and `TRACK_STATE`.

Correctness must compare the candidate with the exact baseline and an FP32
oracle, with a non-zero case count. Precision must cover the declared seeds and
state layouts. Timing must use paired CUDA-event measurements with fixed
warmups/samples and state restoration before every replay. NCU/NSYS data must
be collected on the same source/workload pair before and after the structural
candidate.

Promote only after two independent five-trial paired runs show at least 3%
geometric-mean improvement and no stable row regression above 5%, with all
correctness and precision gates passing. Keep failed candidates and reasons in
the lineage. Serving integration remains a separate status.

## First execution steps

1. Verify the exact source commit and prepare an isolated task workspace.
2. Resolve the workload and add a state-restoring target-kernel harness.
3. Run baseline correctness, precision, paired timing, NSYS and NCU.
4. Implement only register-lifetime candidate 1.
5. Validate correctness and precision before any sweep.
6. Run two paired benchmark sets and compare NCU counters.
7. Keep, revise or reject the candidate with a recorded reason; do not start
   candidate 2 until candidate 1 has a decision.
