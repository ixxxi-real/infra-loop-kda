# GPU0 iteration: BV16 control candidate

## Baseline profile

The baseline was profiled on GB300 GPU0 through `<gateway> -> <gpu-node>`
using the isolated KDA runner and `uniform_b1_s16384`. The target kernel used:

| Metric | BV32 / warps4 / stages2 |
|---|---:|
| Grid | 4 × 12 = 48 CTAs |
| Registers/thread | 137 (144 allocated) |
| Dynamic shared memory | 45,056 B (46,080 B allocated) |
| Register occupancy limit | 3 CTAs/SM |
| Waves/SM | 0.1053 |
| Active warps | 6.23% |
| Tensor-pipe activity | 1.77% |
| DRAM activity | 5.59% |
| SM throughput | 3.63% |

Correctness passed before profiling. Raw NCU artifacts and gate reports are
kept in the ignored Infra Loop evidence run
`chunk-gated-delta-h/baseline-gpu0`.

## Candidate

The first candidate changed only the default `BV` from 32 to 16. The kernel
still used four warps and two stages. Correctness and precision passed. The
candidate NCU profile for the same long single-sequence row showed:

| Metric | BV16 |
|---|---:|
| Grid | 8 × 12 = 96 CTAs |
| Registers/thread | 87 (88 allocated) |
| Dynamic shared memory | 36,872 B (38,016 B allocated) |
| Register occupancy limit | 5 CTAs/SM |
| Waves/SM | 0.1263 |
| Active warps | 6.26% |
| Tensor-pipe activity | 2.02% |
| DRAM activity | 6.06% |
| SM throughput | 7.59% |

This confirms that the long single-sequence gain comes from smaller live state
tiles and more value-row CTAs, rather than an arithmetic change. The full
paired benchmark still produced only `1.021065x` geometric-mean speedup over
51 rows. `uniform_b1_s16384` reached `1.078612x`, while
`uniform_b8_s512` fell to `0.973597x`.

## Decision

Reject BV16 as a global default. Keep it as evidence for a future shape-
dispatched candidate restricted to long, low-batch sequences. Do not promote
that dispatch without a fresh independent paired run; the existing workload
mix shows that the gain does not generalize.

## Next candidate

Move to the structural register-lifetime plan in `docs/plan.md`: reduce the
live FP32 state footprint without changing the recurrence. Keep the baseline
state restoration and the full state/precision gates. A shape-dispatch BV16
candidate can be revisited after the structural candidate, using the same
GPU0 evidence protocol.
