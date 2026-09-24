# KDA H kernel iteration — 2026-09-24 GPU2

## Scope

This iteration targets `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on the
Kimi K3 prefill path. The source base is commit
`9ac2710bd37622f38edb078cc753244a3c38c334`; the frozen full workload hash is
`4ae576f6b2a21ce6b0f7f0b7f0681077785759af6d4e0803434f5a930faec4ed`.

All remote runs used `<gateway> -> <gpu-node>` and the explicit GPU2 UUID
`<redacted-gpu-uuid>`. GPU0 was not used, no TP8 service
was restarted, and no online deployment was changed. The profiler container
was isolated to GPU2 and is removed after evidence collection.

## Baseline evidence

The fresh baseline NCU capture for `uniform_b1_s16384` measured 595,392 ns,
137 registers/thread, 45,056 B dynamic shared memory, grid 48 and 0.1053
waves/SM. SM throughput was only 3.63% of peak. This is a low-grid,
latency-bound state recurrence; changing occupancy limits alone is not a sound
optimization target.

## Candidate: long-shape K-major dataflow

For K=V=128 and at least 64 chunks per sequence, the candidate keeps the
physical state layout unchanged but holds the recurrence tile as `[K, BV]`.
Both state dot products therefore consume the accumulator directly and remove
the two `tl.trans` materializations. All other shapes use the original
V-major kernel and launch geometry.

The full correctness gate passed 11 oracle cases and 51 deployment rows. The
three-seed precision diagnostic passed. One paired full-workload run measured
1.009822x geometric-mean speedup; representative long rows improved by about
4.8–14.4%, including 1.144x for `resumed_chunk_s16384`. The isolated target
kernel improved from 595,392 ns to 427,840 ns (1.392x), but the edit changes
only seven of 51 rows, so it does not meet the fixed 1.03 promotion gate.

NCU also shows the mechanism: registers rise from 137 to 159 and dynamic
shared memory from 45,056 to 66,096 B, while instruction count falls by only
1.6%; the speed comes from removing a layout-conversion serialization point,
not from higher occupancy (waves/SM remains 0.1053).

## Configuration probes

The same K-major candidate was tested with explicit source defaults for
`num_warps=8` and `num_warps=2` on the same GPU2 shape. The w8 probe measured
428,416 ns with 103 registers/thread and 32.9M instructions, slightly slower
than w4. The w2 probe measured 643,552 ns with 255 registers/thread. Neither
was run through the full workload.

## Decision

Keep the w4 K-major result as diagnostic evidence only; reject it and both warp
probes for promotion. The active source workspace is restored to the exact
base commit after the measurements. The patches and complete evidence remain
in the task records and `runtime/remote-evidence/` for a future candidate.
