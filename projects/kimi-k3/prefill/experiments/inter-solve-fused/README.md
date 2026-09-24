# `chunk_kda_fwd_kernel_inter_solve_fused` workload

The [workloads.resolved.json](workloads.resolved.json) manifest is a new
47-case workload for the inter-solve fused kernel. It is derived from the
frozen Kimi K3 manifest but is not H-state kernel evidence and must be
validated independently.

The packed harness passes varlen input as `B=1`; the target launch is `(NT_total, 1*H)`, with `BT=64`, `BC=16`, `H=12`,
`K=V=128`, and Kimi K3's safe gate (`lower_bound=-5`). The wrapper enables
`FUSE_RECOMPUTE` and `FUSE_DIAGONAL` when packed `NT_total*H <= 256`; the manifest marks
the expected regime on each deployment row.

Coverage includes:

- 63/64/65 and 127/128/129 chunk boundaries;
- fused small-grid rows through the packed `NT_total*H=256` threshold;
- non-fused long prefills at 2048/4096/8192/16384 tokens;
- batch 2/8/32/128 grid saturation controls;
- ragged/varlen rows, resumed state, random initial state, and tracked state.

The manifest has 11 correctness cases and 36 deployment-grid cases. Its hash
is `915b967f3b9cd30945cfc8218101dbb226f7ba1fb144216e3408869712d08c9d`.
The parent full workload hash is
`4ae576f6b2a21ce6b0f7f0b7f0681077785759af6d4e0803434f5a930faec4ed`.

This file is a workload artifact only. A future optimization task must use a
fresh baseline, correctness, precision, NCU/NSYS and paired benchmark evidence
for this kernel; previous H-state measurements cannot be reused.
