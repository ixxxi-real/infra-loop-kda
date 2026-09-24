# Source trace: inter-solve-fused-kernel

Base commit: `9ac2710bd37622f38edb078cc753244a3c38c334`.

The target definition is in
`python/sglang/kernels/ops/attention/fla/chunk_intra.py`:
`chunk_kda_fwd_kernel_inter_solve_fused`. The wrapper
`chunk_kda_fwd_intra` launches it with grid `(NT, B*H)` and selects
`FUSE_RECOMPUTE`/`FUSE_DIAGONAL` from the Kimi K3 prefill grid heuristic in
`python/sglang/kernels/ops/attention/fla/kda.py`.

This task uses a new inter-solve workload. Previous H-state kernel evidence is
not acceptance evidence for this task.
