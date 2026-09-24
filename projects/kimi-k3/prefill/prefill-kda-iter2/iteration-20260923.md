# Iteration record: 2026-09-23

The original accepted `prefill-kda` task was left untouched. A fresh
`prefill-kda-iter2` workspace was prepared at base commit
`9ac2710bd37622f38edb078cc753244a3c38c334`.

The Humanize/KDA loop confirmed the Kimi K3 extend path statically:
`KimiLinearConfig -> KDAAttnBackend -> TritonKDAKernel.extend -> chunk_kda`.
The loop did not edit source because the local machine had no CUDA runtime.

The frozen base was then exercised on GB300 GPU
`<redacted-gpu-uuid>` in the already-running `kda_test`
container. The 11 oracle and 51 deployment-grid correctness cases passed, and
the three-seed precision/A-A diagnostic passed.

An Nsight Systems capture for `uniform_b1_s16384` showed
`chunk_gla_fwd_kernel_o` at about 68.6% of GPU kernel time (about 593 us per
call), while both `l2norm_fwd_kernel` launches together were about 27.5 us
(about 2.6%). The proposed Q/K materialization removal is therefore below the
3% promotion gate unless a later profile shows a different shape-specific
limiter.

Two measured wrapper candidates were rejected by correctness:

- `chunk_size=128`: nonzero-state output relative RMS was about 1.0.
- `chunk_size=32`: non-finite output appeared in state and FP32-track cases.

No kernel source was promoted or changed. The next useful iteration needs to
target the dominant `chunk_gla_fwd_kernel_o` path, or expand the allowed source
scope to the implementation that owns that kernel, while retaining the same
correctness and precision gates.
