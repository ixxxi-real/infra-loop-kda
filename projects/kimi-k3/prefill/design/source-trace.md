# Kimi K3 prefill KDA source trace

## Frozen source

- Repository: `https://github.com/Infrawaves/sglang.git`
- Task base: `8eea3c25a3eaa3c850dbdb0ede2bba7de8f03e93`
- Public submodule baseline: see `config/project.example.json`; it is not silently treated as the task base.
- Kernel entrypoint: `python/sglang/kernels/ops/attention/fla/kda.py::chunk_kda`

The task base is retained from the local Kimi K3 campaign history. Before external reuse, make that exact commit reachable from the configured source remote or create a reviewed port against the public submodule commit.

## Call chain and change

The model's prefill path reaches `chunk_kda`, which receives Q/K/V as strided views into a packed post-convolution tensor. The baseline makes Q and K contiguous before L2 normalization and makes V contiguous separately. The patch adds `kda_prepare_qkv_fwd` in `l2norm.py` and calls it from `chunk_kda`, reducing preparation launches while preserving the V alias required by the output path.

## ABI and risk boundaries

The accepted evidence checked the 17 public parameters, state/index layouts, aliasing and in-place write-back, padding sentinel handling, int64 stride arithmetic and FP32 tracking. Alternative backends, real serving dispatch and TP8 integration were deliberately outside this task's acceptance.

See `../kernels/chunk-kda/patch-manifest.json` for file hashes and `../../../../evidence/manifests/prefill-acceptance.json` for the evidence boundary.
