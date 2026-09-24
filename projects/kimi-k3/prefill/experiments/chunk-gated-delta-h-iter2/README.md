# Task: chunk-gated-delta-h-iter2

This is an independent follow-up to `chunk-gated-delta-h`. It starts from the
exact `9ac2710bd37622f38edb078cc753244a3c38c334` checkout and targets the
state/dataflow implementation of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`.
The parent task and its dirty candidate workspace remain untouched.

Two workload manifests are frozen here:

- `workloads.resolved.json`: 62-case full acceptance workload (11 correctness +
  51 deployment-grid cases). This is the only promotion denominator.
- `workloads.state-dataflow-focused-v1.json`: 33-case diagnostic bucket with
  long low-batch state rows, state/layout/continuation checks and saturation
  guardrails. It is for profiling and hypothesis selection only.

No full serving restart is part of preparation. The first run is an isolated
baseline through the existing `kda_test` harness, followed by targeted NSYS/NCU
only when the baseline reaches the target kernel.

The task carries its own `bench/`, `precision/` and `scripts/` harness copied from
the parent task. They are path-local and hash the exact source and full workload,
so this task does not depend on the parent task directory for CPU checks or
evidence generation.

## Preparation

```bash
python3 tools/k3ctl.py workspace prepare \
  --config config/project.local.json \
  --task chunk-gated-delta-h-iter2
python3 tools/k3ctl.py workspace status \
  --config config/project.local.json \
  --task chunk-gated-delta-h-iter2
python3 tools/k3ctl.py agent plan \
  --config config/project.local.json \
  --task chunk-gated-delta-h-iter2
```

The writer must not start until the workspace is clean and baseline evidence is
recorded.

The site runner is `scripts/kda_remote.py`. Every remote invocation for this
task must use `--target <gpu-node>` and the card-2 UUID
`<redacted-gpu-uuid>`; it verifies the container-visible
UUID and takes a per-GPU lock before running.
