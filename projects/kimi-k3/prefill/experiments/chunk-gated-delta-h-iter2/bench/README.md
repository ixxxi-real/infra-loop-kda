# Prefill benchmark harness

The files in this directory are the reusable correctness and timing harness migrated from the Kimi K3 campaign. They are intentionally source-root based: the harness never imports the repository's SGLang package accidentally and records source/workload hashes in every report.

CPU-only checks:

```bash
python3 -B -m unittest discover -s projects/kimi-k3/prefill/experiments/chunk-gated-delta-h-iter2/bench -p 'test_*.py' -v
python3 -B projects/kimi-k3/prefill/experiments/chunk-gated-delta-h-iter2/bench/preflight.py \
  --source-root external/sglang \
  --workloads projects/kimi-k3/prefill/experiments/chunk-gated-delta-h-iter2/workloads.resolved.json \
  --static-check
```

GPU execution requires a separately prepared baseline and candidate source root. The harness does not select a gateway, GPU UUID, container or checkpoint automatically. Use an environment-specific runner to create those roots and store only a redacted evidence manifest in `evidence/`.
