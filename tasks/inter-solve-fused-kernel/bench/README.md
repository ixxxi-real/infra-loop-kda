# Prefill benchmark harness

The files in this directory are the reusable correctness and timing harness migrated from the Kimi K3 campaign. They are intentionally source-root based: the harness never imports the repository's SGLang package accidentally and records source/workload hashes in every report.

CPU-only checks:

```bash
python3 -B -m unittest discover -s tasks/inter-solve-fused-kernel/bench -p 'test_*.py' -v
python3 -B tasks/inter-solve-fused-kernel/bench/preflight.py \
  --source-root external/sglang \
  --workloads tasks/inter-solve-fused-kernel/workloads.resolved.json \
  --static-check
```

GPU execution requires a separately prepared baseline and candidate source root. The harness does not select a gateway, GPU UUID, container or checkpoint automatically. Use an environment-specific runner to create those roots and store only a redacted evidence manifest in `evidence/`.
