# Precision diagnostic

This directory contains the advisory A/A and A/B precision diagnostic. It reuses the benchmark input and source isolation code, runs each seed in fresh processes and records tolerance-grid metrics. It does not measure performance and does not replace the hard correctness gate.

The diagnostic's cache is deliberately identity-bound to source, workload, device, driver and numerical environment. Keep its cache outside Git.

```bash
python3 -B -m unittest discover -s projects/kimi-k3/prefill/chunk-gated-delta-h-iter2/precision -p 'test_*.py' -v
```
