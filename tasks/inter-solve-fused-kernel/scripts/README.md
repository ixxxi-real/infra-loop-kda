# Task utilities

- `configure_workloads.py` resolves the workload matrix from checkpoint metadata and a deployment description without importing Torch.
- `parse_ncu.py` parses a read-only NCU report and keeps missing metrics distinct from zero.

The resolver can be rehearsed without a model or GPU:

```bash
python3 tasks/inter-solve-fused-kernel/scripts/configure_workloads.py \
  --checkpoint-config projects/kimi-k3/prefill/checkpoint-config.example.json \
  --deployment projects/kimi-k3/prefill/deployment.example.json \
  --output /tmp/kimi-k3-workloads.json
```

The former remote launcher and preparation scripts were intentionally left out of the public task package because they contain site-specific SSH, container and storage policy. Implement that adapter against [`docs/runner-interface.md`](../../../docs/runner-interface.md) and keep its credentials in local configuration.
