# Task utilities

- `configure_workloads.py` resolves the workload matrix from checkpoint metadata and a deployment description without importing Torch.
- `parse_ncu.py` parses a read-only NCU report and keeps missing metrics distinct from zero.

The resolver can be rehearsed without a model or GPU:

```bash
python3 projects/kimi-k3/prefill/chunk-gated-delta-h-iter2/scripts/configure_workloads.py \
  --checkpoint-config projects/kimi-k3/prefill/checkpoint-config.example.json \
  --deployment projects/kimi-k3/prefill/deployment.example.json \
  --output /tmp/kimi-k3-workloads.json
```

The runner is included for this controlled node013 experiment; credentials remain in local SSH configuration.

For the node013 single-GPU route, use `kda_remote.py` with an explicit target
and GPU UUID. It supports `prepare`, `candidate`, `final` and `profile`
workflows; `profile --profile-kind ncu` is used for the target-kernel counters.
Do not use the old default node or GPU values.
