# Task: chunk-gated-delta-h

Fresh scaffold created by `k3ctl task-create`. It is intentionally
**unstartable**: the workload is a placeholder and no plan has been written.

| File | Purpose |
| --- | --- |
| `task.json` | Canonical task state. Nothing else overrides it. |
| `contract.md` | Objective, correctness, commands, promotion rules. |
| `source-trace.md` | Exact base commit and how to verify it. |
| `plan-input.md` | Human input to planning. |
| `prompt.md` | Implementation prompt derived from the KDA basic flow. |
| `workloads.placeholder.json` | Replace with a resolved workload. |
| `model-profile.json` | Model facts this task depends on. |

## Making it startable

```bash
# 1. Resolve the workload, then set workload_status to "resolved" in task.json.
# 2. Write plan-input.md into an executable plan.
# 3. Materialise the exact base commit.
k3ctl workspace prepare --task chunk-gated-delta-h
# 4. Set "status" to "ready" in task.json.
k3ctl agent plan --task chunk-gated-delta-h
```

No acceptance, patch, report or evidence manifest is scaffolded. Those are
produced by real runs and reviewed separately.
