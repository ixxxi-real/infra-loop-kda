# Migration from the current local tree

The migration is additive. Keep the existing SGLang worktree and historical evidence unchanged until the new repository can reproduce the required result.

| Current local material | New public location |
| --- | --- |
| `kda_tasks/kimi_k3/campaign.json` and model notes | `config/project.local.json` plus `docs/model-facts.md` |
| `kda_tasks/kimi_k3_prefill/` control docs | `projects/kimi-k3/prefill/` |
| `bench/` and `precision/` Python harnesses | `projects/kimi-k3/prefill/bench/` and `projects/kimi-k3/prefill/precision/` |
| `scripts/configure_workloads.py`, `scripts/parse_ncu.py` | `projects/kimi-k3/prefill/scripts/` |
| `runs/`, `profile/`, runtime orchestrator output | external storage; commit manifests under `evidence/` |
| `runtime/accepted-kernel.patch` | `projects/kimi-k3/prefill/kernels/chunk-kda/accepted-kernel.patch` |
| SGLang checkout | `external/sglang` at the exact source commit |
| `.humanize/` and gateway transcripts | private archive; link only a redacted review record |

Suggested migration order:

1. Copy facts and contracts, replacing machine-specific values with placeholders.
2. Record the original source commit and patch checksum.
3. Create manifests for historical runs instead of copying raw files.
4. Re-run validation and one small correctness workload from the new task directory.
5. Only then copy an accepted patch into `kernels/<kernel>/` and mark delivery state.
