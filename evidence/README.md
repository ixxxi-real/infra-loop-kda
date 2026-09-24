# Evidence

Large run, profiler and orchestration artifacts stay outside Git and are written under the ignored `.infra/` area. Copy [`manifests/run.example.json`](manifests/run.example.json), fill it with redacted metadata and commit the small manifest under this directory; export bundles themselves live in `.infra/exports/`:

```json
{
  "run_id": "2026-09-20-prefill-kda-r001",
  "task_id": "prefill-kda",
  "source_commit": "<40-character sha>",
  "command": "<redacted reproducible command>",
  "artifacts": [
    {
      "uri": "<external artifact URI>",
      "sha256": "<sha256>",
      "bytes": 0
    }
  ]
}
```

Do not commit raw traces, private paths, credentials or request data.
