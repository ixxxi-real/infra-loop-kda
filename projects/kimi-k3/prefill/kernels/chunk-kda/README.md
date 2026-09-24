# chunk-kda kernel

This directory is the code-facing deliverable for the Kimi K3 prefill
optimization. It contains the accepted source patch and its checksum manifest.

The patch changes the source entrypoints listed in the parent stage task.json
and must be applied only after verifying the exact base commit in the source
trace. A faster standalone kernel is not serving integration; those statuses
remain separate in the task record.
