# Kimi K3

Kimi K3 is the first optimization project in Infra Loop KDA. Its serving
stages are kept separate because prefill and decode have different shapes,
launch patterns and acceptance workloads.

- Prefill contains the accepted chunk-kda kernel package.
- Decode is the next stage and is scaffolded for future work.

Shared model facts and source pins live in the root config/ and docs/;
stage-specific contracts stay beside their kernels.
