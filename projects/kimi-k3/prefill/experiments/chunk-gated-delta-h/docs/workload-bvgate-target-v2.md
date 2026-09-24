# Workload v2: BV-gate target bucket

`workloads.bvgate-target-v2.json` is a new diagnostic workload derived from
the frozen `workloads.resolved.json` manifest. The v1 manifest remains the
full-grid regression and promotion workload; this file does not replace it.

The v2 deployment cases focus on the condition supported by the GB300 NCU
result: `N * local_num_heads <= 24` and at least 64 chunks per sequence. For
Kimi K3 (`local_num_heads=12`, `BT=64`), that means batch 1/2 long prefills.
The workload includes the seven measured winning rows, 63/64/65-chunk
boundaries, the batch-3 slot boundary, saturated batch-8/32/128 guardrails,
small-call controls, random-state cases, and a ragged target case.

The file contains 33 cases: the unchanged 11-case correctness suite plus 22
deployment-grid cases. Its SHA-256 is recorded in the file's metadata and its
parent v1 SHA-256 is `4ae576f6b2a21ce6b0f7f0b7f0681077785759af6d4e0803434f5a930faec4ed`.

Use v2 to evaluate the bucket-scoped optimization and report its result
separately. A v2 result must not be presented as whole-grid v1 acceptance. Any
future GPU run must rerun correctness, precision, and paired timing with this
new workload hash; v1 evidence cannot be reused as if it were v2 evidence.
