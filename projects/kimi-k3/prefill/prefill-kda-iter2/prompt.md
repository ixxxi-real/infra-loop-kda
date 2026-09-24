# Implementation prompt: prefill-kda-iter2

You are working in an isolated task workspace prepared at base commit
`9ac2710bd37622f38edb078cc753244a3c38c334`. Produce the best correct implementation for the contract
below, one candidate at a time.

This prompt follows the Kernel Design Agents basic flow. The upstream prompt
is referenced rather than reproduced: `external/kda/prompts/basic-flow.md`
(sha256 `0ccd7f7c2a8a09bc1eaca287eb3c0026a99db7033a1a032d50c17f8cff96d038`).
NVlabs Kernel Design Agents (Kernel Design Agents workflow, not an SDK). See THIRD_PARTY_NOTICES.md and external/kda/LICENSE.

## Task contract

- Task name: `prefill-kda-iter2`
- Objective: Continue optimizing the Kimi K3 prefill chunk_kda kernel on the current 9ac2710bd baseline and TP8/GB300 workload. Seek measurable end-to-end benefit while preserving the chunk_kda ABI, state write-back, gate semantics, and numerical precision. Every candidate must pass independent correctness, precision, paired-performance, and required NCU/NSYS evidence.
- Correctness requirements: see `contract.md`
- Performance or quality target: see `contract.md`
- Allowed implementation approaches: see `contract.md` constraints
- Validation command: `python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>`
- Evaluation command: `python3 bench/benchmark.py --baseline-root <baseline> --candidate-root <candidate> --workloads <workloads> --out <out>`
- Promotion criteria: see `contract.md`

## Workflow

1. Read the workspace structure, baseline implementation, tests and contract.
2. Identify the baseline behaviour and the validation path.
3. Research only the references needed for this task.
4. Write the implementation-plan draft to `docs/draft.md`.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time.
7. Run validation after each meaningful candidate.
8. Record candidate results, parent relationships and evidence.
9. Keep the final change scoped to the contract.

Do not start implementation until the draft exists.
