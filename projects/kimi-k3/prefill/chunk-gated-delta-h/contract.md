# Task contract: chunk-gated-delta-h

Derived from the Kernel Design Agents basic flow. The upstream prompt is
referenced, not copied: `external/kda/prompts/basic-flow.md`
(sha256 `0ccd7f7c2a8a09bc1eaca287eb3c0026a99db7033a1a032d50c17f8cff96d038`, KDA commit `ef6ce617693ef0782b3ecb9f37e39bbf10226a90`).
NVlabs Kernel Design Agents (Kernel Design Agents workflow, not an SDK). See THIRD_PARTY_NOTICES.md and external/kda/LICENSE.

## Objective

Reduce GB300 Kimi K3 prefill latency for chunk_gated_delta_rule_fwd_kernel_h_blockdim64 by lowering register pressure and increasing useful state-kernel parallelism without changing recurrence state semantics.

## Inputs and outputs

The entrypoint is
`python/sglang/kernels/ops/attention/fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
launched by `chunk_gated_delta_rule_fwd_h`. The target shape is Kimi K3
prefill with local `H=12`, `K=V=128`, `BT=64`, BF16 state/activation and the
existing gate variants (`USE_G`/`USE_GK`). The kernel produces per-chunk `h`,
optional `v_new`, optional tracked intermediate state and an in-place final
state update. The public `chunk_kda` ABI, state layout, slot/index semantics,
gate math, padding sentinel behavior and return aliases must remain unchanged.

## Correctness requirements

The new task must execute a non-zero bounded case set covering chunk boundaries
(`63/64/65`, `127/128/129`), long prefill (`4096/8192/16384`), batch 1 and
larger batches, zero and random initial state, continuation state, padded and
strided state pools, slot permutations, `SAVE_NEW_VALUE`, `TRACK_STATE`, and
both gate paths. Compare with the frozen baseline and an FP32 oracle. Reuse
the accepted KDA tolerances only after confirming that the H-state harness
exercises the target kernel: `atol=0.003`, `rtol=0.03`,
`relative_rms=0.03`, plus the existing FP32 tracking constraint.

## Constraints

- The first candidate may change only `chunk_delta_h.py`.
- Keep the implementation in Triton; defer a CUDA rewrite until a measured
  Triton dataflow candidate is shown to be irreducibly limited.
- Do not add multi-configuration Triton autotuning: the kernel updates
  `initial_state` in place and repeated timing replays can corrupt the state
  pool or exhaust memory while cloning it.
- Every timed invocation must restore the initial-state buffer and any tracked
  outputs before replay.
- Preserve TP8/prefill dispatch, ragged sequence handling and fallback paths.

## Validation command

```
python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>
```

## Evaluation command

```
python3 bench/benchmark.py --baseline-root <baseline> --candidate-root <candidate> --workloads <workloads> --out <out>
```

## Promotion criteria

- The validation command passes with a non-zero, bounded case count.
- Precision passes for the declared seeds and state layouts.
- Two independent five-trial paired measurements on the same GB300 workload
  show at least 3% geometric-mean improvement, with no stable workload row
  regression above 5%.
- Performance evidence records the exact source commit and workload file.
- A rejected candidate records why, in `candidates.jsonl`.
- Human review is required before the task status becomes `accepted`.

## Iteration and stop policy

`workloads.resolved.json` is the immutable full-workload acceptance input. A
candidate that fails only the performance gate remains a rejected lineage node,
not a terminal task state: after recording its reason and restoring the base
workspace, continue to the next one-candidate experiment until the configured
round budget is exhausted. Human review is reserved for a passing candidate,
budget exhaustion, an external blocker, or a contract decision that the
measurements cannot settle. Reduced bucket workloads may be used for diagnosis
but cannot change the full-workload promotion rule.

## Evidence

Candidate lineage lives in `candidates.jsonl`; measurements in
`benchmark.csv`; raw artifacts stay outside Git with a manifest checked in.
