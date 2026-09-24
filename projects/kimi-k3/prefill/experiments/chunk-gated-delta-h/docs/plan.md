# Executable plan: `chunk-gated-delta-h`

## Phase 0 — source and workload freeze

- Verify commit `9ac2710bd37622f38edb078cc753244a3c38c334` is reachable.
- Set the allowed source file to `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`.
- Resolve a workload containing the real GB300 Kimi K3 prefill shape and the
  state/layout cases listed in `contract.md`.
- Record model facts: local heads 12, K/V head dimensions 128, BF16 state,
  `BT=64`, TP8 and full-rank gate.

## Phase 1 — baseline evidence

- Run the non-zero correctness and precision gates.
- Run paired CUDA-event timings with state restoration before every replay.
- Capture NSYS for launch sequence and NCU for registers, local-memory spills,
  achieved occupancy, active warps, tensor-pipe activity, global/shared-memory
  throughput and stall reasons.
- Store raw profiles externally and record checksums plus interpretation in
  the task evidence manifest.

## Phase 2 — candidate 1: register lifetime

- Keep the public signature and recurrence order unchanged.
- Refactor only the state-tile lifetime in `chunk_delta_h.py`; do not change
  gate math, state dtype, slot indexing or write-back semantics.
- Run correctness and precision immediately.
- If the candidate passes, collect paired timings and a matching NCU profile.
- Reject if registers/spills do not improve, if synchronization/traffic grows,
  or if any state/precision case fails.

## Phase 3 — later candidates

- Re-test BV16 only as a real-shape control, one configuration per run.
- Consider K-slice shared staging only if candidate 1 shows register pressure
  is the limiting factor and the expected extra traffic is bounded.
- Consider a split/spill design only after a profile demonstrates that a
  single-CTA recurrence is the remaining bottleneck.

## Decision gate

Require two independent five-trial paired runs, at least 3% geometric-mean
improvement, no stable row regression above 5%, full correctness/precision,
exact source and workload hashes, and human review before acceptance. Do not
mark serving integration complete from this kernel-only result.

## Candidate continuation

The full workload in `workloads.resolved.json` is frozen for the entire search.
Candidate rejection is not loop termination: if correctness and precision pass
but the two full-workload measurements miss the 3% gate, record the candidate
and its evidence in `candidates.jsonl`, restore the candidate workspace to the
base commit, and start the next independent candidate while the configured
round budget remains. The loop stops only on acceptance, budget exhaustion, an
external blocker, or an explicit contract decision that cannot be resolved from
the measurements. Bucket-specific workload files are diagnostic only and cannot
replace the full-workload acceptance denominator.
