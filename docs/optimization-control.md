# Optimization control plane

Infra Loop KDA keeps the model agent separate from the authority that accepts
an optimization. The agent may edit an isolated candidate and attach a
hypothesis; the campaign controller records identity, asks the evaluator to
run the gates, and is the only component that can move evidence to `promoted`.

The offline building blocks are:

- `tools/k3/campaign.py` — resumable campaign state machine. It records the
  immutable base/workload/gate identity, candidate budget, transitions and
  reasons. It never launches a GPU process.
- `tools/k3/diagnosis.py` — normalizes profiler output into a stable
  `diagnosis.json`. A candidate can point to this record with `diagnosis_ref`
  and `limiter_class`.
- `tools/k3/tuning.py` — deterministic Cartesian expansion of a parameter
  manifest. The project benchmark remains responsible for compilation,
  correctness and performance gates.

The intended lifecycle is:

```text
prepared -> running -> candidate_ready -> verified -> promoted
                                      \-> rejected
```

`blocked` may be entered from an active state and must carry a reason before a
campaign is resumed. `promoted` and `rejected` are terminal. A benchmark or an
agent must not skip `verified`.

## Candidate record additions

New candidate records should include:

```json
{
  "candidate_id": "h-bv16",
  "parent_candidate": "h-bv32",
  "search_lane": "structural",
  "diagnosis_ref": "runtime/diagnosis/h-state.json",
  "limiter_class": "shared_memory",
  "repro_count": 5,
  "gate_version": "prefill-kda-v2-full-1.03",
  "negative_evidence": ["long-sequence-regression"]
}
```

The machine-readable shape for the prefill campaign is kept in
`projects/kimi-k3/prefill/experiments/prefill-kda-iter2/candidate.schema.json`.
The existing historical JSONL is intentionally not rewritten; new campaigns
can adopt the schema incrementally.

Existing candidate history remains valid; these fields are required only for
new controlled campaigns. The acceptance gate is unchanged.

## Tuner lane

Use `bench/tuning-manifest.example.json` to declare a finite search space.
`tools.k3.tuning.expand` emits stable trial numbers and configuration hashes.
The manifest's seed is recorded for provenance, but enumeration is deterministic
and does not replace correctness, precision, paired timing or full-workload
geomean checks.
