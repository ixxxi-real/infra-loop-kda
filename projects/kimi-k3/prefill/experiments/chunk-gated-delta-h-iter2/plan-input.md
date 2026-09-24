# Plan input: chunk-gated-delta-h-iter2

## Goal

Reduce GB300 Kimi K3 prefill latency in
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` through a measured Triton
state/dataflow change. The candidate must lower the live register footprint or
otherwise improve useful state-kernel parallelism without changing recurrence
semantics, state write-back, gate math, aliases or the public KDA ABI.

## Evidence carried forward

The parent task measured a shape-gated BV16 path that improved seven long,
low-slot rows but failed the 51-row promotion denominator (`0.99483x` and
`1.00830x` whole-grid runs). A `num_stages=1` probe made the isolated kernel
about 2.10x slower. A previous state/output fusion regressed materially. These
results rule out repeating the same global BV/warps/stages sweep.

Fresh baseline evidence is still required for this task. Parent measurements are
lineage and hypothesis evidence, not acceptance evidence.

## Frozen workloads

- Full gate: `workloads.resolved.json`, exactly 62 cases and its SHA-256. Do not
  change or reorder this file during candidate search.
- Diagnostic: `workloads.state-dataflow-focused-v1.json`, 33 cases. It contains
  all 11 correctness cases, the seven long/low-slot rows that exercise the
  state-kernel opportunity, boundary probes around 4032/4160 tokens, four new
  random-state focus rows, and batch-saturation guardrails. Diagnostic results
  cannot change the full-grid denominator or establish production serving
  benefit.

## Baseline preparation

1. Materialise the exact 9ac base in a fresh workspace; do not use the dirty
   parent workspace.
2. Verify the target function, effective KDA backend, imported module paths,
   local H=12, K=V=128, BF16 state/activation and BT=64.
3. Run baseline correctness on the full 62 cases, then the three-seed precision
   diagnostic with state restoration.
4. Run paired baseline timing on both full and focused workloads. Capture one
   representative long/low-slot and one saturated row with NSYS/NCU before code
   changes.

## Candidate order

1. **Register/dataflow lifetime candidate (first):** change only
   `chunk_delta_h.py`; keep the recurrence loop and state layout fixed. Use the
   fresh NCU register/spill data to select a concrete transformation. Do not
   claim that merely renaming or reordering loop-carried accumulators reduces
   registers.
2. **Explicit state staging candidate (second, only if justified):** test a
   controlled shared-memory or staged state representation only if NCU shows a
   register-limited path and the additional synchronization/traffic is measured.
3. **Producer/consumer split (last resort):** consider an extra kernel only if
   the single-CTA recurrence is proven wave-starved and the added launch/state
   traffic can be bounded.

Do not combine candidates. Do not write CUDA until a Triton dataflow candidate
has a measured reason it cannot meet the target.

## Acceptance

Every candidate must pass the full 62-case correctness gate, three-seed precision
checks and two independent paired five-trial measurements. Promotion requires at
least 1.03 geometric-mean speedup on the full workload and no stable row
regression above 5%. A candidate that passes only the focused diagnostic is
rejected for promotion but retained in `candidates.jsonl`.
