# Executable plan: inter-solve-fused-kernel

## Phase 0: freeze and trace

- Verify base commit `9ac2710bd37622f38edb078cc753244a3c38c334`.
- Use `workloads.resolved.json` unchanged; record its hash.
- Trace `kda.py -> chunk_kda_fwd_intra -> chunk_kda_fwd_kernel_inter_solve_fused`.
- Confirm safe-gate and fusion flags on GB300.

## Phase 1: baseline

- Run correctness and precision on all 47 cases.
- Run paired full-workload timing with state/output restoration.
- Profile one fused small-grid row and one non-fused long row with NCU/NSYS.

## Phase 2: candidates

- Implement one dataflow or layout candidate in `chunk_intra.py` only.
- Validate correctness and precision before timing.
- Run two independent full-workload paired measurements.
- Record rejected candidates and reset to base before the next one.

## Promotion

Require both paired full-workload geomeans >=1.03, no stable row regression >5%,
exact source/workload provenance, and human review. Diagnostic subsets may guide
hypotheses but cannot replace the full workload.
