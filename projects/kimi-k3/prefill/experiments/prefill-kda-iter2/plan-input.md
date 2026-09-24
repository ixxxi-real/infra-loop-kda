# Plan input: prefill-kda-iter2

This is the second optimization task for the Kimi K3 prefill `chunk_kda`
kernel. `prefill-kda` is accepted and immutable; do not edit it in place.

## Goal

Find and validate one additional optimization for `chunk_kda` on the current
SGLang Kimi K3/GB300 TP8 prefill baseline at `9ac2710bd37622f38edb078cc753244a3c38c334`.
The candidate must be useful on the real serving shape and preserve every state
and gate invariant.

## Required evidence

1. Verify the real Kimi K3 caller reaches `chunk_kda` on the target checkout.
2. Run the 62-case correctness matrix, FP32 oracle and fixed-seed precision
   diagnostic on the exact base before editing.
3. Capture paired benchmark plus NCU/NSYS baseline for representative 8K and
   long-prefill shapes, recording the effective backend and launch geometry.
4. Select one candidate from the measured limiter. Candidates may include
   removing redundant work in the fused preparation path, reducing unnecessary
   materialization/loads, or a tightly scoped launch/data-layout change; do not
   combine directions or copy the old accepted patch blindly.
5. Run correctness and precision first, then two independent paired benchmark
   sets with state restoration. Record either promotion or a precise rejection.

## Risks and unknowns

The accepted patch already changed Q/K/V preparation, so a follow-up must prove
that it does not regress strided views, aliasing or fallback layouts. The source
branch has newer serving and SP changes than the accepted base; runtime backend
selection must be confirmed rather than assumed. The state pool is mutable and
benchmark replay can corrupt it if buffers are not restored. Host-side timing or
synthetic-only measurements do not establish GB300 serving benefit.

## Candidate order

- First: profile the current base and map the real prefill call to the kernel.
- Second: implement one low-risk dataflow or launch candidate supported by the
  profile, limited to `kda.py`/`l2norm.py`.
- Third: consider a larger algorithmic change only if the profile shows the
  first candidate cannot meet the gate; keep it as a separate lineage entry.
