# Draft: inter-solve fused kernel optimization

Target: `chunk_kda_fwd_kernel_inter_solve_fused` in
`python/sglang/kernels/ops/attention/fla/chunk_intra.py`.

The workload must distinguish the small-grid fused regime (packed `NT_total*H <= 256`; source heuristic uses `_B*_NT_pr*_H_pr`)
from the large-grid non-fused regime, because `kda.py` selects both
`FUSE_RECOMPUTE` and `FUSE_DIAGONAL` from that threshold. The Kimi K3 target is
`H=12, K=V=128, BT=64, BC=16`, with safe-gate lower bound `-5`.

The first baseline should profile both regimes and record whether time is spent
in the fused inter/solve kernel, the preceding token-parallel diagonal path,
or the later recompute/output stages. Candidates must preserve Aqk/Akk/Akkd,
packed w/u/kg outputs, varlen chunk-index behavior, safe-gate math and all
fusion flags.

Candidate directions remain hypotheses until NCU/NSYS identifies the dominant
term. Test one direction per round and keep the full 47-case workload frozen.

## GPU0 iteration evidence

The corrected manifest hash is `915b967f3b9cd30945cfc8218101dbb226f7ba1fb144216e3408869712d08c9d`.
On GB300 GPU0, the baseline passed all 11 correctness and 36 full-grid comparisons.
Five isolated candidate directions were tested and rejected: explicit `num_stages=1`
(0.95135x), removing the fused barrier (0.97408x), `BV=32` (1.00216x), restricting
`BK=64` (0.95628x), and register-indexed fused diagonal tiles (Triton compilation
failure). The candidate tree was reset to the exact base after each rejection; no
source change is promoted.
