# Task: prefill-kda-iter2

This task records a second, isolated optimization round for the complete
Kimi K3 prefill `chunk_kda` pipeline on GB300. The accepted parent task is
immutable. The exact base is commit
`9ac2710bd37622f38edb078cc753244a3c38c334`; workload and evidence are kept
under `runtime/remote-evidence/`.

The full pipeline profile identified the H-state update as the dominant warmed
stage. Four H tile/warp candidates were tested on GPU2 with correctness,
precision, and paired 51-case benchmarks; none met the 1.03x promotion gate.
The workspace is restored to the base source, and no serving integration was
performed. See `docs/optimization-round-20260928.md` and
`candidates.jsonl`.
