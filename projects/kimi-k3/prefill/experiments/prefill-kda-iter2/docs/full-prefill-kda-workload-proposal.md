# Full KDA prefill workload proposal (DRAFT)

Status: pending human review. This document describes workload and acceptance
rules only. It does not modify `workloads.resolved.json`, start a GPU job, or
restart a service.

## Scope

The optimization target is the complete KDA prefill pipeline:

```text
chunk_kda
├── chunk_kda_fwd_intra              # w/u, A, and gate preparation
├── chunk_gated_delta_rule_fwd_h     # recurrent H state update
└── chunk_gla_fwd_kernel_o            # output projection
```

The test entrypoint remains
`python/sglang/kernels/ops/attention/fla/kda.py::chunk_kda`, rather than a
standalone H-kernel call. Full-operator timing covers all three stages; NSYS and
NCU are used to locate the dominant stage and validate the candidate mechanism.

The baseline is commit `9ac2710bd37622f38edb078cc753244a3c38c334`. The current
62-case workload SHA-256 is
`3452bc547ce0439296444409b9cc651fd5d18ff6e27c66c1f0e59fad27a254c5`.

## Workload layers

### A. ABI and precision correctness: 11 cases

Reuse the existing correctness matrix, covering:

- nonzero and zero state;
- contiguous and strided state pools;
- identity, permuted, and padded slot indices;
- FP32 tracked and intermediate state;
- standard and safe/model gates;
- 63/64/65 and 127/128/129 boundaries;
- continuation splits `63+1+65`, `64+65`, and `1024+1024`.

Both baseline and candidate must pass an independent FP32 recurrence oracle. A
candidate-versus-baseline comparison alone is insufficient.

### B. Frozen full-operator grid: 51 cases

Keep the existing 51 deployment-grid cases as the historical comparison set:

- batch sizes `1/2/8/32/128`;
- sequence/chunk boundaries `1/63/64/65/127/128/129`;
- lengths 512, 1024, 2048, 4096, 8192, and 16384;
- ragged batches, mixed lengths, and resumed state.

This remains the primary promotion denominator so new long-sequence cases do not
break comparability with historical results.

### C. Long-prefill scheduler-chunk diagnostics: 12 cases

Production prompts can total 65536 tokens, while deployment
`max_prefill_tokens=16384`. KDA therefore processes a request in scheduler
chunks rather than one call containing all 39,360 or 26,240 miss tokens. Each
case stays at or below 16,384 tokens, and `initial_state=random` means the chunk
follows an existing prefix/state. Cumulative benefit across chunks is checked
separately in serving replay:

| Case | New tokens / sequence | Prefix hit rate | Batch | Purpose |
|---|---:|---:|---:|---|
| `hit40_b1` / tail | 16384 / 6592 | 40% | 1 | 0.3 QPS, first and final chunks |
| `hit60_b1` / tail | 16384 / 9856 | 60% | 1 | 0.5 QPS, first and final chunks |
| `hit90_b1` | 6554 | 90% | 1 | 2 QPS, single completed chunk |
| `hit40_b2` | [8192,8192] | 40% | 2 | low-concurrency merged request |
| `hit60_b2` | [8192,8192] | 60% | 2 | low-concurrency merged request |
| `hit90_b2` | [3277,3277] | 90% | 2 | short-miss merged request |
| `ragged_hit40_hit60` | [8192,4096] | mixed | 2 | ragged scheduling |
| `hit60_chunk_b4` | [4096] x4 | approximately 60% | 4 | small-batch parallelism |
| `hit60_chunk_b8` | [2048] x8 | approximately 60% | 8 | batched parallelism |

These 12 cases are initially diagnostic and do not replace the historical
51-case denominator. They establish whether a candidate improves the main
production shapes instead of only a boundary or the H sub-kernel.

The logical miss-token count for 40%, 60%, and 90% is rounded from
`65536 * (1 - hit_rate)` and stored in case metadata. The actual `seq_lens` are
scheduler chunks. If a trace provides a more precise prefix/miss histogram,
regenerate the workload hash from that trace.

### D. Independent serving replay

After the operator workload passes, run an external replay against the existing
TP8 service. Do not mix service noise into the operator promotion denominator:

- 40% hit rate at 0.3 QPS, output=1;
- 60% hit rate at 0.5 QPS, output=1;
- 90% hit rate at 2 QPS, output=1;
- at least 500 requests at concurrency 4; add 800 requests at concurrency 16
  when historical replay is required.

Record TTFT P50/P90/P99, prefill throughput, request throughput, GPU utilization,
and each stage's kernel share. Do not restart the complete service in this
stage; use only the confirmed deployment and an isolated candidate service.

## Per-round order

1. Confirm the baseline chunk_kda call chain and module provenance on GPU2; do
   not use GPU0.
2. Run baseline correctness, the FP32 oracle, and three precision seeds.
3. Capture NSYS for representative A/B/C shapes and measure the intra, H, and O
   stage shares.
4. Profile the dominant controllable stage with NCU; candidates must come from
   that evidence.
5. Implement one structural candidate, then run A/B correctness and precision.
6. Run the complete B and diagnostic C workloads with two independent five-trial
   paired benchmarks.
7. Only after those pass, run D serving replay. Record rejected candidates and
   restore the base before the next round.

## Promotion criteria

- A: 11/11 correctness, FP32 oracle, and tracked/state precision pass;
- B: full-operator geometric mean at least `1.03x` in both independent paired
  runs, with no stable single-row regression above 5%;
- C: diagnostic geometric mean at least `1.03x`, with no regression above 5% on
  the three primary hit-rate cases;
- D: TTFT P50/P90/P99 and prefill throughput are no worse than baseline. Pursue
  end-to-end gain only after C passes; D must not change the B denominator;
- any failure retains complete source, workload, device, and profiler provenance.
  Never report only the best single-kernel number.

## Decisions for review

1. Keep historical B=51 as the first promotion denominator and add C=12 as the
   long-prefill diagnostic denominator?
2. Generate 40%, 60%, and 90% from `65536 * miss_rate`, or read exact lengths
   from the new trace's miss-token histogram?
3. Keep both concurrency 4 and concurrency 16 replay in stage D?
4. Permit stage D on the existing TP8 service only after B and C pass?
