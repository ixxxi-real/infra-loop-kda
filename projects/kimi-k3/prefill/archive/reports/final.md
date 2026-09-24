# Kimi K3 prefill KDA result

Status: `optimization_status=accepted`, `integration_status=not_applied`.

The accepted patch replaces three Q/K/V preparation copies and two normalization launches with one fused preparation path for the supported strided packed-QKV layout. It changes only `kda.py` and `l2norm.py`; the public ABI and fallback path remain intact.

## Evidence summary

- Source base: `8eea3c25a3eaa3c850dbdb0ede2bba7de8f03e93`
- Patch SHA-256: `eda8c23b63a08d651c321315d2579b396993aece88714758b83170797f1d81cb`
- Workload SHA-256: `2a8768fe725493776becd50f4a4c8df58775d70e21d77b3fe529d9066396e356`
- Correctness: 11/11 oracle cases and 51/51 baseline/candidate grid cases passed
- Precision diagnostic: 378/378 comparable fields identical for candidate-vs-baseline; A/A control also identical
- Timing: geometric mean ratios 1.1615484367 and 1.1750340327 across two independent five-trial measurements
- Mechanism profile: 12 launches reduced to 8 on the profiled ragged case; profile data is diagnostic only

## Limits

The workload is synthetic and deployment-bounded. Evidence covers one recorded device and frozen seeds. It does not establish production traffic benefit, TP8 serving readiness, or universal equivalence over every layout. The exact task base is currently local history; publish or reconstruct it before asking another engineer to apply the patch.
