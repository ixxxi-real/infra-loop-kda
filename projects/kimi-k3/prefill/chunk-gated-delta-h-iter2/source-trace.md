# Source trace: chunk-gated-delta-h-iter2

| Field | Value |
| --- | --- |
| Repository | `https://github.com/Infrawaves/sglang.git` |
| Base branch | `main` |
| Base commit | `9ac2710bd37622f38edb078cc753244a3c38c334` |
| Availability | verify locally before use |

## Verifying the base

```bash
k3ctl doctor --task chunk-gated-delta-h-iter2
k3ctl workspace prepare --task chunk-gated-delta-h-iter2
```

`workspace prepare` refuses to substitute any other commit. If the exact base
commit is not reachable from the configured local source repository, it fails
with a source-availability error. Publish or fetch the exact ref; the public
submodule pin is a different source base and is not an acceptable stand-in.

## Notes

The commit is currently reachable from the local `external/sglang` checkout;
the checkout reports `9ac2710bd37622f38edb078cc753244a3c38c334`. The public
submodule pin and the earlier accepted QKV-preparation task base are different
source identities and must not be substituted. The target function is present
at `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py` and exposes the
single-configuration `BV`, `num_warps` and `num_stages` controls used by this
strategy. This source ref still needs to be verified by `k3ctl workspace
prepare` before implementation begins.
