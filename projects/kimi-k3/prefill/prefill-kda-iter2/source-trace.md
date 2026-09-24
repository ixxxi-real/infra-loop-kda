# Source trace: prefill-kda-iter2

| Field | Value |
| --- | --- |
| Repository | `https://github.com/Infrawaves/sglang.git` |
| Base branch | `feat/kimi_k3_support_sp` |
| Base commit | `9ac2710bd37622f38edb078cc753244a3c38c334` |
| Availability | verify locally before use |

## Verifying the base

```bash
k3ctl doctor --task prefill-kda-iter2
k3ctl workspace prepare --task prefill-kda-iter2
```

`workspace prepare` refuses to substitute any other commit. If the exact base
commit is not reachable from the configured local source repository, it fails
with a source-availability error. Publish or fetch the exact ref; the public
submodule pin is a different source base and is not an acceptable stand-in.

## Notes

Record here how the base commit was obtained, whether it is published, and any
divergence from the public pin.
