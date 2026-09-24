# Implementation-plan draft: prefill-kda-iter2

Base commit: `9ac2710bd37622f38edb078cc753244a3c38c334`
Allowed files: `python/sglang/kernels/ops/attention/fla/kda.py`,
`python/sglang/kernels/ops/attention/fla/l2norm.py`

Status: **draft only. No kernel file has been edited. No measurement has been
taken on GB300.** Every performance figure below is a static estimate, labelled
as such, and is not evidence.

---

## 1. Baseline behaviour and the validation path

### 1.1 The real Kimi K3 prefill caller reaches Triton `chunk_kda`

Traced statically on this checkout:

| Step | Location |
|------|----------|
| K3 delta-attention layer calls `self.attn(...)` | `srt/models/kimi_k3.py:2042` (`KimiK3DeltaAttention.forward`) |
| Config routes to the KDA backend | `srt/configs/hybrid_arch.py:110` `kimi_linear_config()` matches `KimiLinearConfig`; `srt/layers/attention/attention_registry.py:513` -> `KDAAttnBackend(runner)` |
| Backend resolves per-phase kernels | `srt/layers/attention/linear/kda_backend.py:46` `KDAKernelDispatcher` |
| Extend dispatch | `kda_backend.py:794` `forward_extend` -> `kda_backend.py:892` `kernel_dispatcher.extend(...)` -> `kda_backend.py:354` `effective_extend_kernel(...)` |
| Triton kernel wrapper | `srt/layers/attention/linear/kernels/kda_triton.py:237` `TritonKDAKernel.extend` -> `chunk_kda` |
| Kernel entry | `kernels/ops/attention/fla/kda.py:1202` `chunk_kda` |

Two **independent** reasons the effective extend kernel is Triton for K3:

1. `prefill_default` is only computed when `hybrid_gdn_config(...)` matches
   (`attention_registry.py:425-428`). KDA is not GDN, so `prefill_default` stays
   `None` and `resolve_linear_attn_backends` falls through to
   `mamba.linear_attn_backend` (`linear/utils.py:87-89`), whose default is
   `"triton"` (`arg_groups/fields/exec_.py`, `linear_attn_backend = "triton"`).

2. Stronger, and it holds even against an explicit flag: K3 sets
   `lower_bound = config.linear_attn_config["gate_lower_bound"] = -5.0`
   (`models/kimi_k3.py:1718`; `model-profile.json` `gate_lower_bound: -5.0`,
   basis `checkpoint`). `effective_extend_kernel` reroutes any kernel that does
   not declare `supports_safe_gate` to the Triton kernel
   (`kda_backend.py:333-339`), and `CuteDSLKDAKernel.supports_safe_gate = False`
   (`kernels/kda_cutedsl.py:33`). So `--linear-attn-prefill-backend cutedsl`
   still executes Triton `chunk_kda` for this checkpoint.

**Consequence for the plan's risk list.** The plan flagged "runtime backend
selection must be confirmed rather than assumed". Statically it is confirmed
twice over, and the safe-gate reroute means the kernel under optimization is on
the serving path more robustly than a flag audit alone would show. This still
needs a runtime backend log on GB300 to close AC1 fully — the static trace
cannot observe an out-of-tree override or a divergent launch config.

**Consequence for the safe-gate path.** Because `lower_bound is not None` for
the real checkpoint, `chunk_kda_fwd` takes `safe_gate=True`
(`kda.py:1159`), which selects `kda_gate_chunk_cumsum`'s lower-bound branch
(`kda.py:1005`, `lower_bound * sigmoid(exp(A_log) * s)`) and, when
`fuse_diagonal` is false, the `chunk_kda_fwd_kernel_intra_sub_chunk` path rather
than the token-parallel path (`chunk_intra.py:945-979`). Any candidate must be
measured on the **safe-gate** configuration, not the default-gate one. A
benchmark that omits `lower_bound` measures a different code path than K3 serves.

### 1.2 Launch geometry on the real shape

From `model-profile.json`: `num_heads=96`, `local_num_heads=12`,
`head_k_dim=head_v_dim=128`, `attention_tp_size=8`, activation and state dtype
`bfloat16`, `kernel_chunk_size=64`, `use_qk_l2norm_in_kernel=true`,
`beta_is_raw=false`.

So per rank `H=12`, `K=V=128`, `BT=64`. For a single 8K prefill sequence,
`NT = 8192/64 = 128` and the `fuse_diagonal` / `fuse_recompute` heuristic
(`kda.py:1141-1148`, `_B * _NT_pr * _H_pr <= 256`) gives `1*128*12 = 1536 > 256`,
so **both fusions are off** on the 8K serving shape. They only engage for very
small batches (e.g. `NT*H <= 256`, i.e. under ~1.4K tokens at `H=12`). Worth
recording in the NCU baseline: the fused path that the accepted patch's comment
describes is not the path an 8K prefill takes.

### 1.3 Validation path

Per `contract.md`:
- `python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>`
  at `atol=0.003`, `rtol=0.03`, `relative_rms=0.03`, plus the FP32 tracking
  constraint, against both the frozen baseline and the FP32 oracle.
- `python3 bench/benchmark.py --baseline-root <baseline> --candidate-root <candidate> --workloads <workloads> --out <out>`,
  two independent paired CUDA-event sets, warmups + repeated trials, state
  restored before each timed replay.
- Gate: geomean speedup >= 3%, no stable row regressing > 5%.

**Neither command is runnable on this host.** See section 4.

---

## 2. Measured limiter

**Not yet established.** AC3 is blocked (section 4). The candidate in section 3
is therefore a *hypothesis from static reading*, not a profile-selected
candidate, and must be re-checked against the NCU/NSYS baseline before it is
implemented. If the profile shows a different limiter, C1 is dropped in favour
of what the profile indicates — that ordering is the plan's, and it is not
negotiable on the basis of how plausible C1 looks on paper.

---

## 3. Candidate C1 (proposed, unmeasured): make `l2norm_fwd` stride-aware and
drop the redundant Q/K materialization in `chunk_kda`

### 3.1 The redundancy

`chunk_kda` (`kda.py:1225-1227`):

```python
if use_qk_l2norm_in_kernel:
    q = l2norm_fwd(q.contiguous())
    k = l2norm_fwd(k.contiguous())
```

`q` and `k` arrive from `kda_backend.forward_extend`
(`kda_backend.py:853-857`):

```python
q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
```

`split` along the last dim yields a **narrow view**: shape `[T, q_dim]` with
stride `(total_qkv_dim, 1)` where `total_qkv_dim = q_dim + k_dim + v_dim`.
`unflatten` on the last axis keeps it a view — stride becomes
`(total_qkv_dim, D, 1)` for shape `[T, H, D]` — and `unsqueeze(0)` prepends a
size-1 axis. A genuinely contiguous `[1, T, H, D]` tensor would need stride
`(..., H*D, D, 1)`; here the token stride is `total_qkv_dim > H*D`. So `q` and
`k` are **not** contiguous on the real path, and `.contiguous()` is a real
gather-and-write pass, not a no-op.

`l2norm_fwd` then does `x.view(-1, x.shape[-1])` (`l2norm.py:77`), which
*requires* contiguity — which is exactly why the `.contiguous()` call is there —
and then reads `x` and writes a fresh `y` (`l2norm.py:80`, `97-107`).

Net traffic per tensor today: **read strided + write packed** (the
`.contiguous()`), then **read packed + write packed** (the norm). The first
write and second read exist only to satisfy `view()`.

### 3.2 The change

Give `l2norm_fwd` a strided entry point that consumes the `[1, T, H, D]` view
directly, then drop the two `.contiguous()` calls.

**A single row stride is not enough — this is the trap.** For `q` of shape
`[1, T, H, D]` with stride `(T*tot, tot, D, 1)` (where `tot = q_dim+k_dim+v_dim`),
row `(t, h)` of the flattened `[T*H, D]` matrix begins at byte offset
`t*tot + h*D`. Describing those rows with one stride `r` would require both
`r == D` (to step between heads) and `r == tot` (to step between tokens), i.e.
`tot == H*D` — which is exactly what is false here. So the naive
"`(x_stride_t, 1)` block pointer over `[T*H, D]`" formulation is wrong and will
silently read the wrong rows. The layout has to be addressed as genuinely 3D,
with **two** strides.

The fix is to make the grid 2D over `(token_block, head)` and offset the base
pointer per head, so that within one program the rows are tokens at a single
uniform stride. That is precisely the shape of the existing
`gdn_prefill_qkv_prepare_kernel` (`l2norm.py:122-216`), which already addresses
strided `[T, H, D]` Q/K this way — C1 reuses that proven addressing pattern
rather than inventing one.

New kernel in `l2norm.py` (additive; existing kernels untouched):

```python
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_strided_kernel(
    x, y, eps, T,
    x_stride_t, x_stride_h,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    # Rows within one head are tokens at a single uniform stride.
    p_x = tl.make_block_ptr(
        x + i_h * x_stride_h, (T, D), (x_stride_t, 1),
        (i_t * BT, 0), (BT, BD), (1, 0),
    )
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    # Output is freshly allocated and packed: [T, H, D] contiguous.
    p_y = tl.make_block_ptr(
        y + i_h * D, (T, D), (H * D, 1),
        (i_t * BT, 0), (BT, BD), (1, 0),
    )
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
```

Grid: `(cdiv(T, BT), H)`. Reduction is still per-row over the last axis in fp32,
identical to `l2norm_fwd_kernel` (`l2norm.py:66-68`).

**Allocate the output explicitly, not with `empty_like`.** Today's code uses
`y = torch.empty_like(x)` (`l2norm.py:80`), which defaults to
`memory_format=torch.preserve_format`. On the strided path that is ambiguous —
it must not be relied on to hand back a packed buffer. The strided branch has to
allocate `torch.empty(x.shape, dtype=..., device=...)` so the `(H*D, 1)` output
block pointer above is valid, and the existing `assert y.stride(-1) == 1`
(`l2norm.py:83`) should be tightened to assert the output is contiguous on that
branch.

In `l2norm_fwd`, the **existing contiguous path is left exactly as it is** (early
return), and the strided path is added only for the case that currently pays for
`.contiguous()`:

- If `x.is_contiguous()` → today's code, unchanged, bit-for-bit.
- Else if `x` is a `[..., T, H, D]` view with `x.stride(-1) == 1`, a uniform
  token stride, and `D <= 512` → the new strided kernel.
- Else → `x.contiguous()` and today's code. Every currently supported layout
  therefore keeps working; nothing is narrowed.

`D = 128` on the real shape, comfortably inside the `D <= 512` branch
(`l2norm.py:92`), so the `l2norm_fwd_kernel1` large-`D` branch needs no change.

`kda.py` — then, and only then:

```python
if use_qk_l2norm_in_kernel:
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
```

### 3.3 Why this preserves the invariants

- **Public ABI**: `chunk_kda`'s signature is untouched; `l2norm_fwd` gains no
  required parameter (the stride is derived from the tensor).
- **Numerics**: the reduction is per-row over the last axis, in fp32
  (`l2norm.py:66-68`), and `tl.sum(b_x * b_x, axis=1)` reduces over `BD`
  elements where `BD = next_power_of_2(D) = 128` in both the old and new
  kernels. Two things follow. First, changing the *address* a row is read from
  cannot change that row's reduction tree. Second — the part that matters for
  the new 2D grid — **regrouping which rows share a block is irrelevant to a
  row-wise reduction**: each output row depends only on its own `D` values, so
  blocking 16 consecutive tokens-of-one-head instead of 16 consecutive rows of
  the flattened `[T*H, D]` matrix changes nothing numerically. This is the same
  reasoning the existing `gdn_prefill_qkv_prepare_kernel` comment relies on
  ("Match `l2norm_fwd_kernel`'s block layout so the BF16 reduction tree is
  unchanged for strided inputs", `l2norm.py:151-152`). Expected
  bitwise-identical output; the precision diagnostic must confirm this rather
  than assume it.
- **Aliasing**: output is still a fresh `torch.empty_like`-style allocation, so
  `q`/`k` are not written in place and the caller's `qkv` buffer is untouched.
  Note `chunk_kda_fwd` separately passes `o=v` into `chunk_gla_fwd_o_gk`
  (`kda.py:1185`), i.e. `v` *is* used as the output buffer — which is why `v`'s
  `.contiguous()` (`kda.py:1236`) must **not** be removed. C1 touches Q/K only.
- **Fallback layouts**: non-uniformly-strided inputs fall back to
  `.contiguous()`, i.e. today's exact behaviour.
- **State / gate / FP32 tracking**: untouched. C1 does not enter the gate
  cumsum, the state pool, or the write-back.
- **Other callers**: `l2norm_fwd` is called from `fla/chunk.py:111-112`,
  `kernels/kda_cutedsl.py:131-132`, `kernels/kda_nvidia.py:113`,
  `kernels/gdn_cutedsl.py:78`, `npu/ascend_kda_backend.py:48-49`, and
  `l2norm.py:247,280`. Each was checked individually, because "the contiguous
  path is unchanged" only holds if they are in fact on it:
  - `kda_cutedsl.py:131-132` and `ascend_kda_backend.py:48-49` call
    `.contiguous()` explicitly at the call site.
  - `chunk.py:111-112` passes `q`/`k` with **no** explicit `.contiguous()`, but
    it sits inside `ChunkGatedDeltaRuleFunction.forward`, which is decorated
    with `input_guard` (`fla/utils.py:145-158`) — that decorator replaces every
    tensor argument with `i.contiguous()` before the body runs. So contiguity is
    guaranteed one level up rather than at the call.
  - `l2norm.py:247` is the already-contiguous early-return branch of
    `gdn_prefill_qkv_prepare_fwd`; `l2norm.py:280` consumes the freshly
    allocated packed `q_out`/`k_out`.

  For all of them `x_stride_t == D`, so the stride-aware kernel reduces to
  today's arithmetic exactly and the contiguous path is bitwise unchanged.
  `gdn_prefill_qkv_prepare_fwd` is GDN-only and not on the KDA path; C1 does not
  change it.

### 3.4 Static size estimate (not evidence)

Per rank, `H=12`, `D=128`, bf16, one 8K sequence: `8192*12*128*2 B ≈ 25.2 MB`
per tensor. C1 removes one write + one read of that for each of Q and K, i.e.
roughly `4 * 25.2 MB ≈ 100 MB` of HBM traffic per layer per 8K prefill. At an
assumed ~8 TB/s that is order ~12 µs per layer.

This estimate is the *reason to profile C1*, not a result. Whether it converts
into >= 3% geomean on the contract's workload depends on how much of
`chunk_kda`'s total time is in these two launches versus the chunk pipeline —
precisely what the AC3 NCU/NSYS baseline is for. It is entirely possible the two
l2norm launches are a small fraction of the kernel and C1 lands under the 3%
gate; that would be a legitimate measured rejection, recorded as such.

### 3.5 Relationship to the accepted `prefill-kda` patch

The accepted patch already changed Q/K/V preparation, and the plan warns a
follow-up must not regress strided views, aliasing, or fallback layouts. C1 is
*not* a copy of it: the accepted work added a fused prepare kernel for the GDN
FlashInfer path (`gdn_prefill_qkv_prepare_fwd`, which still does
materialize-then-norm in two passes). C1 instead removes the materialization
step altogether on the KDA path by teaching the norm kernel to address strided
rows. The fallback-layout risk is handled by the uniform-stride check in 3.2.

---

## 3bis. Alternate candidates, pre-analyzed (none selected, none implemented)

The contract's "one candidate at a time" applies to *implementation*. Analyzing
alternatives in advance does not violate it, and it is what lets the next round
match the NCU profile against a menu instead of only confirming or denying C1.
If the profile contradicts C1, one of these is the fallback rather than a fresh
round of reading.

**Scoping note that shapes this list.** Only `kda.py` and `l2norm.py` are
editable. Most of the 8K-path work lives elsewhere and is therefore off-limits
no matter what the profile says: `chunk_intra.py`
(`chunk_kda_fwd_kernel_inter_solve_fused`, the sub-chunk and token-parallel
kernels), `chunk_delta_h.py` (`chunk_gated_delta_rule_fwd_h`),
`chunk_intra_token_parallel.py`, `cumsum.py`. Inside the two allowed files the
kernels actually on the 8K serving path are: `kda_gate_chunk_cumsum_vector_kernel`,
`chunk_gla_fwd_kernel_o`, and the two `l2norm` kernels. That is the whole
candidate space. Worth stating plainly, because a profile that fingers
`inter_solve_fused` or the `h` kernel as the limiter leaves nothing actionable
under this contract, and the honest response would be a measured rejection with
"limiter is out of allowed scope" rather than a reach for second-best.

### C2 (proposed, unmeasured): load `h` pre-transposed in `chunk_gla_fwd_kernel_o`

`kda.py:819-840` builds the `h` block pointer in `(V, K)` layout and then
transposes in-register on every `i_k` iteration:

```python
p_h = tl.make_block_ptr(
    h + (i_tg * H + i_h) * V * K,
    (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0),
)
...
b_h = tl.load(p_h, boundary_check=(0, 1))
b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
```

A transposed block pointer loads it as `[BK, BV]` directly:

```python
p_h = tl.make_block_ptr(
    h + (i_tg * H + i_h) * V * K,
    (K, V), (1, K), (i_k * BK, i_v * BV), (BK, BV), (0, 1),
)
...
b_h = tl.load(p_h, boundary_check=(0, 1))
b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
```

This is already the idiom elsewhere in the same file — `kda.py:288-293` loads
`k` transposed exactly this way (`(K, T), (1, H * K), ..., order (0, 1)`), so it
is not a novel construction.

**Payoff is genuinely uncertain, and I am not going to oversell it.** `tl.trans`
on a register tile may compile to nothing more than a different MMA operand
selection, in which case C2 is worth 0%. It may instead force a shared-memory
round-trip per iteration, in which case it is worth something on tiles up to
`BK×BV = 64×128`. On Blackwell/tcgen05 the operand-layout constraints are what
decide this. That question is answerable *only* from the NCU report (look for
shared-memory traffic and instruction count inside this kernel), which is
precisely why C2 stays a candidate and not a change.

Traps if it is implemented:
- Keep the `.to(b_qg.dtype)` cast. `h = k.new_empty(...)`
  (`chunk_delta_h.py:396`) inherits `k`'s dtype, which is bf16 on the real path
  (so the cast is a no-op there) but would be fp32 if `k` ever is — dropping it
  would silently change the dot's input precision.
- The `if i_k >= 0:` guard at `kda.py:839` is always true (`i_k` is a loop index
  from 0). It is dead scaffolding, not a condition. Leave it alone or remove it
  as part of C2, but do not mistake it for logic that needs preserving.
- `allow_tf32=False` on the second dot (`kda.py:865`) is not on this path and
  must not be "unified" with the first as a drive-by.

### C3 (noted, not worked up): fp32 gate tensor `g` traffic

`kda_gate_chunk_cumsum` allocates `g` as fp32 (`kda.py:1061`, `output_dtype`
default `torch.float`). At `T=8192, H=12, K=128` that is ~50 MB per layer, and
it is then read by `chunk_kda_fwd_intra`, `chunk_gated_delta_rule_fwd_h`, and
`chunk_gla_fwd_kernel_o` — several hundred MB of reads per layer, plausibly
larger than C1's target.

Deliberately **not** worked up into a candidate: `g` holds a log2-space
cumulative sum consumed by `exp2`, and narrowing it is a precision change to the
gate, not a dataflow change. The contract requires gate semantics preserved and
FP32 tracking intact. Listed only so a profile pointing at gate traffic is not
mistaken for an unexplored opportunity — the reason it is unexplored is that it
is very likely out of bounds, and confirming that would take the precision
diagnostic, not a benchmark.

---

## 4. Why sections 2 and 3 cannot be closed on this host

Probed directly, all confirmed:

| Requirement | Probe | Result |
|-------------|-------|--------|
| GB300 GPU | `which nvidia-smi` | not found |
| NCU / NSYS profile | `which ncu nsys nvcc` | none found |
| Host | `uname -srm` | `Darwin 25.2.0 arm64` (macOS, Apple Silicon) |
| Correctness harness | `ls bench/` | no `bench/` directory in the workspace |
| Benchmark harness | `find -name "benchmark.csv" / bench*` | only the upstream SGLang `benchmark/` tree; no `bench/benchmark.py` |
| Resolved workloads | `find -name "workloads*.json"` | only `.kda-task/workloads.json`; the `workloads.resolved.json` named by `task.json` is absent |
| Task CLI | `which k3ctl` | not found |

Consequences, stated plainly:
- AC2 (62-case matrix, FP32 oracle, fixed-seed precision) — **not run**.
- AC3 (paired baseline + NCU/NSYS, limiter identification) — **not run**.
- AC5 (two paired benchmark sets, >= 3% geomean) — **not run**.
- AC1 — static half closed with file:line evidence; runtime backend log
  outstanding.

The contract requires the correctness matrix to pass *on the exact base before
editing*, and requires the candidate to come from the *measured* limiter.
Implementing C1 now would invert that order and produce a diff that cannot be
certified against AC4/AC5. So no kernel file was edited in Round 0.

## 5. Next actions once a GB300 node with the harness is available

1. Log the resolved backend at runtime (`Linear attention kernel backend: ...`
   from `linear/utils.py:99-102`) and the `chunk_kda` launch geometry; close AC1.
2. Run `bench/correctness.py` on the unmodified base: 62 bounded cases,
   non-zero count, zero failures, FP32 oracle + fixed-seed precision. Close AC2.
3. Capture the paired baseline plus NCU/NSYS for 8K and long-prefill, with
   `lower_bound=-5.0` (safe gate) and `fuse_diagonal=fuse_recompute=False`,
   matching section 1.2. Identify the limiter. Close AC3.
4. Confirm-or-drop C1 against that limiter. If confirmed, implement exactly
   section 3.2 and nothing else.
5. Correctness + precision first, then two independent paired benchmark sets
   with verified state restoration. Record promotion or precise rejection in
   `candidates.jsonl`.
