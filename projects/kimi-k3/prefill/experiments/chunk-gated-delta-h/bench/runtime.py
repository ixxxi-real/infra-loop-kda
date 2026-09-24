"""CUDA execution helpers. Importing this file does not import torch."""

import hashlib
import math
import statistics
import time

# Fixed before either implementation is run. Do not tune to candidate output.
TOLERANCE = {"atol": 0.003, "rtol": 0.03, "relative_rms": 0.03}
# Same precision invariant as the registered track-state regression. This is
# intentionally separate from the end-to-end BF16 recurrence error budget.
TRACK_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}
SEED = 17092026


def trial_seed(trial):
    if type(trial) is not int or trial < 0:
        raise ValueError("trial must be a nonnegative integer")
    return SEED + trial * 1000003


def case_seed(case_id, seed):
    return seed + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)


class Inputs:
    def __init__(self, torch, profile, case, seed=SEED):
        self.torch, self.profile, self.case = torch, profile, case
        self.h, self.k, self.vdim = (profile[key] for key in ("local_num_heads", "head_k_dim", "head_v_dim"))
        self.dtype = getattr(torch, profile["activation_dtype"])
        self.state_dtype = getattr(torch, profile["state_dtype"])
        self.seed = case_seed(case["id"], seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def random(shape, dtype, amplitude=1.0):
            # CPU RNG makes inputs independent of GPU, implementation and trial.
            return (torch.randn(shape, generator=generator) * amplitude).to(dtype=dtype, device="cuda")

        tokens, batch = case["total_tokens"], case["batch_size"]
        packed = random((1, tokens, self.h * (2 * self.k + self.vdim)), self.dtype)
        q, k, v = packed.split([self.h * self.k, self.h * self.k, self.h * self.vdim], dim=-1)
        self.q = q.reshape(1, tokens, self.h, self.k)
        self.k_tensor = k.reshape_as(self.q)
        self.v = v.reshape(1, tokens, self.h, self.vdim)
        self.v.mul_(0.1)
        self.original_v = self.v.clone()
        self.g = random(self.q.shape, self.dtype)
        self.beta = random((1, tokens, self.h), self.dtype).float().sigmoid()
        self.a_log = random((self.h,), torch.float32, 0.3)
        self.dt_bias = random((self.h * self.k,), torch.float32, 0.3)
        slots = batch + 3
        self.slot_step = 2 if case["state_layout"] == "strided" else 1
        self.storage = random((slots * self.slot_step, self.h, self.vdim, self.k), self.state_dtype, 0.1)
        self.state = self.storage[::self.slot_step]
        self.indices_cpu = list(range(batch)) if case["index_mode"] == "identity" else list(range(batch, 0, -1))
        if case["index_mode"] == "padded":
            self.indices_cpu[-1] = -1
        if case["initial_state"] == "zero":
            for index in self.indices_cpu:
                if index >= 0:
                    self.state[index].zero_()
        self.original_storage = self.storage.clone()
        self.indices = torch.tensor(self.indices_cpu, device="cuda", dtype=torch.int32)
        cumulative = [0]
        for length in case["seq_lens"]:
            cumulative.append(cumulative[-1] + length)
        self.cu_seqlens = torch.tensor(cumulative, device="cuda", dtype=torch.int64)
        self.bound = {"model": profile["gate_lower_bound"], "safe": -5.0, "standard": None}[case["gate_mode"]]
        self.track_indices_cpu = [(length - 1) // 64 for length in case["seq_lens"]]
        if batch > 1:
            self.track_indices_cpu[-1] = -1
        self.track_indices = torch.tensor(self.track_indices_cpu, dtype=torch.int32, device="cuda")
        self.track = torch.full((batch, self.h, self.vdim, self.k), 17.0, dtype=torch.float32, device="cuda") if case["track_state"] else None

    def reset(self):
        self.storage.copy_(self.original_storage)
        self.v.copy_(self.original_v)
        if self.track is not None:
            self.track.fill_(17.0)

    def invoke(self, kernel, begin=0, end=None, cu_seqlens=None, track=True):
        end = self.case["total_tokens"] if end is None else end
        return kernel(
            q=self.q[:, begin:end], k=self.k_tensor[:, begin:end], v=self.v[:, begin:end],
            g=self.g[:, begin:end], beta=self.beta[:, begin:end],
            initial_state=self.state, initial_state_indices=self.indices,
            cu_seqlens=self.cu_seqlens if cu_seqlens is None else cu_seqlens,
            A_log=self.a_log, dt_bias=self.dt_bias, lower_bound=self.bound,
            use_qk_l2norm_in_kernel=True, beta_is_raw=False,
            output_intermediate_states=self.case["output_intermediate_states"],
            track_state=self.track if track else None,
            track_chunk_idx=self.track_indices if track and self.track is not None else None,
        )

    def describe(self):
        return {"seed": self.seed, "activation_dtype": str(self.dtype), "state_dtype": str(self.state.dtype),
                "q_shape": list(self.q.shape), "q_stride": list(self.q.stride()),
                "state_shape": list(self.state.shape), "state_stride": list(self.state.stride()),
                "state_indices": self.indices_cpu, "beta_dtype": str(self.beta.dtype),
                "beta_is_raw": False, "lower_bound": self.bound, "qk_l2norm": True,
                "input_layout": "packed post-convolution QKV views"}

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in (self.q, self.k_tensor, self.original_v, self.g, self.beta,
                      self.a_log, self.dt_bias, self.original_storage, self.indices, self.cu_seqlens):
            cpu = value.detach().cpu().contiguous()
            digest.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
            digest.update(cpu.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def recurrence(inputs, continuation_splits=None):
    """Independent CPU FP32 delta recurrence; no SGLang or Triton reference."""
    torch = inputs.torch
    q, k = inputs.q.cpu().float(), inputs.k_tensor.cpu().float()
    # l2norm_fwd returns activation dtype; preserve that public-input boundary.
    q = (q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(inputs.dtype).float()
    k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(inputs.dtype).float()
    values = inputs.original_v.cpu().float()
    raw = inputs.g.cpu().float() + inputs.dt_bias.cpu().reshape(inputs.h, inputs.k)
    a = inputs.a_log.cpu().exp().reshape(inputs.h, 1)
    gates = -a * torch.nn.functional.softplus(raw) if inputs.bound is None else inputs.bound * torch.sigmoid(a * raw)
    beta = inputs.beta.cpu().float()
    storage = inputs.original_storage.cpu().clone()
    pool = storage[::inputs.slot_step]
    output = torch.empty_like(values)
    intermediates = []
    tracks = torch.full_like(inputs.track.cpu(), 17.0) if inputs.track is not None else None
    commit_boundaries = set()
    if continuation_splits:
        position = 0
        for length in continuation_splits[:-1]:
            position += length
            commit_boundaries.add(position)
    offset = 0
    for sequence, length in enumerate(inputs.case["seq_lens"]):
        slot = inputs.indices_cpu[sequence]
        state = pool[slot].float().clone() if slot >= 0 else torch.zeros_like(pool[0], dtype=torch.float32)
        for local in range(length):
            token = offset + local
            if local % 64 == 0:
                intermediates.append(state.clone())
                if tracks is not None and local // 64 == inputs.track_indices_cpu[sequence]:
                    tracks[sequence].copy_(state)
            key = k[0, token]
            state = state * gates[0, token].exp()[:, None, :]
            prediction = torch.einsum("hvk,hk->hv", state, key)
            residual = (values[0, token] - prediction) * beta[0, token, :, None]
            state = state + residual[:, :, None] * key[:, None, :]
            output[0, token] = torch.einsum("hvk,hk->hv", state, q[0, token]) * inputs.k ** -0.5
            if token + 1 in commit_boundaries:
                state = state.to(inputs.state_dtype).float() if slot >= 0 else torch.zeros_like(state)
        if slot >= 0:
            pool[slot].copy_(state.to(inputs.state_dtype))
        offset += length
    return {"output": output, "storage": storage, "intermediates": torch.stack(intermediates).unsqueeze(0), "track": tracks}


def check_tensor(torch, actual, expected, label):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: nonfinite values")
    difference = actual - expected
    rms = difference.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    relative = rms / max(reference_rms, 1e-8)
    if relative > TOLERANCE["relative_rms"] and rms > 1e-6:
        raise AssertionError(f"{label}: relative RMS {relative:.6g} exceeds {TOLERANCE['relative_rms']}")
    torch.testing.assert_close(actual, expected, atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"], msg=label)
    return {"max_abs": difference.abs().max().item(), "rms": rms, "relative_rms": relative}


def track_boundaries(lengths, track_indices):
    """Locate pre-chunk snapshots in packed tokens and returned h rows."""
    if len(lengths) != len(track_indices):
        raise AssertionError("Track index count differs from sequence count")
    token_offset, chunk_offset, boundaries = 0, 0, []
    for sequence, (length, chunk) in enumerate(zip(lengths, track_indices)):
        if chunk < -1 or chunk >= (length + 63) // 64:
            raise AssertionError("Track chunk index lies outside its sequence")
        if chunk >= 0:
            boundaries.append((sequence, token_offset, chunk * 64, chunk_offset + chunk))
        token_offset += length
        chunk_offset += (length + 63) // 64
    return boundaries


def truncated_fp32_state(inputs, kernel, sequence, begin, prefix):
    """Separate invocation ending exactly BEFORE the tracked chunk starts.

    Promote the original BF16 cache value, not the already-updated pool. A
    negative source slot is a zero initial state; remap it to writable slot 0
    so the prefix run can expose its FP32 final accumulator.
    """
    torch = inputs.torch
    slot = inputs.indices_cpu[sequence]
    initial = inputs.original_storage[max(slot, 0) * inputs.slot_step].float().clone()
    if slot < 0:
        initial.zero_()
    state = initial.unsqueeze(0)
    if prefix:
        end = begin + prefix
        kernel(
            q=inputs.q[:, begin:end].clone(), k=inputs.k_tensor[:, begin:end].clone(),
            v=inputs.original_v[:, begin:end].clone(), g=inputs.g[:, begin:end].clone(),
            beta=inputs.beta[:, begin:end].clone(), initial_state=state,
            initial_state_indices=torch.tensor([0], dtype=torch.int32, device=inputs.state.device),
            cu_seqlens=torch.tensor([0, prefix], dtype=torch.int64, device=inputs.state.device),
            A_log=inputs.a_log.clone(), dt_bias=inputs.dt_bias.clone(), lower_bound=inputs.bound,
            use_qk_l2norm_in_kernel=True, beta_is_raw=False,
            output_intermediate_states=False, track_state=None, track_chunk_idx=None,
        )
        torch.cuda.synchronize()
    return state[0]


def assert_track_precision(torch, actual, reference, intermediate, evolved):
    """Reject routing track scratch through a BF16 intermediate, even if close."""
    if actual.dtype != torch.float32 or reference.dtype != torch.float32:
        raise AssertionError("Track scratch and truncated final-state reference must be FP32")
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError("Track scratch or truncated final-state reference is nonfinite")
    torch.testing.assert_close(actual, reference, **TRACK_TOLERANCE,
                               msg="track vs independently truncated FP32 final state")
    beyond_bf16 = not torch.equal(actual, actual.to(torch.bfloat16).float())
    # Chunk zero legitimately contains the original BF16 cache. Only evolved
    # states must demonstrate that the FP32 scratch retains extra precision.
    if evolved and not beyond_bf16:
        raise AssertionError("Tracked FP32 accumulator lost all precision beyond BF16")
    if intermediate is not None:
        intermediate = intermediate.detach().cpu()
        if not torch.equal(intermediate.float(), actual.to(intermediate.dtype).float()):
            raise AssertionError("Returned h must exactly equal activation-dtype rounding of track scratch")
    return {"truncated_fp32_tolerance": dict(TRACK_TOLERANCE),
            "retains_beyond_bfloat16": beyond_bf16,
            "intermediate_rounding_exact": intermediate is not None}


def validate_track_snapshots(inputs, kernel, intermediate):
    rows = []
    boundaries = track_boundaries(inputs.case["seq_lens"], inputs.track_indices_cpu)
    if not any(prefix > 0 for _, _, prefix, _ in boundaries):
        raise AssertionError("Track precision regression needs an evolved tracked boundary")
    for sequence, begin, prefix, h_row in boundaries:
        reference = truncated_fp32_state(inputs, kernel, sequence, begin, prefix)
        row = assert_track_precision(inputs.torch, inputs.track[sequence], reference,
                                     intermediate[0, h_row] if intermediate is not None else None,
                                     evolved=prefix > 0)
        row.update(sequence=sequence, prefix_tokens=prefix, intermediate_row=h_row)
        rows.append(row)
    return rows


def correctness_case(torch, kernel, profile, case):
    inputs = Inputs(torch, profile, case)
    expected = recurrence(inputs)
    inputs.reset()
    result = inputs.invoke(kernel)
    torch.cuda.synchronize()
    output, intermediate = result if isinstance(result, tuple) else (result, None)
    metrics = {"output": check_tensor(torch, output, expected["output"], "output")}
    active = [index * inputs.slot_step for index in inputs.indices_cpu if index >= 0]
    inactive = [index for index in range(inputs.storage.shape[0]) if index not in active]
    if active:
        metrics["state"] = check_tensor(torch, inputs.storage[active], expected["storage"][active], "state")
    if not torch.equal(inputs.storage[inactive].cpu(), expected["storage"][inactive]):
        raise AssertionError("Inactive state slots or strided pool padding were overwritten")
    if case["output_intermediate_states"]:
        if intermediate is None:
            raise AssertionError("Missing requested intermediate states")
        metrics["intermediates"] = check_tensor(torch, intermediate, expected["intermediates"], "intermediates")
    if inputs.track is not None:
        metrics["track_fp32"] = check_tensor(torch, inputs.track, expected["track"], "track_fp32")
        metrics["track_precision"] = validate_track_snapshots(inputs, kernel, intermediate)
        # Validate the final cast used by the radix-cache path, separately from h.
        metrics["track_pool_cast"] = check_tensor(torch, inputs.track.to(inputs.state_dtype), expected["track"].to(inputs.state_dtype), "track_pool_cast")
        for index, chunk in enumerate(inputs.track_indices_cpu):
            if chunk < 0 and not torch.equal(inputs.track[index].cpu(), expected["track"][index]):
                raise AssertionError("Untracked row was overwritten")
    splits = case.get("continuation_splits")
    if splits:
        expected_split = recurrence(inputs, splits)
        inputs.reset()
        outputs, offset = [], 0
        for length in splits:
            cu = torch.tensor([0, length], dtype=torch.int64, device="cuda")
            result = inputs.invoke(kernel, offset, offset + length, cu, track=False)
            outputs.append((result[0] if isinstance(result, tuple) else result).clone())
            offset += length
        torch.cuda.synchronize()
        metrics["continuation_output"] = check_tensor(torch, torch.cat(outputs, dim=1), expected_split["output"], "continuation_output")
        metrics["continuation_state"] = check_tensor(torch, inputs.storage, expected_split["storage"], "continuation_state")
    return {"id": case["id"], "status": "passed", "inputs": inputs.describe(), "metrics": metrics}


def capture_case(torch, kernel, profile, case, seed=SEED):
    """Capture every deployed-grid row for a separate implementation to check."""
    inputs = Inputs(torch, profile, case, seed=seed)
    fingerprint = inputs.fingerprint()
    inputs.reset()
    result = inputs.invoke(kernel)
    torch.cuda.synchronize()
    output, intermediate = result if isinstance(result, tuple) else (result, None)
    snapshot = {"output": output.detach().cpu(), "storage": inputs.storage.detach().cpu()}
    if case["output_intermediate_states"]:
        if intermediate is None:
            raise AssertionError("Missing requested intermediate states")
        snapshot["intermediates"] = intermediate.detach().cpu()
    if inputs.track is not None:
        snapshot["track"] = inputs.track.detach().cpu()
        track_precision = validate_track_snapshots(inputs, kernel, intermediate)
        for index, chunk in enumerate(inputs.track_indices_cpu):
            if chunk < 0 and not torch.equal(snapshot["track"][index], torch.full_like(snapshot["track"][index], 17.0)):
                raise AssertionError("Untracked row was overwritten")
    active = [index * inputs.slot_step for index in inputs.indices_cpu if index >= 0]
    inactive = [index for index in range(inputs.storage.shape[0]) if index not in active]
    if not torch.equal(snapshot["storage"][inactive], inputs.original_storage[inactive].cpu()):
        raise AssertionError("Inactive state slots or strided pool padding were overwritten")
    for name, value in snapshot.items():
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name}: nonfinite values")
    row = {"id": case["id"], "status": "passed", "inputs": inputs.describe(),
           "input_sha256": fingerprint, "inactive_storage_exact": True,
           "active_storage_indices": active}
    if inputs.track is not None:
        row["track_precision"] = track_precision
    return row, snapshot


def compare_snapshot(torch, actual, expected, active):
    if set(actual) != set(expected):
        raise AssertionError("Baseline/candidate snapshot fields differ")
    metrics = {}
    for name in actual:
        left, right = actual[name], expected[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{name}: baseline/candidate shape or dtype differs")
        if name == "storage":
            inactive = [index for index in range(left.shape[0]) if index not in active]
            if not torch.equal(left[inactive], right[inactive]):
                raise AssertionError("Baseline/candidate inactive storage differs")
            if not active:
                continue
            left, right = left[active], right[active]
        metrics[name] = check_tensor(torch, left, right, f"baseline_comparison_{name}")
    return metrics


def summarize(samples):
    values = sorted(samples)
    def percentile(fraction):
        location = (len(values) - 1) * fraction
        left, right = math.floor(location), math.ceil(location)
        return values[left] + (values[right] - values[left]) * (location - left)
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values),
            "median": statistics.median(values), "p10": percentile(0.1), "p90": percentile(0.9), "min": min(values), "max": max(values)}


def timed_case(torch, kernel, profile, case, warmup, samples, seed=SEED):
    inputs = Inputs(torch, profile, case, seed=seed)
    for _ in range(warmup):
        inputs.reset()
        result = inputs.invoke(kernel)
        torch.cuda.synchronize()
        del result
    start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize event resources before the first reported sample.
    start.record()
    finish.record()
    finish.synchronize()
    cuda_ms, host_ms = [], []
    for _ in range(samples):
        inputs.reset()
        torch.cuda.synchronize()  # Restore copies and their CPU enqueue cost excluded.
        host_start = time.perf_counter_ns()
        start.record()
        result = inputs.invoke(kernel)
        finish.record()
        finish.synchronize()
        host_ms.append((time.perf_counter_ns() - host_start) / 1e6)
        cuda_ms.append(start.elapsed_time(finish))
        del result
    return {"id": case["id"], "status": "passed", "inputs": inputs.describe(),
            "cuda_event_ms": cuda_ms, "synchronized_host_ms": host_ms,
            "cuda_event_summary_ms": summarize(cuda_ms), "synchronized_host_summary_ms": summarize(host_ms)}
