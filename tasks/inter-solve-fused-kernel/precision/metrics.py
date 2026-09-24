"""Tolerance-grid error metrics.

The reported grid has one plain-Python definition (`sweep_reference`) and one
chunked torch implementation (`Sweep`). The torch path exists only for speed on
tensors with tens of millions of elements; `self_check` asserts the two agree
before any real comparison is reported, so the fast path cannot silently drift
away from the definition the report claims to use.
"""

import hashlib
from pathlib import Path

# Predeclared before either implementation was run in this diagnostic. Do not
# tune to observed output.
ATOL = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
RTOL = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2]
# bench/runtime.TOLERANCE, the unchanged original acceptance pair. It is the
# loosest row of this grid; the diagnostic grid does not replace it. The worker
# asserts these against the harness so they cannot drift apart silently.
ACCEPTANCE = {"atol": 3e-3, "rtol": 3e-2}
TOLERANCE_RELATIVE_RMS = 0.03
TRACK_ACCEPTANCE = {"atol": 1e-5, "rtol": 1e-5}
PERCENTILES = [0.5, 0.9, 0.99, 0.999]
NEAR_ZERO = 1e-6
# torch.quantile rejects inputs beyond 2**24 elements, so percentiles come from
# a fixed-stride subsample above this size and are labelled as such. Counts,
# maxima, sums and the violation grid are always exact over every element.
SAMPLE_LIMIT = 1 << 22
CHUNK = 1 << 20


def script_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def key(atol, rtol):
    return f"atol={atol:g},rtol={rtol:g}"


def bucket_index(value, boundaries):
    """Pure-Python torch.bucketize(right=False): boundaries strictly below value.

    For an ascending grid this is also the number of atol rows the value
    violates, which is what makes the histogram identity below correct.
    """
    return sum(1 for bound in boundaries if bound < value)


def violations_from_histogram(histogram):
    """Suffix sums: a value in bucket i violates exactly rows 0..i-1."""
    counts, running = [0] * len(ATOL), 0
    for index in range(len(ATOL), 0, -1):
        running += histogram[index]
        counts[index - 1] = running
    return counts


def sweep_reference(errors, references):
    """Exact definition of the reported grid, in plain Python floats."""
    counts, minimum = {}, {}
    for rtol in RTOL:
        shifted = [abs(error) - rtol * abs(reference)
                   for error, reference in zip(errors, references)]
        minimum[f"rtol={rtol:g}"] = max([max(value, 0.0) for value in shifted], default=0.0)
        for atol in ATOL:
            counts[key(atol, rtol)] = sum(1 for value in shifted if value > atol)
    return counts, minimum


class Sweep:
    """Per-rtol histogram over the atol grid, accumulated chunk by chunk."""

    def __init__(self, torch):
        self.torch = torch
        self.bounds = torch.tensor(ATOL, dtype=torch.float64)
        self.histogram = torch.zeros((len(RTOL), len(ATOL) + 1), dtype=torch.int64)
        self.minimum = torch.zeros(len(RTOL), dtype=torch.float64)

    def update(self, error, reference):
        """error and reference are float64, already absolute values."""
        for index, rtol in enumerate(RTOL):
            shifted = error - rtol * reference if rtol else error
            self.minimum[index] = self.torch.maximum(self.minimum[index], shifted.max().clamp_min(0.0))
            buckets = self.torch.bucketize(shifted, self.bounds, right=False)
            self.histogram[index] += self.torch.bincount(buckets, minlength=len(ATOL) + 1)

    def result(self, total):
        counts, minimum = {}, {}
        for index, rtol in enumerate(RTOL):
            minimum[f"rtol={rtol:g}"] = self.minimum[index].item()
            for position, atol in enumerate(ATOL):
                violations = violations_from_histogram(self.histogram[index].tolist())[position]
                counts[key(atol, rtol)] = {"violations": violations,
                                           "fraction": violations / total if total else 0.0}
        return counts, minimum


def self_check(torch):
    """Assert the torch path reproduces the plain-Python definition exactly."""
    references = [0.0, 1.0, 0.5, 1e-7, 2.0, 0.0, 100.0, 3e-3, -4.0]
    nominal = [0.0, 1e-8, 5e-7, 2e-6, 4e-5, 8e-4, 2e-3, 1e-2, -1e-3]
    actual_values = [reference + error for reference, error in zip(references, nominal)]
    # Compare against the errors float64 actually realizes, not the nominal ones.
    realized = [value - reference for value, reference in zip(actual_values, references)]
    expected_counts, expected_minimum = sweep_reference(realized, references)

    for value in (0.0, 1e-8, 1e-7, 3e-3, 1.0):
        tensor = torch.tensor([value], dtype=torch.float64)
        torch_index = int(torch.bucketize(tensor, torch.tensor(ATOL, dtype=torch.float64), right=False)[0])
        if torch_index != bucket_index(value, ATOL):
            raise AssertionError(f"torch.bucketize disagrees with the grid definition at {value!r}")

    record = tensor_metrics(
        torch,
        torch.tensor(actual_values, dtype=torch.float64),
        torch.tensor(references, dtype=torch.float64),
        label="self_check")
    for name, expected in expected_counts.items():
        if record["violations"][name]["violations"] != expected:
            raise AssertionError(f"violation count mismatch at {name}")
    for name, expected in expected_minimum.items():
        if record["minimum_atol"][name] != expected:
            raise AssertionError(f"minimum atol mismatch at {name}")
    if record["max_abs"] != max(abs(value) for value in realized):
        raise AssertionError("max_abs mismatch in self check")
    if record["exact_equal_fraction"] != 1 / len(references) or record["dtype_match"] is not True:
        raise AssertionError("exact equality or dtype reporting is wrong in self check")

    # Identical tensors must take the bitwise fast path and report a zero grid.
    same = torch.tensor([1.5, -2.5, 0.0], dtype=torch.float32)
    identical = tensor_metrics(torch, same.clone(), same.clone(), label="self_check.identical")
    if not identical["bitwise_identical"] or not identical["violations_all_zero"]:
        raise AssertionError("identical tensors did not take the bitwise-identical path")
    if identical["exact_equal_fraction"] != 1.0:
        raise AssertionError("identical tensors did not report full exact equality")

    # Oracle comparisons are BF16 against FP32; that must be measurable, not a hash.
    reference = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    mixed = tensor_metrics(torch, reference.to(torch.bfloat16), reference,
                           label="self_check.mixed", allow_mixed_dtype=True)
    if mixed["dtype_match"] or mixed["bitwise_comparable"] or mixed["bitwise_identical"] is not None:
        raise AssertionError("mixed-dtype comparison made a bitwise claim")
    if mixed["exact_equality_basis"] != "float64 cast" or mixed["max_abs"] != 0.0:
        raise AssertionError("mixed-dtype comparison of exactly representable values is wrong")

    # A dtype change between baseline and candidate must still fail loudly.
    try:
        tensor_metrics(torch, reference.to(torch.bfloat16), reference, label="self_check.strict")
    except AssertionError:
        pass
    else:
        raise AssertionError("strict comparison accepted a dtype mismatch")

    # The direct float32 acceptance verdict, on values straddling the pair.
    inside = torch.tensor([1.0, 1.0], dtype=torch.float32)
    verdict = acceptance_direct(torch, inside + torch.tensor([0.03, -0.03], dtype=torch.float32), inside)
    if verdict["violations"] or not verdict["passes"]:
        raise AssertionError("acceptance_direct rejected an error inside atol + rtol * abs(expected)")
    outside = acceptance_direct(torch, inside + torch.tensor([0.05, 0.0], dtype=torch.float32), inside)
    if outside["violations"] != 1:
        raise AssertionError("acceptance_direct did not count an error outside the pair")
    # Zero reference: only atol protects, so rtol cannot absorb anything.
    zero = torch.zeros(2, dtype=torch.float32)
    tight = acceptance_direct(torch, torch.tensor([0.004, 0.0], dtype=torch.float32), zero)
    if tight["violations"] != 1:
        raise AssertionError("acceptance_direct absorbed a near-zero-reference error it should not")

    return {"checked_elements": len(references), "grid_cells": len(expected_counts),
            "bucketize_semantics": "verified against plain-Python definition",
            "checked": ["grid against sweep_reference", "bitwise-identical fast path",
                        "mixed-dtype oracle path", "strict dtype mismatch rejection"]}


def normalize_device_uuid(value):
    """torch reports the bare UUID; nvidia-smi and CUDA_VISIBLE_DEVICES prefix it.

    torch.cuda.get_device_properties().uuid is `0e049deb-...`, while the runner's
    GPU lock, CUDA_VISIBLE_DEVICES and nvidia-smi all use `the configured GPU UUID`.
    Compare the identity, not the spelling. The original spellings are kept in the
    report so the device claim stays auditable.
    """
    return str(value).strip().lower().removeprefix("gpu-")


def same_device_uuid(actual, expected):
    return normalize_device_uuid(actual) == normalize_device_uuid(expected)


def byte_hash(torch, tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


# The grid subtracts rtol * abs(reference) in float64 and compares against atol,
# which is the right space for a tolerance envelope but is not bit-for-bit the
# predicate the original acceptance gate evaluates.
GRID_ARITHMETIC = ("float64; error and rtol * abs(reference) are subtracted before comparison to atol, "
                   "which can differ from the float32 atol + rtol * abs(reference) form by about 1 ULP "
                   "exactly at a boundary. The acceptance verdict is computed separately and directly.")


def acceptance_direct(torch, actual, expected, rows=None):
    """Evaluate the unchanged acceptance pair directly, never read off the grid.

    bench/runtime.check_tensor casts both sides to float32 and lets
    torch.testing.assert_close evaluate abs(actual - expected) <= atol +
    rtol * abs(expected) in float32. This reproduces that float32 predicate, so
    the acceptance verdict cannot be an artifact of the grid's float64 form.
    Aggregates are accumulated in float64 for stability, which is labelled.
    """
    atol, rtol = ACCEPTANCE["atol"], ACCEPTANCE["rtol"]
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if rows is not None:
        actual, expected = actual[rows], expected[rows]
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    total = flat_actual.numel()
    if total == 0:
        return {"violations": 0, "passes": True, "count": 0}
    violations, maximum, sum_square, reference_square = 0, 0.0, 0.0, 0.0
    for begin in range(0, total, CHUNK):
        left = flat_actual[begin:begin + CHUNK].to(torch.float32)
        right = flat_expected[begin:begin + CHUNK].to(torch.float32)
        error = (left - right).abs()
        violations += int((error > atol + rtol * right.abs()).sum().item())
        maximum = max(maximum, error.max().item())
        wide = error.to(torch.float64)
        sum_square += wide.square().sum().item()
        reference_square += right.to(torch.float64).square().sum().item()
    rms = (sum_square / total) ** 0.5
    reference_rms = (reference_square / total) ** 0.5
    relative = rms / max(reference_rms, 1e-8)
    return {"atol": atol, "rtol": rtol, "count": total, "violations": violations,
            "fraction": violations / total, "max_abs": maximum, "relative_rms": relative,
            # Same two-part verdict as bench/runtime.check_tensor.
            "passes": violations == 0 and not (relative > TOLERANCE_RELATIVE_RMS and rms > 1e-6),
            "predicate": "float32 atol + rtol * abs(expected), as torch.testing.assert_close evaluates it",
            "aggregate_arithmetic": "float64 accumulation"}


def tensor_metrics(torch, actual, expected, *, label, rows=None, scope="all_elements",
                   allow_mixed_dtype=False):
    """Per-field error statistics with the full predeclared tolerance grid.

    `expected` is the reference on the right-hand side of
    abs(actual - expected) <= atol + rtol * abs(expected).

    The independent CPU oracle returns FP32 output and intermediates while the
    public call returns activation-dtype tensors, so oracle comparisons pass
    `allow_mixed_dtype=True`: both storage dtypes are recorded, the difference is
    taken in float64, and no bitwise claim is made. A/B comparisons leave it
    False, so a baseline/candidate dtype change still fails loudly.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    same_dtype = actual.dtype == expected.dtype
    if not same_dtype and not allow_mixed_dtype:
        raise AssertionError(f"{label}: dtype {actual.dtype} != {expected.dtype}")
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if rows is not None:
        actual, expected = actual[rows], expected[rows]
    record = {"label": label, "scope": scope,
              "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
              "dtype_match": same_dtype, "shape": list(actual.shape), "count": actual.numel(),
              "nonfinite_actual": int((~torch.isfinite(actual)).sum().item()),
              "nonfinite_expected": int((~torch.isfinite(expected)).sum().item())}
    if record["count"] == 0:
        record.update(empty=True, bitwise_identical=True, violations_all_zero=True)
        return record
    if same_dtype:
        # A byte hash is the only honest bitwise claim. Value equality is
        # reported separately and never described as bitwise.
        record["actual_sha256"] = byte_hash(torch, actual)
        record["expected_sha256"] = byte_hash(torch, expected)
        record["bitwise_comparable"] = True
        record["bitwise_identical"] = record["actual_sha256"] == record["expected_sha256"]
    else:
        record["bitwise_comparable"] = False
        record["bitwise_identical"] = None
        record["dtype_note"] = ("different storage dtypes; difference taken in float64 and no bitwise "
                                "comparison is possible")
    if record["nonfinite_actual"] or record["nonfinite_expected"]:
        # Undefined ordering would make the grid meaningless; fail instead.
        record.update(status="failed", error="nonfinite values present")
        return record
    if same_dtype and record["bitwise_identical"]:
        # Zero error satisfies every row, including atol=0 and rtol=0.
        record.update(max_abs=0.0, mean_abs=0.0, rmse=0.0, relative_rms=0.0,
                      exact_equal_fraction=1.0, exact_equality_basis="native dtype",
                      reference_rms=expected.to(torch.float64).square().mean().sqrt().item(),
                      violations_all_zero=True, acceptance_violations=0,
                      acceptance_row=key(ACCEPTANCE["atol"], ACCEPTANCE["rtol"]),
                      minimum_atol={f"rtol={rtol:g}": 0.0 for rtol in RTOL},
                      percentiles={f"p{value:g}": 0.0 for value in PERCENTILES},
                      percentile_basis="exact (bitwise identical)",
                      # Zero error passes the float32 predicate identically; no
                      # arithmetic can make an identical pair violate it.
                      original_acceptance_direct={"atol": ACCEPTANCE["atol"], "rtol": ACCEPTANCE["rtol"],
                                                  "count": record["count"], "violations": 0, "fraction": 0.0,
                                                  "max_abs": 0.0, "relative_rms": 0.0, "passes": True,
                                                  "predicate": "trivially satisfied; tensors are byte identical"})
        return record

    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    total = flat_actual.numel()
    maximum = sum_abs = sum_square = reference_square = near_zero_max = 0.0
    near_zero_count = equal_count = 0
    sweep = Sweep(torch)
    for begin in range(0, total, CHUNK):
        left = flat_actual[begin:begin + CHUNK].to(torch.float64)
        right = flat_expected[begin:begin + CHUNK].to(torch.float64)
        error, reference = (left - right).abs(), right.abs()
        # float64 widening is lossless for every dtype here, so this equality is
        # native for matched dtypes and a post-cast value equality for mixed ones.
        equal_count += int((left == right).sum().item())
        maximum = max(maximum, error.max().item())
        sum_abs += error.sum().item()
        sum_square += error.square().sum().item()
        reference_square += reference.square().sum().item()
        near = reference < NEAR_ZERO
        count = int(near.sum().item())
        if count:
            near_zero_count += count
            near_zero_max = max(near_zero_max, error[near].max().item())
        sweep.update(error, reference)
    record["exact_equal_fraction"] = equal_count / total
    record["exact_equality_basis"] = "native dtype" if same_dtype else "float64 cast"
    rmse = (sum_square / total) ** 0.5
    reference_rms = (reference_square / total) ** 0.5
    counts, minimum = sweep.result(total)
    stride = max(1, -(-total // SAMPLE_LIMIT))
    sample = (flat_actual[::stride].to(torch.float64) - flat_expected[::stride].to(torch.float64)).abs()
    # One sort for every percentile rather than one per percentile.
    quantiles = torch.quantile(sample, torch.tensor(PERCENTILES, dtype=torch.float64)).tolist()
    record.update(
        max_abs=maximum, mean_abs=sum_abs / total, rmse=rmse, reference_rms=reference_rms,
        # Same guarded ratio as bench/runtime.check_tensor.
        relative_rms=rmse / max(reference_rms, 1e-8),
        near_zero={"threshold": NEAR_ZERO, "count": near_zero_count,
                   "fraction": near_zero_count / total, "max_abs": near_zero_max},
        violations=counts, minimum_atol=minimum, grid_arithmetic=GRID_ARITHMETIC,
        percentiles=dict(zip((f"p{value:g}" for value in PERCENTILES), quantiles)),
        percentile_basis=("exact" if stride == 1 else f"fixed stride {stride} sample of {sample.numel()} elements"),
        acceptance_row=key(ACCEPTANCE["atol"], ACCEPTANCE["rtol"]),
        # Read off the grid, so it inherits the grid's float64 form.
        acceptance_violations=counts[key(ACCEPTANCE["atol"], ACCEPTANCE["rtol"])]["violations"],
        # Computed directly in float32, independent of the grid. This is the
        # verdict about the unchanged original acceptance gate.
        original_acceptance_direct=acceptance_direct(torch, actual, expected))
    return record


def exactness(torch, actual, expected, label):
    """Slots that must not have been written at all."""
    equal = actual.shape == expected.shape and torch.equal(actual, expected)
    return {"label": label, "exactly_unchanged": bool(equal),
            "actual_sha256": byte_hash(torch, actual), "expected_sha256": byte_hash(torch, expected)}
