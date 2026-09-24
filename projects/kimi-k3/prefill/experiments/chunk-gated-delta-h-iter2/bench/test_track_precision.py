"""CPU-only checks for the FP32 track regression; no torch dependency."""

import math
import struct
import types
import unittest
from unittest.mock import MagicMock

import runtime


class ScalarTensor:
    """One FP32 value, with real round-to-nearest-even BF16 conversion."""

    def __init__(self, value, dtype="float32"):
        bits = struct.unpack("I", struct.pack("f", value))[0]
        if dtype == "bfloat16":
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        self.value = struct.unpack("f", struct.pack("I", bits))[0]
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, dtype):
        return ScalarTensor(self.value, dtype)

    def float(self):
        return self.to("float32")


def scalar_assert_close(actual, expected, atol, rtol, msg):
    if abs(actual.value - expected.value) > atol + rtol * abs(expected.value):
        raise AssertionError(msg)


SCALAR_TORCH = types.SimpleNamespace(
    float32="float32", bfloat16="bfloat16",
    equal=lambda left, right: left.value == right.value,
    isfinite=lambda value: types.SimpleNamespace(all=lambda: math.isfinite(value.value)),
    testing=types.SimpleNamespace(assert_close=scalar_assert_close),
)


class TrackPrecisionTests(unittest.TestCase):
    def test_bf16_roundtrip_is_rejected_even_inside_strict_numeric_budget(self):
        reference = ScalarTensor(1.000001)
        rounded = reference.to("bfloat16").float()
        # This is the P1 regression: both the former 3% check and even the new
        # 1e-5 check alone accept the precision loss. The invariant must reject.
        scalar_assert_close(rounded, reference, atol=runtime.TOLERANCE["atol"],
                            rtol=runtime.TOLERANCE["rtol"], msg="fixture must pass old budget")
        scalar_assert_close(rounded, reference, **runtime.TRACK_TOLERANCE, msg="fixture must be numerically close")
        with self.assertRaisesRegex(AssertionError, "precision beyond BF16"):
            runtime.assert_track_precision(SCALAR_TORCH, rounded, reference,
                                           reference.to("bfloat16"), evolved=True)

    def test_full_precision_snapshot_matches_truncated_state_and_rounded_h(self):
        actual = ScalarTensor(1.001)
        result = runtime.assert_track_precision(SCALAR_TORCH, actual, ScalarTensor(1.001),
                                                actual.to("bfloat16"), evolved=True)
        self.assertTrue(result["retains_beyond_bfloat16"])
        self.assertTrue(result["intermediate_rounding_exact"])

    def test_broadly_close_but_wrong_truncated_state_fails_strict_check(self):
        actual, reference = ScalarTensor(1.001), ScalarTensor(1.002)
        scalar_assert_close(actual, reference, atol=runtime.TOLERANCE["atol"],
                            rtol=runtime.TOLERANCE["rtol"], msg="fixture must pass old budget")
        with self.assertRaisesRegex(AssertionError, "independently truncated"):
            runtime.assert_track_precision(SCALAR_TORCH, actual, reference, None, evolved=True)

    def test_wrong_h_row_or_non_fp32_scratch_is_rejected(self):
        actual = ScalarTensor(1.001)
        with self.assertRaisesRegex(AssertionError, "exactly equal"):
            runtime.assert_track_precision(SCALAR_TORCH, actual, actual,
                                           ScalarTensor(1.015625, "bfloat16"), evolved=True)
        with self.assertRaisesRegex(AssertionError, "must be FP32"):
            runtime.assert_track_precision(SCALAR_TORCH, actual.to("bfloat16"), actual,
                                           None, evolved=True)

    def test_chunk_zero_can_equal_original_bf16_state(self):
        actual = ScalarTensor(1.0)
        result = runtime.assert_track_precision(SCALAR_TORCH, actual, actual,
                                                actual.to("bfloat16"), evolved=False)
        self.assertFalse(result["retains_beyond_bfloat16"])

    def test_ragged_offsets_use_per_sequence_chunk_counts(self):
        self.assertEqual(runtime.track_boundaries([1, 129, 64, 257], [-1, 1, -1, 4]),
                         [(1, 1, 64, 2), (3, 194, 256, 9)])
        for lengths, indices in (([64], [1]), ([129], [-2]), ([64, 64], [0])):
            with self.assertRaises(AssertionError):
                runtime.track_boundaries(lengths, indices)

    def test_truncated_invocation_uses_original_slot_and_fresh_prefix(self):
        torch = types.SimpleNamespace(int32="int32", int64="int64", tensor=MagicMock(),
                                      cuda=types.SimpleNamespace(synchronize=MagicMock()))
        for slot in (3, -1):
            inputs = types.SimpleNamespace(
                torch=torch, indices_cpu=[slot], slot_step=2, bound=-5.0,
                state=types.SimpleNamespace(device="cpu"),
                **{name: MagicMock(name=name) for name in
                   ("original_storage", "q", "k_tensor", "original_v", "g", "beta", "a_log", "dt_bias")},
            )
            kernel = MagicMock()
            result = runtime.truncated_fp32_state(inputs, kernel, sequence=0, begin=17, prefix=128)
            inputs.original_storage.__getitem__.assert_called_once_with(max(slot, 0) * 2)
            original = inputs.original_storage.__getitem__.return_value
            initial = original.float.return_value.clone.return_value
            self.assertEqual(initial.zero_.call_count, 1 if slot < 0 else 0)
            self.assertIs(result, initial.unsqueeze.return_value.__getitem__.return_value)
            kwargs = kernel.call_args.kwargs
            self.assertIs(kwargs["initial_state"], initial.unsqueeze.return_value)
            for name in ("q", "k_tensor", "original_v", "g", "beta"):
                value = getattr(inputs, name)
                value.__getitem__.assert_called_once_with((slice(None), slice(17, 145)))
                value.__getitem__.return_value.clone.assert_called_once_with()
            self.assertFalse(kwargs["beta_is_raw"])
            self.assertTrue(kwargs["use_qk_l2norm_in_kernel"])
            self.assertEqual(kwargs["lower_bound"], -5.0)
            self.assertFalse(kwargs["output_intermediate_states"])
            self.assertIsNone(kwargs["track_state"])
            self.assertIsNone(kwargs["track_chunk_idx"])
            self.assertEqual(torch.tensor.call_args_list[-2].args, ([0],))
            self.assertEqual(torch.tensor.call_args_list[-1].args, ([0, 128],))

    def test_no_evolved_boundary_fails_instead_of_silently_skipping_precision(self):
        inputs = types.SimpleNamespace(case={"seq_lens": [63, 64]}, track_indices_cpu=[0, -1])
        with self.assertRaisesRegex(AssertionError, "evolved tracked boundary"):
            runtime.validate_track_snapshots(inputs, MagicMock(), None)


if __name__ == "__main__":
    unittest.main()
