"""Standard-library-only checks of the tolerance-grid arithmetic.

Runs without numpy or torch. The torch fast path is not exercised here; it is
checked against `sweep_reference` on the GPU node by `metrics.self_check`,
which also asserts torch.bucketize matches `bucket_index` below.
"""

import unittest
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import metrics

# _worker imports only standard-library modules plus metrics at import time, so
# the tracked-row derivation is testable here with a stub.
_worker_spec = importlib.util.spec_from_file_location("precision_worker", Path(__file__).with_name("_worker.py"))
_worker = importlib.util.module_from_spec(_worker_spec)
_worker_spec.loader.exec_module(_worker)


class StubInputs:
    def __init__(self, track, indices):
        self.track, self.track_indices_cpu = track, indices


class TestTrackRows(unittest.TestCase):
    """The tracked-row masking that must match runtime.Inputs exactly."""

    def rows(self, lengths, batch, track_state):
        return _worker.track_rows_for({"track_state": track_state, "seq_lens": lengths,
                                       "batch_size": batch})

    def test_no_tracked_rows_when_tracking_is_disabled(self):
        """runtime.Inputs builds indices unconditionally but leaves track = None."""
        # The harness still computes [0, -1] here; with no scratch tensor there
        # are no tracked rows. The second launch failed by comparing these.
        self.assertEqual(self.rows([129, 257], 2, False), [])
        self.assertEqual(_worker.harness_track_rows(StubInputs(None, [0, -1])), [])

    def test_tracked_rows_exclude_the_sentinel_row(self):
        # Batched cases force the last index to -1, which stays at 17.0.
        self.assertEqual(self.rows([129, 257], 2, True), [0])
        self.assertEqual(_worker.harness_track_rows(StubInputs(object(), [0, -1])), [0])

    def test_derivation_matches_the_harness_for_every_frozen_shape(self):
        for lengths, batch in (([129, 257], 2), ([2048], 1), ([1, 63, 64, 65, 127, 128, 129], 7)):
            indices = [(length - 1) // 64 for length in lengths]
            if batch > 1:
                indices[-1] = -1
            expected = [index for index, chunk in enumerate(indices) if chunk >= 0]
            self.assertEqual(self.rows(lengths, batch, True), expected)
            self.assertEqual(_worker.harness_track_rows(StubInputs(object(), indices)), expected)

    def test_inactive_rows_are_the_complement(self):
        self.assertEqual(_worker.inactive_rows(4, [0]), [1, 2, 3])
        self.assertEqual(_worker.inactive_rows(4, [0, 2]), [1, 3])


class TestPreparation(unittest.TestCase):
    def test_candidate_without_fused_preparation_uses_original_path(self):
        class Tensor:
            def contiguous(self): return self
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return self
        inputs = SimpleNamespace(q=Tensor(), k_tensor=Tensor(), v=Tensor())
        module = SimpleNamespace(l2norm_fwd=lambda value: value)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, {"sglang.kernels.ops.attention.fla.l2norm": module}):
            path, values = _worker.prepare(torch, inputs, "candidate")
        self.assertIn("l2norm_fwd", path)
        self.assertEqual(values, {"prep_q": inputs.q, "prep_k": inputs.k_tensor, "prep_v": inputs.v})


class TestGrid(unittest.TestCase):
    def test_device_uuid_comparison_ignores_prefix_and_case(self):
        """The first launch failed here: torch reports the bare UUID."""
        bare = "00000000-0000-0000-0000-000000000000"
        prefixed = "GPU-00000000-0000-0000-0000-000000000000"
        self.assertTrue(metrics.same_device_uuid(bare, prefixed))
        self.assertTrue(metrics.same_device_uuid(prefixed, bare))
        self.assertTrue(metrics.same_device_uuid(prefixed.upper(), bare))
        self.assertTrue(metrics.same_device_uuid(f" {prefixed} ", bare))
        # A genuinely different device must still be rejected.
        self.assertFalse(metrics.same_device_uuid("GPU-11111111-1111-1111-1111-111111111111", bare))

    def test_grid_is_the_predeclared_one(self):
        self.assertEqual(metrics.ATOL, [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3])
        self.assertEqual(metrics.RTOL, [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2])
        self.assertEqual(metrics.ATOL, sorted(metrics.ATOL))
        # The unchanged acceptance pair must be a row of the grid, and the loosest one.
        self.assertEqual((metrics.ACCEPTANCE["atol"], metrics.ACCEPTANCE["rtol"]),
                         (metrics.ATOL[-1], metrics.RTOL[-1]))

    def test_bucket_index_counts_boundaries_strictly_below(self):
        self.assertEqual(metrics.bucket_index(0.0, metrics.ATOL), 0)
        self.assertEqual(metrics.bucket_index(1e-8, metrics.ATOL), 1)
        # A value equal to a boundary does not violate that boundary.
        self.assertEqual(metrics.bucket_index(1e-7, metrics.ATOL), 1)
        self.assertEqual(metrics.bucket_index(2e-7, metrics.ATOL), 2)
        self.assertEqual(metrics.bucket_index(1.0, metrics.ATOL), len(metrics.ATOL))

    def test_histogram_suffix_sums_equal_brute_force_counts(self):
        """The identity the chunked torch histogram relies on."""
        shifted = [0.0, 1e-9, 1e-7, 4e-7, 1e-5, 2e-4, 6e-4, 2e-3, 5e-3, 1.0]
        histogram = [0] * (len(metrics.ATOL) + 1)
        for value in shifted:
            histogram[metrics.bucket_index(value, metrics.ATOL)] += 1
        counts = metrics.violations_from_histogram(histogram)
        for position, atol in enumerate(metrics.ATOL):
            self.assertEqual(counts[position], sum(1 for value in shifted if value > atol),
                             f"atol={atol:g}")

    def test_sweep_reference_hand_computed(self):
        counts, minimum = metrics.sweep_reference([0.0, 1e-8], [1.0, 1.0])
        # rtol=0: only the 1e-8 error exceeds atol=0, and nothing exceeds atol=1e-7.
        self.assertEqual(counts[metrics.key(0.0, 0.0)], 1)
        self.assertEqual(counts[metrics.key(1e-7, 0.0)], 0)
        self.assertEqual(minimum["rtol=0"], 1e-8)
        # rtol=1e-5 against a reference of 1.0 absorbs both errors entirely.
        self.assertEqual(counts[metrics.key(0.0, 1e-5)], 0)
        self.assertEqual(minimum["rtol=1e-05"], 0.0)

    def test_minimum_atol_is_the_tight_envelope(self):
        """minimum_atol must be the smallest atol that passes, for each rtol."""
        errors, references = [3e-4, 1e-5, 2e-3], [1.0, 0.0, 0.5]
        _, minimum = metrics.sweep_reference(errors, references)
        for rtol in metrics.RTOL:
            needed = minimum[f"rtol={rtol:g}"]
            self.assertTrue(all(abs(error) <= needed + rtol * abs(reference)
                                for error, reference in zip(errors, references)),
                            f"rtol={rtol:g}: envelope {needed!r} does not pass")
            if needed > 0:
                tighter = needed * (1 - 1e-12)
                self.assertFalse(all(abs(error) <= tighter + rtol * abs(reference)
                                     for error, reference in zip(errors, references)),
                                 f"rtol={rtol:g}: envelope {needed!r} is not tight")

    def test_rtol_relaxation_is_monotone(self):
        errors = [1e-4, 5e-4, 1e-3]
        references = [1.0, 0.2, 0.05]
        counts, minimum = metrics.sweep_reference(errors, references)
        for rtol in metrics.RTOL:
            row = [counts[metrics.key(atol, rtol)] for atol in metrics.ATOL]
            self.assertEqual(row, sorted(row, reverse=True), "violations must fall as atol rises")
        envelopes = [minimum[f"rtol={rtol:g}"] for rtol in metrics.RTOL]
        self.assertEqual(envelopes, sorted(envelopes, reverse=True),
                         "a larger rtol cannot need a larger atol")


if __name__ == "__main__":
    unittest.main()
