"""CPU-only original-target alignment and visible-context repetition checks."""

import unittest

import numpy as np

from long_context_diagnostics.score_local_control import build_local_control
from long_context_diagnostics.panel_readout import panel_identity


class LocalControlTests(unittest.TestCase):
    def test_default_targets_and_lengths(self):
        ids = np.arange(32768, dtype=np.int64)
        donor = ids + 100000
        plan, positions, bins, history = build_local_control(ids, donor)
        np.testing.assert_array_equal(positions, np.arange(31744, 32768))
        for name in ("full", "short", "foreign"):
            np.testing.assert_array_equal(
                plan.inputs[name][plan.target_starts[name] :], ids[31744:32768]
            )
        self.assertEqual(history["input_lengths"], {"full": 2048, "short": 1537, "foreign": 2048})
        self.assertEqual(history["full_history_start"], 30721)
        self.assertEqual(history["retained_start"], 31232)
        self.assertEqual(history["native_end_exclusive"], 2047)
        self.assertEqual(history["end_exclusive"], 32768)
        np.testing.assert_array_equal(plan.inputs["full"][1:], ids[30721:32768])
        np.testing.assert_array_equal(plan.inputs["foreign"][1:512], donor[:511])

    def test_hidden_occurrence_not_far_repeat(self):
        ids = np.arange(100, 140, dtype=np.int64)
        ids[:4] = ids[-4:] = [1, 2, 3, 4]
        _, positions, bins, _ = build_local_control(
            ids, ids + 1000, end=40, recent=3, targets=4, far=5
        )
        np.testing.assert_array_equal(positions, [36, 37, 38, 39])
        self.assertEqual(bins["4"][-1], "never")
        ids[28:32] = [1, 2, 3, 4]
        _, _, bins, _ = build_local_control(ids, ids + 1000, end=40, recent=3, targets=4, far=5)
        self.assertTrue(bins["4"][-1].startswith("far:"))

    def test_local_panel_identity(self):
        ids = np.arange(32768)
        _, _, _, history = build_local_control(ids, ids + 100000)
        metadata = {"history": history, "model": {}, "selected_execution": "full"}
        local = panel_identity({"metadata": metadata})
        metadata["history"] = {
            key: value
            for key, value in history.items()
            if key not in ("full_history_start", "repeat_label_scope")
        }
        full = panel_identity({"metadata": metadata})
        self.assertNotEqual(local, full)
        self.assertEqual(full["full_history_start"], 0)


if __name__ == "__main__":
    unittest.main()
