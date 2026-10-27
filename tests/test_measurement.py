"""Small contract checks; no model execution, download or GPU initialization."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from long_context_diagnostics.measurement import (
    build_histories,
    evaluate_histories,
    load_pair,
    save_pair,
    validate_arms,
)


class MeasurementTests(unittest.TestCase):
    def test_fixed_block_and_foreign_prefix(self):
        plan = build_histories(
            np.arange(30),
            np.arange(100, 130),
            end=24,
            recent_tokens=8,
            target_tokens=4,
            prefix_id=99,
        )
        np.testing.assert_array_equal(plan.inputs["full"], np.r_[99, np.arange(24)])
        np.testing.assert_array_equal(plan.inputs["short"], np.r_[99, np.arange(12, 24)])
        np.testing.assert_array_equal(
            plan.inputs["foreign"], np.r_[99, np.arange(100, 112), np.arange(12, 24)]
        )
        for name, seq in plan.inputs.items():
            np.testing.assert_array_equal(seq[plan.target_starts[name] :], [20, 21, 22, 23])
        self.assertEqual(len(plan.inputs["short"]), 1 + 8 + 4)
        self.assertEqual(plan.retained_start, 12)
        # Documentary prefix for target j grows as C+j, rather than rolling C.
        self.assertEqual(plan.target_starts["short"] - 1 + 3, 8 + 3)

    def test_recent_prefix_excludes_targets(self):
        ids = np.arange(180000)
        plan = build_histories(
            ids, ids + 200000, end=180000, recent_tokens=131072, target_tokens=16384
        )
        self.assertEqual(len(plan.inputs["short"]), 147456)
        self.assertEqual(plan.target_starts["short"], 131072)

    def test_save_load_and_misaligned_targets(self):
        plan = build_histories(
            np.arange(30), np.arange(40, 70), end=24, recent_tokens=8, target_tokens=4
        )
        calls = []

        def score(seq, start):
            calls.append((seq.copy(), start))
            return np.array([1.0, 1.1, 1.2, 1.3])

        arms = evaluate_histories(plan, score)
        self.assertEqual(len(calls), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.npz"
            save_pair(
                path,
                document_id="doc",
                arms=arms,
                repeat_bins={"4": np.array(["never"] * 4)},
                metadata={"kind": "fixture"},
            )
            restored = load_pair(path)
            np.testing.assert_array_equal(restored["target_ids"], [20, 21, 22, 23])
            self.assertEqual(restored["full"].dtype, np.float32)
            self.assertEqual(restored["metadata"]["kind"], "fixture")
            arms["short"]["target_ids"] = arms["short"]["target_ids"][::-1]
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_arms(arms)

    def test_positions_catch_misalignment(self):
        arms = {
            name: {
                "target_ids": np.array([1, 1]),
                "target_positions": np.array([10, 11]),
                "losses": np.array([1.0, 1.0]),
            }
            for name in ("full", "short", "foreign")
        }
        arms["foreign"]["target_positions"] = np.array([11, 12])
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_arms(arms)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            build_histories(np.arange(30), [9], end=24, recent_tokens=8, target_tokens=4)
        with self.assertRaisesRegex(ValueError, "preceding"):
            build_histories([1, 2], [3, 4], end=2, recent_tokens=0, target_tokens=2)


if __name__ == "__main__":
    unittest.main()
