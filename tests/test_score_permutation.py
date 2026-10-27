"""CPU contracts for old-prefix permutation and retained repetition labels."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from long_context_diagnostics.score_permutation import build_permutation_pair, permute_old_prefix
from long_context_diagnostics.measurement import CONDITIONS, load_pair, save_pair
from long_context_diagnostics.readout import classify_repeats


class PermutationTests(unittest.TestCase):
    def test_default_permutation_invariants(self):
        ids = np.arange(32768, dtype=np.int64)
        donor = ids + 100000
        before_ids, before_donor = ids.copy(), donor.copy()
        plans, bins, info = build_permutation_pair(ids, donor, prefix_id=900000)
        original, changed = plans["original"], plans["permuted"]
        self.assertEqual(info["blocks"], 60)
        self.assertEqual(info["shuffled_blocks"], 59)
        self.assertEqual(info["anchored_tail_tokens"], 512)
        self.assertEqual(info["order"][-1], 59)
        self.assertEqual(sorted(info["order"]), list(range(60)))
        self.assertNotEqual(info["order"][:59], list(range(59)))
        for condition, source in (("full", ids), ("foreign", donor)):
            np.testing.assert_array_equal(
                changed.inputs[condition][1:30721].reshape(60, 512),
                source[:30720].reshape(60, 512)[info["order"]],
            )
            np.testing.assert_array_equal(
                changed.inputs[condition][1 + 30208 :], original.inputs[condition][1 + 30208 :]
            )
            self.assertEqual(changed.inputs[condition][0], 900000)
            self.assertTrue(all(info["verification"][condition].values()))
        np.testing.assert_array_equal(changed.inputs["short"], original.inputs["short"])
        for plan in plans.values():
            for condition in CONDITIONS:
                np.testing.assert_array_equal(
                    plan.inputs[condition][plan.target_starts[condition] :], ids[31744:32768]
                )
            np.testing.assert_array_equal(plan.target_positions, np.arange(31744, 32768))
        np.testing.assert_array_equal(ids, before_ids)
        np.testing.assert_array_equal(donor, before_donor)
        again, again_info = permute_old_prefix(original)
        self.assertEqual(again_info["order"], info["order"])
        np.testing.assert_array_equal(again.inputs["full"], changed.inputs["full"])
        for n in (4, 8):
            np.testing.assert_array_equal(
                bins[str(n)], classify_repeats(ids, original.target_positions, 30720, n=n)
            )

    def test_repeat_survival_report(self):
        ids = np.arange(100, 128, dtype=np.int64)
        ids[:4] = ids[20:24] = [1, 2, 3, 4]
        ids[6:10] = ids[24:28] = [5, 6, 7, 8]
        plans, bins, info = build_permutation_pair(
            ids, ids + 1000, end=28, recent=4, targets=8, block_size=4
        )
        self.assertEqual(info["order"], [0, 2, 1, 3])
        self.assertTrue(bins["4"][3].startswith("far:"))
        self.assertTrue(bins["4"][7].startswith("far:"))
        survival = info["far_repeat_survival"]["4"]
        self.assertEqual(survival["original_far_targets"], 2)
        self.assertEqual(survival["original_far_still_matching_permuted_far"], 1)
        self.assertEqual(survival["original_far_lost_match_positions"], [27])
        self.assertEqual(survival["prefix_matches"]["full"]["lost_target_positions"], [27])
        with tempfile.TemporaryDirectory() as directory:
            for variant, plan in plans.items():
                path = Path(directory) / f"{variant}.npz"
                save_pair(
                    path,
                    document_id="fixture",
                    arms={
                        key: {
                            "target_ids": plan.inputs[key][plan.target_starts[key] :],
                            "target_positions": plan.target_positions,
                            "losses": np.ones(8),
                        }
                        for key in CONDITIONS
                    },
                    repeat_bins=bins,
                    metadata={"intervention": dict(info, variant=variant)},
                )
                loaded = load_pair(path)
                np.testing.assert_array_equal(loaded["repeat_bins"]["4"], bins["4"])
                np.testing.assert_array_equal(loaded["target_ids"], ids[20:28])
                self.assertEqual(loaded["metadata"]["intervention"]["variant"], variant)

    def test_rejects_invalid_blocks(self):
        ids = np.arange(30)
        with self.assertRaisesRegex(ValueError, "whole blocks"):
            build_permutation_pair(ids, ids + 100, end=29, recent=4, targets=8, block_size=4)
        with self.assertRaisesRegex(ValueError, "two shuffled"):
            build_permutation_pair(
                ids, ids + 100, end=28, recent=4, targets=8, block_size=4, keep_tail_blocks=3
            )


if __name__ == "__main__":
    unittest.main()
