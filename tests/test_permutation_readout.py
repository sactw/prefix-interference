"""Focused CPU checks for paired shuffle effects and fixed original strata."""

import copy
import json
import unittest

import numpy as np

from long_context_diagnostics.permutation_readout import summarize_permutation


def pair(doc, full=2.0, short=3.0, foreign=4.0, full_change=-0.5, foreign_change=0.25):
    info = {
        "name": "old_prefix_block_permutation",
        "seed": 20261024,
        "rng": "NumPy PCG64",
        "block_tokens": 4,
        "far_tokens": 16,
        "blocks": 4,
        "shuffled_blocks": 3,
        "keep_tail_blocks": 1,
        "anchored_tail_tokens": 4,
        "order": [0, 2, 1, 3],
        "same_order_for_actual_and_foreign": True,
        "short_input_equal": True,
        "verification": {
            condition: {
                key: True
                for key in (
                    "token_multiset_equal",
                    "boundary_equal",
                    "recent_and_targets_equal",
                    "anchored_tail_equal",
                )
            }
            for condition in ("full", "foreign")
        },
    }
    row = {
        "document_id": doc,
        "target_ids": np.array([10, 11]),
        "target_positions": np.array([18, 19]),
        "repeat_bins": {"4": np.array(["far:(4,16]", "never"])},
        **{
            k: np.full(2, v, dtype=np.float64)
            for k, v in (("full", full), ("short", short), ("foreign", foreign))
        },
        "metadata": {
            "donor_id": "d1",
            "paired_attempt": "attempt-" + doc,
            "text_sha256": "text-" + doc,
            "donor_text_sha256": "donor-1",
            "selected_execution": "full",
            "model": {"model_id": "fixture", "revision": "v1"},
            "history": {
                "end_exclusive": 20,
                "recent_prefix_tokens": 2,
                "target_tokens": 2,
                "far_prefix_tokens": 16,
                "prefix_id": 0,
            },
            "intervention": {**info, "variant": "original"},
        },
    }
    altered = copy.deepcopy(row)
    altered["metadata"]["intervention"]["variant"] = "permuted"
    altered["full"] += full_change
    altered["foreign"] += foreign_change
    return row, altered


class PermutationReadoutTests(unittest.TestCase):
    def test_known_paired_changes(self):
        rows = [
            row
            for d, change in (("a", -0.5), ("b", -1.0), ("c", -2.0))
            for row in pair(d, full_change=change)
        ]
        report = summarize_permutation(rows, bootstrap_samples=83, seed=11, expected_documents=3)
        values = np.array([-0.5, -1.0, -2.0])
        draws = np.random.default_rng(11).integers(0, 3, size=(83, 3))
        np.testing.assert_allclose(
            report["delta_ce"]["full"]["ci95"],
            np.percentile(values[draws].mean(axis=1), [2.5, 97.5]),
        )
        self.assertAlmostEqual(report["delta_effects"]["N"]["mean"], -values.mean())
        self.assertAlmostEqual(report["delta_effects"]["B"]["mean"], 0.25 - values.mean())
        self.assertEqual(report["delta_effects"]["P"]["ci95"], [0.25, 0.25])
        self.assertEqual(report["distinct_targets"], 6)
        self.assertTrue(report["coverage"]["complete_counts"])
        json.dumps(report, allow_nan=False)

    def test_original_far_stratum(self):
        a, b = pair("a")
        a["full"][:] = [1, 9]
        b["full"][:] = [0.5, 20]
        report = summarize_permutation([a, b])
        far = report["original_far_strata"]["far4"]
        self.assertEqual(far["targets"], 1)
        self.assertEqual(far["delta_ce"]["full"], {"mean": -0.5})
        self.assertEqual(report["original_far_strata"]["far8"]["status"], "labels_unavailable")
        b["repeat_bins"]["4"][0] = "recent"
        with self.assertRaisesRegex(ValueError, "repeat labels"):
            summarize_permutation([a, b])

    def test_mismatched_pairs(self):
        a, b = pair("a")
        for change, message in (
            (lambda r: r["target_ids"].__setitem__(0, 99), "target_ids"),
            (lambda r: r["target_positions"].__setitem__(0, 17), "target_positions"),
            (lambda r: r["metadata"]["model"].update(revision="v2"), "configuration"),
            (lambda r: r["metadata"].update(paired_attempt="other"), "paired_attempt"),
            (lambda r: r["metadata"]["intervention"].update(seed=12), "permutation"),
        ):
            changed = copy.deepcopy(b)
            change(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                summarize_permutation([a, changed])
        other = pair("b")
        for row in other:
            row["metadata"]["intervention"]["order"] = [1, 0, 2, 3]
        with self.assertRaisesRegex(ValueError, "fixed permutation"):
            summarize_permutation([a, b, *other])

    def test_failed_input_check(self):
        a, b = pair("a")
        b["metadata"]["intervention"]["verification"]["full"]["anchored_tail_equal"] = False
        with self.assertRaisesRegex(ValueError, "input matching checks"):
            summarize_permutation([a, b])

    def test_short_noise_and_code_revision(self):
        a, b = pair("a")
        a["metadata"]["hardware"] = {"code_revision": "before"}
        b["metadata"]["hardware"] = {"code_revision": "after"}
        b["short"][0] += 0.001
        report = summarize_permutation([a, b], bootstrap_samples=0)
        self.assertAlmostEqual(report["max_abs_short_target_difference"], 0.001)
        self.assertAlmostEqual(report["delta_ce"]["short"]["mean"], 0.0005)

    def test_partial_and_duplicate_pairs(self):
        a, b = pair("a")
        c, _ = pair("c")
        report = summarize_permutation([a, b, c], bootstrap_samples=0)
        self.assertEqual(report["coverage"]["paired_documents"], 1)
        self.assertEqual(report["coverage"]["missing_variants"], {"c": ["permuted"]})
        empty = summarize_permutation([c], bootstrap_samples=0)
        self.assertIsNone(empty["delta_ce"]["full"]["mean"])
        with self.assertRaisesRegex(ValueError, "duplicate attempt"):
            summarize_permutation([a, b, copy.deepcopy(a)])


if __name__ == "__main__":
    unittest.main()
