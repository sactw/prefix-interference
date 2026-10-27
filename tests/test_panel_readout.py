"""Focused fixtures for donor clustering and panel effects, CPU only."""

import copy
import json
import unittest

import numpy as np

from long_context_diagnostics.panel_readout import summarize_panel
from long_context_diagnostics.readout import summarize


def record(document, donor, full, short, foreign, end=20, recent=4, labels=None):
    count = len(full)
    return {
        "document_id": document,
        "target_ids": np.arange(count),
        "target_positions": np.arange(end - count, end),
        **{
            key: np.asarray(values, dtype=np.float32)
            for key, values in (("full", full), ("short", short), ("foreign", foreign))
        },
        "repeat_bins": {n: np.asarray(labels or ["never"] * count) for n in ("4", "8")},
        "metadata": {
            "donor_id": donor,
            "text_sha256": "text-" + document,
            "model": {"model_id": "public-fixture", "revision": "v1", "tokenizer_revision": "t1"},
            "history": {
                "end_exclusive": end,
                "recent_prefix_tokens": recent,
                "target_tokens": count,
                "prefix_id": 0,
            },
        },
    }


class PanelReadoutTests(unittest.TestCase):
    def test_donor_average_before_bootstrap(self):
        rows = []
        expected = []
        for doc, net, donors in (("a", 1, (1, 3)), ("b", 3, (5, 7)), ("c", 6, (2, 8))):
            rows.extend(
                record(doc, f"donor{i}", [2, 2], [2 + net] * 2, [2 + effect] * 2)
                for i, effect in enumerate(donors)
            )
            expected.append(
                record(doc, "average", [2, 2], [2 + net] * 2, [2 + np.mean(donors)] * 2)
            )
        result = summarize_panel(rows, 79, 31, 3, 2)["panels"][0]
        reference = summarize(expected, 79, 31)
        self.assertEqual(result["targets"], 6)
        self.assertEqual(result["documents"], 3)
        self.assertEqual(result["blocks"], 3)
        self.assertEqual(result["effects"], reference["effects"])
        self.assertTrue(result["coverage"]["complete_counts"])

    def test_cancellation_and_donor_sensitivity(self):
        labels = ["far:(4,16]", "recent", "never"]
        rows = [
            record("a", "d1", [3, 3, 3], [2, 3, 3], [4, 4, 2], labels=labels),
            record("a", "d2", [3, 3, 3], [2, 3, 3], [5, 2, 2], labels=labels),
        ]
        panel = summarize_panel(rows, 20)["panels"][0]
        far = panel["groups"]["far4"]
        self.assertEqual(far["targets"], 1)
        self.assertEqual(far["effects"]["N"]["mean"], -1)
        self.assertEqual(far["effects"]["content"]["mean"], 1.5)
        self.assertEqual(far["effects"]["prefix_penalty"]["mean"], 2.5)
        sensitivity = panel["donor_sensitivity"]["panel"]
        self.assertTrue(sensitivity["far4"]["effects"]["content"]["sign_stable"])
        self.assertFalse(sensitivity["recent4"]["effects"]["content"]["sign_stable"])
        self.assertEqual(sensitivity["recent4"]["effects"]["content"]["mean_range"], [-1, 1])
        json.dumps(panel, allow_nan=False)

    def test_unequal_donor_counts(self):
        rows = [
            record("a", "d1", [2], [1], [3]),
            record("b", "d1", [2], [4], [4]),
            record("b", "d2", [2], [4], [6]),
        ]
        result = summarize_panel(rows, 0)["panels"][0]
        self.assertEqual(result["targets"], 2)
        self.assertEqual(result["effects"]["N"]["mean"], 0.5)
        self.assertEqual(result["effects"]["content"]["mean"], 2.0)

    def test_groups_keep_documents_without_repeats(self):
        rows = [
            record("a", "d1", [3], [2], [4], labels=["far:(4,16]"]),
            record("b", "d1", [3], [2], [4], labels=["never"]),
        ]
        group = summarize_panel(rows, 100, 5)["panels"][0]["groups"]["far4"]
        self.assertEqual(group["documents"], 1)
        self.assertGreater(group["effects"]["N"]["bootstrap_empty"], 0)
        self.assertEqual(group["weighted_contribution"]["N"]["mean"], -0.5)

    def test_separate_panels(self):
        row = record("a", "d", [3], [2], [4])
        different_end = record("a", "d", [3], [2], [4], end=40)
        different_recent = record("a", "d", [3], [2], [4], recent=8)
        different_model = copy.deepcopy(row)
        different_model["metadata"]["model"]["revision"] = "v2"
        result = summarize_panel(
            [row, different_end, different_recent, different_model], 0, 0, 8, 2
        )
        self.assertEqual(len(result["panels"]), 4)
        self.assertTrue(all(not panel["coverage"]["complete_counts"] for panel in result["panels"]))
        self.assertTrue(all(panel["targets"] == 1 for panel in result["panels"]))

    def test_mismatch_and_duplicate_donor(self):
        a = record("a", "d1", [3], [2], [4])
        b = record("a", "d2", [3], [2], [4])
        b["target_ids"][0] = 9
        with self.assertRaisesRegex(ValueError, "different target_ids"):
            summarize_panel([a, b])
        with self.assertRaisesRegex(ValueError, "duplicate donor"):
            summarize_panel([a, a])

    def test_repeat_spread_reported(self):
        a = record("a", "d1", [3], [2], [4])
        b = record("a", "d2", [3.1], [2.1], [4])
        panel = summarize_panel([a, b], 0)["panels"][0]
        self.assertEqual(panel["targets"], 1)
        self.assertAlmostEqual(panel["ce"]["full"]["mean"], 3.05, places=6)
        self.assertAlmostEqual(
            panel["documents_detail"]["a"]["reference_repeat_spread"]["full"][
                "maximum_per_target_range"
            ],
            0.1,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
