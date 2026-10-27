"""Focused paired-target and shared document-bootstrap checks; CPU only."""

import copy
import json
import unittest

import numpy as np

from long_context_diagnostics.ladder_readout import summarize_ladder


def record(doc, far, *, full=2.0, short=3.0, foreign=4.0, donor="d1"):
    return {
        "document_id": doc,
        "target_ids": np.array([10, 11]),
        "target_positions": np.array([18, 19]),
        **{
            key: np.full(2, value, dtype=np.float64)
            for key, value in (("full", full), ("short", short), ("foreign", foreign))
        },
        "repeat_bins": {"4": np.array(["never", "never"])},
        "metadata": {
            "donor_id": donor,
            "text_sha256": "text-" + doc,
            "donor_text_sha256": "donor-" + donor,
            "selected_execution": "full",
            "model": {"model_id": "fixture", "revision": "v1", "tokenizer_revision": "t1"},
            "history": {
                "far_prefix_tokens": far,
                "full_history_start": 15 - far,
                "end_exclusive": 20,
                "recent_prefix_tokens": 3,
                "target_tokens": 2,
                "retained_start": 15,
                "prefix_id": 0,
                "repeat_label_scope": "visible_full_slice",
            },
        },
    }


class LadderReadoutTests(unittest.TestCase):
    def summarize(self, rows, **kwargs):
        return summarize_ladder(
            rows, donor_id="d1", fars=[2, 5], baseline_far=2, expected_documents=3, **kwargs
        )

    def test_known_changes_share_draws(self):
        rows = []
        for doc, net in (("a", 1.0), ("b", 2.0), ("c", 5.0)):
            rows.extend([record(doc, 2, short=2 + net), record(doc, 5, full=1.5, short=2 + net)])
        result = self.summarize(rows, bootstrap_samples=67, seed=13)
        draws = np.random.default_rng(13).integers(0, 3, size=(67, 3))
        expected_ci = np.percentile(np.array([1.0, 2.0, 5.0])[draws].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose(result["lengths"][0]["effects"]["N"]["ci95"], expected_ci)
        self.assertEqual(
            result["lengths"][1]["change_vs_baseline"]["N"], {"mean": 0.5, "ci95": [0.5, 0.5]}
        )
        self.assertEqual(result["lengths"][1]["change_vs_baseline"]["B"]["mean"], 0.5)
        self.assertEqual(result["lengths"][1]["change_vs_baseline"]["P"]["mean"], 0)
        self.assertEqual(result["distinct_targets"], 6)
        self.assertTrue(result["coverage"]["complete_counts"])
        json.dumps(result, allow_nan=False)

    def test_donor_selection(self):
        rows = [
            record("a", far, donor=donor, foreign=4 if donor == "d1" else 100)
            for far in (2, 5)
            for donor in ("d1", "d2")
        ]
        result = self.summarize(rows, bootstrap_samples=0)
        self.assertEqual(result["lengths"][0]["effects"]["B"]["mean"], 2)
        self.assertEqual(result["coverage"]["excluded_other_donor_records"], 2)
        with self.assertRaisesRegex(ValueError, "select one donor"):
            summarize_ladder(rows, fars=[2, 5], baseline_far=2)

    def test_mismatches_not_paired(self):
        a, b = record("a", 2), record("a", 5)
        for change, message in (
            (lambda r: r["target_ids"].__setitem__(0, 99), "target_ids"),
            (lambda r: r["target_positions"].__setitem__(0, 17), "positions"),
            (lambda r: r["metadata"]["model"].update(revision="v2"), "model"),
            (lambda r: r["metadata"]["history"].update(recent_prefix_tokens=4), "fixed E/R/T"),
            (lambda r: r["metadata"].update(text_sha256="different"), "text_sha256"),
        ):
            altered = copy.deepcopy(b)
            change(altered)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.summarize([a, altered])

    def test_short_drift_and_code_revision(self):
        a, b = record("a", 2), record("a", 5)
        a["metadata"]["hardware"] = {"code_revision": "old"}
        b["metadata"]["hardware"] = {"code_revision": "new"}
        b["short"][0] += 0.001
        b["repeat_bins"]["4"][0] = "recent"  # visibility may change with F
        result = self.summarize([a, b], bootstrap_samples=0)
        self.assertAlmostEqual(result["max_short_per_target_range"], 0.001)
        self.assertAlmostEqual(
            result["lengths"][1]["short_mean_change_vs_baseline"]["mean"], 0.0005
        )

    def test_missing_cells_not_imputed(self):
        rows = [record("a", 2), record("a", 5), record("b", 2, full=50)]
        result = self.summarize(rows, bootstrap_samples=0)
        self.assertEqual(result["coverage"]["paired_documents"], 1)
        self.assertEqual(result["coverage"]["unpaired_missing_fars"], {"b": [5]})
        self.assertEqual(result["lengths"][0]["ce"]["full"]["mean"], 2)
        empty = self.summarize([record("a", 2)], bootstrap_samples=0)
        self.assertIsNone(empty["lengths"][0]["effects"]["N"]["mean"])
        self.assertIsNone(empty["max_short_per_target_range"])

    def test_duplicate_attempt(self):
        a = record("a", 2)
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            self.summarize([a, copy.deepcopy(a), record("a", 5)])


if __name__ == "__main__":
    unittest.main()
