"""Small CPU-only fixtures for the independent statistical readout."""

import json
import unittest

import numpy as np

from long_context_diagnostics.readout import classify_repeats, summarize


def block(document_id, full, short, foreign, labels=None, start=10):
    count = len(full)
    return {
        "document_id": document_id,
        "target_ids": np.arange(count, dtype=np.int64),
        "target_positions": np.arange(start, start + count, dtype=np.int64),
        "full": np.asarray(full, dtype=np.float32),
        "short": np.asarray(short, dtype=np.float32),
        "foreign": np.asarray(foreign, dtype=np.float32),
        "repeat_bins": {"4": np.asarray(labels if labels is not None else ["never"] * count)},
    }


class RepeatTests(unittest.TestCase):
    def test_recent_and_boundary_labels(self):
        # Prior 4-grams end at 3 and 7; target occurrence ends at 11.
        ids = np.tile([10, 20, 30, 40], 3)
        self.assertEqual(classify_repeats(ids, [11], 4).tolist(), ["recent"])
        self.assertEqual(classify_repeats(ids, [11], 5).tolist(), ["boundary"])
        self.assertEqual(classify_repeats(ids, [11], 8).tolist(), ["far:(0,1024]"])

    def test_large_vocabulary_eight_grams(self):
        first = list(range(200000, 200008))
        second = first[:-1] + [999999]
        ids = np.asarray(first + second + first, dtype=np.int64)
        labels = classify_repeats(ids, [7, 15, 23], 0, n=8)
        self.assertEqual(labels.tolist(), ["never", "never", "recent"])
        # A future matching sequence cannot retroactively make the first a repeat.
        self.assertEqual(classify_repeats(ids, [7], 0, n=8).tolist(), ["never"])

    def test_incomplete_overlap_and_edges(self):
        self.assertEqual(
            classify_repeats([1, 1, 1, 1, 1], [0, 2, 3, 4], 0).tolist(),
            ["incomplete_ngram", "incomplete_ngram", "never", "recent"],
        )
        ids = [1, 2, 3, 4, 9, 1, 2, 3, 4]
        self.assertEqual(
            classify_repeats(ids, [8], 5, distance_edges=(4, 5)).tolist(),
            ["far:(4,5]"],
        )
        self.assertEqual(
            classify_repeats(ids, [8], 5, distance_edges=(4,)).tolist(),
            ["far:(4,inf)"],
        )

    def test_target_order_is_preserved(self):
        self.assertEqual(
            classify_repeats([1] * 8, [7, 3, 4], 0).tolist(),
            ["recent", "never", "recent"],
        )


class ReadoutTests(unittest.TestCase):
    def test_signs_decomposition_and_contributions(self):
        rows = [
            block("a", [2, 4], [3, 2], [5, 3], ["recent", "never"]),
            block("b", [1], [2], [4], ["recent"]),
        ]
        report = summarize(rows, bootstrap_samples=50)
        self.assertEqual(report["targets"], 3)
        self.assertAlmostEqual(report["ce"]["full"]["mean"], 7 / 3)
        self.assertAlmostEqual(report["effects"]["N"]["mean"], 0)
        self.assertAlmostEqual(report["effects"]["content"]["mean"], 5 / 3)
        self.assertAlmostEqual(report["effects"]["prefix_penalty"]["mean"], 5 / 3)
        for name, overall in report["effects"].items():
            contribution = sum(
                value["weighted_contribution"][name]["mean"]
                for value in report["repeat_bins"]["4"].values()
            )
            self.assertAlmostEqual(contribution, overall["mean"])
        json.dumps(report, allow_nan=False)

    def test_bootstrap_multiplicity_and_weights(self):
        rows = [
            block("a", [1], [2], [4]),
            block("b", [1, 1, 1], [4, 4, 4], [5, 5, 5]),
            block("c", [1, 1], [7, 7], [8, 8]),
        ]
        samples, seed = 67, 31
        draws = np.random.default_rng(seed).integers(0, 3, size=(samples, 3))
        sums, counts = np.asarray([1, 9, 12]), np.asarray([1, 3, 2])
        expected = sums[draws].sum(1) / counts[draws].sum(1)
        result = summarize(rows, bootstrap_samples=samples, seed=seed)["effects"]["N"]
        self.assertAlmostEqual(result["mean"], 22 / 6)
        np.testing.assert_allclose(result["ci95"], np.percentile(expected, [2.5, 97.5]))
        # Replacing each resample with its set of IDs changes the interval.
        deduplicated = np.asarray(
            [sums[np.unique(draw)].sum() / counts[np.unique(draw)].sum() for draw in draws]
        )
        self.assertFalse(
            np.allclose(
                np.percentile(expected, [2.5, 97.5]), np.percentile(deduplicated, [2.5, 97.5])
            )
        )

    def test_blocks_cluster_by_document(self):
        split = [
            block("a", [1], [2], [4], start=10),
            block("a", [3, 3], [7, 7], [8, 8], start=20),
            block("b", [2], [1], [3]),
        ]
        merged = [
            block("a", [1, 3, 3], [2, 7, 7], [4, 8, 8]),
            block("b", [2], [1], [3]),
        ]
        left, right = (summarize(rows, 200, 8) for rows in (split, merged))
        self.assertEqual(left["documents"], 2)
        self.assertEqual(left["blocks"], 3)
        for key in ("ce", "effects", "repeat_bins"):
            self.assertEqual(left[key], right[key])

    def test_empty_bin_draws_counted(self):
        rows = [
            block("a", [1], [3], [4], ["recent"]),
            block("b", [1], [2], [4], ["never"]),
        ]
        report = summarize(rows, 100, 5)
        recent = report["repeat_bins"]["4"]["recent"]["effects"]["N"]
        self.assertEqual(recent["mean"], 2)
        self.assertEqual(recent["ci95"], [2, 2])
        self.assertGreater(recent["bootstrap_empty"], 0)
        self.assertEqual(recent["bootstrap_empty"] + recent["bootstrap_valid"], 100)
        empty = report["repeat_bins"]["4"]["boundary"]
        self.assertIsNone(empty["effects"]["N"]["mean"])
        self.assertIsNone(empty["effects"]["N"]["ci95"])
        self.assertEqual(empty["effects"]["N"]["bootstrap_empty"], 100)
        self.assertEqual(empty["weighted_contribution"]["N"]["mean"], 0)
        json.dumps(report, allow_nan=False)

    def test_null_intervals(self):
        report = summarize([], 10)
        self.assertEqual(report["targets"], 0)
        self.assertIsNone(report["effects"]["N"]["mean"])
        self.assertIsNone(report["effects"]["N"]["ci95"])
        self.assertEqual(report["effects"]["N"]["bootstrap_empty"], 10)
        report = summarize([block("a", [1], [2], [3])], 0)
        self.assertEqual(report["effects"]["N"]["mean"], 1)
        self.assertIsNone(report["effects"]["N"]["ci95"])
        json.dumps(report, allow_nan=False)

    def test_rejects_misaligned_inputs(self):
        row = block("a", [1, 2], [2, 3], [3, 4])
        row["short"] = row["short"][:1]
        with self.assertRaisesRegex(ValueError, "align"):
            summarize([row])
        a = block("a", [1], [2], [3])
        b = block("b", [1], [2], [3])
        a["repeat_bins"]["8"] = np.asarray(["incomplete_ngram"])
        with self.assertRaisesRegex(ValueError, "same repeat-bin"):
            summarize([a, b])


if __name__ == "__main__":
    unittest.main()
