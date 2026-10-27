"""Focused constructed-fixture tests for byte ratios and paired document draws."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from long_context_diagnostics.matched_panel_readout import main, summarize_matched_panel
from long_context_diagnostics.measurement import load_pair, save_pair


def fixture(doc, name, size, totals, *, code="current"):
    count, start = {"mamba": (2, 10), "gdn": (4, 25), "rwkv": (8, 40)}[name]
    losses = {
        key: np.full(count, value / count, dtype=np.float32)
        for key, value in zip(("full", "short", "foreign"), totals)
    }
    full, short, foreign = totals
    summed = dict(
        zip(
            ("full", "short", "foreign", "N", "content", "prefix_penalty"),
            (full, short, foreign, short - full, foreign - full, foreign - short),
        )
    )
    reference = {
        "model_id": name,
        "revision": "weights",
        "tokenizer_id": name,
        "tokenizer_revision": "tokens",
    }
    history = {
        "protocol": "matched_text_span_v1",
        "model_name": name,
        "document_id": doc,
        "text_sha256": doc,
        "donor_text_sha256": "donor-text",
        "target_text_sha256": doc + "-target",
        "source_char_spans": {"far": [0, 200], "recent": [200, 300], "target": [300, 300 + size]},
        "requested_source_char_spans": {
            "far": [0, 200],
            "recent": [200, 300],
            "target": [300, 300 + size],
        },
        "donor_id": "donor",
        "target_utf8_bytes": size,
        "target_characters": size,
        "foreign_policy": "character-matched",
        "normalization": "none",
        "reference_span": {"E": 32768},
        "model_reference": reference,
        "prefix_id": 0 if name == "mamba" else 1,
        "tokenizer_json_sha256": name + "-tokenizer",
        "target_tokens": count,
        "documentary_input_tokens": {
            "full": start + count,
            "short": count + 2,
            "foreign": start + count,
        },
    }
    record = {
        "document_id": doc,
        "target_ids": np.arange(count) + start,
        "target_positions": np.arange(start, start + count),
        "repeat_bins": {},
        **losses,
        "metadata": {
            "history": history,
            "code_revision": code,
            "model": dict(reference, dtype="float32", loss_dtype="float32", backend=name),
            "selected_execution": "full",
            "matched_text": {
                "counts": {"tokens": count, "characters": size, "utf8_bytes": size},
                "total_nll_nats": summed,
                "nats_per_utf8_byte": {key: value / size for key, value in summed.items()},
            },
        },
    }
    return record


def records():
    return [
        fixture("a", "mamba", 4, (4, 6, 8), code="old-revision"),
        fixture("a", "gdn", 4, (6, 10, 12)),
        fixture("b", "mamba", 16, (8, 8, 12)),
        fixture("b", "gdn", 16, (12, 10, 12)),
    ]


def three_model_records():
    return records() + [
        fixture("a", "rwkv", 4, (8, 10, 12)),
        fixture("b", "rwkv", 16, (16, 18, 20)),
    ]


class MatchedPanelTests(unittest.TestCase):
    def test_byte_ratio_and_bootstrap(self):
        report = summarize_matched_panel(records())
        self.assertEqual(report["target_utf8_bytes"], 20)
        self.assertEqual(report["models"]["mamba"]["primary_byte_weighted"]["N"]["mean"], 0.1)
        self.assertEqual(report["models"]["gdn"]["primary_byte_weighted"]["N"]["mean"], 0.1)
        delta = report["paired_model_difference"]
        self.assertEqual(delta["primary_byte_weighted"]["N"]["mean"], 0)
        self.assertEqual(delta["primary_byte_weighted"]["N"]["ci95"], [-0.125, 0.5])
        self.assertEqual(delta["secondary_equal_document"]["N"]["mean"], 0.1875)
        self.assertEqual(report, summarize_matched_panel(records()))

    def test_pair_files_keep_native_positions(self):
        loaded = []
        with tempfile.TemporaryDirectory() as directory:
            for i, record in enumerate(records()):
                path = Path(directory) / f"{i}.npz"
                arms = {
                    key: {
                        "target_ids": record["target_ids"],
                        "target_positions": record["target_positions"],
                        "losses": record[key],
                    }
                    for key in ("full", "short", "foreign")
                }
                save_pair(
                    path,
                    document_id=record["document_id"],
                    arms=arms,
                    repeat_bins={},
                    metadata=record["metadata"],
                )
                loaded.append(load_pair(path))
        self.assertEqual(summarize_matched_panel(loaded)["documents"], 2)

    def test_mismatches_not_pooled(self):
        for field, value in [
            ("target_text_sha256", "different"),
            ("text_sha256", "different"),
            ("donor_text_sha256", "different"),
            ("source_char_spans", {"far": [1, 200], "recent": [200, 300], "target": [300, 304]}),
        ]:
            with self.subTest(field=field):
                rows = records()
                rows[1]["metadata"]["history"][field] = value
                with self.assertRaises(ValueError):
                    summarize_matched_panel(rows)
        rows = records()
        rows[3]["metadata"]["model"]["dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "mixed model revision/precision"):
            summarize_matched_panel(rows)
        rows = records()
        rows[3]["metadata"]["model"]["revision"] = "other-weights"
        rows[3]["metadata"]["history"]["model_reference"]["revision"] = "other-weights"
        with self.assertRaisesRegex(ValueError, "mixed model revision/precision"):
            summarize_matched_panel(rows)

    def test_missing_duplicate_and_corrupted_records(self):
        with self.assertRaisesRegex(ValueError, "missing matched model"):
            summarize_matched_panel(records()[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_matched_panel(records() + [deepcopy(records()[0])])
        rows = records()
        rows[0]["metadata"]["matched_text"]["total_nll_nats"]["full"] += 1
        with self.assertRaisesRegex(ValueError, "saved total NLL"):
            summarize_matched_panel(rows)
        rows = records()
        rows[1]["target_positions"] += 1
        with self.assertRaisesRegex(ValueError, "native target positions"):
            summarize_matched_panel(rows)

    def test_one_document_no_interval(self):
        report = summarize_matched_panel(records()[:2])
        self.assertIsNone(report["models"]["mamba"]["primary_byte_weighted"]["N"]["ci95"])
        self.assertIsNone(report["paired_model_difference"]["primary_byte_weighted"]["N"]["ci95"])

    def test_two_model_output_fields(self):
        report = summarize_matched_panel(records())
        self.assertEqual(report, summarize_matched_panel(records(), models=("mamba", "gdn")))
        self.assertEqual(report, summarize_matched_panel(records(), models=("gdn", "mamba")))
        self.assertEqual(report["paired_model_difference"]["direction"], "gdn_minus_mamba")
        self.assertNotIn("pairwise_model_differences", report)
        self.assertNotIn("pairwise_model_effects", report["per_document"][0])
        self.assertEqual(
            report["per_document"][0]["gdn_minus_mamba_effects"], {"N": 0.5, "B": 0.5, "P": 0}
        )

    def test_three_models_share_draws(self):
        rows = three_model_records()
        report = summarize_matched_panel(rows, models=("mamba", "gdn", "rwkv"))
        self.assertEqual(list(report["models"]), ["mamba", "gdn", "rwkv"])
        self.assertEqual(report["models"]["rwkv"]["native_target_tokens"], 16)
        self.assertEqual(
            report["paired_model_difference"],
            summarize_matched_panel(records())["paired_model_difference"],
        )
        self.assertEqual(
            set(report["pairwise_model_differences"]),
            {"gdn_minus_mamba", "rwkv_minus_mamba", "rwkv_minus_gdn"},
        )
        for pair in (("mamba", "gdn"), ("mamba", "rwkv"), ("gdn", "rwkv")):
            direction = f"{pair[1]}_minus_{pair[0]}"
            pair_rows = [row for row in rows if row["metadata"]["history"]["model_name"] in pair]
            pair_report = summarize_matched_panel(pair_rows, models=pair)
            contrast = report["pairwise_model_differences"][direction]
            self.assertEqual(contrast, pair_report["paired_model_difference"])
            for aggregation in ("primary_byte_weighted", "secondary_equal_document"):
                self.assertEqual(set(contrast[aggregation]), {"N", "B", "P"})
            for original, paired in zip(report["per_document"], pair_report["per_document"]):
                self.assertEqual(
                    original["pairwise_model_effects"][direction], paired[f"{direction}_effects"]
                )
        # This paired effect is identical per document despite different native
        # token counts, absolute losses and document byte lengths.
        self.assertEqual(
            report["pairwise_model_differences"]["rwkv_minus_mamba"]["primary_byte_weighted"]["B"],
            {"mean": 0.0, "ci95": [0.0, 0.0]},
        )

    def test_third_model_requirements(self):
        rows = three_model_records()
        rows[-1]["metadata"]["history"]["target_text_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different target_text_sha256"):
            summarize_matched_panel(rows, models=("mamba", "gdn", "rwkv"))
        with self.assertRaisesRegex(ValueError, "missing matched model"):
            summarize_matched_panel(three_model_records()[:-1], models=("mamba", "gdn", "rwkv"))
        for models in (("mamba",), ("mamba", "mamba"), ("mamba", "unknown")):
            with (
                self.subTest(models=models),
                self.assertRaisesRegex(ValueError, "distinct available"),
            ):
                summarize_matched_panel(records(), models=models)

    def test_three_model_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, record in enumerate(three_model_records()):
                path = Path(directory) / f"{i}.npz"
                arms = {
                    key: {
                        "target_ids": record["target_ids"],
                        "target_positions": record["target_positions"],
                        "losses": record[key],
                    }
                    for key in ("full", "short", "foreign")
                }
                save_pair(
                    path,
                    document_id=record["document_id"],
                    arms=arms,
                    repeat_bins={},
                    metadata=record["metadata"],
                )
                paths.append(str(path))
            output = Path(directory) / "report.json"
            with redirect_stdout(StringIO()):
                main([*paths, "--output", str(output), "--models", "mamba", "gdn", "rwkv"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["records"], 6)
            self.assertEqual(len(report["pairwise_model_differences"]), 3)


if __name__ == "__main__":
    unittest.main()
