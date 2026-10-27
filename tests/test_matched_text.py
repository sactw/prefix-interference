"""CPU tests for cross-tokenizer source boundaries and byte denominators."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from long_context_diagnostics.matched_text import (
    load_text_plan,
    plan_matched_text,
    save_text_plan,
    summarize_matched_losses,
)


class ChunkTokenizer:
    """Lossless fixture with native one- or two-character token boundaries."""

    def __init__(self, width):
        self.width, self.vocab = width, []

    def encode(self, value, add_special_tokens=False):
        ids, offsets = [], []
        for begin in range(0, len(value), self.width):
            token = value[begin : begin + self.width]
            if token not in self.vocab:
                self.vocab.append(token)
            ids.append(self.vocab.index(token))
            offsets.append((begin, min(begin + self.width, len(value))))
        return SimpleNamespace(ids=ids, offsets=offsets)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[index] for index in ids)


class MatchedTextTests(unittest.TestCase):
    def test_boundary_skips_straddling_token(self):
        text, donor = "abΓδεζηθ", "12345678"
        tokenizers = {"one": ChunkTokenizer(1), "two": ChunkTokenizer(2)}
        plans, report = plan_matched_text(
            text,
            donor,
            recent_start=2,
            target_start=3,
            end=8,
            tokenizers=tokenizers,
            prefix_ids={"one": 99, "two": 99},
        )
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["source_char_spans"]["target"], [4, 8])
        self.assertEqual(report["target_start_adjustment_characters"], 1)
        self.assertEqual(report["target_utf8_bytes"], 8)
        self.assertEqual([len(plans[name].target_ids) for name in tokenizers], [4, 2])
        expected_prefix = {"full": text[:4], "short": text[2:4], "foreign": donor[:2] + text[2:4]}
        for name, plan in plans.items():
            for condition, ids in plan.inputs.items():
                start = plan.target_starts[condition]
                self.assertEqual(tokenizers[name].decode(ids[1:start]), expected_prefix[condition])
                self.assertEqual(tokenizers[name].decode(ids[start:]), text[4:8])
                np.testing.assert_array_equal(ids[start:], plan.target_ids)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.npz"
                save_text_plan(path, plan)
                restored = load_text_plan(path)
                self.assertEqual(restored.metadata, plan.metadata)
                for key in plan.inputs:
                    np.testing.assert_array_equal(restored.inputs[key], plan.inputs[key])

    def test_common_byte_denominator(self):
        plans, _ = plan_matched_text(
            "abΓδεζηθ",
            "12345678",
            recent_start=2,
            target_start=3,
            end=8,
            tokenizers={"one": ChunkTokenizer(1), "two": ChunkTokenizer(2)},
            prefix_ids={"one": 99, "two": 99},
        )
        summaries = {}
        for name, plan in plans.items():
            count = len(plan.target_ids)
            summaries[name] = summarize_matched_losses(
                plan,
                {
                    condition: np.full(count, total / count)
                    for condition, total in (("full", 8), ("short", 12), ("foreign", 16))
                },
            )
        for summary in summaries.values():
            self.assertEqual(summary["nats_per_utf8_byte"]["full"], 1)
            self.assertEqual(summary["nats_per_utf8_byte"]["N"], 0.5)
            self.assertEqual(summary["nats_per_utf8_byte"]["content"], 1)
            self.assertEqual(summary["nats_per_utf8_byte"]["prefix_penalty"], 0.5)
        self.assertNotEqual(
            summaries["one"]["nats_per_native_token"]["full"],
            summaries["two"]["nats_per_native_token"]["full"],
        )

    def test_lossy_normalization_unsupported(self):
        tokenizer = ChunkTokenizer(1)
        decode = tokenizer.decode
        tokenizer.decode = lambda ids, **kwargs: decode(ids, **kwargs).lower()
        plans, report = plan_matched_text(
            "abCDefgh",
            "12345678",
            recent_start=2,
            target_start=3,
            end=8,
            tokenizers={"lossy": tokenizer},
            prefix_ids={"lossy": 99},
        )
        self.assertEqual(plans, {})
        self.assertEqual(report["status"], "unsupported_text_normalization")


if __name__ == "__main__":
    unittest.main()
