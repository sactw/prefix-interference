"""CPU checks for reference offsets, plan metadata and scoring commands."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from long_context_diagnostics.matched_text import load_text_plan, save_text_plan
from long_context_diagnostics.prepare_replication_panel import (
    MODEL_REFERENCES,
    prepare_case,
    reference_char_spans,
)
from long_context_diagnostics.score_matched_text import scoring_command
from .test_matched_text import ChunkTokenizer


class ReplicationPreparationTests(unittest.TestCase):
    def test_unicode_offsets_and_metadata(self):
        doc = {
            "document_id": "target",
            "text": "abΓδεζηθ",
            "source": {"url": "https://example.org/target"},
        }
        donor = {
            "document_id": "donor",
            "text": "12345678",
            "source": {"url": "https://example.org/donor"},
        }
        tokenizers = {"mamba": ChunkTokenizer(1), "gdn": ChunkTokenizer(2)}
        self.assertEqual(
            reference_char_spans(doc["text"], tokenizers["mamba"], end=8, recent=1, targets=5),
            {"far": [0, 2], "recent": [2, 3], "target": [3, 8]},
        )
        plans, report = prepare_case(
            doc,
            donor,
            tokenizers,
            {"mamba": "sha1", "gdn": "sha2"},
            source_panel_sha256="panel",
            end=8,
            recent=1,
            targets=5,
            code_revision="test",
        )
        self.assertEqual(report["source_char_spans"]["target"], [4, 8])
        self.assertEqual(report["target_utf8_bytes"], 8)
        for name, plan in plans.items():
            self.assertEqual(plan.metadata["model_reference"], MODEL_REFERENCES[name])
            self.assertEqual(plan.metadata["source"], doc["source"])
            self.assertEqual(plan.metadata["donor_source"], donor["source"])
            self.assertEqual(plan.metadata["preparation_revision"], "test")
            for condition, ids in plan.inputs.items():
                np.testing.assert_array_equal(ids[plan.target_starts[condition] :], plan.target_ids)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.npz"
                save_text_plan(path, plan)
                loaded = load_text_plan(path)
                self.assertEqual(loaded.metadata, plan.metadata)
                np.testing.assert_array_equal(loaded.target_ids, plan.target_ids)

    def test_short_reference_and_scoring_command(self):
        with self.assertRaisesRegex(ValueError, "needs 12"):
            reference_char_spans("abcdefgh", ChunkTokenizer(1), end=12, recent=2, targets=2)
        command = scoring_command(
            "/srv/study",
            name="gdn",
            plan_path="/data/new.npz",
            result_path="/results/new.npz",
            revision="rev",
            python_executable="/opt/gdn/bin/python",
        )
        self.assertIn("/opt/gdn/bin/python", command)
        self.assertEqual(command[0], "env")
        self.assertEqual(command[command.index("--plan") + 1], "/data/new.npz")
        self.assertEqual(command[command.index("--output") + 1], "/results/new.npz")
        self.assertEqual(command[command.index("--solve-precision") + 1], "native")
        self.assertIn("OMP_NUM_THREADS=2", command)
        self.assertIn("HF_HOME=/srv/study/hf", command)
        self.assertIn("TRITON_CACHE_DIR=/srv/study/cache/triton-gdn", command)
        self.assertNotIn("\\", " ".join(command))


if __name__ == "__main__":
    unittest.main()
