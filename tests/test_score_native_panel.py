"""Light persistence/length checks; no model, torch, GPU or downloads."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from long_context_diagnostics import score_native_panel as panel
from long_context_diagnostics.measurement import load_pair


def arguments(output):
    return SimpleNamespace(
        output=Path(output),
        model_id=panel.DEFAULT_MODEL_ID,
        revision=panel.DEFAULT_REVISION,
        tokenizer_revision=panel.DEFAULT_TOKENIZER_REVISION,
        ends=[6],
        recent=2,
        targets=2,
        document_id=None,
        code_revision="first",
        max_memory_gb=14,
        chunk_size=2,
        repeat_first_block=False,
    )


class NativePanelTests(unittest.TestCase):
    def test_record_paths_and_length_check(self):
        rows = [
            {"document_id": "a", "role": "target", "donor_id": "d", "text": "a"},
            {"document_id": "d", "role": "donor", "text": "d"},
        ]
        args = arguments("unused")
        first = panel.plan_cases(rows, args)[0]
        args.code_revision = "operational_fix"
        self.assertEqual(first["path"], panel.plan_cases(rows, args)[0]["path"])
        reason = panel.eligibility(first, {"a": np.arange(5), "d": np.arange(3)}, 2, 2)
        self.assertEqual(reason["status"], "insufficient_text")
        self.assertIsNone(panel.eligibility(first, {"a": np.arange(6), "d": np.arange(2)}, 2, 2))
        reason = panel.eligibility(first, {"a": np.arange(6), "d": np.arange(1)}, 2, 2)
        self.assertEqual(reason["required_donor_tokens"], 2)

    def test_oom_keeps_completed_triplets(self):
        class FixtureOOM(Exception):
            pass

        cuda = SimpleNamespace(
            OutOfMemoryError=FixtureOOM,
            empty_cache=lambda: None,
            max_memory_allocated=lambda: 10,
            max_memory_reserved=lambda: 12,
        )
        adapter = SimpleNamespace(
            prefix_id=0,
            metadata={"fixture": True},
            _torch=SimpleNamespace(cuda=cuda),
            score_full=None,
        )

        def score_fixture(adapter, method, ids, start, chunk_size):
            if ids[-1] == 2:
                raise FixtureOOM("constructed allocator failure")
            return np.ones(len(ids) - start, dtype=np.float32), {"seconds": 0.01}

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = arguments(directory)
            rows = [
                {"document_id": name, "role": "target", "donor_id": "d", "text": name}
                for name in ("a", "b", "c")
            ]
            rows.append({"document_id": "d", "role": "donor", "text": "d"})
            tokens = {name: np.full(6, i + 1) for i, name in enumerate(("a", "b", "c", "d"))}
            cases = panel.plan_cases(rows, args)
            status = {"blocks": []}
            with patch.object(panel, "timed", side_effect=score_fixture):
                panel.run_cases(adapter, cases, tokens, args, {"code_revision": "first"}, status)
            self.assertEqual(
                [row["status"] for row in status["blocks"]],
                ["completed", "out_of_memory", "completed"],
            )
            self.assertEqual([case["path"].exists() for case in cases], [True, False, True])
            for case in (cases[0], cases[2]):
                self.assertEqual(load_pair(case["path"])["target_positions"].tolist(), [4, 5])
            source = Path(directory) / "documents.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.object(
                panel, "MambaAdapter", side_effect=AssertionError("unexpected model load")
            ):
                panel.main(
                    [
                        "--documents",
                        str(source),
                        "--output",
                        directory,
                        "--end",
                        "6",
                        "--recent",
                        "2",
                        "--targets",
                        "2",
                        "--document-id",
                        "a",
                        "--document-id",
                        "c",
                        "--code-revision",
                        "operational_fix",
                        "--bootstrap-samples",
                        "8",
                    ]
                )
            summary = json.loads((Path(directory) / "end-6" / "summary.json").read_text())
            self.assertEqual(summary["documents"], 2)
            self.assertEqual(summary["targets"], 4)


if __name__ == "__main__":
    unittest.main()
