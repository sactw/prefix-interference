"""Checks for record names and agreement metrics; no model is run."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from long_context_diagnostics.check_mamba import compare_modes, record_path


class CheckMambaTests(unittest.TestCase):
    def test_parity_includes_effects(self):
        full = {name: np.ones(3) for name in ("full", "short", "foreign")}
        recurrent = {name: values.copy() for name, values in full.items()}
        recurrent["full"] -= 0.00008
        recurrent["short"] += 0.00008
        comparison = compare_modes(full, recurrent, 1e-4, 1e-3)
        self.assertLess(comparison["per_condition_nll"]["short"]["mean_abs"], 1e-4)
        self.assertGreater(comparison["mean_effect_absolute_difference"]["N"], 1e-4)
        self.assertFalse(comparison["within_provisional_tolerances"])

    def test_exact_parity(self):
        full = {
            name: np.array([1, 2, 3], dtype=np.float32) for name in ("full", "short", "foreign")
        }
        self.assertTrue(compare_modes(full, full, 1e-4, 1e-3)["within_provisional_tolerances"])

    def test_record_path_ignores_code_revision(self):
        args = SimpleNamespace(
            model_id="model",
            revision="weights1",
            tokenizer_revision="tok1",
            end=2048,
            recent=512,
            targets=128,
            code_revision="old",
        )
        row = {"document_id": "target", "text": "target text"}
        donor = {"document_id": "donor", "text": "donor text"}
        path = record_path(Path("runs"), row, donor, args)
        args.code_revision = "hotfix"
        self.assertEqual(record_path(Path("runs"), row, donor, args), path)
        args.end = 4096
        self.assertNotEqual(record_path(Path("runs"), row, donor, args), path)
        args.end = 2048
        donor["text"] = "changed source bytes"
        self.assertNotEqual(record_path(Path("runs"), row, donor, args), path)


if __name__ == "__main__":
    unittest.main()
