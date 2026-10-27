"""Focused cache ownership and causal alignment checks, without torch/GPU."""

from types import SimpleNamespace
import unittest

import numpy as np

from long_context_diagnostics.score_state_attenuation import (
    clone_cache,
    common_suffix_boundaries,
    continuation_losses,
)


class TensorDouble:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float64)

    def clone(self):
        return TensorDouble(self.values.copy())

    def mul_(self, value):
        self.values *= value
        return self


class BoundaryTests(unittest.TestCase):
    def test_cache_scaling_keeps_conv_state(self):
        state = SimpleNamespace(
            seqlen_offset=9,
            batch_size_offset=0,
            lengths_per_sample=TensorDouble([9]),
            key_value_memory_dict={2: (TensorDouble([4, 5]), TensorDouble([6, 8]))},
        )
        half, zero, identity = [clone_cache(state, a) for a in (0.5, 0, 1)]
        for branch, expected in ((half, [3, 4]), (zero, [0, 0]), (identity, [6, 8])):
            self.assertEqual(branch.seqlen_offset, 9)
            np.testing.assert_array_equal(branch.key_value_memory_dict[2][0].values, [4, 5])
            np.testing.assert_array_equal(branch.key_value_memory_dict[2][1].values, expected)
        half.key_value_memory_dict[2][0].mul_(7)
        half.lengths_per_sample.mul_(2)
        np.testing.assert_array_equal(state.key_value_memory_dict[2][0].values, [4, 5])
        np.testing.assert_array_equal(state.key_value_memory_dict[2][1].values, [6, 8])
        np.testing.assert_array_equal(state.lengths_per_sample.values, [9])

    def test_common_suffix_skips_tokenizer_seam(self):
        plan = SimpleNamespace(
            target_ids=np.array([8, 9]),
            inputs={
                "full": np.array([0, 1, 21, 4, 5, 8, 9]),
                "foreign": np.array([0, 31, 32, 4, 5, 8, 9]),
                "short": np.array([0, 2, 3, 4, 5, 8, 9]),
            },
        )
        boundaries, recent = common_suffix_boundaries(plan)
        self.assertEqual(boundaries, {"full": 3, "foreign": 3, "short": 3})
        self.assertEqual(recent, 2)
        np.testing.assert_array_equal(plan.inputs["full"], [0, 1, 21, 4, 5, 8, 9])

    def test_continuation_keeps_base_cache(self):
        ids = np.array([0, 2, 3, 5, 7, 11, 13], dtype=np.int64)
        base = SimpleNamespace(
            seqlen_offset=3,
            lengths_per_sample=None,
            key_value_memory_dict={0: (TensorDouble([99]), TensorDouble([5]))},
        )
        calls, scored = [], []

        def forward(tokens, state):
            self.assertEqual(len(tokens), 1)
            calls.append((state.seqlen_offset, int(tokens[0])))
            ssm = state.key_value_memory_dict[0][1]
            ssm.values += tokens[0]
            return ssm.values.copy()[None, :, None]

        def readout(hidden, targets):
            scored.extend(targets.tolist())
            return hidden[0, :, 0] * 10 + targets

        for alpha in (1, 0.5, 0):
            calls.clear()
            scored.clear()
            branch = clone_cache(base, alpha)
            result = continuation_losses(
                ids,
                5,
                3,
                branch,
                forward,
                readout,
                lambda rows: np.concatenate(rows, axis=1),
                logits_chunk=1,
            )
            expected = (5 * alpha + np.cumsum(ids[3:-1]))[1:] * 10 + ids[5:]
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(scored, [11, 13])
            self.assertEqual(calls, [(3, 5), (4, 7), (5, 11)])
            self.assertEqual(branch.seqlen_offset, 6)
            self.assertEqual(base.seqlen_offset, 3)
            np.testing.assert_array_equal(base.key_value_memory_dict[0][1].values, [5])


if __name__ == "__main__":
    unittest.main()
