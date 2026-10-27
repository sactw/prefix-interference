"""Causal alignment and state-carry checks; kernel parity is covered by check_rwkv."""

from types import SimpleNamespace
import unittest

import numpy as np

from long_context_diagnostics.rwkv_adapter import _chunked_target_losses


class RWKVScheduleTests(unittest.TestCase):
    def test_chunk_seams_and_fresh_state(self):
        ids = np.asarray([7, 3, 11, 2, 9, 4, 1], dtype=np.int64)
        states = []

        def forward(tokens, state, full_output):
            if state is None:
                state = SimpleNamespace(total=0, consumed=0)
                states.append(state)
            values = state.total + np.cumsum(tokens)
            state.total += int(tokens.sum())
            state.consumed += len(tokens)
            return values[:, None] if full_output else values[-1:], state

        def readout(logits, targets):
            return logits[:, 0] * 100 + targets

        for start in (1, 2, 4, 6):
            for chunk in (1, 2, 4, 20):
                for prefill in (1, 3, 20):
                    actual = _chunked_target_losses(ids, start, chunk, prefill, forward, readout)
                    expected = np.cumsum(ids[:-1])[start - 1 :] * 100 + ids[start:]
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(states[-1].consumed, len(ids) - 1)
                    self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(len({id(state) for state in states}), 48)

    def test_empty_target_is_noop(self):
        def unexpected(*args):
            self.fail("empty target range should not execute the model")

        result = _chunked_target_losses(np.asarray([1, 2]), 2, 8, 16, unexpected, unexpected)
        self.assertEqual(result.shape, (0,))

    def test_prefill_reference_skips_prefix_logits(self):
        ids = np.arange(12, dtype=np.int64)
        calls = []

        def forward(tokens, state, full_output):
            calls.append((tokens.tolist(), full_output))
            return tokens[:, None] if full_output else tokens[-1:], None

        def readout(logits, targets):
            np.testing.assert_array_equal(logits[:, 0] + 1, targets)
            return targets.astype(np.float32)

        result = _chunked_target_losses(ids, 8, 3, 7, forward, readout)
        np.testing.assert_array_equal(result, ids[8:])
        self.assertEqual(calls, [(list(range(7)), False), ([7, 8, 9], True), ([10], True)])


if __name__ == "__main__":
    unittest.main()
