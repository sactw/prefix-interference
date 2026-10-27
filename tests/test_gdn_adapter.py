"""CPU alignment and state-carry checks; GPU checks use check_gdn."""

from types import SimpleNamespace
import unittest

import numpy as np

from long_context_diagnostics.gdn_adapter import _cached_target_losses


class CachedScheduleTests(unittest.TestCase):
    def test_chunk_alignment_and_reset(self):
        ids = np.asarray([7, 3, 11, 2, 9, 4, 1], dtype=np.int64)
        states, seen = [], []

        def forward(tokens, state):
            if state is None:
                state = SimpleNamespace(total=0, consumed=0)
                states.append(state)
            hidden = (state.total + np.cumsum(tokens))[None, :, None]
            state.total += int(tokens.sum())
            state.consumed += len(tokens)
            return hidden, state

        def readout(hidden, targets):
            seen.extend(targets.tolist())
            return hidden[0, :, 0] * 100 + targets

        for start in (1, 2, 4, 6):
            for chunk in (1, 2, 4, 20):
                seen.clear()
                actual = _cached_target_losses(ids, start, chunk, forward, readout)
                expected = np.cumsum(ids[:-1])[start - 1 :] * 100 + ids[start:]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(seen, ids[start:].tolist())
                self.assertEqual(states[-1].consumed, len(ids) - 1)
                self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(len({id(state) for state in states}), 16)

    def test_empty_targets_do_not_execute(self):
        def unexpected(*args):
            self.fail("empty target block must not execute")

        result = _cached_target_losses(np.asarray([1, 2]), 2, 16, unexpected, unexpected)
        self.assertEqual(result.shape, (0,))


if __name__ == "__main__":
    unittest.main()
