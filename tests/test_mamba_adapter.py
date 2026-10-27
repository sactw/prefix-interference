"""Light scheduling checks with NumPy doubles; no torch or model execution."""

from types import SimpleNamespace
import unittest

import numpy as np

from long_context_diagnostics.mamba_adapter import _stream_target_losses


class NativeScheduleTests(unittest.TestCase):
    def test_alignment_carry_and_step_width(self) -> None:
        ids = np.asarray([4, 2, 8, 3, 1, 9], dtype=np.int64)
        calls = []
        states = []
        readout_targets = []

        def new_state(length):
            state = SimpleNamespace(seqlen_offset=0, total=0, length=length)
            states.append(state)
            return state

        def forward(tokens, state):
            calls.append((state.seqlen_offset, tokens.tolist()))
            if state.seqlen_offset:
                self.assertEqual(len(tokens), 1)
            hidden = (state.total + np.cumsum(tokens))[None, :, None]
            state.total += int(tokens.sum())
            return hidden

        def readout(hidden, targets):
            readout_targets.extend(targets.tolist())
            # Deliberately not a language model: exposes which causal prefix and
            # target reach each readout without importing torch.
            return hidden[0, :, 0] * 10 + targets

        expected = np.cumsum(ids[:-1])[2:] * 10 + ids[3:]
        for prefill in (1, 3, 4, 20):
            calls.clear()
            readout_targets.clear()
            losses = _stream_target_losses(ids, 3, prefill, new_state, forward, readout)
            np.testing.assert_array_equal(losses, expected)
            self.assertEqual(losses.dtype, np.float32)
            self.assertEqual(readout_targets, ids[3:].tolist())
            self.assertEqual(calls[0][0], 0)
            self.assertEqual(states[-1].seqlen_offset, len(ids) - 1)
        self.assertEqual(len({id(state) for state in states}), 4)

    def test_first_and_empty_targets(self) -> None:
        ids = np.asarray([7, 2], dtype=np.int64)

        def readout(hidden, targets):
            np.testing.assert_array_equal(hidden, [[[7]]])
            np.testing.assert_array_equal(targets, [2])
            return np.asarray([5], dtype=np.float32)

        losses = _stream_target_losses(
            ids,
            1,
            256,
            lambda _: SimpleNamespace(seqlen_offset=0),
            lambda tokens, state: tokens[None, :, None],
            readout,
        )
        np.testing.assert_array_equal(losses, [5])

        def unexpected(*args):
            self.fail("empty target block must not run the model")

        empty = _stream_target_losses(ids, 2, 1, unexpected, unexpected, unexpected)
        self.assertEqual(empty.shape, (0,))


if __name__ == "__main__":
    unittest.main()
