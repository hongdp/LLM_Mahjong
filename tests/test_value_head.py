"""Unit tests for the state-value head (src/core/value_head.py)."""

import os
import tempfile
import unittest

import torch

from src.core.value_head import (ValueHead, explained_variance,
                                 last_prompt_hidden, load_value_head,
                                 save_value_head)


class TestValueHead(unittest.TestCase):

    def test_near_zero_init(self):
        """Fresh head predicts ~0 so cold-start advantages equal raw returns."""
        head = ValueHead(hidden_size=64)
        h = torch.randn(32, 64)
        self.assertLess(head(h).abs().max().item(), 0.05)

    def test_learns_linear_value(self):
        """Head must fit a simple linear value function from hidden states."""
        torch.manual_seed(0)
        head = ValueHead(hidden_size=16)
        w = torch.randn(16)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(300):
            h = torch.randn(64, 16)
            target = h @ w
            loss = torch.nn.functional.mse_loss(head(h), target)
            opt.zero_grad(); loss.backward(); opt.step()
        h = torch.randn(256, 16)
        ev = explained_variance(head(h).detach(), h @ w)
        self.assertGreater(ev, 0.9, f"explained variance only {ev}")

    def test_last_prompt_hidden_gather(self):
        hs = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        out = last_prompt_hidden(hs, torch.tensor([3, 5]))
        self.assertTrue(torch.equal(out[0], hs[0, 2]))
        self.assertTrue(torch.equal(out[1], hs[1, 4]))

    def test_bf16_hidden_accepted(self):
        """Trunk runs bf16; head must upcast internally without dtype errors."""
        head = ValueHead(hidden_size=8)
        out = head(torch.randn(4, 8, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.float32)

    def test_explained_variance_bounds(self):
        g = torch.randn(100)
        self.assertAlmostEqual(explained_variance(g, g), 1.0, places=5)
        self.assertLessEqual(explained_variance(torch.zeros(100), g), 0.01)
        self.assertEqual(explained_variance(torch.zeros(5), torch.ones(5)), 0.0)

    def test_save_load_roundtrip(self):
        head = ValueHead(hidden_size=12, inner=32)
        h = torch.randn(7, 12)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "vh.pt")
            save_value_head(head, p)
            head2 = load_value_head(p)
        self.assertTrue(torch.allclose(head(h), head2(h), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
