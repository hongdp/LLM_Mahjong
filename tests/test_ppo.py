"""Unit tests for the PPO clipped-surrogate loss (src/core/ppo.py)."""

import unittest

import torch

from src.core.ppo import ppo_clip_loss


def make_batch(B=3, T=6, g_lens=(4, 6, 2), advs=(1.5, -2.0, 0.5), seed=0):
    g = torch.Generator().manual_seed(seed)
    new_lp = -torch.rand((B, T), generator=g)  # logprobs are negative
    mask = torch.zeros((B, T))
    for b, gl in enumerate(g_lens):
        mask[b, :gl] = 1.0
    adv = torch.tensor(advs)
    return new_lp, mask, adv


class TestPPOClipLoss(unittest.TestCase):

    def test_ratio_one_gradient_equals_reinforce(self):
        """At old==new the PPO gradient must equal the advantage-weighted
        NLL (REINFORCE) gradient — same per-sequence token-mean weighting."""
        new_lp, mask, adv = make_batch()
        x = new_lp.clone().requires_grad_(True)
        old = new_lp.clone()

        loss, _ = ppo_clip_loss(x, old, adv, mask, clip_eps=0.2)
        loss.backward()
        grad_ppo = x.grad.clone()

        y = new_lp.clone().requires_grad_(True)
        counts = mask.sum(dim=1).clamp(min=1)
        nll = (-(y) * mask).sum(dim=1) / counts        # mean NLL over tokens
        reinforce = (nll * adv).mean()
        reinforce.backward()
        grad_rf = y.grad.clone()

        self.assertTrue(torch.allclose(grad_ppo, grad_rf, atol=1e-6),
                        f"max diff {(grad_ppo - grad_rf).abs().max()}")

    def test_clip_kills_gradient_for_large_positive_ratio(self):
        """A>0 and ratio above 1+eps → the clipped branch is active and the
        token contributes zero gradient (no runaway reinforcement)."""
        new_lp = torch.tensor([[-0.5, -0.5]], requires_grad=True)
        old_lp = new_lp.detach() - 1.0          # ratio = e ≈ 2.72 > 1.2
        adv = torch.tensor([2.0])
        mask = torch.ones((1, 2))
        loss, stats = ppo_clip_loss(new_lp, old_lp, adv, mask, clip_eps=0.2)
        loss.backward()
        self.assertTrue(torch.allclose(new_lp.grad, torch.zeros_like(new_lp)),
                        f"grad {new_lp.grad}")
        self.assertEqual(stats["clip_frac"], 1.0)

    def test_negative_advantage_large_ratio_still_penalized(self):
        """A<0 with ratio above 1+eps takes the UNCLIPPED branch (min of two
        negatives) — gradient keeps flowing to push the action down."""
        new_lp = torch.tensor([[-0.5]], requires_grad=True)
        old_lp = new_lp.detach() - 1.0
        adv = torch.tensor([-2.0])
        mask = torch.ones((1, 1))
        loss, _ = ppo_clip_loss(new_lp, old_lp, adv, mask, clip_eps=0.2)
        loss.backward()
        self.assertGreater(new_lp.grad.abs().item(), 0.1)

    def test_mask_excludes_padding(self):
        """Tokens outside the mask must not affect loss or gradient."""
        new_lp, mask, adv = make_batch()
        x1 = new_lp.clone().requires_grad_(True)
        loss1, _ = ppo_clip_loss(x1, new_lp.clone(), adv, mask)
        loss1.backward()
        # Perturb masked-out entries wildly: loss must be identical.
        noisy = new_lp + (1 - mask) * 123.0
        x2 = noisy.clone().requires_grad_(True)
        loss2, _ = ppo_clip_loss(x2, new_lp.clone(), adv, mask)
        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-6))
        self.assertTrue(torch.all(x1.grad * (1 - mask) == 0))

    def test_kl_and_clipfrac_ranges(self):
        new_lp, mask, adv = make_batch(seed=7)
        old = new_lp + 0.3 * torch.randn(new_lp.shape,
                                         generator=torch.Generator().manual_seed(1))
        _, stats = ppo_clip_loss(new_lp, old, adv, mask)
        self.assertGreaterEqual(stats["approx_kl"], 0.0)
        self.assertGreaterEqual(stats["clip_frac"], 0.0)
        self.assertLessEqual(stats["clip_frac"], 1.0)


if __name__ == "__main__":
    unittest.main()


class TestKLRefPenalty(unittest.TestCase):
    def test_zero_at_identical(self):
        from src.core.ppo import kl_ref_penalty
        lp = -torch.rand((2, 5))
        mask = torch.ones((2, 5))
        self.assertAlmostEqual(kl_ref_penalty(lp, lp.clone(), mask).item(), 0.0, places=7)

    def test_positive_and_grows_with_divergence(self):
        from src.core.ppo import kl_ref_penalty
        lp = -torch.rand((1, 6))
        mask = torch.ones((1, 6))
        small = kl_ref_penalty(lp, lp - 0.1, mask).item()
        big = kl_ref_penalty(lp, lp - 0.5, mask).item()
        self.assertGreater(small, 0.0)
        self.assertGreater(big, small)

    def test_gradient_pulls_toward_ref(self):
        from src.core.ppo import kl_ref_penalty
        new = (-torch.rand((1, 4))).requires_grad_(True)
        ref = new.detach() - 0.3          # ref says these tokens LESS likely
        mask = torch.ones((1, 4))
        kl_ref_penalty(new, ref, mask).backward()
        # d k3 / d new = (1 - exp(ref-new)); ref<new => exp<1 => grad>0
        # minimizing pushes new DOWN toward ref
        self.assertTrue(torch.all(new.grad > 0))

    def test_mask_respected(self):
        from src.core.ppo import kl_ref_penalty
        new = -torch.rand((1, 4))
        ref = new - 1.0
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        full = kl_ref_penalty(new, ref, torch.ones_like(mask)).item()
        half = kl_ref_penalty(new, ref, mask).item()
        self.assertAlmostEqual(full, half, places=6)  # same per-token value
