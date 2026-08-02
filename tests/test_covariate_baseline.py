"""Tests for the deal-luck control-variate baseline in ReplayBuffer."""

import unittest

import torch

from src.core.rollout import ReplayBuffer, TrajectoryStep
from src.tasks.mahjong.rewards import initial_hand_energy


def mk_episode(total_reward, n_steps=3):
    """Episode whose undiscounted reward sums to total_reward."""
    per = total_reward / n_steps
    return [TrajectoryStep(prompt_text=f"p{i}", action_text="a",
                           reward=per, is_terminal=(i == n_steps - 1))
            for i in range(n_steps)]


def episode_mean_advantages(samples, n_eps, n_steps):
    out = []
    for e in range(n_eps):
        advs = [s["advantage"] for s in samples[e * n_steps:(e + 1) * n_steps]]
        out.append(sum(advs) / len(advs))
    return out


def corr(xs, ys):
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return (x @ y).item() / denom if denom > 1e-12 else 0.0


class TestCovariateBaseline(unittest.TestCase):

    def _build(self, use_cov):
        torch.manual_seed(0)
        buf = ReplayBuffer(gamma=1.0)
        covs = [-8, -6, -4, -2, 0, 2, 4, 6]
        # return = 2*cov + small policy signal alternating ±1
        for i, c in enumerate(covs):
            ep = mk_episode(2 * c + (1 if i % 2 == 0 else -1))
            buf.add_episode(ep, covariate=c if use_cov else None)
        return buf, covs

    def test_baseline_removes_covariate_correlation(self):
        buf, covs = self._build(use_cov=True)
        samples = buf.calculate_advantages_and_flatten()
        means = episode_mean_advantages(samples, len(covs), 3)
        self.assertLess(abs(corr(covs, means)), 0.15,
                        f"cov still correlated: {corr(covs, means):.3f}")
        # policy signal survives: alternating ±1 pattern still separates
        even = sum(means[0::2]) / 4
        odd = sum(means[1::2]) / 4
        self.assertGreater(even, odd)

    def test_no_covariates_is_correlated(self):
        buf, covs = self._build(use_cov=False)
        samples = buf.calculate_advantages_and_flatten()
        means = episode_mean_advantages(samples, len(covs), 3)
        self.assertGreater(abs(corr(covs, means)), 0.9)

    def test_off_path_unchanged(self):
        """Without covariates the numbers are identical to the legacy path."""
        b1, _ = self._build(use_cov=False)
        s1 = b1.calculate_advantages_and_flatten()
        b2 = ReplayBuffer(gamma=1.0)
        for i, c in enumerate([-8, -6, -4, -2, 0, 2, 4, 6]):
            b2.add_episode(mk_episode(2 * c + (1 if i % 2 == 0 else -1)))
        s2 = b2.calculate_advantages_and_flatten()
        for a, b in zip(s1, s2):
            self.assertAlmostEqual(a["advantage"], b["advantage"], places=9)

    def test_initial_hand_energy_orders_hand_quality(self):
        good = ("### 当前状态：\n私有 (Private)： 自风: 东, 点数: 25000, "
                "手牌: 2m 3m 4m 4p 5p 6p 6s 7s 8s 9s 9s 1z 2z 5z, 副露: 无\n")
        bad = ("### 当前状态：\n私有 (Private)： 自风: 东, 点数: 25000, "
               "手牌: 1m 9m 2p 7p 9p 1s 4s 8s 9s 1z 2z 4z 6z 7z, 副露: 无\n")
        self.assertGreater(initial_hand_energy(good), initial_hand_energy(bad))


if __name__ == "__main__":
    unittest.main()
