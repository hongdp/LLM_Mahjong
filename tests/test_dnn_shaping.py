"""PBRS shaping for the from-scratch DNN arm.

The whole point of a potential-based term is that it adds a dense gradient
WITHOUT changing what the optimal policy is. The telescoping property is
what guarantees that, so it is what these tests pin down.
"""

import unittest

from src.agents.dnn.selfplay import DnnStep, apply_shaping, returns_to_go


def mk(rewards, phis):
    return [DnnStep(planes=None, scalars=None, mask=None, action_idx=0,
                    logprob=0.0, reward=r, phi=p)
            for r, p in zip(rewards, phis)]


class TestShaping(unittest.TestCase):

    def test_telescopes_to_minus_phi0(self):
        """Discounted sum of shaping alone must equal -Phi(s_0) exactly."""
        gamma = 0.995
        phis = [-4.0, -2.0, -1.5, 0.5]
        steps = mk([0.0] * 4, phis)
        apply_shaping(steps, gamma)
        total = returns_to_go(steps, gamma)[0]
        self.assertAlmostEqual(total, -phis[0], places=6,
                               msg=f"shaping did not telescope: {total}")

    def test_adds_to_existing_reward(self):
        """With real rewards present, the shaped return is the original
        return plus the deal constant -Phi(s_0) — not something else."""
        gamma = 0.99
        rewards = [0.0, 0.0, 3.0]
        phis = [-6.0, -3.0, -1.0]
        plain = returns_to_go(mk(rewards, phis), gamma)[0]
        steps = mk(rewards, phis)
        apply_shaping(steps, gamma)
        shaped = returns_to_go(steps, gamma)[0]
        self.assertAlmostEqual(shaped - plain, -phis[0], places=6)

    def test_terminal_potential_is_zero(self):
        """The last step must be credited against Phi=0, otherwise a hand
        left mid-progress would be rewarded for never finishing."""
        steps = mk([0.0], [-5.0])
        apply_shaping(steps, 1.0)
        self.assertAlmostEqual(steps[0].reward, 5.0, places=6)

    def test_single_step_zero_potential_is_noop(self):
        steps = mk([1.0], [0.0])
        apply_shaping(steps, 0.99)
        self.assertAlmostEqual(steps[0].reward, 1.0, places=6)

    def test_potential_prefers_closer_to_tenpai(self):
        """Sanity on the potential itself: a 1-shanten hand must score
        above a 4-shanten hand, or the shaping points the wrong way."""
        import random
        from src.agents.dnn.selfplay import potential
        from src.tasks.mahjong.table import PyMahjongTable
        random.seed(0)
        table = PyMahjongTable(randomize_round=False)
        table.hands[0] = ["1m", "1m", "1m", "2p", "3p", "4p", "5s", "6s",
                          "7s", "9s", "9s", "1z", "2z"]
        near = potential(table, 0)
        table.hands[0] = ["1m", "4m", "7m", "1p", "4p", "7p", "1s", "4s",
                          "7s", "1z", "3z", "5z", "7z"]
        far = potential(table, 0)
        self.assertGreater(near, far, f"near={near} far={far}")


if __name__ == "__main__":
    unittest.main()
