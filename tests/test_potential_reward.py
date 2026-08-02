"""Tests for MahjongPotentialReward — the energy-consistent PBRS shaping.

Core invariant under test: the discounted sum of shaping terms over any
trajectory telescopes to -Phi(initial hand), a constant of the deal that no
policy can inflate. That is the "intermediate rewards consistent with the
final reward" guarantee.
"""

import unittest

from src.tasks.mahjong.rewards import MahjongPotentialReward


def mk_prompt(hand, melds="无"):
    return (
        "### 当前状态：\n"
        f"私有 (Private)： 自风: 东, 点数: 25000, 手牌: {' '.join(hand)}, 副露: {melds}\n"
        "公共 (Public)：\n"
    )


def mk_discard(tile):
    return f"<think>x</think>\n<action type=\"discard\" tile=\"{tile}\" />"


# A clean 1-shanten 14-tile hand: 234m 456p 678s 99s + 2z 5z floaters-ish
HAND14 = "2m 3m 4m 4p 5p 6p 6s 7s 8s 9s 9s 1z 2z 5z".split()


def rollforward(hand14, discard, draw):
    """Afterstate of `discard` from hand14, then draw `draw` -> next 14-tile hand."""
    h = list(hand14)
    h.remove(discard)
    return h + [draw]


class TestPotentialReward(unittest.TestCase):

    def setUp(self):
        self.rm = MahjongPotentialReward(gamma=0.99, device="cpu")

    def _total(self, rewards):
        return sum(self.rm.gamma ** i * r.item() for i, r in enumerate(rewards))

    def test_telescoping_to_initial_energy(self):
        """Discounted shaping sum == -Phi_pre(initial hand), exactly."""
        h0 = HAND14
        h1 = rollforward(h0, "5z", "3p")
        h2 = rollforward(h1, "2z", "1m")
        prompts = [mk_prompt(h) for h in (h0, h1, h2)]
        responses = [mk_discard("5z"), mk_discard("2z"), mk_discard("1z")]
        rewards = self.rm.compute_reward(prompts, responses)
        psi_pre = self.rm._pre_energy(h0, 0)
        self.assertIsNotNone(psi_pre)
        self.assertAlmostEqual(self._total(rewards), -psi_pre, places=6)

    def test_policy_independent_total(self):
        """Two different (legal) action sequences from the same deal produce
        the SAME discounted shaping total — farming is impossible."""
        h0 = HAND14
        # Trajectory A: good discards (floaters first)
        a1 = rollforward(h0, "5z", "3p")
        prompts_a = [mk_prompt(h0), mk_prompt(a1)]
        responses_a = [mk_discard("5z"), mk_discard("2z")]
        # Trajectory B: hand-wrecking discards (break the 234m run)
        b1 = rollforward(h0, "3m", "3p")
        prompts_b = [mk_prompt(h0), mk_prompt(b1)]
        responses_b = [mk_discard("3m"), mk_discard("2m")]
        tot_a = self._total(self.rm.compute_reward(prompts_a, responses_a))
        tot_b = self._total(self.rm.compute_reward(prompts_b, responses_b))
        self.assertAlmostEqual(tot_a, tot_b, places=6)

    def test_bad_discard_scores_lower_at_the_step(self):
        """Immediate shaping still discriminates: wrecking the hand at step 0
        yields a strictly lower step-0 reward than the efficient discard."""
        h0 = HAND14
        good = self.rm.compute_reward([mk_prompt(h0)], [mk_discard("5z")])
        bad = self.rm.compute_reward([mk_prompt(h0)], [mk_discard("3m")])
        # single-step episodes: F_0 = -psi_pre + 0 (terminal) => equal.
        # Use 2-step episodes so step 0 carries gamma*psi_0 - psi_pre.
        h1g = rollforward(h0, "5z", "3p")
        h1b = rollforward(h0, "3m", "3p")
        good = self.rm.compute_reward([mk_prompt(h0), mk_prompt(h1g)],
                                      [mk_discard("5z"), mk_discard("2z")])
        bad = self.rm.compute_reward([mk_prompt(h0), mk_prompt(h1b)],
                                     [mk_discard("3m"), mk_discard("2m")])
        self.assertGreater(good[0].item(), bad[0].item())

    def test_format_violation_constraint(self):
        """A response with no action tag gets the -10 constraint on top of
        shaping; the shaping part still telescopes."""
        h0 = HAND14
        rewards = self.rm.compute_reward([mk_prompt(h0)], ["3s 4s 6s no tag"])
        psi_pre = self.rm._pre_energy(h0, 0)
        # single terminal step: shaping = -psi_pre; constraint = -10
        self.assertAlmostEqual(rewards[0].item(), -psi_pre - 10.0, places=5)

    def test_ghost_tile_constraint(self):
        """Discarding a tile not in hand adds -5 and keeps hand unchanged."""
        h0 = HAND14
        h1 = h0[:13] + ["1p"]  # arbitrary continuation
        rewards = self.rm.compute_reward(
            [mk_prompt(h0), mk_prompt(h1)],
            [mk_discard("9p"), mk_discard("1p")])  # 9p not in hand
        base = self.rm.compute_reward(
            [mk_prompt(h0), mk_prompt(h1)],
            [f"<think>x</think>\n<action type=\"skip\" />", mk_discard("1p")])
        # ghost discard = same afterstate as skip (hand unchanged) - 5.0
        self.assertAlmostEqual(rewards[0].item(), base[0].item() - 5.0, places=5)

    def test_ukeire_term_never_reorders_shanten(self):
        """C_UKEIRE * 34 must stay strictly below C_SHANTEN."""
        self.assertLess(MahjongPotentialReward.C_UKEIRE * 34,
                        MahjongPotentialReward.C_SHANTEN)


if __name__ == "__main__":
    unittest.main()
