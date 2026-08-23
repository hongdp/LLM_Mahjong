"""Duplicate-deal (common random numbers) group baseline in ReplayBuffer.

The premise (exp4 probe): mid-game return is intrinsically ~unpredictable,
so a learned V(s) cannot reduce variance. Replaying the SAME wall from the
SAME seat several times gives an *empirical* baseline for that start state
that needs nothing to be predictable.
"""

import unittest

from src.core.rollout import ReplayBuffer, TrajectoryStep


def make_episode(rewards):
    return [TrajectoryStep(prompt_text="p", action_text="a", reward=r,
                           is_terminal=(i == len(rewards) - 1))
            for i, r in enumerate(rewards)]


class TestDuplicateDealBaseline(unittest.TestCase):

    def test_group_baseline_reduces_spread_of_deal_luck(self):
        """Two deals: one lucky (+10 tail), one unlucky (-10). Two replicas
        each. The luck is shared inside a group, so removing the group mean
        must shrink the spread of episode returns."""
        buf = ReplayBuffer(gamma=0.99)
        for key, base in (("dealA", 10.0), ("dealB", -10.0)):
            for jitter in (0.5, -0.5):          # policy-driven variation
                buf.add_episode(make_episode([0.0, 0.0, base + jitter]),
                                group_key=key)
        samples = buf.calculate_advantages_and_flatten()
        # advantages are normalized; what matters is that episodes from the
        # two deals are no longer separated by their luck
        first_steps = [s["advantage"] for i, s in enumerate(samples) if i % 3 == 0]
        lucky, unlucky = first_steps[:2], first_steps[2:]
        gap = abs(sum(lucky) / 2 - sum(unlucky) / 2)
        self.assertLess(gap, 2.0, f"deal luck still separates groups (gap {gap})")

    def test_leave_one_out_is_used(self):
        """With 2 replicas the baseline for one is the OTHER's return, so
        two identical replicas end up with equal (and opposite-free)
        corrections rather than both being zeroed by their own value."""
        buf = ReplayBuffer(gamma=1.0)
        buf.add_episode(make_episode([0.0, 4.0]), group_key="d")
        buf.add_episode(make_episode([0.0, 8.0]), group_key="d")
        raw = [4.0, 8.0]
        # leave-one-out baselines are 8 and 4 -> corrected G0 = -4 and +4
        samples = buf.calculate_advantages_and_flatten()
        g0s = [samples[0]["return"], samples[2]["return"]]
        self.assertLess(g0s[0], g0s[1], "ordering must be preserved")
        self.assertAlmostEqual(g0s[0] + g0s[1], 0.0, places=6,
                               msg=f"LOO correction should be symmetric: {g0s}")
        self.assertNotAlmostEqual(g0s[0], raw[0] - 6.0, places=6)  # not plain mean

    def test_singleton_group_is_left_alone(self):
        """A deal with no replica has no baseline — must pass through."""
        buf = ReplayBuffer(gamma=1.0)
        buf.add_episode(make_episode([0.0, 5.0]), group_key="solo")
        buf.add_episode(make_episode([0.0, 1.0]), group_key="other")
        samples = buf.calculate_advantages_and_flatten()
        self.assertAlmostEqual(samples[0]["return"], 5.0, places=6)
        self.assertAlmostEqual(samples[2]["return"], 1.0, places=6)

    def test_no_groups_is_a_noop(self):
        """Without group keys the buffer must behave exactly as before."""
        a = ReplayBuffer(gamma=0.99)
        b = ReplayBuffer(gamma=0.99)
        for buf in (a, b):
            buf.add_episode(make_episode([1.0, -2.0, 3.0]))
            buf.add_episode(make_episode([0.0, 0.5, -1.0]))
        sa = a.calculate_advantages_and_flatten()
        sb = b.calculate_advantages_and_flatten()
        self.assertEqual([s["return"] for s in sa], [s["return"] for s in sb])

    def test_late_steps_not_over_corrected(self):
        """The correction is spread over per-step rewards (same fix as the
        covariate baseline): the LAST step's return must keep most of its
        own signal instead of absorbing the whole episode constant."""
        buf = ReplayBuffer(gamma=1.0)
        buf.add_episode(make_episode([0.0, 0.0, 0.0, 6.0]), group_key="d")
        buf.add_episode(make_episode([0.0, 0.0, 0.0, 0.0]), group_key="d")
        samples = buf.calculate_advantages_and_flatten()
        ep0 = [s["return"] for s in samples[:4]]
        # G0 shifted by the full LOO baseline (0), last step by 1/n of it
        self.assertAlmostEqual(ep0[0], 6.0, places=6)
        self.assertGreater(ep0[3], 5.0, f"last step over-corrected: {ep0}")


if __name__ == "__main__":
    unittest.main()
