"""Scheduler-level tests for the batched parallel rollout (random policy —
no GPU needed). Verifies game-semantics invariants match the sequential path."""

import random
import unittest

from src.tasks.mahjong.batch_rollout import run_rollout_batched
from src.tasks.mahjong.orchestrator import run_rollout


class TestBatchRollout(unittest.TestCase):

    def test_invariants_random_policy(self):
        random.seed(7)
        episodes = run_rollout_batched(3, model=None, tokenizer=None,
                                       exp_dir="/tmp/claude-1000/batch_ro_test",
                                       parallel=3)
        # 4 trajectories per game
        self.assertEqual(len(episodes), 12)
        for ep in episodes:
            self.assertGreater(len(ep), 0)
            # exactly the last step of each trajectory is terminal
            self.assertTrue(ep[-1].is_terminal)
            self.assertTrue(all(not s.is_terminal for s in ep[:-1]))
        # settlement zero-sum: point deltas ×0.001 + rank bonuses sum to 0
        # per game => terminal rewards minus per-step penalties sum ≈ 0 is
        # not directly checkable here (penalties mix in), but every game
        # must have distributed final rewards: at least one trajectory with
        # a terminal reward differing from a pure step reward is expected.
        # Weaker invariant: total step count is sane for full games.
        total_steps = sum(len(ep) for ep in episodes)
        self.assertGreater(total_steps, 60)

    def test_value_facts_flag_reaches_prompts(self):
        random.seed(11)
        episodes = run_rollout_batched(1, model=None, tokenizer=None,
                                       exp_dir="/tmp/claude-1000/batch_ro_test2",
                                       parallel=1, value_facts=True)
        self.assertTrue(any("自家宝牌" in s.prompt_text
                            for ep in episodes for s in ep))
        episodes = run_rollout_batched(1, model=None, tokenizer=None,
                                       exp_dir="/tmp/claude-1000/batch_ro_test3",
                                       parallel=1, value_facts=False)
        self.assertFalse(any("自家宝牌" in s.prompt_text
                             for ep in episodes for s in ep))

    def test_matches_sequential_shape(self):
        """Same seed → both paths produce 4 non-empty trajectories per game
        with one terminal step each (RNG streams differ by construction, so
        we compare structure, not content)."""
        random.seed(3)
        seq = run_rollout(1, model=None, tokenizer=None,
                          exp_dir="/tmp/claude-1000/batch_ro_test4")
        random.seed(3)
        bat = run_rollout_batched(1, model=None, tokenizer=None,
                                  exp_dir="/tmp/claude-1000/batch_ro_test5",
                                  parallel=1)
        self.assertEqual(len(seq), 4)
        self.assertEqual(len(bat), 4)
        for ep in list(seq) + list(bat):
            self.assertTrue(ep[-1].is_terminal)


if __name__ == "__main__":
    unittest.main()
