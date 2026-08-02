"""Tests for the value bundle: dora mapping, dora energy term (PBRS-safe),
value-facts prompt line, and the value-aware teacher tie-break."""

import unittest

from src.tasks.mahjong.shanten import dora_from_indicator
from src.tasks.mahjong.rewards import MahjongPotentialReward, REWARD_MODELS
from src.tasks.mahjong.table import PyMahjongTable

import scripts.generate_sft_data as gen


def mk_prompt(hand, melds="无", dora_ind="1p"):
    return (
        "### 当前状态：\n"
        f"场况 (Global)： 场风: 东, 局数: 东1局, 宝牌指示牌: {dora_ind}, "
        "供托: 0, 剩余牌数: 41\n"
        f"私有 (Private)： 自风: 东, 点数: 25000, 手牌: {' '.join(hand)}, 副露: {melds}\n"
        "公共 (Public)：\n"
    )


def mk_discard(tile):
    return f"<think>x</think>\n<action type=\"discard\" tile=\"{tile}\" />"


HAND14 = "2m 3m 4m 4p 5p 6p 6s 7s 8s 9s 9s 1z 2z 5z".split()


class TestDoraMapping(unittest.TestCase):
    def test_wraps(self):
        cases = {"3p": "4p", "9m": "1m", "9s": "1s",
                 "1z": "2z", "4z": "1z",          # winds cycle
                 "5z": "6z", "7z": "5z"}          # dragons cycle
        for ind, dora in cases.items():
            self.assertEqual(dora_from_indicator(ind), dora, ind)


class TestDoraEnergy(unittest.TestCase):
    def setUp(self):
        self.rm = REWARD_MODELS["potential_value"](device="cpu")

    def test_registry_weight(self):
        self.assertGreater(self.rm.dora_weight, 0)
        # dora term must stay below one shanten step even at 4 copies held
        self.assertLess(self.rm.dora_weight * 4, self.rm.C_SHANTEN)

    def test_keeping_dora_scores_higher(self):
        """Same structural discard candidates; afterstate keeping the dora
        has exactly dora_weight more energy."""
        hand13 = list(HAND14)
        # indicator 1z -> dora 2z; hand holds 2z and 5z floaters
        keep = [t for t in hand13 if t != "5z"]   # discarded 5z, kept 2z
        drop = [t for t in hand13 if t != "2z"]   # discarded the dora
        e_keep = self.rm._energy(keep, 0, dora_tiles=("2z",))
        e_drop = self.rm._energy(drop, 0, dora_tiles=("2z",))
        self.assertAlmostEqual(e_keep - e_drop, self.rm.dora_weight, places=6)

    def test_telescoping_still_exact_with_dora(self):
        h0 = HAND14
        h1 = [t for t in h0 if t != "5z"] + ["3p"]
        prompts = [mk_prompt(h, dora_ind="1z") for h in (h0, h1)]
        responses = [mk_discard("5z"), mk_discard("1z")]
        rewards = self.rm.compute_reward(prompts, responses)
        total = sum(self.rm.gamma ** i * r.item()
                    for i, r in enumerate(rewards))
        psi_pre = self.rm._pre_energy(h0, 0, dora_tiles=("2z",))
        self.assertAlmostEqual(total, -psi_pre, places=6)


class TestValueFactsPrompt(unittest.TestCase):
    def test_line_present_only_when_enabled(self):
        t_on = PyMahjongTable(value_facts=True)
        t_off = PyMahjongTable()
        self.assertIn("自家宝牌", t_on._format_state(0))
        self.assertNotIn("自家宝牌", t_off._format_state(0))

    def test_lists_held_dora(self):
        t = PyMahjongTable(value_facts=True)
        t.dora_indicators = ["1m"]           # dora = 2m
        t.hands[0] = "2m 2m 5p 6p 7p 3s 4s 5s 6s 7s 8s 1z 1z".split()
        state = t._format_state(0)
        self.assertIn("自家宝牌: 2m 2m", state)
        t.hands[0] = "5p 6p 7p 3s 4s 5s 6s 7s 8s 9s 9s 1z 1z".split()
        self.assertIn("自家宝牌: 无", t._format_state(0))


class TestTeacherTieBreak(unittest.TestCase):
    def test_prefers_keeping_dora_and_says_so(self):
        t = PyMahjongTable()
        t.dora_indicators = ["4z"]           # dora = 1z (default pick w/o value)
        t.melds[0] = []
        hand = list(HAND14)                  # floaters 1z 2z 5z tie on (sh,uk)
        old_flag = gen.VALUE_AWARE
        try:
            gen.VALUE_AWARE = False
            tile_plain, _ = gen.discard_decision(t, 0, hand)
            self.assertEqual(tile_plain, "1z")   # 34-order tie-break → dora
            gen.VALUE_AWARE = True
            tile_val, think = gen.discard_decision(t, 0, hand)
            self.assertNotEqual(tile_val, "1z")  # dora retained
            self.assertIn("保留宝牌1z", think)
        finally:
            gen.VALUE_AWARE = old_flag


if __name__ == "__main__":
    unittest.main()
