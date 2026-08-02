"""Unit tests for the mahjong table engine.

Run:  python -m unittest tests.test_engine -v
"""

import random
import unittest

from src.tasks.mahjong.table import PyMahjongTable
from src.tasks.mahjong.orchestrator import run_rollout, _extract_action


def rig(table, pid, tiles, drawn=None, melds=None):
    """Overwrite a player's hand for scenario testing."""
    table.hands[pid] = sorted(tiles, key=table_sort)
    table.last_drawn[pid] = drawn
    if melds is not None:
        table.melds[pid] = melds


def table_sort(t):
    from src.tasks.mahjong.table import sort_key
    return sort_key(t)


TANYAO_14 = ['2m', '3m', '4m', '3p', '4p', '5p', '5p', '6p', '7p',
             '6s', '7s', '8s', '8s', '8s']
KANCHAN_TENPAI_13 = ['2p', '4p', '2m', '3m', '4m', '5m', '6m', '7m',
                     '3s', '4s', '5s', '8s', '8s']  # waits on 3p, tanyao


class TestSetup(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_deal(self):
        self.assertEqual(len(self.t.hands[0]), 14)
        for pid in (1, 2, 3):
            self.assertEqual(len(self.t.hands[pid]), 13)
        # 136 - 14 dead - 53 dealt
        self.assertEqual(len(self.t.wall), 69)
        self.assertEqual(len(self.t.dora_indicators), 1)
        self.assertEqual(sum(self.t.points) + self.t.kyotaku, 100000)


class TestWinValidation(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_tsumo_offered_and_settles(self):
        rig(self.t, 0, TANYAO_14, drawn='5p')
        actions = self.t.get_legal_actions(0)
        self.assertIn('<action type="tsumo" />', actions)
        _, _, done, _ = self.t.step(0, '<action type="tsumo" />')
        self.assertTrue(done)
        self.assertTrue(self.t.finished)
        self.assertGreater(self.t.points[0], 25000)
        self.assertEqual(sum(self.t.points) + self.t.kyotaku, 100000)
        self.assertIsNotNone(self.t.final_rewards)
        self.assertGreater(self.t.final_rewards[0], 0)
        self.assertLess(min(self.t.final_rewards[1:]), 0)

    def test_false_tsumo_penalized(self):
        # Non-winning hand: tsumo must not be offered and must be punished.
        junk = ['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z',
                '4z', '5z', '6z', '7z', '2m']
        rig(self.t, 0, junk, drawn='2m')
        actions = self.t.get_legal_actions(0)
        self.assertNotIn('<action type="tsumo" />', actions)
        _, rewards, done, info = self.t.step(0, '<action type="tsumo" />')
        self.assertFalse(done)
        self.assertEqual(rewards[0], self.t.ILLEGAL_PENALTY)
        self.assertTrue(info['discarded'])  # forced discard keeps game going

    def test_open_hand_without_yaku_cannot_ron(self):
        rig(self.t, 1,
            ['2p', '3p', '4p', '4s', '5s', '6s', '6s', '7s', '2z', '2z'],
            melds=[{"type": "pon", "tiles": ['5m'] * 3, "opened": True}])
        self.t.last_discard = '8s'
        self.t.last_discarder = 0
        self.assertFalse(self.t._can_ron(1, '8s'))
        actions = self.t.get_interrupt_actions(1)
        self.assertNotIn('<action type="ron" />', actions)

    def test_closed_tanyao_can_ron_and_furiten_blocks(self):
        rig(self.t, 1, KANCHAN_TENPAI_13)
        self.t.last_discard = '3p'
        self.t.last_discarder = 0
        self.assertTrue(self.t._can_ron(1, '3p'))
        # Furiten: same wait in own river blocks ron entirely.
        self.t.discards[1] = ['3p']
        self.assertFalse(self.t._can_ron(1, '3p'))

    def test_ron_settlement_charges_discarder_only(self):
        rig(self.t, 1, KANCHAN_TENPAI_13)
        self.t.last_discard = '3p'
        self.t.last_discarder = 0
        _, _, done, info = self.t.step_interrupt(1, '<action type="ron" />')
        self.assertTrue(done and info['interrupt'])
        self.assertLess(self.t.points[0], 25000)
        self.assertEqual(self.t.points[2], 25000)
        self.assertEqual(self.t.points[3], 25000)
        # Deal-in gets the extra houjuu penalty on top of the point loss.
        self.assertLess(self.t.final_rewards[0],
                        (self.t.points[0] - 25000) * self.t.REWARD_SCALE + 0.001)


class TestRiichi(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_riichi_only_for_tenpai_discards(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        actions = self.t.get_legal_actions(0)
        self.assertIn('<action type="riichi" tile="1z" />', actions)
        # Discarding a core tile breaks tenpai — no riichi option for it.
        self.assertNotIn('<action type="riichi" tile="2p" />', actions)

    def test_riichi_deposit_and_lock(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        self.t.step(0, '<action type="riichi" tile="1z" />')
        self.assertTrue(self.t.riichi[0])
        self.assertEqual(self.t.points[0], 24000)
        self.assertEqual(self.t.kyotaku, 1000)
        self.assertIn('1z*', self.t.discards[0])
        # Locked: next draw allows only tsumogiri (plus tsumo if winning).
        self.t.hands[0].append('9m')
        self.t.hands[0].sort(key=table_sort)
        self.t.last_drawn[0] = '9m'
        actions = self.t.get_legal_actions(0)
        self.assertEqual(actions, ['<action type="discard" tile="9m" />'])

    def test_no_riichi_for_open_hand(self):
        rig(self.t, 0,
            ['2p', '4p', '2m', '3m', '4m', '3s', '4s', '5s', '8s', '8s', '1z'],
            drawn='1z',
            melds=[{"type": "pon", "tiles": ['5m'] * 3, "opened": True}])
        actions = self.t.get_legal_actions(0)
        self.assertFalse(any('riichi' in a for a in actions))


class TestInterrupts(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_claims_ignore_model_tile_attr(self):
        rig(self.t, 1, ['5m', '5m', '1p', '2p', '3p', '4p', '5p', '6p',
                        '7p', '8p', '9p', '1s', '2s'])
        self.t.last_discard = '5m'
        self.t.last_discarder = 0
        self.t.discards[0] = ['5m']
        # Model lies about the tile — the engine must claim the real discard.
        _, _, _, info = self.t.step_interrupt(
            1, '<action type="pon" tile="9z" />')
        self.assertTrue(info['interrupt'])
        self.assertEqual(self.t.melds[1][0]['tiles'], ['5m', '5m', '5m'])
        self.assertNotIn('5m', self.t.hands[1])

    def test_ron_with_fake_tile_attr_rejected(self):
        # Hand waits on 3p but the actual discard is 9z: ron must fail
        # even if the model claims tile="3p".
        rig(self.t, 1, KANCHAN_TENPAI_13)
        self.t.last_discard = '1z'
        self.t.last_discarder = 0
        _, rewards, done, info = self.t.step_interrupt(
            1, '<action type="ron" tile="3p" />')
        self.assertFalse(done)
        self.assertFalse(info['interrupt'])
        self.assertEqual(rewards[1], self.t.ILLEGAL_PENALTY)

    def test_chi_with_pair_selection(self):
        rig(self.t, 1, ['3p', '4p', '6p', '7p', '1m', '2m', '3m', '5s',
                        '6s', '7s', '9m', '9m', '1z'])
        self.t.last_discard = '5p'
        self.t.last_discarder = 0
        self.t.discards[0] = ['5p']
        self.t.turn = 0
        options = self.t.get_interrupt_actions(1)
        chi_opts = [a for a in options if 'chi' in a]
        self.assertEqual(len(chi_opts), 3)  # 3p4p / 4p6p / 6p7p
        self.t.step_interrupt(
            1, '<action type="chi" tile="5p" with="6p 7p" />')
        self.assertNotIn('6p', self.t.hands[1])
        self.assertNotIn('7p', self.t.hands[1])
        self.assertIn('3p', self.t.hands[1])
        self.assertEqual(self.t.melds[1][0]['tiles'], ['5p', '6p', '7p'])

    def test_meld_cap(self):
        melds = [{"type": "pon", "tiles": [f'{i}m'] * 3, "opened": True}
                 for i in (1, 2, 3, 4)]
        rig(self.t, 1, ['5m', '5m'], melds=melds)
        self.t.last_discard = '5m'
        self.t.last_discarder = 0
        actions = self.t.get_interrupt_actions(1)
        self.assertFalse(any('pon' in a for a in actions))


class TestRyuukyoku(unittest.TestCase):
    def test_tenpai_payments(self):
        random.seed(7)
        t = PyMahjongTable()
        # Scattered simples + honor singles: genuinely noten (NOT kokushi
        # tenpai — thirteen distinct orphans would count as tenpai).
        junk = ['2m', '5m', '8m', '2p', '5p', '8p', '2s', '5s', '8s',
                '1z', '2z', '3z', '4z']
        rig(t, 0, KANCHAN_TENPAI_13)
        for pid in (1, 2, 3):
            rig(t, pid, list(junk))
        t.wall = []
        _, done = t.advance_turn()
        self.assertTrue(done)
        self.assertEqual(t.points[0], 28000)
        for pid in (1, 2, 3):
            self.assertEqual(t.points[pid], 24000)
        self.assertEqual(sum(t.points) + t.kyotaku, 100000)


class TestOrchestrator(unittest.TestCase):
    def test_think_embedded_actions_not_extracted(self):
        raw = ('<think>\n可以考虑 <action type="ron" />，但风险太大，'
               '还是跳过。\n</think>\n<action type="skip" />')
        self.assertEqual(_extract_action(raw), '<action type="skip" />')
        only_think = '<think>\n执行 <action type="ron" />。\n</think>'
        self.assertIsNone(_extract_action(only_think))

    def test_random_rollout_terminates_and_distributes(self):
        random.seed(11)
        episodes = run_rollout(
            3, exp_dir="/tmp/claude-1000/-home-hongdp-Workspace-LLM-Mahjong/"
                       "2451e9eb-8241-4b3b-85ea-3e34222dad3a/scratchpad/"
                       "engine_test_rollout")
        self.assertEqual(len(episodes), 12)  # 4 players x 3 games
        for ep in episodes:
            self.assertTrue(ep[-1].is_terminal)


if __name__ == "__main__":
    unittest.main()
