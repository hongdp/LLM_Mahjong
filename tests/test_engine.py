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
        # Furiten: same wait in own river blocks ron entirely (RCR 3.13.1).
        self.t.furiten_river[1] = ['3p']
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
        # RCR 3.12: the stick is only paid once the declaration tile
        # survives the interrupt window.
        self.assertEqual(self.t.points[0], 25000)
        self.assertEqual(self.t.riichi_pending, 0)
        self.t.advance_turn()
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


YAKUHAI_TENPAI_13 = ['5z', '5z', '2p', '3p', '4p', '5p', '6p', '7p',
                     '2s', '3s', '4s', '9s', '9s']   # shanpon 5z / 9s


class TestWallAndKan(unittest.TestCase):
    """RCR 3.7.1 / 3.7.2 / 3.14."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_kan_moves_live_wall_tail_into_dead_wall(self):
        rig(self.t, 0, ['1m'] * 4 + ['2p', '3p', '4p', '5p', '6p', '7p',
                                     '8p', '9p', '1s', '2s'], drawn='1m')
        wall_before = len(self.t.wall)
        self.t.step(0, '<action type="kan" tile="1m" />')
        # The replacement tile comes from the dead wall and the live wall
        # loses its tail, so the round still yields exactly 70 draws.
        self.assertEqual(len(self.t.wall), wall_before - 1)
        self.assertEqual(self.t._rinshan_idx, 1)
        self.assertEqual(len(self.t.dora_indicators), 2)
        self.assertEqual(len(self.t.ura_indicators), 2)
        self.assertEqual(len(self.t.hands[0]), 11)   # 14 - 4 kan + 1 rinshan

    def test_four_kans_per_round_cap(self):
        for pid, tile in enumerate(['1m', '2m', '3m', '4m']):
            rig(self.t, pid, [tile] * 4 + ['2p', '3p', '4p', '5p', '6p',
                                           '7p', '8p', '9p', '1s', '2s'],
                drawn=tile)
            self.assertTrue(self.t._can_ankan(pid, tile))
            self.t.step(pid, f'<action type="kan" tile="{tile}" />')
        self.assertEqual(self.t.kan_count, 4)
        # RCR 3.7.2: a fifth kan is impossible for anyone.
        rig(self.t, 0, ['9s'] * 4 + ['2p', '3p', '4p', '5p', '6p', '7p'],
            drawn='9s')
        self.assertFalse(self.t._can_ankan(0, '9s'))
        self.assertFalse(any('kan' in a for a in self.t.get_legal_actions(0)))
        # Dora indicators cap at 5 (1 initial + 4 kan).
        self.assertEqual(len(self.t.dora_indicators), 5)

    def test_no_kan_on_haitei_and_no_call_on_houtei(self):
        self.t.wall = []
        rig(self.t, 0, ['1m'] * 4 + ['2p', '3p', '4p', '5p', '6p', '7p',
                                     '8p', '9p', '1s', '2s'], drawn='1m')
        self.assertFalse(any('kan' in a for a in self.t.get_legal_actions(0)))
        # The houtei discard may only be ronned (RCR 3.14).
        self.t.last_discard, self.t.last_discarder = '9p', 0
        rig(self.t, 1, ['9p', '9p'] + ['1s'] * 11)
        opts = self.t.get_interrupt_actions(1)
        self.assertFalse(any('pon' in a or 'chi' in a or 'kan' in a for a in opts))


class TestFuriten(unittest.TestCase):
    """RCR 3.13."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_furiten_survives_the_tile_being_called(self):
        rig(self.t, 2, KANCHAN_TENPAI_13)
        self.t.furiten_river[2] = ['3p']
        self.t.discards[2] = ['3p']
        self.assertTrue(self.t._is_furiten(2))
        # Player 3 pons the 3p out of player 2's river.
        self.t.last_discard, self.t.last_discarder = '3p', 2
        rig(self.t, 3, ['3p', '3p'] + ['1s'] * 11)
        self.t.step_interrupt(3, '<action type="pon" tile="3p" />')
        self.assertEqual(self.t.discards[2], [])          # visually gone
        self.assertEqual(self.t.furiten_river[2], ['3p'])  # never forgotten
        self.assertTrue(self.t._is_furiten(2))
        self.assertFalse(self.t._can_ron(2, '3p'))

    def test_passed_ron_causes_same_turn_furiten(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['3p'], drawn='3p')
        rig(self.t, 2, KANCHAN_TENPAI_13)
        self.t.step(0, '<action type="discard" tile="3p" />')
        self.assertIn(2, self.t._ron_chance)
        self.t.advance_turn()          # nobody ronned
        self.assertTrue(self.t.temp_furiten[2])
        self.assertFalse(self.t._can_ron(2, '3p'))
        # ... and it lifts once player 2 draws again.
        self.t.advance_turn()
        self.assertEqual(self.t.turn, 2)
        self.assertFalse(self.t.temp_furiten[2])

    def test_riichi_player_passing_a_ron_is_permanently_furiten(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['3p'], drawn='3p')
        rig(self.t, 2, KANCHAN_TENPAI_13)
        self.t.riichi[2] = True
        self.t.step(0, '<action type="discard" tile="3p" />')
        self.t.advance_turn()
        self.assertTrue(self.t.perm_furiten[2])
        for _ in range(4):             # drawing never clears it again
            self.t.advance_turn()
        self.assertTrue(self.t.perm_furiten[2])
        self.assertFalse(self.t._can_ron(2, '3p'))


class TestRiichiRules(unittest.TestCase):
    """RCR 3.12."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_no_riichi_with_fewer_than_four_tiles_left(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        self.t.wall = self.t.wall[:3]
        self.assertFalse(any('riichi' in a for a in self.t.get_legal_actions(0)))
        _, rewards, _, _ = self.t.step(0, '<action type="riichi" tile="1z" />')
        self.assertEqual(rewards[0], self.t.ILLEGAL_PENALTY)
        self.assertFalse(self.t.riichi[0])

    def test_riichi_void_when_declaration_tile_is_ronned(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['5z'], drawn='5z')
        rig(self.t, 1, YAKUHAI_TENPAI_13)
        self.t.step(0, '<action type="riichi" tile="5z" />')
        self.assertEqual(self.t.points[0], 25000)     # stick not paid yet
        self.t.step_interrupt(1, '<action type="ron" />')
        self.assertTrue(self.t.finished)
        self.assertFalse(self.t.riichi[0])            # riichi never happened
        self.assertEqual(self.t.kyotaku, 0)
        self.assertNotIn('5z*', self.t.discards[0])
        # Player 0 only paid the hand value, never the 1000 stick.
        self.assertEqual(self.t.points[0] + self.t.points[1], 50000)
        self.assertEqual(sum(self.t.points), 100000)

    def test_riichi_stick_paid_once_the_tile_survives(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        self.t.step(0, '<action type="riichi" tile="1z" />')
        self.t.advance_turn()
        self.assertEqual(self.t.points[0], 24000)
        self.assertEqual(self.t.kyotaku, 1000)
        self.assertIsNone(self.t.riichi_pending)

    def test_riichi_ankan_allowed_only_when_the_wait_is_unchanged(self):
        # Honor ankou + tanki wait: kan changes nothing -> allowed.
        rig(self.t, 0, ['1z'] * 4 + ['2m', '3m', '4m', '5m', '6m', '7m',
                                     '2p', '3p', '4p', '9s'], drawn='1z')
        self.t.riichi[0] = True
        self.assertTrue(self.t._can_ankan(0, '1z'))
        self.assertIn('<action type="kan" tile="1z" />',
                      self.t.get_legal_actions(0))
        # RCR 3.12 example: 2333s waits 1s/2s/4s, the kan would narrow it.
        t2 = PyMahjongTable()
        rig(t2, 0, ['2s', '3s', '3s', '3s', '3s', '2m', '3m', '4m',
                    '5m', '6m', '7m', '2p', '3p', '4p'], drawn='3s')
        t2.riichi[0] = True
        self.assertFalse(t2._can_ankan(0, '3s'))

    def test_ippatsu_and_ura_dora_are_scored(self):
        rig(self.t, 0, TANYAO_14, drawn='5p')
        self.t.discard_count = [1, 1, 1, 1]   # mid-game, so not tenhou
        self.t.riichi[0] = True
        self.t.ippatsu[0] = True
        self.t.dora_indicators = ['1z']      # dora 2z: not in hand
        self.t.ura_indicators = ['1z']
        base = self.t._win_result(0, '5p', is_tsumo=True)
        names = {str(y) for y in base.yaku}
        self.assertIn('Riichi', names)
        self.assertIn('Ippatsu', names)
        self.assertIn('Menzen Tsumo', names)
        # An ura indicator pointing at a held tile adds han (RCR 3.12).
        self.t.ura_indicators = ['1m']       # ura dora 2m, one in hand
        with_ura = self.t._win_result(0, '5p', is_tsumo=True)
        self.assertGreater(with_ura.han, base.han)
        # A non-riichi hand never sees the ura indicators (RCR 3.12).
        self.t.riichi[0] = False
        self.t.ippatsu[0] = False
        plain = self.t._win_result(0, '5p', is_tsumo=True)
        self.assertNotIn('Riichi', {str(y) for y in plain.yaku})
        self.assertEqual(plain.han, with_ura.han - 3)   # riichi + ippatsu + ura

    def test_double_riichi_flag_on_first_discard(self):
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        self.t.step(0, '<action type="riichi" tile="1z" />')
        self.assertTrue(self.t.daburu[0])
        rig(self.t, 1, KANCHAN_TENPAI_13 + ['1z'], drawn='1z')
        self.t.discard_count[1] = 1          # not their first discard
        self.t.step(1, '<action type="riichi" tile="1z" />')
        self.assertFalse(self.t.daburu[1])


class TestKuikae(unittest.TestCase):
    """RCR 3.8."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_chi_forbids_the_called_tile_and_the_swap_end(self):
        # 3m4m is a ryanmen on 2m/5m: after calling 5m, discarding 2m
        # would be the swap (RCR 3.8 first case).
        rig(self.t, 1, ['3m', '4m', '1p', '2p', '3p', '5p', '6p', '7p',
                        '9s', '9s', '9s', '2s', '2m'])
        self.t.last_discard, self.t.last_discarder = '5m', 0
        self.t.discards[0] = ['5m']
        self.t.step_interrupt(1, '<action type="chi" tile="5m" with="3m 4m" />')
        actions = self.t.get_legal_actions(1)
        self.assertFalse(any('tile="5m"' in a for a in actions))  # same tile
        self.assertFalse(any('tile="2m"' in a for a in actions))  # swap end
        self.assertTrue(any('tile="2s"' in a for a in actions))
        _, rewards, _, _ = self.t.step(1, '<action type="discard" tile="2m" />')
        self.assertEqual(rewards[1], self.t.ILLEGAL_PENALTY)
        # The restriction is lifted after that discard.
        self.assertIsNone(self.t.kuikae)

    def test_pon_forbids_the_fourth_identical_tile(self):
        rig(self.t, 1, ['7z', '7z', '7z', '1p', '2p', '3p', '5p', '6p',
                        '7p', '9s', '9s', '9s', '2s'])
        self.t.last_discard, self.t.last_discarder = '7z', 0
        self.t.discards[0] = ['7z']
        self.t.step_interrupt(1, '<action type="pon" tile="7z" />')
        actions = self.t.get_legal_actions(1)
        self.assertFalse(any('tile="7z"' in a for a in actions))


class TestMultipleRonAndChankan(unittest.TestCase):
    """RCR 3.11 / 4.2.1.12."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def test_double_ron_pays_both_winners(self):
        rig(self.t, 1, list(YAKUHAI_TENPAI_13))
        rig(self.t, 2, list(YAKUHAI_TENPAI_13))
        self.t.last_discard, self.t.last_discarder = '5z', 0
        self.t.points[3] -= 1000       # player 3 had declared riichi earlier
        self.t.kyotaku = 1000
        _, rewards, done, info = self.t.step_ron([1, 2])
        self.assertTrue(done)
        self.assertEqual(info["winners"], [1, 2])
        self.assertGreater(self.t.points[1], 25000)
        self.assertGreater(self.t.points[2], 25000)
        self.assertEqual(self.t.points[3], 24000)      # bystander untouched
        self.assertLess(self.t.points[0], 25000)       # discarder pays both
        # Riichi sticks go to the winner closest to the discarder.
        self.assertEqual(self.t.kyotaku, 0)
        self.assertEqual(sum(self.t.points), 100000)
        self.assertIn('双响', self.t.result_summary)

    def test_added_kan_opens_a_chankan_window(self):
        self.t.melds[0] = [{"type": "pon", "tiles": ['3z'] * 3, "opened": True}]
        rig(self.t, 0, ['3z', '2p', '3p', '4p', '5p', '6p', '7p',
                        '2s', '3s', '4s', '9s'], drawn='3z')
        rig(self.t, 1, ['3z', '3z', '2p', '3p', '4p', '5p', '6p', '7p',
                        '2s', '3s', '4s', '9s', '9s'])
        _, _, done, info = self.t.step(0, '<action type="kan" tile="3z" />')
        self.assertFalse(done)
        self.assertEqual(info.get('chankan'), '3z')
        self.assertIsNotNone(self.t.pending_kan)
        # Nothing has been mutated yet: still a pon, no new dora.
        self.assertEqual(self.t.melds[0][0]['type'], 'pon')
        self.assertEqual(len(self.t.dora_indicators), 1)
        self.assertIn('<action type="ron" />',
                      self.t.get_interrupt_actions(1))
        # The kan player is never asked.
        self.assertEqual(self.t.get_interrupt_actions(0),
                         ['<action type="skip" />'])
        _, _, done, info = self.t.step_ron([1])
        self.assertTrue(done)
        self.assertIn('Chankan', self.t.result_summary)
        self.assertIn('抢杠', self.t.result_summary)
        # The robbed kan never completed, so no extra dora was flipped.
        self.assertEqual(len(self.t.dora_indicators), 1)

    def test_unrobbed_added_kan_completes(self):
        self.t.melds[0] = [{"type": "pon", "tiles": ['3z'] * 3, "opened": True}]
        rig(self.t, 0, ['3z', '2p', '3p', '4p', '5p', '6p', '7p',
                        '2s', '3s', '4s', '9s'], drawn='3z')
        self.t.step(0, '<action type="kan" tile="3z" />')
        self.t.resolve_pending_kan()
        self.assertIsNone(self.t.pending_kan)
        self.assertEqual(self.t.melds[0][0]['type'], 'shouminkan')
        self.assertEqual(self.t.kan_count, 1)
        # Majsoul timing: the added-kan dora is turned over after the discard
        self.assertEqual(len(self.t.dora_indicators), 1)
        self.assertEqual(self.t.pending_dora_reveal, 1)
        self.assertTrue(self.t.rinshan[0])
        self.t.step(0, f'<action type="discard" tile="{self.t.last_drawn[0]}" />')
        self.assertEqual(len(self.t.dora_indicators), 2)
        self.assertEqual(self.t.pending_dora_reveal, 0)


class TestPao(unittest.TestCase):
    """RCR 4.2.5.10."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()

    def _daisangen_setup(self):
        self.t.melds[1] = [
            {"type": "pon", "tiles": ['5z'] * 3, "opened": True},
            {"type": "pon", "tiles": ['6z'] * 3, "opened": True},
        ]
        rig(self.t, 1, ['7z', '7z', '2m', '3m', '4m', '1p'])
        self.t.last_discard, self.t.last_discarder = '7z', 0
        self.t.discards[0] = ['7z']

    def test_third_dragon_pon_records_liability(self):
        self._daisangen_setup()
        self.t.step_interrupt(1, '<action type="pon" tile="7z" />')
        self.assertEqual(self.t.pao.get(1), 0)

    def test_liable_player_pays_the_whole_yakuman_on_tsumo(self):
        self._daisangen_setup()
        self.t.step_interrupt(1, '<action type="pon" tile="7z" />')
        rig(self.t, 1, ['2m', '3m', '4m', '1p', '1p'], drawn='1p')
        result = self.t._win_result(1, '1p', is_tsumo=True)
        self.assertIn('Daisangen', {str(y) for y in result.yaku})
        self.t._settle_tsumo(1, result)
        self.assertEqual(self.t.points[2], 25000)   # innocent bystanders
        self.assertEqual(self.t.points[3], 25000)
        self.assertEqual(self.t.points[0], 25000 - 32000)   # liable pays all
        self.assertIn('包牌', self.t.result_summary)
        self.assertEqual(sum(self.t.points), 100000)

    def test_liability_is_split_with_the_discarder_on_ron(self):
        self._daisangen_setup()
        self.t.step_interrupt(1, '<action type="pon" tile="7z" />')
        rig(self.t, 1, ['2m', '3m', '4m', '1p'])
        result = self.t._win_result(1, '1p', is_tsumo=False)
        self.t._settle_ron([(1, result)], discarder=3)
        self.assertEqual(self.t.points[0], 25000 - 16000)   # liable: half
        self.assertEqual(self.t.points[3], 25000 - 16000)   # discarder: half
        self.assertEqual(self.t.points[2], 25000)
        self.assertEqual(sum(self.t.points), 100000)


class TestRoundRandomization(unittest.TestCase):
    def test_round_wind_and_dealer_vary(self):
        random.seed(3)
        dealers, winds = set(), set()
        for _ in range(40):
            t = PyMahjongTable(randomize_round=True)
            dealers.add(t.dealer)
            winds.add(t.round_wind_idx)
            self.assertEqual(t.turn, t.dealer)
            self.assertEqual(len(t.hands[t.dealer]), 14)
            state = t._format_state(t.dealer)
            self.assertIn(f"{t.round_number}局", state)
            self.assertIn("自风: 东", state)      # the dealer's seat wind
        self.assertEqual(dealers, {0, 1, 2, 3})
        self.assertTrue({0, 1} <= winds <= {0, 1, 2})   # West 10% since epoch 4

    def test_default_is_deterministic_east_one(self):
        t = PyMahjongTable()
        self.assertEqual((t.dealer, t.round_wind_idx, t.round_number), (0, 0, 1))


class TestOrchestratorRuleRouting(unittest.TestCase):
    """The chankan window and multiple ron have to survive the real node
    loop, not just direct engine calls: a driver that mishandles
    `pending_kan` spins forever on the un-mutated kan."""

    def setUp(self):
        random.seed(7)
        self.t = PyMahjongTable()
        self.state = None

    def _run(self, scripted, node):
        import src.tasks.mahjong.orchestrator as orch
        if self.state is None:
            self.state = orch.MahjongState({
                "table": self.t, "trajectories": {i: [] for i in range(4)},
                "model": None, "tokenizer": None, "done": False,
                "last_player": -1, "needs_interrupt": False,
                "exp_dir": "/tmp/mahjong_routing", "capture_logprobs": False,
            })
        seq = iter(scripted)
        real = orch._query
        orch._query = lambda st, pid, legal: (
            lambda a: ("prompt", a, a, None, None))(next(seq))
        try:
            node(self.state)
        finally:
            orch._query = real
        return self.state

    def _rig_added_kan(self):
        self.t.melds[0] = [{"type": "pon", "tiles": ['3z'] * 3, "opened": True}]
        rig(self.t, 0, ['3z', '2p', '3p', '4p', '5p', '6p', '7p',
                        '2s', '3s', '4s', '9s'], drawn='3z')
        # Only player 1 can rob it; 2 and 3 hold nothing relevant.
        rig(self.t, 1, ['3z', '3z', '2p', '3p', '4p', '5p', '6p', '7p',
                        '2s', '3s', '4s', '9s', '9s'])
        for pid in (2, 3):
            rig(self.t, pid, ['1m', '4m', '7m', '1p', '4p', '7p', '1s',
                              '4s', '7s', '1z', '2z', '4z', '6z'])

    def test_robbed_added_kan_ends_the_game_through_the_nodes(self):
        from src.tasks.mahjong.orchestrator import (turn_node, interrupt_node,
                                                    should_continue)
        self._rig_added_kan()
        self._run(['<action type="kan" tile="3z" />'], turn_node)
        self.assertTrue(self.state['needs_interrupt'])
        self.assertEqual(should_continue(self.state), "interrupt")
        self._run(['<action type="ron" />'], interrupt_node)
        self.assertTrue(self.state['done'])
        self.assertIn('抢杠', self.t.result_summary)
        self.assertIn('Chankan', self.t.result_summary)
        self.assertGreater(self.t.points[1], 25000)
        self.assertLess(self.t.points[0], 25000)

    def test_unrobbed_added_kan_returns_to_the_same_player(self):
        from src.tasks.mahjong.orchestrator import (turn_node, interrupt_node,
                                                    should_continue)
        self._rig_added_kan()
        self._run(['<action type="kan" tile="3z" />'], turn_node)
        self._run(['<action type="skip" />'], interrupt_node)
        self.assertFalse(self.state['done'])
        self.assertFalse(self.state['needs_interrupt'])
        self.assertEqual(should_continue(self.state), "turn")
        self.assertIsNone(self.t.pending_kan)
        self.assertEqual(self.t.melds[0][0]['type'], 'shouminkan')
        self.assertEqual(len(self.t.dora_indicators), 1)   # flipped after the discard
        self.assertEqual(self.t.pending_dora_reveal, 1)
        self.assertEqual(self.t.turn, 0)          # same player discards next
        self.assertIsNotNone(self.t.last_drawn[0])

    def test_double_ron_through_the_nodes(self):
        from src.tasks.mahjong.orchestrator import turn_node, interrupt_node
        rig(self.t, 0, KANCHAN_TENPAI_13 + ['5z'], drawn='5z')
        rig(self.t, 1, list(YAKUHAI_TENPAI_13))
        rig(self.t, 2, list(YAKUHAI_TENPAI_13))
        rig(self.t, 3, ['1m', '4m', '7m', '1p', '4p', '7p', '1s',
                        '4s', '7s', '1z', '2z', '4z', '6z'])
        self._run(['<action type="discard" tile="5z" />'], turn_node)
        self._run(['<action type="ron" />', '<action type="ron" />'],
                  interrupt_node)
        self.assertTrue(self.state['done'])
        self.assertIn('双响', self.t.result_summary)
        self.assertGreater(self.t.points[1], 25000)
        self.assertGreater(self.t.points[2], 25000)
        self.assertEqual(self.t.points[3], 25000)
        self.assertEqual(sum(self.t.points) + self.t.kyotaku, 100000)
        # Both winners' steps must be marked terminal for the return calc.
        for pid in (1, 2):
            self.assertTrue(self.state['trajectories'][pid][-1].is_terminal)


class TestInvariants(unittest.TestCase):
    """Random-policy games must never break tile or point conservation."""

    def test_random_games_conserve_tiles_and_points(self):
        from src.tasks.mahjong.orchestrator import build_mahjong_graph, MahjongState
        random.seed(5)
        graph = build_mahjong_graph()
        for _ in range(25):
            table = PyMahjongTable(randomize_round=True)
            final = graph.invoke(MahjongState({
                "table": table, "trajectories": {i: [] for i in range(4)},
                "model": None, "tokenizer": None, "done": False,
                "last_player": -1, "needs_interrupt": False,
                "exp_dir": "/tmp/mahjong_invariants", "capture_logprobs": False,
            }), config={"recursion_limit": 2000})
            t = final['table']
            self.assertTrue(t.finished)
            # random context: initial kyotaku sticks come from previous
            # (unplayed) hands, so the invariant is relative to the start
            self.assertEqual(sum(t.points) + t.kyotaku,
                             sum(t.start_points) + t.start_kyotaku)
            self.assertLessEqual(t.kan_count, t.MAX_KANS)
            self.assertLessEqual(len(t.dora_indicators), 5)
            self.assertIsNone(t.riichi_pending)
            tiles = (sum(len(h) for h in t.hands.values())
                     + sum(len(m['tiles']) for ms in t.melds.values() for m in ms)
                     + sum(len(d) for d in t.discards.values())
                     + len(t.wall) + 14)
            self.assertEqual(tiles, 136)
            for pid in range(4):
                self.assertLessEqual(len(t.melds[pid]), t.MAX_MELDS)


if __name__ == "__main__":
    unittest.main()


class TestRiichiAnkanExactTripletReading(unittest.TestCase):
    """RCR 3.12 (2) is checked by enumerating every winning reading, not by
    the old "any neighbouring tile in hand" approximation (fixed 2026-08-23)."""

    def test_neighbours_in_hand_but_only_triplet_reading_is_allowed(self):
        # 700k seed 77000002 step 59: 234s + 555s + 345m + 456m + 5p tanki,
        # draws the 4th 5s. The only reading has 5s as a koutsu -> legal.
        t = PyMahjongTable()
        rig(t, 0, ['2s', '3s', '4s', '5s', '5s', '5s', '5s',
                   '3m', '4m', '4m', '5m', '5m', '6m', '5p'], drawn='5s')
        t.riichi[0] = True
        self.assertTrue(t._can_ankan(0, '5s'))
        self.assertIn('<action type="kan" tile="5s" />', t.get_legal_actions(0))

    def test_tile_readable_as_pair_in_some_wait_is_refused(self):
        # 111m 23m | 456p | 789s | 2z2z waits 1m / 4m / 2z: the 2z wait reads
        # 11m(pair) + 123m + 222z, so 1m is not a triplet in every reading.
        t = PyMahjongTable()
        rig(t, 0, ['1m', '1m', '1m', '1m', '2m', '3m', '4p', '5p', '6p',
                   '7s', '8s', '9s', '2z', '2z'], drawn='1m')
        t.riichi[0] = True
        self.assertFalse(t._can_ankan(0, '1m'))
        # RCR 3.12 example: 2333s waits 1s/2s/4s, 33s can be the pair.
        t2 = PyMahjongTable()
        rig(t2, 0, ['2s', '3s', '3s', '3s', '3s', '2m', '3m', '4m',
                    '5m', '6m', '7m', '2p', '3p', '4p'], drawn='3s')
        t2.riichi[0] = True
        self.assertFalse(t2._can_ankan(0, '3s'))

    def test_run_reading_refused_but_fourth_copy_in_run_tolerated(self):
        from src.tasks.mahjong.table import _tile_only_as_triplet
        # 34555m 234p 678p 11z: wait 1z reads 345m + 55m(pair) -> refused overall.
        t = PyMahjongTable()
        rig(t, 0, ['3m', '4m', '5m', '5m', '5m', '5m', '2p', '3p', '4p',
                   '6p', '7p', '8p', '1z', '1z'], drawn='5m')
        t.riichi[0] = True
        self.assertFalse(t._can_ankan(0, '5m'))
        # Per-wait readings: 2m -> 234m + 555m (triplet only) ; 5m -> the
        # 13-tile shape is 34m + 555m, the run 345m only uses the drawn 4th
        # copy, so the triplet is intact ; 1z -> 55m is the pair.
        base = ['3m', '4m', '5m', '5m', '5m', '2p', '3p', '4p', '6p', '7p', '8p', '1z', '1z']
        self.assertTrue(_tile_only_as_triplet(base + ['2m'], '5m', 4))
        self.assertTrue(_tile_only_as_triplet(base + ['5m'], '5m', 4))
        self.assertFalse(_tile_only_as_triplet(base + ['1z'], '5m', 4))
        # A run reading with no triplet at all: 456m + 5m... 3456m? use
        # 55m pair + 456m... simplest: 5m in a run, no koutsu -> refused.
        self.assertFalse(_tile_only_as_triplet(
            ['4m', '5m', '6m', '5m', '5m', '2p', '3p', '4p', '6p', '7p', '8p', '1z', '1z', '1z'], '5m', 4))


class TestExp45ActionGaps(unittest.TestCase):
    """Two action-space gaps found by exp45 human-log replay (2026-08-26).

    Both scenarios are verbatim from 凤凰卓 logs where the human's action
    was legal on tenhou but missing from our legal set."""

    def setUp(self):
        random.seed(11)
        self.t = PyMahjongTable()

    def _four_pons(self, pid):
        self.t.melds[pid] = [
            {"type": "pon", "tiles": [x] * 3, "opened": True, "from": 1}
            for x in ('6z', '1m', '1s', '5z')]

    def test_kakan_allowed_with_four_melds(self):
        # 2026080109gm-...-0ffd660c: toitoi, 4 pons, draws the 4th 1m.
        self._four_pons(0)
        rig(self.t, 0, ['1m', '7z'], drawn='1m')
        self.assertTrue(self.t._can_shouminkan(0, '1m'))
        self.assertIn('<action type="kan" tile="1m" />',
                      self.t.get_legal_actions(0))
        self.t.step(0, '<action type="kan" tile="1m" />')
        self.t.resolve_pending_kan()                  # chankan window closes
        self.assertEqual(len(self.t.melds[0]), 4)     # upgraded, not added
        self.assertEqual(self.t.melds[0][1]["type"], "shouminkan")
        self.assertEqual(self.t.kan_count, 1)

    def test_kakan_still_capped_by_max_kans(self):
        self._four_pons(0)
        rig(self.t, 0, ['1m', '7z'], drawn='1m')
        self.t.kan_count = self.t.MAX_KANS
        self.assertFalse(self.t._can_shouminkan(0, '1m'))

    def test_new_meld_kans_still_capped_by_max_melds(self):
        # ankan as a 5th meld stays impossible
        self._four_pons(0)
        rig(self.t, 0, ['7z'] * 4 + ['1m'], drawn='7z')
        self.assertFalse(self.t._can_ankan(0, '7z'))

    def test_riichi_offered_from_completed_hand(self):
        # 2026060121gm-...-20043503: complete 2m2m 567m 789m 234s 567s;
        # the human declines the tsumo and riichis discarding 2s.
        rig(self.t, 0, ['2m', '2m', '5m', '6m', '7m', '7m', '8m', '9m',
                        '2s', '3s', '4s', '5s', '6s', '7s'], drawn='2m')
        acts = self.t.get_legal_actions(0)
        self.assertIn('<action type="tsumo" />', acts)
        self.assertIn('<action type="riichi" tile="2s" />', acts)
