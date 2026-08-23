"""Single-hand rules aligned to Majsoul (docs/design_majsoul_rules.md)."""
import unittest

from src.tasks.mahjong.table import PyMahjongTable
from tests.test_engine import rig

TANYAO_13 = ['2m', '3m', '4m', '3p', '4p', '5p', '5p', '6p', '7p', '6s', '7s', '8s', '8s']


def _fresh(dealer=0):
    t = PyMahjongTable(randomize_round=False)
    t.text_obs = False
    return t


class TestYakuChanges(unittest.TestCase):
    def test_no_renhou(self):
        t = _fresh()
        # non-dealer, first go-around, ron on a yaku-less but complete hand:
        # without renhou there is no yaku -> no ron offered.
        rig(t, 1, ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '3z', '3z', '3z', '4z'])
        t.last_discard, t.last_discarder, t.turn = '4z', 0, 0
        t.discard_count = [1, 0, 0, 0]
        self.assertNotIn('<action type="ron" />', t.get_interrupt_actions(1))

    def test_double_yakuman_suuankou_tanki(self):
        t = _fresh()
        rig(t, 0, ['1m', '1m', '1m', '3p', '3p', '3p', '7s', '7s', '7s', '2z', '2z', '2z', '5z', '5z'], drawn='5z')
        t.discard_count = [1, 1, 1, 1]
        res = t._win_result(0, '5z', is_tsumo=True)
        self.assertIsNotNone(res)
        self.assertEqual(res.han, 26)          # 四暗刻单骑 = double yakuman


class TestNagashiMangan(unittest.TestCase):
    def test_all_terminal_uncalled_river_pays_mangan(self):
        t = _fresh()
        t.wall = []
        for p in range(4):
            t.river_events[p] = [['2m', False, False, False, 0]]
        t.river_events[1] = [[x, False, False, False, i] for i, x in enumerate(['1m', '9p', '1z', '7z', '9s'])]
        t._ryuukyoku()
        self.assertIn('流局满贯', t.result_summary)
        self.assertEqual(t.points[1], 25000 + 8000)          # non-dealer mangan tsumo
        self.assertEqual(t.points[0], 25000 - 4000)          # dealer pays 4000

    def test_called_tile_voids_it(self):
        t = _fresh()
        t.wall = []
        for p in range(4):
            t.river_events[p] = [['2m', False, False, False, 0]]
        t.river_events[1] = [['1m', False, False, True, 0], ['9p', False, False, False, 1]]
        t._ryuukyoku()
        self.assertIn('流局 |', t.result_summary)


class TestAbortiveDraws(unittest.TestCase):
    def test_kyuushu_kyuuhai(self):
        t = _fresh()
        rig(t, 0, ['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z', '5z', '2m', '3m', '4m'], drawn='4m')
        acts = t.get_legal_actions(0)
        self.assertIn('<action type="kyuushu" />', acts)
        _, _, done, _ = t.step(0, '<action type="kyuushu" />')
        self.assertTrue(done)
        self.assertIn('途中流局(九种九牌)', t.result_summary)
        self.assertEqual(t.points, [25000] * 4)
        # not offered once a call happened or after the first discard
        t2 = _fresh()
        rig(t2, 0, ['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z', '5z', '2m', '3m', '4m'], drawn='4m')
        t2.any_call = True
        self.assertNotIn('<action type="kyuushu" />', t2.get_legal_actions(0))

    def test_suufon_renda(self):
        t = _fresh()
        for p in range(4):
            h = list(TANYAO_13) + ['1z']
            rig(t, p, h, drawn='1z')
        for p in range(4):
            t.turn = p
            t.last_drawn[p] = '1z'
            t.step(p, '<action type="discard" tile="1z" />')
            _, done = t.advance_turn()
        self.assertTrue(done)
        self.assertIn('四风连打', t.result_summary)

    def test_suucha_riichi(self):
        t = _fresh()
        t.wall = t.wall[:40]
        for p in range(4):
            rig(t, p, ['2m', '3m', '4m', '3p', '4p', '5p', '6s', '7s', '8s', '2z', '2z', '5s', '5s', '9p'], drawn='9p')
        for p in range(4):
            t.turn = p
            t.last_drawn[p] = '9p'
            _, _, done, _ = t.step(p, '<action type="riichi" tile="9p" />')
            self.assertFalse(done)
            _, done = t.advance_turn()
        self.assertTrue(done)
        self.assertIn('四家立直', t.result_summary)
        self.assertEqual(t.kyotaku, 4000)

    def test_suukaikan_two_players(self):
        t = _fresh()
        t.kan_count, t.kan_players = 4, {0, 1}
        rig(t, 2, list(TANYAO_13) + ['1z'], drawn='1z')
        t.turn = 2
        t.step(2, '<action type="discard" tile="1z" />')
        _, done = t.advance_turn()
        self.assertTrue(done)
        self.assertIn('四杠散了', t.result_summary)
        # one player holding all four kans plays on
        t2 = _fresh()
        t2.kan_count, t2.kan_players = 4, {0}
        rig(t2, 2, list(TANYAO_13) + ['1z'], drawn='1z')
        t2.turn = 2
        t2.step(2, '<action type="discard" tile="1z" />')
        _, done = t2.advance_turn()
        self.assertFalse(done)

    def test_triple_ron_is_a_draw(self):
        t = _fresh()
        win = ['2m', '3m', '4m', '3p', '4p', '5p', '6s', '7s', '8s', '2z', '2z', '5s', '5s']
        for p in (1, 2, 3):
            rig(t, p, list(win))
        t.riichi = [False, True, True, True]
        t.discard_count = [2, 2, 2, 2]
        t.last_discard, t.last_discarder, t.turn = '5s', 0, 0
        _, _, done, info = t.step_ron([1, 2, 3])
        self.assertTrue(done)
        self.assertEqual(info.get("abort"), "三家和了")
        self.assertEqual(t.points, [25000] * 4)
        # double ron still pays
        t2 = _fresh()
        for p in (1, 2):
            rig(t2, p, list(win))
        t2.riichi = [False, True, True, False]
        t2.discard_count = [2, 2, 2, 2]
        t2.last_discard, t2.last_discarder, t2.turn = '5s', 0, 0
        _, _, done, _ = t2.step_ron([1, 2])
        self.assertTrue(done)
        self.assertIn('双响', t2.result_summary)


class TestKanDoraTiming(unittest.TestCase):
    def test_ankan_immediate_open_kan_after_discard(self):
        t = _fresh()
        rig(t, 0, ['1z'] * 4 + ['2m', '3m', '4m', '5m', '6m', '7m', '2p', '3p', '4p', '9s'], drawn='1z')
        t.step(0, '<action type="kan" tile="1z" />')
        self.assertEqual(len(t.dora_indicators), 2)
        t2 = _fresh()
        t2.melds[1] = [{"type": "pon", "tiles": ['3z'] * 3, "opened": True, "from": 0}]
        rig(t2, 1, ['3z', '2p', '3p', '4p', '5p', '6p', '7p', '2s', '3s', '4s', '9s'], drawn='3z')
        t2.turn = 1
        t2.step(1, '<action type="kan" tile="3z" />')
        t2.resolve_pending_kan()
        self.assertEqual(len(t2.dora_indicators), 1)
        self.assertEqual(t2.pending_dora_reveal, 1)
        t2.step(1, f'<action type="discard" tile="{t2.last_drawn[1]}" />')
        self.assertEqual(len(t2.dora_indicators), 2)


if __name__ == "__main__":
    unittest.main()
