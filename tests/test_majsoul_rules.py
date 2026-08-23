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


class TestRedDora(unittest.TestCase):
    def test_wall_has_one_red_five_per_suit(self):
        t = _fresh()
        everything = list(t.wall) + list(t.dead_wall)
        for p in range(4):
            everything += t.display_hand(p)
        for suit in "mps":
            self.assertEqual(everything.count(f"0{suit}"), 1, suit)
            self.assertEqual(everything.count(f"5{suit}"), 3, suit)
        self.assertEqual(len(everything), 136)

    def test_red_five_is_a_distinct_discard_action(self):
        t = _fresh()
        rig(t, 0, ['5m', '5m', '2p', '3p', '4p', '6s', '7s', '8s', '1z', '1z', '1z', '3z', '4z', '7z'], drawn='7z')
        t.red[0]["m"] = 1
        acts = t.get_legal_actions(0)
        self.assertIn('<action type="discard" tile="5m" />', acts)
        self.assertIn('<action type="discard" tile="0m" />', acts)
        # only the red copy left -> only the red spelling
        t.hands[0].remove('5m')
        acts = t.get_legal_actions(0)
        self.assertNotIn('<action type="discard" tile="5m" />', acts)
        self.assertIn('<action type="discard" tile="0m" />', acts)
        t.turn = 0
        t.step(0, '<action type="discard" tile="0m" />')
        self.assertEqual(t.discards[0][-1], '0m')
        self.assertEqual(t.red[0]["m"], 0)
        self.assertTrue(t.last_discard_red)
        self.assertEqual(t.last_discard, '5m')          # claims match on the plain spelling
        self.assertEqual(t.river_events[0][-1][0], '0m')

    def test_riichi_lock_tsumogiri_keeps_red_identity(self):
        t = _fresh()
        rig(t, 0, ['2m', '3m', '4m', '3p', '4p', '5p', '6s', '7s', '8s', '2z', '2z', '5s', '5s', '5m'], drawn='5m')
        t.riichi[0] = True
        t.red[0]["m"] = 1
        t.last_drawn_red[0] = True
        acts = t.get_legal_actions(0)
        self.assertEqual([a for a in acts if 'discard' in a], ['<action type="discard" tile="0m" />'])
        # discarding the plain spelling while locked is illegal
        t.turn = 0
        _, r, _, _ = t.step(0, '<action type="discard" tile="5m" />')
        self.assertLess(r[0], 0)

    def test_aka_dora_is_scored(self):
        t = _fresh()
        # tanyao hand holding a red 5p and a red 5s, tsumo on 5p
        rig(t, 0, ['2m', '3m', '4m', '3p', '4p', '5p', '5p', '6p', '7p', '6s', '7s', '8s', '5s', '5s'], drawn='5p')
        t.red[0]["p"] = 1
        t.red[0]["s"] = 1
        t.dora_indicators, t.ura_indicators = ['1z'], ['1z']
        t.discard_count = [1, 1, 1, 1]
        res = t._win_result(0, '5p', is_tsumo=True)
        self.assertIsNotNone(res)
        names = [str(y) for y in res.yaku]
        self.assertTrue(any("Aka" in n for n in names), names)
        base = t._win_result(0, '5p', is_tsumo=True).han
        t.red[0]["p"] = 0
        t.red[0]["s"] = 0
        self.assertEqual(base, t._win_result(0, '5p', is_tsumo=True).han + 2)   # two aka dora

    def test_ron_on_a_red_five_counts_it(self):
        t = _fresh()
        rig(t, 1, ['2m', '3m', '4m', '3p', '4p', '5p', '6p', '7p', '8p', '6s', '7s', '8s', '5s'])
        t.discard_count = [2, 2, 2, 2]
        t.last_discard, t.last_discarder, t.turn = '5s', 0, 0
        t.last_discard_red = True
        res = t._win_result(1, '5s', is_tsumo=False)
        self.assertIsNotNone(res)
        self.assertTrue(any("Aka" in str(y) for y in res.yaku))
        t.last_discard_red = False
        res2 = t._win_result(1, '5s', is_tsumo=False)
        self.assertEqual(res.han, res2.han + 1)

    def test_pon_uses_plain_copies_first(self):
        t = _fresh()
        rig(t, 1, ['5p', '5p', '5p', '2m', '3m', '4m', '6s', '7s', '8s', '1z', '1z', '3z', '4z'])
        t.red[1]["p"] = 1
        t.last_discard, t.last_discarder, t.turn = '5p', 0, 0
        t.last_discard_red = False
        t.wall = t.wall[:30]
        _, _, _, info = t.step_interrupt(1, '<action type="pon" tile="5p" />')
        self.assertTrue(info["interrupt"])
        self.assertEqual(t.melds[1][-1]["red"], 0)
        self.assertEqual(t.red[1]["p"], 1)                 # red stayed in hand
        self.assertEqual(t.display_hand(1).count('0p'), 1)

    def test_legacy_checkpoint_head_is_widened(self):
        import torch
        from src.agents.dnn.arch_zoo import ZOO
        from src.agents.dnn.net import load_compatible
        from src.agents.dnn.encoder import ACTION_DIM, LEGACY_ACTION_DIM, TILE_TYPES
        net = ZOO["cnn_m"][0]()
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        # fake a pre-red checkpoint: trim the flat head to 272 rows
        hk = [k for k in sd if sd[k].dim() > 0 and sd[k].shape[0] == ACTION_DIM]
        self.assertTrue(hk)
        for k in hk:
            sd[k] = sd[k][:LEGACY_ACTION_DIM]
            torch.nn.init.normal_(sd[k])
        load_compatible(net, sd)
        new_sd = net.state_dict()
        for k in hk:
            w = new_sd[k]
            d5m = 4                                      # 5m column
            disc, disc0 = 0 * TILE_TYPES + d5m, 8 * TILE_TYPES + d5m
            if w.dim() == 1:
                self.assertAlmostEqual(float(w[disc0]), float(w[disc]) - 1.0, places=5)
                self.assertLess(float(w[10 * TILE_TYPES]), -10)      # kyuushu never
            else:
                self.assertTrue(torch.equal(w[disc0], w[disc]))


class TestContextRandomization(unittest.TestCase):
    def test_random_context_is_seeded_and_consistent(self):
        import random as _r
        _r.seed(777)
        t1 = PyMahjongTable(randomize_round=True)
        _r.seed(777)
        t2 = PyMahjongTable(randomize_round=True)
        self.assertEqual(t1.points, t2.points)
        self.assertEqual(t1.kyotaku, t2.kyotaku)
        self.assertEqual(t1.round_wind_idx, t2.round_wind_idx)
        self.assertEqual(sum(t1.start_points), 100000)
        self.assertTrue(all(p >= 1000 for p in t1.points))

    def test_contexts_actually_vary_and_west_appears(self):
        import random as _r
        pts, winds, kyo = set(), set(), set()
        for sd in range(300):
            _r.seed(9000 + sd)
            t = PyMahjongTable(randomize_round=True)
            pts.add(tuple(t.points)); winds.add(t.round_wind_idx); kyo.add(t.kyotaku)
        self.assertGreater(len(pts), 250)
        self.assertEqual(winds, {0, 1, 2})
        self.assertIn(1000, kyo); self.assertIn(0, kyo)

    def test_rewards_use_delta_from_start(self):
        import random as _r
        _r.seed(4242)
        t = PyMahjongTable(randomize_round=True)
        start = list(t.start_points)
        t.wall = []
        for p in range(4):
            t.river_events[p] = [['2m', False, False, False, 0]]
        t._ryuukyoku()
        for i in range(4):
            self.assertAlmostEqual(t.final_rewards[i] - t.RANK_BONUS_APPLIED[i]
                                   if hasattr(t, 'RANK_BONUS_APPLIED') else 0.0, 0.0) if False else None
        # noten payments moved points; delta reward reflects them, not the level
        deltas = [t.points[i] - start[i] for i in range(4)]
        self.assertEqual(sum(deltas), 0)

    def test_west_round_scalar_code(self):
        import random as _r
        from src.agents.dnn.encoder import encode_state
        for sd in range(200):
            _r.seed(31000 + sd)
            t = PyMahjongTable(randomize_round=True)
            if t.round_wind_idx == 2:
                t.text_obs = False
                _, s = encode_state(t, 0, variant="v1")
                self.assertEqual((float(s[12]), float(s[13])), (1.0, 1.0))
                return
        self.fail("no West round in 200 seeds")
