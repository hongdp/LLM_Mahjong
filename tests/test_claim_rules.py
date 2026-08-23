"""Audit (user 2026-08-23): claim priority on one tile / multiple ron,
chankan, furiten — Majsoul rules."""
import unittest

from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.table import PyMahjongTable
from tests.test_engine import rig

TENPAI_5s = ['2m', '3m', '4m', '3p', '4p', '5p', '6s', '7s', '8s', '2z', '2z', '5s', '5s']   # waits 5s (shanpon 2z/5s)


def _table():
    t = PyMahjongTable(randomize_round=False)
    t.text_obs = False
    t.discard_count = [2, 2, 2, 2]
    return t


def _cands(*specs):
    out = []
    for pid, a in specs:
        if a == "chi":
            xml = '<action type="chi" tile="5s" with="3s 4s" />'
        elif a in ("pon", "kan"):
            xml = f'<action type="{a}" tile="5s" />'
        else:
            xml = f'<action type="{a}" />'
        out.append({"player_id": pid, "parsed": xml, "type": a, "reward": 0.0})
    return out


class TestPriority(unittest.TestCase):
    def test_ron_beats_pon_beats_chi(self):
        t = _table()
        rig(t, 1, ['3s', '4s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z'])   # next seat: chi
        rig(t, 2, ['5s', '5s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z'])   # pon
        rig(t, 3, list(TENPAI_5s))                                                                # ron
        t.riichi[3] = True
        t.last_discard, t.last_discarder, t.turn = '5s', 0, 0
        t.hands[0] = ['1p'] * 1 + ['2p', '3p', '4p', '6p', '7p', '8p', '9p', '1s', '2s', '3s', '9m', '9m', '8m']
        executed, done = _resolve_claims(t, _cands((1, "chi"), (2, "pon"), (3, "ron")))
        self.assertTrue(done and executed[0]["player_id"] == 3)
        self.assertIn('玩家3 荣和', t.result_summary)
        # pon beats chi when nobody rons
        t2 = _table()
        rig(t2, 1, ['3s', '4s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z'])
        rig(t2, 2, ['5s', '5s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z'])
        t2.last_discard, t2.last_discarder, t2.turn = '5s', 0, 0
        t2.wall = t2.wall[:30]
        executed, done = _resolve_claims(t2, _cands((1, "chi"), (2, "pon")))
        self.assertFalse(done)
        self.assertEqual(executed[0]["player_id"], 2)
        self.assertEqual(t2.melds[2][-1]["type"], "pon")
        self.assertEqual(t2.melds[1], [])

    def test_double_ron_sticks_go_to_nearest_seat(self):
        t = _table()
        for p in (1, 3):
            rig(t, p, list(TENPAI_5s))
        t.riichi = [False, True, False, True]
        t.kyotaku = 2000
        t.last_discard, t.last_discarder, t.turn = '5s', 0, 0
        _, _, done, info = t.step_ron([1, 3])          # seat order from the discarder
        self.assertTrue(done and set(info["winners"]) == {1, 3})
        self.assertIn('双响', t.result_summary)
        self.assertEqual(t.kyotaku, 0)
        # winner 1 (closest counter-clockwise from seat 0) took the 2000
        self.assertGreater(t.points[1], t.points[3])

    def test_illegal_ron_falls_through_to_melds(self):
        t = _table()
        rig(t, 2, ['5s', '5s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z'])
        rig(t, 3, list(TENPAI_5s))
        t.furiten_river[3] = ['5s']                    # seat 3 is furiten: its ron is illegal
        t.last_discard, t.last_discarder, t.turn = '5s', 0, 0
        t.wall = t.wall[:30]
        cands = _cands((2, "pon"), (3, "ron"))
        executed, done = _resolve_claims(t, cands)
        self.assertFalse(done)
        self.assertEqual(executed[0]["player_id"], 2)
        self.assertLess(cands[1]["reward"], 0)


class TestChankan(unittest.TestCase):
    def _kakan(self):
        t = _table()
        t.melds[0] = [{"type": "pon", "tiles": ['5s'] * 3, "opened": True, "from": 1, "red": 0}]
        rig(t, 0, ['5s', '2p', '3p', '4p', '5p', '6p', '7p', '2m', '3m', '4m', '9m'], drawn='5s')
        rig(t, 2, list(TENPAI_5s))
        t.riichi[2] = True
        t.turn = 0
        _, _, done, info = t.step(0, '<action type="kan" tile="5s" />')
        self.assertFalse(done); self.assertEqual(info.get("chankan"), '5s')
        return t

    def test_chankan_ron_scores_the_yaku(self):
        t = self._kakan()
        self.assertIn('<action type="ron" />', t.get_interrupt_actions(2))
        _, _, done, _ = t.step_ron([2])
        self.assertTrue(done)
        self.assertIn('抢杠', t.result_summary); self.assertIn('Chankan', t.result_summary)
        self.assertEqual(t.melds[0][0]["type"], "pon")          # the kan never completed

    def test_passing_a_chankan_makes_you_furiten(self):
        t = self._kakan()
        t.resolve_pending_kan()                                   # seat 2 passed
        self.assertTrue(t.perm_furiten[2])                        # riichi -> permanent
        self.assertFalse(t._can_ron(2, '5s'))
        t2 = self._kakan(); t2.riichi[2] = False
        t2.resolve_pending_kan()
        self.assertTrue(t2.temp_furiten[2])                       # same-turn only
        t2.turn = 1; t2.advance_turn()                            # seat 2 draws -> lifted
        self.assertFalse(t2.temp_furiten[2])

    def test_kokushi_may_rob_an_ankan_others_may_not(self):
        t = _table()
        rig(t, 0, ['1z'] * 4 + ['2m', '3m', '4m', '5m', '6m', '7m', '2p', '3p', '4p', '9s'], drawn='1z')
        rig(t, 2, ['1m', '9m', '1p', '9p', '1s', '9s', '2z', '3z', '4z', '5z', '6z', '7z', '7z'])   # kokushi waiting 1z
        rig(t, 3, ['2z', '2z', '1z', '2m', '3m', '4m', '5m', '6m', '7m', '2p', '3p', '4p', '9s'])  # plain hand waiting 1z/9s
        t.turn = 0
        _, _, done, info = t.step(0, '<action type="kan" tile="1z" />')
        self.assertFalse(done); self.assertEqual(info.get("chankan"), '1z')
        self.assertIn('<action type="ron" />', t.get_interrupt_actions(2))
        self.assertNotIn('<action type="ron" />', t.get_interrupt_actions(3))
        _, _, done, _ = t.step_ron([2])
        self.assertTrue(done); self.assertIn('Kokushi', t.result_summary)
        # nobody robs -> the ankan completes normally
        t2 = _table()
        rig(t2, 0, ['1z'] * 4 + ['2m', '3m', '4m', '5m', '6m', '7m', '2p', '3p', '4p', '9s'], drawn='1z')
        rig(t2, 2, ['1m', '9m', '1p', '9p', '1s', '9s', '2z', '3z', '4z', '5z', '6z', '7z', '7z'])
        t2.turn = 0
        t2.step(0, '<action type="kan" tile="1z" />'); t2.resolve_pending_kan()
        self.assertEqual(t2.melds[0][0]["type"], "ankan"); self.assertTrue(t2.rinshan[0])


class TestFuriten(unittest.TestCase):
    def test_own_discard_furiten_includes_called_away_tiles(self):
        t = _table()
        rig(t, 1, list(TENPAI_5s))
        t.furiten_river[1] = ['2z']            # discarded earlier and later called away (visible river empty)
        t.discards[1] = []
        self.assertTrue(t._is_furiten(1))      # 2z is one of the waits
        t.furiten_river[1] = ['9p']
        self.assertFalse(t._is_furiten(1))

    def test_same_turn_furiten_lifts_only_on_own_draw(self):
        t = _table()
        rig(t, 2, list(TENPAI_5s))
        rig(t, 0, ['5s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z', '3s', '4s'], drawn='4s')
        t.turn = 0
        t.step(0, '<action type="discard" tile="5s" />')   # seat 2 could ron, passes
        t.advance_turn()                                     # seat 1 draws
        self.assertTrue(t.temp_furiten[2])
        self.assertFalse(t._can_ron(2, '5s'))
        t.turn = 1; t.hands[1].pop(); t.advance_turn()       # seat 2 draws -> lifted
        self.assertFalse(t.temp_furiten[2])

    def test_riichi_missed_ron_is_permanent(self):
        t = _table()
        rig(t, 2, list(TENPAI_5s)); t.riichi[2] = True
        rig(t, 0, ['5s', '1m', '2m', '3m', '7p', '8p', '9p', '1z', '1z', '1z', '6z', '6z', '3s', '4s'], drawn='4s')
        t.turn = 0
        t.step(0, '<action type="discard" tile="5s" />')
        t.advance_turn()
        self.assertTrue(t.perm_furiten[2])
        t.turn = 1; t.hands[1].pop(); t.advance_turn()
        self.assertTrue(t.perm_furiten[2]); self.assertFalse(t._can_ron(2, '2z'))

    def test_furiten_never_blocks_tsumo(self):
        t = _table()
        rig(t, 2, list(TENPAI_5s) + ['5s'], drawn='5s')
        t.furiten_river[2] = ['2z']; t.riichi[2] = True; t.perm_furiten[2] = True
        t.turn = 2
        self.assertIn('<action type="tsumo" />', t.get_legal_actions(2))


if __name__ == "__main__":
    unittest.main()
