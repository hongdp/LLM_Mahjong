"""Unit tests for the tenhou mjlog -> MJAI converter (exp45 pipeline).

Meld-bitfield vectors are pinned from a real 凤凰卓 log (replay-verified:
every decoded meld passed the engine legal-set gate in the harvester).
The full-log fidelity test lives in the harvester itself and runs over
data/tenhou/raw/, which is local-only — here we keep pure-function pins
that need no fixture files.
"""

import unittest

from tools.tenhou.mjlog_to_mjai import decode_meld, parse_mjlog, tile136_to_mjai


class Tile136Test(unittest.TestCase):
    def test_reds_and_plains(self):
        self.assertEqual(tile136_to_mjai(16), "5mr")   # copy 0 of 5m
        self.assertEqual(tile136_to_mjai(52), "5pr")
        self.assertEqual(tile136_to_mjai(88), "5sr")
        self.assertEqual(tile136_to_mjai(17), "5m")
        self.assertEqual(tile136_to_mjai(0), "1m")
        self.assertEqual(tile136_to_mjai(35), "9m")

    def test_honors(self):
        self.assertEqual([tile136_to_mjai(108 + 4 * i) for i in range(7)],
                         ["E", "S", "W", "N", "P", "F", "C"])


class MeldDecodeTest(unittest.TestCase):
    # vectors from 2026082710gm-00a9-0000-66062311 (replay-verified)
    PINS = [
        (3, 49705, {"type": "pon", "actor": 3, "target": 0,
                    "pai": "F", "consumed": ["F", "F"]}),
        (1, 47721, {"type": "pon", "actor": 1, "target": 2,
                    "pai": "P", "consumed": ["P", "P"]}),
        (1, 30219, {"type": "pon", "actor": 1, "target": 0,
                    "pai": "2s", "consumed": ["2s", "2s"]}),
    ]

    def test_pinned_melds(self):
        for who, m, want in self.PINS:
            self.assertEqual(decode_meld(who, m), want)

    def test_chi_shape(self):
        # any chi must consume exactly 2 tiles forming a run with pai
        ev = decode_meld(2, 0b0000000001010111)          # low bits: chi, from kamicha
        self.assertEqual(ev["type"], "chi")
        self.assertEqual(len(ev["consumed"]), 2)
        self.assertEqual(ev["target"], (2 + 3) % 4)


class ParseSmokeTest(unittest.TestCase):
    XML = ('<mjloggm ver="2.3"><GO type="169" lobby="0"/><TAIKYOKU oya="0"/>'
           '<INIT seed="0,0,0,2,1,86" ten="250,250,250,250" oya="0" '
           'hai0="0,4,8,12,16,20,24,28,32,36,40,44,48" '
           'hai1="1,5,9,13,17,21,25,29,33,37,41,45,49" '
           'hai2="2,6,10,14,18,22,26,30,34,38,42,46,50" '
           'hai3="3,7,11,15,19,23,27,31,35,39,43,47,51"/>'
           '<T52/><D52/><REACH who="1" step="1"/>'
           '<RYUUKYOKU type="yao9"/></mjloggm>')

    def test_events(self):
        g = parse_mjlog(self.XML)
        types = [e["type"] for e in g["events"]]
        self.assertEqual(types, ["start_kyoku", "tsumo", "dahai", "reach",
                                 "ryukyoku", "end_kyoku", "end_game"])
        sk = g["events"][0]
        self.assertEqual(sk["scores"], [25000] * 4)
        self.assertEqual(sk["dora_marker"], tile136_to_mjai(86))
        d = g["events"][2]
        self.assertTrue(d["tsumogiri"])
        self.assertEqual(d["pai"], "5pr")                # tile 52 drawn+cut

    def test_three_player_rejected(self):
        with self.assertRaises(ValueError):
            parse_mjlog('<mjloggm ver="2.3"><GO type="185" lobby="0"/></mjloggm>')


if __name__ == "__main__":
    unittest.main()
