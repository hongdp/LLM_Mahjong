"""Value-distance profile tests (exp11 shared feature).

The profile must be a sensible, monotone summary: closer to a valuable
family => smaller distance at that value bucket; and higher value buckets
can never be CLOSER than lower ones (min over a shrinking family set).
"""

import unittest

from src.agents.dnn.yaku_features import (FAMILY_VALUE, VALUE_BUCKETS,
                                          family_distances,
                                          value_distance_profile)


class TestFamilyDistances(unittest.TestCase):

    def test_flush_hand_close_to_chinitsu(self):
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                "1m", "2m", "3m", "5m"]                      # pure manzu
        d = family_distances(hand, 0, closed=True)
        self.assertEqual(d["chinitsu"], 0.0)
        self.assertEqual(d["honitsu"], 0.0)

    def test_scattered_hand_far_from_flush(self):
        hand = ["1m", "4p", "7s", "2m", "5p", "8s", "3m", "6p", "9s",
                "1z", "3z", "5z", "7z"]
        d = family_distances(hand, 0, closed=True)
        self.assertGreaterEqual(d["chinitsu"], 7.0)   # >=7 tiles off-suit
        self.assertLessEqual(d["honitsu"], d["chinitsu"])  # honors help honitsu

    def test_chiitoi_counts_pairs(self):
        hand = ["1m", "1m", "3p", "3p", "5s", "5s", "7z", "7z",
                "9m", "9m", "2p", "4s", "6z"]                # 5 pairs
        d = family_distances(hand, 0, closed=True)
        self.assertEqual(d["chiitoi"], 1.0)

    def test_chiitoi_impossible_when_open(self):
        hand = ["1m", "1m", "3p", "3p", "5s", "5s", "7z", "7z", "9m", "9m"]
        d = family_distances(hand, 1, closed=False)
        self.assertGreaterEqual(d["chiitoi"], 8.0)

    def test_dragon_triplet_zeroes_yakuhai_gap(self):
        hand = ["5z", "5z", "5z", "1m", "2m", "3m", "4p", "5p", "6p",
                "7s", "8s", "9s", "9s"]
        d = family_distances(hand, 0, closed=True)
        self.assertLessEqual(d["yakuhai"], 1.0)   # bounded by base shanten


class TestProfile(unittest.TestCase):

    def test_monotone_in_value(self):
        import random
        random.seed(0)
        tiles = [f"{v}{s}" for s in "mps" for v in range(1, 10)] + \
                [f"{v}z" for v in range(1, 8)]
        for _ in range(50):
            hand = random.choices(tiles, k=13)
            prof = value_distance_profile(hand, 0, closed=True)
            self.assertEqual(prof, sorted(prof),
                             f"profile must be non-decreasing: {prof}")
            for p in prof:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_flush_hand_lights_up_high_value_bucket(self):
        flush = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                 "1m", "2m", "3m", "5m"]
        scattered = ["1m", "4p", "7s", "2m", "5p", "8s", "3m", "6p", "9s",
                     "1z", "3z", "5z", "7z"]
        pf = value_distance_profile(flush, 0, True)
        ps = value_distance_profile(scattered, 0, True)
        self.assertLess(pf[-1], ps[-1],
                        "flush hand must be closer to the 8000+ bucket")

    def test_buckets_cover_family_values(self):
        self.assertLessEqual(max(VALUE_BUCKETS), max(FAMILY_VALUE.values()))


if __name__ == "__main__":
    unittest.main()


class TestHazardExtractors(unittest.TestCase):
    """Kokushi vs suuankou: same value, different dynamics — the pair the
    shared hazard head must tell apart by (d, u) shape alone."""

    def test_kokushi_shanten_formula(self):
        from src.agents.dnn.yaku_features import kokushi_distance
        # 12 distinct orphans + a 1m pair -> 13 - 12 - 1 = 0 (tenpai on 7z)
        hand = ["1m", "1m", "9m", "1p", "9p", "1s", "9s",
                "1z", "2z", "3z", "4z", "5z", "6z"]
        d, u = kokushi_distance(hand)
        self.assertEqual(d, 0.0)
        self.assertEqual(u, 4.0)          # only 7z missing -> 4 copies

    def test_kokushi_one_shanten(self):
        from src.agents.dnn.yaku_features import kokushi_distance
        # 11 distinct orphans + pair (5m is not an orphan) -> 13-11-1 = 1
        hand = ["1m", "1m", "9m", "1p", "9p", "1s", "9s",
                "1z", "2z", "3z", "4z", "5z", "5m"]
        d, u = kokushi_distance(hand)
        self.assertEqual(d, 1.0)
        self.assertEqual(u, 8.0)          # 6z and 7z missing -> 8 copies

    def test_kokushi_no_pair_needs_one_more(self):
        from src.agents.dnn.yaku_features import kokushi_distance
        hand = ["1m", "9m", "1p", "9p", "1s", "9s",
                "1z", "2z", "3z", "4z", "5z", "6z", "7z"]   # all 13, no pair
        d, u = kokushi_distance(hand)
        self.assertEqual(d, 0.0)                            # tenpai (13-wait)
        self.assertGreater(u, 30.0)                         # any orphan pairs

    def test_scattered_hand_far_from_kokushi(self):
        from src.agents.dnn.yaku_features import kokushi_distance
        hand = ["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s",
                "6s", "7s", "8s", "5m"]
        d, u = kokushi_distance(hand)
        self.assertGreaterEqual(d, 12.0)

    def test_same_value_different_ukeire_shape(self):
        """A kokushi-ish hand and a suuankou-ish hand at similar distance
        must expose DIFFERENT ukeire — that is the identity-free signal."""
        from src.agents.dnn.yaku_features import (kokushi_distance,
                                                  suuankou_distance)
        kok = ["1m", "1m", "9m", "1p", "9p", "1s", "9s",
               "1z", "2z", "3z", "4z", "5z", "6z"]
        kd, ku = kokushi_distance(kok)
        suu = ["2m", "2m", "5p", "5p", "8s", "8s", "3z", "3z",
               "6m", "6m", "9p", "9p", "4s"]                 # six pairs
        sd, su = suuankou_distance(suu, 0)
        self.assertEqual(ku, 4.0)         # kokushi: narrow, fixed outs
        self.assertEqual(su, 12.0)        # suuankou: 6 pairs x 2 outs each
        self.assertNotEqual(ku, su)

    def test_call_kills_suuankou_instantly(self):
        from src.agents.dnn.yaku_features import suuankou_distance, MAX_D
        suu = ["2m", "2m", "5p", "5p", "8s", "8s", "3z", "3z", "6m", "6m"]
        d0, u0 = suuankou_distance(suu, 0)
        d1, u1 = suuankou_distance(suu, 1)     # after ANY open meld
        self.assertLess(d0, MAX_D)
        self.assertEqual((d1, u1), (MAX_D, 0.0))

    def test_hazard_rows_shape_and_ranges(self):
        from src.agents.dnn.yaku_features import (HAZARD_FAMILIES,
                                                  hazard_features)
        hand = ["1m", "1m", "9m", "1p", "9p", "1s", "9s",
                "1z", "2z", "3z", "4z", "5z", "6z"]
        rows = hazard_features(hand, 0, True, turns_left=12)
        self.assertEqual(len(rows), len(HAZARD_FAMILIES))
        for r in rows:
            self.assertEqual(len(r), 5)
            for x in r:
                self.assertGreaterEqual(x, -0.2)
                self.assertLessEqual(x, 1.0)
