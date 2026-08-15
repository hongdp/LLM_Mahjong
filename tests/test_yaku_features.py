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
