"""Encoder / action-space tests for the conventional-DNN baseline.

The critical invariant is round-tripping: every action string the engine
offers must map to a distinct index, and that index must map back to the
same string. A silent collision would make the agent play a different
move than the one it scored.
"""

import random
import unittest

import torch

from src.agents.dnn.encoder import (ACTION_DIM, N_PLANES, N_SCALARS,
                                    TILE_TYPES, action_to_index, encode_state,
                                    legal_mask, tile_to_34)
from src.tasks.mahjong.table import PyMahjongTable


class TestTileIndex(unittest.TestCase):
    def test_all_34_tiles_distinct(self):
        tiles = [f"{v}{s}" for s in "mps" for v in range(1, 10)]
        tiles += [f"{v}z" for v in range(1, 8)]
        idx = [tile_to_34(t) for t in tiles]
        self.assertEqual(len(set(idx)), 34)
        self.assertEqual(min(idx), 0)
        self.assertEqual(max(idx), 33)

    def test_riichi_star_marker_ignored(self):
        self.assertEqual(tile_to_34("3p*"), tile_to_34("3p"))


class TestActionSpace(unittest.TestCase):
    def test_types_map_to_disjoint_ranges(self):
        a = action_to_index('<action type="discard" tile="1m" />')
        b = action_to_index('<action type="riichi" tile="1m" />')
        self.assertIsNotNone(a)
        self.assertNotEqual(a, b, "discard and riichi of the same tile must differ")

    def test_chi_variants_distinct(self):
        """Three chi shapes on the same called tile must not collide."""
        acts = ['<action type="chi" tile="3m" with="1m 2m" />',
                '<action type="chi" tile="3m" with="2m 4m" />',
                '<action type="chi" tile="3m" with="4m 5m" />']
        idx = [action_to_index(a) for a in acts]
        self.assertEqual(len(set(idx)), 3, f"chi collision: {idx}")

    def test_typeless_actions_parse(self):
        for a in ('<action type="skip" />', '<action type="ron" />',
                  '<action type="tsumo" />'):
            self.assertIsNotNone(action_to_index(a), a)

    def test_mask_roundtrip_on_real_engine_actions(self):
        """Play random games; every legal list must round-trip losslessly."""
        random.seed(7)
        checked = 0
        for game in range(6):
            table = PyMahjongTable(randomize_round=True)
            for _ in range(400):
                if table.finished:
                    break
                pid = table.turn
                actions = table.get_legal_actions(pid)
                if not actions:
                    break
                mask, lookup = legal_mask(actions)
                self.assertEqual(len(lookup), len(actions),
                                 f"index collision in {actions}")
                self.assertEqual(int(mask.sum()), len(actions))
                for idx, a in lookup.items():
                    self.assertEqual(action_to_index(a), idx)
                checked += 1
                table.step(pid, random.choice(actions))
                # exercise the interrupt list too
                for other in range(4):
                    if other == pid:
                        continue
                    iacts = table.get_interrupt_actions(other)
                    m2, l2 = legal_mask(iacts)
                    self.assertEqual(len(l2), len(iacts),
                                     f"interrupt collision in {iacts}")
        self.assertGreater(checked, 50, "test did not exercise enough states")


class TestStateEncoding(unittest.TestCase):
    def test_shapes_and_ranges(self):
        random.seed(1)
        table = PyMahjongTable(randomize_round=True)
        planes, scalars = encode_state(table, 0)
        self.assertEqual(planes.shape, (N_PLANES, TILE_TYPES))
        self.assertEqual(scalars.shape, (N_SCALARS,))
        self.assertTrue(torch.isfinite(planes).all())
        self.assertTrue(torch.isfinite(scalars).all())
        self.assertLessEqual(planes.max().item(), 1.0)

    def test_hand_planes_match_hand(self):
        random.seed(2)
        table = PyMahjongTable(randomize_round=False)
        planes, _ = encode_state(table, 0)
        hand = table.hands[0]
        for t in set(hand):
            self.assertEqual(planes[0][tile_to_34(t)].item(), 1.0,
                             f"{t} missing from hand plane")
        # a tile held twice must light the >=2 plane
        for t in set(hand):
            expect = 1.0 if hand.count(t) >= 2 else 0.0
            self.assertEqual(planes[1][tile_to_34(t)].item(), expect)

    def test_perspective_rotation(self):
        """Seat 1's own-hand plane must equal seat 1's hand, not seat 0's."""
        random.seed(3)
        table = PyMahjongTable(randomize_round=False)
        p0, _ = encode_state(table, 0)
        p1, _ = encode_state(table, 1)
        self.assertFalse(torch.equal(p0[0], p1[0]))
        for t in set(table.hands[1]):
            self.assertEqual(p1[0][tile_to_34(t)].item(), 1.0)

    def test_no_shanten_leak(self):
        """Information fairness: the encoder must not exceed prompt content.
        Scalars are bounded small numbers; a shanten value would break that."""
        random.seed(4)
        table = PyMahjongTable(randomize_round=False)
        _, s = encode_state(table, 0)
        self.assertLessEqual(s.abs().max().item(), 2.0)


class TestNetwork(unittest.TestCase):
    def test_masked_sampling_only_picks_legal(self):
        from src.agents.dnn.net import MahjongPolicyNet
        random.seed(5)
        torch.manual_seed(0)
        net = MahjongPolicyNet(channels=16, blocks=1).eval()
        table = PyMahjongTable(randomize_round=False)
        actions = table.get_legal_actions(table.turn)
        mask, lookup = legal_mask(actions)
        planes, scalars = encode_state(table, table.turn)
        for _ in range(20):
            idx, lp = net.act(planes[None], scalars[None], mask[None])
            self.assertIn(int(idx), lookup, "sampled an illegal action")
            self.assertTrue(torch.isfinite(lp).all())

    def test_logits_are_minus_inf_off_mask(self):
        from src.agents.dnn.net import MahjongPolicyNet
        net = MahjongPolicyNet(channels=16, blocks=1).eval()
        mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
        mask[0, 5] = True
        out = net(torch.zeros(1, N_PLANES, TILE_TYPES),
                  torch.zeros(1, N_SCALARS), mask)
        self.assertTrue(torch.isinf(out[0, 0]) and out[0, 0] < 0)
        self.assertTrue(torch.isfinite(out[0, 5]))


if __name__ == "__main__":
    unittest.main()
