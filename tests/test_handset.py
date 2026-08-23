"""exp27 HandSet encoder: permutation-invariant over tile instances, sees copies."""
import random
import unittest

import torch

from src.agents.dnn.arch_zoo import ZOO, HandSetEncoder
from src.agents.dnn.encoder import encode_state, N_PLANES_V1R, ACTION_DIM, TILE_TYPES
from src.tasks.mahjong.table import PyMahjongTable
from tests.test_engine import rig


class TestHandSet(unittest.TestCase):
    def _planes(self, hand, pid=0):
        t = PyMahjongTable(randomize_round=False)
        t.text_obs = False
        rig(t, pid, hand, drawn=hand[-1])
        p, s = encode_state(t, pid, variant="v1r")
        return p[None], s[None]

    def test_tokens_follow_instances_not_types(self):
        enc = HandSetEncoder(32, 1, 2).eval()
        p1, _ = self._planes(['3s', '3s', '2m', '5m', '8m', '1p', '4p', '7p', '2s', '6s', '9s', '1z', '4z', '7z'])
        p2, _ = self._planes(['3s', '4s', '2m', '5m', '8m', '1p', '4p', '7p', '2s', '6s', '9s', '1z', '4z', '7z'])
        self.assertEqual(int((p1[0, :4].sum())), 14)
        self.assertFalse(torch.allclose(enc(p1), enc(p2)))

    def test_permutation_invariance_via_planes(self):
        # the count planes are already order-free; the encoder must be a
        # function of them only — rigging the same multiset in any order
        # gives bit-identical planes and output
        enc = HandSetEncoder(32, 2, 2).eval()
        hand = ['3s', '3s', '2m', '5m', '8m', '1p', '4p', '7p', '2s', '6s', '9s', '1z', '4z', '7z']
        random.seed(1)
        outs = []
        for _ in range(3):
            h = list(hand); random.shuffle(h)
            p, _ = self._planes(h)
            outs.append(enc(p))
        self.assertTrue(torch.allclose(outs[0], outs[1]) and torch.allclose(outs[1], outs[2]))

    def test_zoo_nets_forward_and_act(self):
        for name in ("handset_cnn_m_r", "handset_pure_cnn_m_r"):
            net = ZOO[name][0]().eval()
            p = torch.rand(3, N_PLANES_V1R, TILE_TYPES).round()
            s = torch.rand(3, 20)
            m = torch.zeros(3, ACTION_DIM, dtype=torch.bool); m[:, :5] = True
            logits, v = net.forward_with_value(p, s, m)
            self.assertEqual(logits.shape, (3, ACTION_DIM)); self.assertEqual(v.shape, (3,))
            idx, lp = net.act(p, s, m, temperature=1.0)
            self.assertTrue(bool((idx < 5).all()))
            n = sum(x.numel() for x in net.parameters())
            self.assertLess(n, 6_000_000, name)


if __name__ == "__main__":
    unittest.main()
