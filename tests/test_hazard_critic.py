"""exp11 A1/A2 integration tests.

Locks the four properties the hazard-critic design depends on:
  1. completion labels parse correctly out of result_summary (winners only,
     double ron split, 放铳 mention never matches, draws all-zero);
  2. the POLICY path is bit-identical with and without critic features —
     privileged inputs reach only the value side (information parity);
  3. V decomposes as hazard component + residual and moves when a family's
     completion probability input moves;
  4. shape-aware loading lets a plain rollout worker consume A1/A2
     checkpoints (skipping critic keys) but refuses policy-key mismatches.
"""

import unittest

import numpy as np
import torch
from src.agents.dnn.encoder import ACTION_DIM

from src.agents.dnn.net import HazardHead, MahjongPolicyNet, load_compatible
from src.agents.dnn.selfplay import CFEAT_DIM, play_game
from src.agents.dnn.yaku_features import (HAZARD_FAMILIES, completion_labels)

IDX = {f: i for i, f in enumerate(HAZARD_FAMILIES)}


class TestCompletionLabels(unittest.TestCase):

    def test_tsumo_single_winner(self):
        s = "玩家2 自摸 | 6番30符 | Riichi, Menzen Tsumo, Chinitsu | 点数: [24000, 24000, 31000, 21000]"
        lab = completion_labels(s)
        self.assertEqual(lab[2][IDX["riichi_menzen"]], 1.0)
        self.assertEqual(lab[2][IDX["chinitsu"]], 1.0)
        self.assertEqual(lab[2][IDX["tanyao"]], 0.0)
        for p in (0, 1, 3):
            self.assertEqual(sum(lab[p]), 0.0)

    def test_ron_discarder_mention_never_matches(self):
        s = "玩家3 荣和(放铳:玩家1) | 2番40符 | Yakuhai (hatsu) | 点数: [25000, 22400, 25000, 27600]"
        lab = completion_labels(s)
        self.assertEqual(lab[3][IDX["yakuhai"]], 1.0)
        self.assertEqual(sum(lab[1]), 0.0)      # the deal-in seat gets nothing

    def test_double_ron_splits_segments(self):
        s = ("玩家0 荣和(放铳:玩家2) | 1番40符 | Tanyao ; "
             "玩家3 荣和(放铳:玩家2) | 2番25符 | Chiitoitsu | 双响 | 点数: [26300, 25000, 20500, 28200]")
        lab = completion_labels(s)
        self.assertEqual(lab[0][IDX["tanyao"]], 1.0)
        self.assertEqual(lab[0][IDX["chiitoi"]], 0.0)   # not cross-assigned
        self.assertEqual(lab[3][IDX["chiitoi"]], 1.0)
        self.assertEqual(lab[3][IDX["tanyao"]], 0.0)

    def test_yakuman_names(self):
        for name, fam in (("Kokushi Musou", "kokushi"), ("Suu Ankou", "suuankou")):
            lab = completion_labels(f"玩家1 自摸 | 13番0符 | {name} | 点数: [1,2,3,4]")
            self.assertEqual(lab[1][IDX[fam]], 1.0, name)

    def test_daburu_riichi_maps_to_riichi_family(self):
        lab = completion_labels("玩家0 荣和(放铳:玩家3) | 3番40符 | Daburu Riichi | 点数: [1,2,3,4]")
        self.assertEqual(lab[0][IDX["riichi_menzen"]], 1.0)

    def test_draw_and_empty_all_zero(self):
        for s in ("流局 | 听牌: [0, 2] | 点数: [26000, 24000, 26000, 24000]", ""):
            lab = completion_labels(s)
            self.assertEqual(sum(sum(v) for v in lab.values()), 0.0)


class TestHazardHead(unittest.TestCase):

    def test_shapes_and_value_component(self):
        head = HazardHead()
        cf = torch.rand(7, CFEAT_DIM["hazard"])
        logits = head(cf)
        self.assertEqual(logits.shape, (7, 9))
        v = head.value_component(cf)
        self.assertEqual(v.shape, (7,))
        # P<=1 and value feature <=1 bound the component by 9*32 return units
        self.assertTrue(bool((v.abs() <= 9 * 32.0).all()))

    def test_value_component_tracks_probability(self):
        """Pushing a family's distance from tenpai to hopeless must lower V's
        hazard component once the head has (briefly) learned d is bad."""
        torch.manual_seed(0)
        head = HazardHead()
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        close = torch.tensor([[0.0, 0.3, 1.0, 1.0, 0.5]]).repeat(64, 9).view(64, 45)
        far = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.5]]).repeat(64, 9).view(64, 45)
        for _ in range(200):
            x = torch.cat([close, far])
            y = torch.cat([torch.ones(64, 9), torch.zeros(64, 9)])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        self.assertGreater(float(head.value_component(close[:1])),
                           float(head.value_component(far[:1])))


class TestPolicyPurity(unittest.TestCase):
    """Critic features must never leak into the policy distribution."""

    def _batch(self, n=5):
        torch.manual_seed(1)
        from src.agents.dnn.encoder import ACTION_DIM, N_PLANES, N_SCALARS, TILE_TYPES
        planes = torch.rand(n, N_PLANES, TILE_TYPES)
        scal = torch.rand(n, N_SCALARS)
        mask = torch.zeros(n, ACTION_DIM, dtype=torch.bool)
        mask[:, :40] = True
        return planes, scal, mask

    def test_a1_logits_independent_of_cfeats(self):
        net = MahjongPolicyNet(channels=16, blocks=1, critic_feat_dim=4).eval()
        planes, scal, mask = self._batch()
        l1, v1 = net.forward_with_value(planes, scal, mask, cfeats=torch.zeros(5, 4))
        l2, v2 = net.forward_with_value(planes, scal, mask, cfeats=torch.rand(5, 4))
        self.assertTrue(torch.equal(l1, l2))
        self.assertFalse(torch.equal(v1, v2))   # the critic DOES react

    def test_a2_logits_independent_of_cfeats(self):
        net = MahjongPolicyNet(channels=16, blocks=1, hazard=True).eval()
        planes, scal, mask = self._batch()
        cf1 = torch.zeros(5, 45); cf2 = torch.rand(5, 45)
        l1, _ = net.forward_with_value(planes, scal, mask, cfeats=cf1)
        l2, _ = net.forward_with_value(planes, scal, mask, cfeats=cf2)
        self.assertTrue(torch.equal(l1, l2))
        self.assertTrue(torch.equal(l1, net(planes, scal, mask)))


class TestLoadCompatible(unittest.TestCase):

    def test_worker_net_consumes_a1_and_a2_checkpoints(self):
        for kw in (dict(critic_feat_dim=4), dict(hazard=True)):
            src = MahjongPolicyNet(channels=16, blocks=1, **kw)
            plain = MahjongPolicyNet(channels=16, blocks=1)
            skipped = load_compatible(plain, src.state_dict())
            for k in skipped:
                self.assertTrue(k.startswith("value"), k)
            x = torch.rand(2, plain.stem.in_channels, 34)
            s = torch.rand(2, plain.scalar_fc[0].in_features)
            m = torch.ones(2, ACTION_DIM, dtype=torch.bool)
            self.assertTrue(torch.allclose(plain(x, s, m), src(x, s, m)))

    def test_policy_mismatch_raises(self):
        big = MahjongPolicyNet(channels=32, blocks=1)
        small = MahjongPolicyNet(channels=16, blocks=1)
        with self.assertRaises(RuntimeError):
            load_compatible(small, big.state_dict())

    def test_arena_loader_handles_arch_none_and_critic_variants(self):
        """The arena loader must route arch=None checkpoints (every default
        CNN since exp10 stores the key) to MahjongPolicyNet, and consume
        exp11 critic-variant checkpoints (mismatched value head)."""
        import tempfile
        from scripts.run_arena_dnn import load_dnn
        for kw in (dict(), dict(critic_feat_dim=4), dict(hazard=True)):
            src = MahjongPolicyNet(channels=16, blocks=1, **kw).eval()
            blob = {"state_dict": src.state_dict(), "channels": 16,
                    "blocks": 1, "arch": None,
                    "critic_feats": "any", "games": 0, "iter": 0}
            with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                torch.save(blob, f.name)
                net = load_dnn(f.name, "cpu")
            x = torch.rand(2, net.stem.in_channels, 34)
            s = torch.rand(2, net.scalar_fc[0].in_features)
            m = torch.ones(2, ACTION_DIM, dtype=torch.bool)
            self.assertTrue(torch.allclose(net(x, s, m), src(x, s, m)),
                            str(kw))


class TestTrainerCallCompat(unittest.TestCase):
    """Every net the PPO trainer can construct must accept the cfeats kwarg
    (None included) — the trainer passes it unconditionally. The vit-r3
    cloud run died on exactly this (TypeError at startup)."""

    def test_all_zoo_nets_accept_cfeats_kwarg(self):
        from src.agents.dnn.arch_zoo import ZOO
        from src.agents.dnn.encoder import (ACTION_DIM, N_PLANES, N_PLANES_V2,
                                            N_PLANES_V3, N_SCALARS,
                                            N_SCALARS_V3, TILE_TYPES)
        for name, (factory, order) in ZOO.items():
            net = factory().eval()
            variant = getattr(net, "encoder_variant", "v1")
            from src.agents.dnn.encoder import VARIANT_SHAPE
            _, n_sc = VARIANT_SHAPE.get(variant, (None, N_SCALARS))
            n_pl = getattr(net, "in_planes", None) or (N_PLANES_V2 if order else N_PLANES)
            p = torch.rand(2, n_pl, TILE_TYPES)
            s = torch.rand(2, n_sc)
            m = torch.ones(2, ACTION_DIM, dtype=torch.bool)
            logits, v = net.forward_with_value(p, s, m, cfeats=None)
            self.assertEqual(v.shape, (2,), name)


class TestRolloutCarriesFeatures(unittest.TestCase):

    def test_play_game_hazard_features_shape(self):
        torch.manual_seed(0)
        net = MahjongPolicyNet(channels=16, blocks=1).eval()
        g = play_game(net, deal_seed=123, critic_feats="hazard")
        steps = [s for pid in range(4) for s in g.trajectories[pid]]
        self.assertGreater(len(steps), 0)
        for s in steps:
            self.assertIsNotNone(s.cfeats)
            self.assertEqual(tuple(s.cfeats.shape), (45,))
            self.assertTrue(bool(torch.isfinite(s.cfeats).all()))

    def test_play_game_none_mode_has_no_cfeats(self):
        torch.manual_seed(0)
        net = MahjongPolicyNet(channels=16, blocks=1).eval()
        g = play_game(net, deal_seed=123, critic_feats="none")
        for pid in range(4):
            for s in g.trajectories[pid]:
                self.assertIsNone(s.cfeats)


if __name__ == "__main__":
    unittest.main()
