"""Stable entry points (2026-11-01 structure clean-up).

Resident evaluators, the Majsoul service and the cloud ops scripts import
these by path and signature; the clean-up turned several of them into thin
wrappers over src/, and this test pins the public surface so a later
refactor cannot silently move or re-shape them.
"""

import inspect
import unittest

import torch


def _params(fn):
    return list(inspect.signature(fn).parameters)


class TestStableEntryPoints(unittest.TestCase):

    def test_play_pair_vector_signature(self):
        from scripts.run_elo_league import play_pair_vector
        self.assertEqual(_params(play_pair_vector),
                         ["path_a", "path_b", "deals", "seed0", "parallel",
                          "device", "temp_a", "temp_b", "hanchan"])

    def test_load_dnn_signature_and_identity(self):
        from scripts.run_arena_dnn import load_dnn
        from scripts.run_elo_league import load_dnn as via_league
        self.assertEqual(_params(load_dnn), ["path", "device"])
        self.assertIs(via_league, load_dnn)

    def test_eval_arena_signatures(self):
        from src.eval.arena import _cfg, play_table
        self.assertEqual(_params(_cfg), ["paths", "temps", "device", "seed0", "parallel"])
        self.assertEqual(_params(play_table),
                         ["eids", "walls", "seed0", "device", "parallel", "rotations"])

    def test_rollout_entry_points_exist(self):
        from src.agents.dnn.arch_zoo import ZOO
        from src.agents.dnn.parallel_rollout import collect_parallel
        from src.agents.dnn.rust_rollout import collect_rust
        self.assertTrue(callable(collect_parallel))
        self.assertTrue(callable(collect_rust))
        self.assertIsInstance(ZOO, dict)
        self.assertIn("cnn_m_v3r", ZOO)


class TestSharedImplementations(unittest.TestCase):

    def test_arena_cfg_matches_pair_cfg(self):
        """src.eval.arena._cfg and play_pair_vector build their rollout config
        from the same base; the table driver adds only its seating keys."""
        from src.eval.vec_arena import base_cfg
        net = torch.nn.Module()
        net.encoder_variant = "v3r"
        net.action_space = "mortal46"
        blob = {"channels": 32, "blocks": 2, "arch": "cnn_m_v3r", "symmetrize": None}
        cfg = base_cfg(net, blob, "cpu", 123)
        self.assertEqual(cfg["arch"], "cnn_m_v3r")
        self.assertEqual(cfg["encoder_variant"], "v3r")
        self.assertEqual(cfg["action_space"], "mortal46")
        self.assertEqual(cfg["seed"], 123)
        self.assertTrue(cfg["no_episodes"] and cfg["gpu_infer"])
        for k in ("arena", "table", "league", "hanchan", "symmetrize"):
            self.assertNotIn(k, cfg)          # driver-specific keys stay with the driver

    def test_probe_loop_reproduces_play_game(self):
        """The probes' shared loop makes the same engine calls as the trainer's
        play_game: at T=0 both reproduce the same deal from the same seed."""
        from src.agents.dnn.arch_zoo import ZOO
        from src.agents.dnn.selfplay import play_game
        from src.eval.probe_loop import play_probe_game
        torch.manual_seed(0)
        net = ZOO["cnn_m_v3r"][0]().eval()
        for seed in (11, 12):
            ref = play_game(net, 0.0, "cpu", deal_seed=seed)
            n_turns = []
            table = play_probe_game(net, seed, 0.0, "cpu",
                                    before_turn=lambda t, p, a: n_turns.append(p))
            self.assertEqual(list(table.points), ref.points)
            self.assertEqual(table.result_summary, ref.result)
            self.assertGreater(len(n_turns), 0)


if __name__ == "__main__":
    unittest.main()
