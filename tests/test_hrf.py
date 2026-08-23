"""exp30 HandRiverFormer: v4 event buffer fidelity + model API."""
import random
import unittest

import torch

from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn.encoder import (encode_state, EV_MAX, EV_F, N_PLANES_V1R,
                                    N_PLANES_V4, ACTION_DIM, TILE_TYPES)
from src.tasks.mahjong.table import PyMahjongTable
from src.agents.dnn.selfplay import play_game


class TestV4Events(unittest.TestCase):
    def _played_table(self, seed):
        net = ZOO["cnn_m_r"][0]().eval()
        random.seed(seed)
        t = PyMahjongTable(randomize_round=True)
        t.text_obs = False
        # play a few turns with the small net to populate rivers
        for _ in range(40):
            if t.finished:
                break
            pid = t.turn
            acts = t.get_legal_actions(pid)
            if not acts:
                break
            p, s = encode_state(t, pid, variant="v1r")
            from src.agents.dnn.encoder import legal_mask
            m, lk = legal_mask(acts)
            i, _ = net.act(p[None], s[None], m[None], temperature=1.0)
            t.step(pid, lk[int(i)])
            if not t.finished:
                t.advance_turn()
        return t

    def test_event_buffer_matches_river_events(self):
        t = self._played_table(11)
        for me in range(4):
            p, s = encode_state(t, me, variant="v4")
            buf = p[N_PLANES_V1R:].reshape(-1)
            n = int(buf[-1])
            ev = buf[:EV_MAX * EV_F].reshape(EV_MAX, EV_F)[:n]
            n_disc = sum(len(t.river_events[q]) for q in range(4))
            disc = [row for row in ev.tolist() if row[3] == 0]
            self.assertEqual(len(disc), n_disc)
            # recency: exactly one newest discard with recency 0 & last flag consistency
            recs = sorted(int(r[6]) for r in disc)
            self.assertEqual(recs, list(range(n_disc)))
            last = [r for r in disc if int(r[7]) & 16]
            if t.last_discard:
                self.assertEqual(len(last), 1)
                self.assertEqual(int(last[0][6]), 0)
            # junme/seat spot check: token seat offsets map back to the actual rivers
            for row in disc[:5]:
                off, j = int(row[4]), int(row[5])
                pid = (me + off) % 4
                tile, *_ = t.river_events[pid][j]
                i34_row = (int(row[1]) * 9 + int(row[0]) - 1) if int(row[0]) else 27 + int(row[2]) - 1
                from src.agents.dnn.encoder import tile_to_34
                self.assertEqual(i34_row, tile_to_34(tile.replace('*', '')))

    def test_model_forward_act_and_play(self):
        net = ZOO["hrf_xl_v4"][0]().eval()
        self.assertEqual(net.encoder_variant, "v4")
        t = self._played_table(12)
        me = t.turn
        acts = t.get_legal_actions(me) or ['<action type="skip" />']
        from src.agents.dnn.encoder import legal_mask
        p, s = encode_state(t, me, variant="v4")
        m, _ = legal_mask(acts)
        logits, v = net.forward_with_value(p[None], s[None], m[None])
        self.assertEqual(logits.shape, (1, ACTION_DIM))
        self.assertTrue(bool(torch.isfinite(v).all()))
        g = play_game(net, temperature=1.0, deal_seed=77)
        self.assertTrue(g.result is not None or sum(len(x) for x in g.trajectories.values()) > 40)

    def test_ablation_variants_forward(self):
        t = self._played_table(13)
        p, s = encode_state(t, 0, variant="v4")
        m = torch.zeros(ACTION_DIM, dtype=torch.bool); m[:6] = True
        for name in ("hrf_xl_nocross_v4", "hrf_xl_notime_v4", "hrf_xl_freerank_v4"):
            net = ZOO[name][0]().eval()
            lg, v = net.forward_with_value(p[None], s[None], m[None])
            self.assertTrue(bool(torch.isfinite(lg[m[None]]).all()), name)


if __name__ == "__main__":
    unittest.main()
