"""Privileged (oracle) CRITIC-ONLY features (exp67, 2026-09-05).

The literature's consistent answer to mahjong's terminal-reward noise is an
asymmetric critic: the value function sees the hidden state (Suphx oracle
guiding, RVR's relative value network, PerfectDou's perfect-information
critic), the policy never does. These features ride in the `cfeats` tensor
that only `forward_with_value` consumes, exactly like exp11's profile/hazard
features, so the actor keeps its information parity.

Layout (relative seats off = 1..3 downstream of `pid`, all values in [0, 1]):
  [0:102)    opponents' concealed hand counts / 4        (3 x 34)
  [102:111)  opponents' red fives in hand (m, p, s)        (3 x 3)
  [111:213)  opponents' waits (1 = this tile completes)    (3 x 34)
  [213:216)  opponents' shanten / 6, clamped               (3)
  [216:250)  live wall composition counts / 4              (34)
  [250]      live wall length / 70
  [251:285)  next kan-dora INDICATOR (hidden) one-hot      (34)
  [285:319)  ura indicator under the first dora one-hot    (34)
"""
from __future__ import annotations

import numpy as np
import torch

ORACLE_DIM = 319
_SUITS = "mps"


def _t34(tile: str) -> int:
    t = tile.replace("*", "")
    v, s = int(t[0]) or 5, t[1]
    return 27 + v - 1 if s == "z" else _SUITS.index(s) * 9 + v - 1


def oracle_features(table, pid: int) -> torch.Tensor:
    """Privileged view of `table` from seat `pid` (never shown to the policy)."""
    f = np.zeros(ORACLE_DIM, dtype=np.float32)
    for off in range(1, 4):
        opp = (pid + off) % 4
        base = (off - 1) * 34
        for t in table.hands[opp]:
            f[base + _t34(t)] += 0.25
        red = table.red[opp]
        for k, s in enumerate(_SUITS):
            f[102 + (off - 1) * 3 + k] = float(red[s])
        try:
            if len(table.hands[opp]) % 3 == 1:            # 13-tile state: waits are defined
                for w in table._waits(opp):
                    f[111 + base + _t34(w)] = 1.0
            sh = table._shanten(list(table.hands[opp]), len(table.melds[opp]))
            f[213 + off - 1] = min(max(sh + 1, 0), 6) / 6.0   # -1 (complete) -> 0
        except Exception:                                  # noqa: BLE001
            pass
    for t in table.wall:
        f[216 + _t34(t)] += 0.25
    f[250] = len(table.wall) / 70.0
    try:
        nxt = 4 + len(table.dora_indicators)
        if nxt < 9:
            f[251 + _t34(table.dead_wall[nxt])] = 1.0
        f[285 + _t34(table.dead_wall[9])] = 1.0
    except Exception:                                      # noqa: BLE001
        pass
    return torch.from_numpy(f)
