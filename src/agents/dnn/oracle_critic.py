"""exp108: SEPARATE oracle critics for the pure-line PPO (2026-10-12).

exp67 (2026-09-05) gave the privileged 319-dim oracle vector to a value head that
shared the policy's trunk, trained it one PPO pass per iteration for 1M games,
and measured zero gain (EV 0.102 vs 0.107). This module retests the idea in the
form that could actually win:

  * its own trunk (same CNN family as the policy), its own Adam, several passes
    per iteration -> the critic can be trained to convergence and never fights
    the policy for representation;
  * two heads: V(s) (asymmetric baseline + GAE bootstrap) and Q(s, a) over the
    374 native slots.  The Q form yields an all-action advantage
    A(s, a) = Q(s, a) - sum_a' pi(a'|s) Q(s, a') per state, so the policy gradient
    no longer rides one sampled action's +-5k-point return;
  * `blind=True` keeps the identical architecture and training but zeroes the
    oracle input - the control that separates "separate critic" from "oracle
    information".

The policy never reads the oracle vector; at play time nothing here exists.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.agents.dnn.encoder import ACTION_DIM, N_PLANES_V3R, N_SCALARS_V3, TILE_TYPES
from src.agents.dnn.net import ResBlock

ORACLE_DIM = 319


class SeparateCritic(nn.Module):
    def __init__(self, mode: str, channels: int = 128, blocks: int = 4, in_planes: int = N_PLANES_V3R,
                 in_scalars: int = N_SCALARS_V3, oracle_dim: int = ORACLE_DIM, action_dim: int = ACTION_DIM,
                 blind: bool = False):
        super().__init__()
        assert mode in ("v", "q")
        self.mode, self.blind, self.oracle_dim, self.action_dim = mode, blind, oracle_dim, action_dim
        self.stem = nn.Conv1d(in_planes, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.scalar_fc = nn.Sequential(nn.Linear(in_scalars, 64), nn.ReLU())
        self.oracle_fc = nn.Sequential(nn.Linear(oracle_dim, 128), nn.ReLU())
        d = channels * TILE_TYPES + 64 + 128
        out = 1 if mode == "v" else action_dim
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, out))

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor, cfeats: torch.Tensor) -> torch.Tensor:
        """planes [B,P,34] float, scalars [B,S], cfeats [B,319] -> V [B] or Q [B,A] (unmasked)."""
        h = torch.relu(self.stem(planes))
        h = self.blocks(h).flatten(1)
        if self.blind:
            cfeats = torch.zeros_like(cfeats)
        g = torch.cat([h, self.scalar_fc(scalars), self.oracle_fc(cfeats.to(h.dtype))], 1)
        out = self.head(g)
        return out.squeeze(-1) if self.mode == "v" else out


@torch.no_grad()
def q_advantage(q: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
    """All-action advantage from a Q table and the sampling policy's (masked) logits.
    Returns (adv [B], v_pi [B]) with v_pi = sum_a pi(a|s) Q(s,a) over legal actions."""
    pi = torch.softmax(logits.masked_fill(~mask, float("-inf")), 1)
    v_pi = (pi * q.masked_fill(~mask, 0.0)).sum(1)
    return q.gather(1, actions[:, None]).squeeze(1) - v_pi, v_pi


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - (target - pred).var() / (target.var() + 1e-9))
