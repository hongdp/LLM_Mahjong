"""Conventional policy network for the mahjong baseline.

Deliberately small and ordinary: a 1-D residual CNN over the 34 tile types
(the standard trick in mahjong DNNs — convolving across tile index lets
the net share structure between e.g. 3m4m5m and 6p7p8p), then a head that
scores the fixed action space. ~250k parameters: trains in hours on one
consumer GPU, which is the whole point of the comparison.
"""

import torch
import torch.nn as nn

from src.agents.dnn.encoder import ACTION_DIM, N_PLANES, N_SCALARS, TILE_TYPES


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n1 = nn.BatchNorm1d(ch)
        self.n2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        h = torch.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return torch.relu(x + h)


class HazardHead(nn.Module):
    """Shared completion-hazard head (exp11 A2, docs/design_hazard_critic.md).

    ONE MLP scores EVERY family row [d, u, closed_ok, value, turns_left] to
    P(complete family | s) — families have no identity, only dynamics, which
    is what lets rates learned on common families transfer to kokushi and
    suuankou (same 32000 value, different (d, u) shape). The family value is
    a MULTIPLIER on P, never a regression target, so a yakuman's 32000
    reaches V(s) without surviving advantage clipping.
    """

    N_FAMILIES, ROW_DIM = 9, 5
    # rows carry value/32000; convert to return units (points x REWARD_SCALE)
    VALUE_SCALE = 32000 * 0.001

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(self.ROW_DIM, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))

    def forward(self, cfeats: torch.Tensor) -> torch.Tensor:
        """cfeats [B, 45] -> per-family completion logits [B, 9]."""
        rows = cfeats.view(-1, self.N_FAMILIES, self.ROW_DIM)
        return self.g(rows).squeeze(-1)

    def value_component(self, cfeats: torch.Tensor) -> torch.Tensor:
        """sum_y P_y(s) * value_y, in return units. [B]"""
        rows = cfeats.view(-1, self.N_FAMILIES, self.ROW_DIM)
        p = torch.sigmoid(self.g(rows).squeeze(-1))
        return (p * rows[:, :, 3] * self.VALUE_SCALE).sum(dim=1)


class MahjongPolicyNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 3,
                 critic_feat_dim: int = 0, hazard: bool = False):
        super().__init__()
        self.stem = nn.Conv1d(N_PLANES, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.scalar_fc = nn.Sequential(nn.Linear(N_SCALARS, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(channels * TILE_TYPES + 64, 512), nn.ReLU(),
            nn.Linear(512, ACTION_DIM),
        )
        # Optional critic. Measured first (four probes, EV ~0.02-0.03), so it
        # is NOT here for variance reduction — V's spread is only ~15% of the
        # return's, which is what lets a 0-return draw carry a non-zero
        # advantage instead of vanishing.
        # exp11 privileged variants (critic-only inputs; the POLICY path never
        # touches cfeats, preserving information parity with the LLM):
        #   critic_feat_dim>0 (A1): cfeats concatenated into the value input.
        #   hazard (A2): V = hazard value component + residual(trunk); the
        #   residual keeps plain trunk input so the decomposition stays clean.
        self.critic_feat_dim = critic_feat_dim
        self.hazard = hazard
        self.hazard_head = HazardHead() if hazard else None
        self.value = nn.Sequential(
            nn.Linear(channels * TILE_TYPES + 64 + critic_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def trunk(self, planes: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(planes))
        h = self.blocks(h).flatten(1)
        return torch.cat([h, self.scalar_fc(scalars)], dim=1)

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """planes [B,P,34], scalars [B,S], mask [B,A] -> masked logits [B,A]."""
        logits = self.head(self.trunk(planes, scalars))
        # -inf on illegal actions so softmax/sampling can never pick them
        return logits.masked_fill(~mask, float("-inf"))

    def forward_with_value(self, planes, scalars, mask, cfeats=None):
        h = self.trunk(planes, scalars)
        logits = self.head(h).masked_fill(~mask, float("-inf"))
        if self.hazard:
            v = self.value(h).squeeze(-1) + self.hazard_head.value_component(cfeats)
        elif self.critic_feat_dim:
            v = self.value(torch.cat([h, cfeats], dim=1)).squeeze(-1)
        else:
            v = self.value(h).squeeze(-1)
        return logits, v

    @torch.no_grad()
    def act(self, planes, scalars, mask, temperature: float = 1.0):
        """Sample one legal action index per row. Returns (idx, logprob).

        Every row must have at least one legal action: an all-masked row
        makes softmax return NaN and multinomial raise a device-side
        assert. Callers handle the no-legal-action case themselves.
        """
        if not bool(mask.any(dim=1).all()):
            raise ValueError("act() got a row with no legal actions; "
                             "the caller must handle empty legal lists")
        logits = self.forward(planes, scalars, mask)
        if temperature <= 0:
            idx = logits.argmax(dim=1)
        else:
            probs = torch.softmax(logits / temperature, dim=1)
            idx = torch.multinomial(probs, 1).squeeze(1)
        logprob = torch.log_softmax(logits, dim=1).gather(1, idx[:, None]).squeeze(1)
        return idx, logprob


def load_compatible(net: nn.Module, state: dict) -> list:
    """Load every key whose shape matches; return the skipped model keys.

    strict=False does NOT excuse shape mismatches, and exp11's critic
    variants change the value head's input width — so rollout workers
    (which never call the value path) and cross-variant warm starts load
    through this instead. Skipping critic keys ("value*", "hazard_head*")
    is expected; skipping a POLICY key means the checkpoint doesn't match
    the net and raises. Works for any zoo net whose critic params are
    namespaced under "value"/"hazard_head" (all of ours are).
    """
    model_sd = net.state_dict()
    ok = {k: v for k, v in state.items()
          if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    net.load_state_dict(ok, strict=False)
    bad = [k for k in model_sd if k not in ok
           and not k.startswith(("value", "hazard_head"))]
    if bad:
        raise RuntimeError(f"policy keys failed to load: {bad[:5]}")
    return [k for k in model_sd if k not in ok]
