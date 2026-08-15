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


class MahjongPolicyNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 3):
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
        self.value = nn.Sequential(
            nn.Linear(channels * TILE_TYPES + 64, 256), nn.ReLU(),
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

    def forward_with_value(self, planes, scalars, mask):
        h = self.trunk(planes, scalars)
        logits = self.head(h).masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)

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
