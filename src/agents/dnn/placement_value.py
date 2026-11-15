"""Placement-value network W: between-deal match state -> E[final uma | state].

Small MLP with two heads on a shared trunk: placement 4-way CE (calibration
diagnostics) and uma regression (the reward signal). Per-deal credit is the
telescoping difference W(after) - W(before).

The class lives here because library code loads its checkpoints
(`hanchan_credit.PureW`, `tasks/mahjong/hanchan.py`); the trainers that fit
it are `scripts/train_placement_value.py` (human houou states, exp55-D) and
`scripts/fit_pure_w.py` (pure self-play states). The feature layout is owned
by `scripts/extract_placement_states.py`.
"""
import torch.nn as nn


class PlacementValue(nn.Module):
    def __init__(self, d_in=12, width=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.place = nn.Linear(width, 4)
        self.uma = nn.Linear(width, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.place(h), self.uma(h).squeeze(-1)
