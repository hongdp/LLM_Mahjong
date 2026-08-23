"""State-value head for variance reduction (exp4_critic).

V(s) is read from the policy trunk's hidden state at the LAST PROMPT TOKEN
(the state summary right before generation starts). The head is a small
fp32 MLP trained with MSE against return-to-go; the trunk is NOT trained
through the value loss (hidden states are detached) so the policy
representation cannot be degraded by value fitting.

Motivation (exp2): initial-hand energy explains only 2% of settlement
variance and arena CIs are +-1500-1800 points at 64 duplicate deals —
the missing variance lives in mid-game state, which is exactly what the
prompt hidden state encodes.
"""

from typing import Dict

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, inner: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        # near-zero init on the output layer: V starts ~0 so early advantages
        # equal the raw returns (no cold-start distortion of the policy loss).
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [..., H] (any leading shape) -> V: [...] (squeezed)."""
        return self.net(hidden.float()).squeeze(-1)


def last_prompt_hidden(hidden_states: torch.Tensor,
                       prompt_lens: torch.Tensor) -> torch.Tensor:
    """Gather the hidden vector at index p_len-1 for each row.

    hidden_states: [B, T, H]; prompt_lens: [B] (int). Returns [B, H].
    """
    idx = (prompt_lens.long() - 1).clamp(min=0)
    b = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[b, idx]


def explained_variance(v_pred: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var[G - V]/Var[G]; <=0 means the head is useless, 1 is perfect."""
    var_g = returns.float().var(unbiased=False)
    if var_g < 1e-12:
        return 0.0
    return (1.0 - (returns.float() - v_pred.float()).var(unbiased=False)
            / var_g).item()


def save_value_head(head: ValueHead, path: str) -> None:
    torch.save({"state_dict": head.state_dict(),
                "hidden_size": head.net[0].in_features,
                "inner": head.net[0].out_features}, path)


def load_value_head(path: str, device="cpu") -> ValueHead:
    blob: Dict = torch.load(path, map_location=device)
    head = ValueHead(blob["hidden_size"], blob["inner"])
    head.load_state_dict(blob["state_dict"])
    return head.to(device)
