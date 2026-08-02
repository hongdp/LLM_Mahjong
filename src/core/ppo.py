"""PPO clipped-surrogate loss over action-token logprobs.

Kept as a pure-tensor function so the math is unit-testable without a model.
Convention: sequence-level advantage broadcast to that sequence's action
tokens; per-sequence token-mean, then batch mean (matches the REINFORCE
path's per-sequence mean NLL weighting).
"""

from typing import Dict, Tuple

import torch


def ppo_clip_loss(new_logprobs: torch.Tensor,
                  old_logprobs: torch.Tensor,
                  advantages: torch.Tensor,
                  mask: torch.Tensor,
                  clip_eps: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Args:
        new_logprobs: [B, T] current-policy logprobs of the taken tokens.
        old_logprobs: [B, T] behavior-policy logprobs recorded at rollout.
        advantages:   [B]    per-sequence advantage (already normalized/clipped).
        mask:         [B, T] 1.0 on action tokens, 0.0 elsewhere (padding/prompt).
        clip_eps:     PPO clip range epsilon.

    Returns:
        (loss, stats) — stats carries approx_kl and clip_frac for early-stop
        and monitoring. At new==old the gradient equals the REINFORCE
        (advantage-weighted NLL) gradient; that equivalence is unit-tested.
    """
    adv = advantages.unsqueeze(1)
    log_ratio = (new_logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped) * mask

    token_counts = mask.sum(dim=1).clamp(min=1.0)
    loss = (per_token.sum(dim=1) / token_counts).mean()

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denom
        clip_frac = ((torch.abs(ratio - 1.0) > clip_eps).float() * mask).sum() / denom

    return loss, {"approx_kl": approx_kl.item(), "clip_frac": clip_frac.item()}
