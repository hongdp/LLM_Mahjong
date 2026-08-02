import re
from typing import List

import torch

from src.core.base_reward import BaseRewardModel
from src.core.chat_format import visible_text
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds
from src.tasks.mahjong.table import ACTION_RE


class MahjongStepReward(BaseRewardModel):
    """
    Step-level tile-efficiency shaping.

    Only discard-quality shaping and format penalties live here — game
    actions (riichi/melds/wins) carry NO prior bonuses; their value must
    come from the end-of-game settlement distributed by the engine.
    """

    HAND_RE = re.compile(r'手牌: ((?:[1-9][mpsz] )*[1-9][mpsz])')
    FULU_RE = re.compile(r'私有[^\n]*?副露: ([^\n]*)')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.te = TileEfficiency()

    def compute_reward(self, prompts: List[str], responses: List[str],
                       **kwargs) -> List[torch.Tensor]:
        rewards = []
        for prompt, response in zip(prompts, responses):
            score = 0.0
            match = ACTION_RE.search(visible_text(response))

            if not match:
                score -= 10.0  # No action tag outside <think>
            else:
                action_type, tile, _with = match.groups()
                if action_type in ("discard", "riichi") and tile:
                    hand_match = self.HAND_RE.search(prompt)
                    hand = hand_match.group(1).split() if hand_match else []
                    if hand and tile not in hand:
                        score -= 5.0  # Discarding a tile not in hand
                    elif hand:
                        fulu_match = self.FULU_RE.search(prompt)
                        n_melds = fulu_match.group(1).count('(') if fulu_match else 0
                        try:
                            padded = pad_for_melds(hand, n_melds)
                            ranked = {
                                t: v for t, v in
                                self.te.evaluate_discards_ranked(padded).items()
                                if t in hand
                            }
                            if ranked and tile in ranked:
                                # Shanten first, ukeire second — rewarding
                                # raw ukeire alone favours hand regression.
                                min_sh = min(sh for sh, _ in ranked.values())
                                max_uk = max(len(uk) for sh, uk in ranked.values()
                                             if sh == min_sh)
                                ch_sh, ch_uk = (ranked[tile][0],
                                                len(ranked[tile][1]))
                                if ch_sh > min_sh:
                                    score -= 2.0 * (ch_sh - min_sh)
                                elif ch_uk == max_uk:
                                    score += 2.0
                                else:
                                    score -= (max_uk - ch_uk) * 0.5
                        except Exception:
                            pass  # Shaping is best-effort; never crash a rollout

            rewards.append(torch.tensor(score, device=self.device,
                                        dtype=torch.float32))
        return rewards
