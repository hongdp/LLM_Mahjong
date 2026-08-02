import re
from typing import List

import torch

from src.core.base_reward import BaseRewardModel
from src.core.chat_format import visible_text
from src.tasks.mahjong.shanten import (TileEfficiency, pad_for_melds,
                                       dora_from_indicator)
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


class MahjongPotentialReward(BaseRewardModel):
    """
    Energy-consistent shaping via potential-based reward shaping (PBRS,
    Ng/Harada/Russell 1999). Replaces MahjongStepReward's absolute per-step
    scores, which are inconsistent with the terminal settlement: +2 per
    optimal discard accumulates with episode length (a 20-step trajectory
    farms +40 of shaping against a settlement scale of ±4).

    Energy of a 13-tile-equivalent afterstate hand h:

        Phi(h) = -C_SHANTEN * shanten(h) + C_UKEIRE * |ukeire(h)|

    C_UKEIRE * 34 < C_SHANTEN, so the ukeire term can never reorder shanten
    levels (same lexicographic principle as evaluate_discards_ranked).

    Shaping over one player's trajectory with afterstate energies
    psi_0..psi_{n-1} and pre-deal reference psi_pre (best reachable energy
    of the step-0 hand):

        F_0     = gamma * psi_0    - psi_pre
        F_i     = gamma * psi_i    - psi_{i-1}     (0 < i < n-1)
        F_{n-1} =        0         - psi_{n-2}     (terminal energy := 0;
                                                    the settlement carries
                                                    all terminal value)

    The discounted sum telescopes to exactly -psi_pre — a constant of the
    deal, independent of the actions taken. Hence the shaped return equals
    the true (settlement) return plus a policy-independent constant:
    intermediate and final rewards are consistent by construction and the
    shaping cannot be farmed.

    Format/legality penalties are kept as explicit *constraint* terms
    outside the energy (same constants as MahjongStepReward); they vanish
    once compliance reaches 1.0.

    Approximations (documented, deliberate):
    - Illegal actions are rolled back by the engine; we still use the
      attempted afterstate. The -5 constraint penalty dominates the error.
    - Interrupt decisions (pon/chi/kan/ron/skip prompts) keep the hand
      unchanged; their energy is the hand's own energy.
    """

    C_SHANTEN = 2.0
    C_UKEIRE = 0.05
    FORMAT_PENALTY = -10.0
    GHOST_TILE_PENALTY = -5.0

    HAND_RE = MahjongStepReward.HAND_RE
    FULU_RE = MahjongStepReward.FULU_RE
    DORA_IND_RE = re.compile(r'宝牌指示牌: ([^,\n]+)')

    def __init__(self, gamma: float = 0.99, dora_weight: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        # Optional value term in the energy: dora_weight * (#dora held in
        # hand + own melds). Any state function is a valid PBRS potential,
        # so this redirects credit toward keeping value tiles WITHOUT
        # changing the telescoped total or the optimal policy. Kept well
        # below C_SHANTEN so speed pressure still dominates.
        self.dora_weight = dora_weight
        self.te = TileEfficiency()

    # ---- energy helpers -------------------------------------------------

    def _dora_bonus(self, tiles, dora_tiles, meld_dora=0):
        if not self.dora_weight or not dora_tiles:
            return 0.0
        held = sum(1 for t in tiles if t in dora_tiles) + meld_dora
        return self.dora_weight * held

    def _energy(self, tiles, n_melds, dora_tiles=(), meld_dora=0):
        """Phi of a 13-tile-equivalent hand; None if not computable."""
        try:
            padded = pad_for_melds(tiles, n_melds)
            sh = self.te.calculate_shanten(padded)
            uk = len(self.te.calculate_ukeire(padded))
            return (-self.C_SHANTEN * sh + self.C_UKEIRE * uk
                    + self._dora_bonus(tiles, dora_tiles, meld_dora))
        except Exception:
            return None

    def _pre_energy(self, tiles, n_melds, dora_tiles=(), meld_dora=0):
        """Best reachable energy of a pre-action hand (state function)."""
        if len(tiles) % 3 == 2:  # 14-tile decision: best post-discard energy
            try:
                padded = pad_for_melds(tiles, n_melds)
                ranked = self.te.evaluate_discards_ranked(padded)
                cands = []
                for t, (sh, uk) in ranked.items():
                    if t not in tiles:
                        continue
                    after = list(tiles)
                    after.remove(t)
                    cands.append(-self.C_SHANTEN * sh
                                 + self.C_UKEIRE * len(uk)
                                 + self._dora_bonus(after, dora_tiles,
                                                    meld_dora))
                if cands:
                    return max(cands)
            except Exception:
                return None
        return self._energy(tiles, n_melds, dora_tiles, meld_dora)

    MELD_TILES_RE = re.compile(r'\w+\(([^)]*)\)')

    def _parse_step(self, prompt, response):
        hand_match = self.HAND_RE.search(prompt)
        hand = hand_match.group(1).split() if hand_match else []
        fulu_match = self.FULU_RE.search(prompt)
        fulu_txt = fulu_match.group(1) if fulu_match else ""
        n_melds = fulu_txt.count('(')
        dora_tiles, meld_dora = (), 0
        if self.dora_weight:
            ind_match = self.DORA_IND_RE.search(prompt)
            if ind_match:
                dora_tiles = tuple(dora_from_indicator(i)
                                   for i in ind_match.group(1).split())
                meld_tiles = [t for m in self.MELD_TILES_RE.finditer(fulu_txt)
                              for t in m.group(1).split()]
                meld_dora = sum(1 for t in meld_tiles if t in dora_tiles)
        match = ACTION_RE.search(visible_text(response))
        return hand, n_melds, match, dora_tiles, meld_dora

    # ---- main entry ------------------------------------------------------

    def compute_reward(self, prompts, responses, **kwargs):
        """Treats (prompts, responses) as ONE ordered player trajectory —
        exactly how MahjongTask.collect_rollouts calls it per episode."""
        n = len(prompts)
        constraint = [0.0] * n     # format/legality penalties
        psi = [None] * n           # afterstate energies

        psi_pre = None
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            hand, n_melds, match, dora_tiles, meld_dora = self._parse_step(
                prompt, response)
            if i == 0 and hand:
                psi_pre = self._pre_energy(hand, n_melds, dora_tiles,
                                           meld_dora)

            after = hand
            if not match:
                constraint[i] += self.FORMAT_PENALTY
            else:
                action_type, tile, _with = match.groups()
                if action_type in ("discard", "riichi") and tile:
                    if hand and tile not in hand:
                        constraint[i] += self.GHOST_TILE_PENALTY
                    elif hand:
                        after = list(hand)
                        after.remove(tile)
            psi[i] = (self._energy(after, n_melds, dora_tiles, meld_dora)
                      if after else None)

        # Fill unparseable energies by carrying the previous value forward
        # (contributes only (gamma-1)*psi of drift, negligible at 0.99).
        prev = psi_pre if psi_pre is not None else 0.0
        for i in range(n):
            if psi[i] is None:
                psi[i] = prev
            prev = psi[i]

        if psi_pre is None:
            psi_pre = psi[0] if n else 0.0

        rewards = []
        for i in range(n):
            prev_e = psi_pre if i == 0 else psi[i - 1]
            if i == n - 1:
                shaping = -prev_e            # terminal afterstate energy := 0
            else:
                shaping = self.gamma * psi[i] - prev_e
            rewards.append(torch.tensor(shaping + constraint[i],
                                        device=self.device,
                                        dtype=torch.float32))
        return rewards


class MahjongSettlementOnly(BaseRewardModel):
    """Pure-objective training: NO shaping. Steps carry only the legality
    constraints (no-action-tag -10, ghost discard -5); all learning signal
    comes from the engine's terminal settlement already merged into the
    trajectory. Motivation (2026-08-02): with a competent SFT prior the
    dense PBRS channel dominates transient learning dynamics (arena showed
    large style migration with no strength gain over the SFT anchor); this
    model lets the settlement gradient speak alone. Pair with
    --covariate_baseline and large game batches to manage variance."""

    FORMAT_PENALTY = MahjongPotentialReward.FORMAT_PENALTY
    GHOST_TILE_PENALTY = MahjongPotentialReward.GHOST_TILE_PENALTY
    HAND_RE = MahjongStepReward.HAND_RE

    def compute_reward(self, prompts, responses, **kwargs):
        rewards = []
        for prompt, response in zip(prompts, responses):
            score = 0.0
            match = ACTION_RE.search(visible_text(response))
            if not match:
                score += self.FORMAT_PENALTY
            else:
                a_type, tile, _w = match.groups()
                if a_type in ("discard", "riichi") and tile:
                    hm = self.HAND_RE.search(prompt)
                    if hm and tile not in hm.group(1).split():
                        score += self.GHOST_TILE_PENALTY
            rewards.append(torch.tensor(score, device=self.device,
                                        dtype=torch.float32))
        return rewards


# Modular reward selection (CLAUDE.md: registry + BaseRewardModel, never
# hardcoded into the training loop). Configs pick via "reward_model".
def _potential_value(**kwargs):
    kwargs.setdefault("dora_weight", 0.3)
    return MahjongPotentialReward(**kwargs)


REWARD_MODELS = {
    "step": MahjongStepReward,            # legacy absolute shaping (v2 runs)
    "potential": MahjongPotentialReward,  # energy-consistent PBRS
    "potential_value": _potential_value,  # PBRS + dora term in the energy
    "settlement": MahjongSettlementOnly,  # constraints only; pure objective
}


# ---------------------------------------------------------------------------
# Deal-quality covariate (for the buffer-level control-variate baseline).
# Action-independent by construction: computed from the FIRST observation of
# an episode, before any policy decision — subtracting any function of it
# from returns leaves the expected policy gradient unchanged.
_COV_MODEL = None


def initial_hand_energy(prompt: str) -> float:
    """Best reachable structural energy of the episode's starting hand
    (dora_weight=0: pure speed quality). Returns 0.0 if unparsable."""
    global _COV_MODEL
    if _COV_MODEL is None:
        _COV_MODEL = MahjongPotentialReward(device="cpu", dora_weight=0.0)
    hand_match = _COV_MODEL.HAND_RE.search(prompt)
    if not hand_match:
        return 0.0
    hand = hand_match.group(1).split()
    fulu_match = _COV_MODEL.FULU_RE.search(prompt)
    n_melds = fulu_match.group(1).count('(') if fulu_match else 0
    e = _COV_MODEL._pre_energy(hand, n_melds)
    return float(e) if e is not None else 0.0
