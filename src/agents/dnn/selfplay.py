"""Self-play rollout driver for the conventional-DNN baseline.

Mirrors the LLM orchestrator's game loop exactly — same engine, same turn
/interrupt structure, and the SAME claim resolution helper
(`orchestrator._resolve_claims`) so multi-ron, meld priority and riichi
sticks behave identically. Only the policy differs, which is the point of
the comparison.
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.agents.dnn.encoder import encode_state, legal_mask
from src.agents.dnn.yaku_features import hazard_features, value_distance_profile
from src.tasks.mahjong.claims import _resolve_claims
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds
from src.tasks.mahjong.table import PyMahjongTable

_TE = TileEfficiency()
C_SHANTEN, C_UKEIRE = 2.0, 0.05

# Privileged CRITIC-ONLY features (exp11 A1/A2). The policy never sees them:
# they ride in a separate tensor consumed only by the value path, so the
# actor keeps information parity with the LLM baseline.
CFEAT_DIM = {"none": 0, "profile": 4, "hazard": 45}   # hazard: 9 families x 5


def critic_features(table, pid, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    hand = table.hands[pid]
    n_melds = len(table.melds[pid])
    closed = n_melds == 0                 # proxy: ankan also counts as open
    try:
        if mode == "profile":
            vals = value_distance_profile(hand, n_melds, closed)
        else:                             # "hazard"
            rows = hazard_features(hand, n_melds, closed,
                                   turns_left=len(table.wall) / 4.0)
            vals = [x for row in rows for x in row]
    except Exception:
        vals = [0.0] * CFEAT_DIM[mode]
    return torch.tensor(vals, dtype=torch.float32)


def potential(table, pid) -> float:
    """Phi(s) = -2.0*shanten + 0.05*|ukeire| for this seat's hand.

    Same potential the LLM runs used (rewards.MahjongPotentialReward), so
    the two shaping setups stay comparable. Computed straight off the
    table instead of parsing prompt text.
    """
    try:
        tiles = table.hands[pid]
        padded = pad_for_melds(tiles, len(table.melds[pid]))
        if len(tiles) % 3 == 2:          # 14 tiles: score the best discard
            ranked = _TE.evaluate_discards_ranked(padded)
            cands = [(sh, len(uk)) for t, (sh, uk) in ranked.items() if t in tiles]
            if not cands:
                return 0.0
            sh, uk = min(cands, key=lambda c: (c[0], -c[1]))
        else:
            sh = _TE.calculate_shanten(padded)
            uk = len(_TE.calculate_ukeire(padded))
        return -C_SHANTEN * sh + C_UKEIRE * uk
    except Exception:
        return 0.0

ACTION_RE = re.compile(r'type="(\w+)"')


@dataclass
class DnnStep:
    planes: torch.Tensor
    scalars: torch.Tensor
    mask: torch.Tensor
    action_idx: int
    logprob: float
    reward: float = 0.0
    is_terminal: bool = False
    phi: float = 0.0          # potential at this decision point (PBRS)
    cfeats: Optional[torch.Tensor] = None   # privileged critic-only features


@dataclass
class DnnGame:
    trajectories: dict = field(default_factory=lambda: {i: [] for i in range(4)})
    result: Optional[str] = None
    points: Optional[List[int]] = None
    # end-of-game style facts (eval_style_profile): riichi declared per seat,
    # meld count per seat (ankan included — noted as a proxy for 副露)
    riichi: Optional[List[bool]] = None
    n_melds: Optional[List[int]] = None


def _choose(net, table, pid, actions, temperature, device, cmode="none"):
    planes, scalars = encode_state(table, pid,
                                   variant=getattr(net, "encoder_variant", "v1"))
    mask, lookup = legal_mask(actions)
    idx, lp = net.act(planes[None].to(device), scalars[None].to(device),
                      mask[None].to(device), temperature=temperature)
    i = int(idx)
    return (DnnStep(planes=planes, scalars=scalars, mask=mask, action_idx=i,
                    logprob=float(lp),
                    cfeats=critic_features(table, pid, cmode)), lookup[i])


def play_game(net, temperature: float = 1.0, device="cpu",
              deal_seed: Optional[int] = None,
              randomize_round: bool = True,
              shaping: bool = False,
              critic_feats: str = "none",
              seat_nets: Optional[dict] = None) -> DnnGame:
    """seat_nets (exp22 league): {pid: net} overrides `net` per seat, so a
    game can mix the learner with frozen opponents. Trajectories are still
    recorded for every seat; the caller drops the opponents' ones."""
    if deal_seed is not None:
        random.seed(deal_seed)          # common random numbers
    seat_nets = seat_nets or {}
    # temperature may be a per-seat dict (exp28 mixed-temperature rollouts)
    seat_temp = temperature if isinstance(temperature, dict) else {p: temperature for p in range(4)}
    table = PyMahjongTable(randomize_round=randomize_round)
    table.text_obs = False          # DNN path never reads text obs
    game = DnnGame()

    guard = 0
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        step, action_str = _choose(seat_nets.get(pid, net), table, pid, actions,
                                   seat_temp[pid], device, cmode=critic_feats)
        if shaping:
            step.phi = potential(table, pid)
        _, rewards, done, info = table.step(pid, action_str)
        step.reward = rewards[pid]
        step.is_terminal = done
        game.trajectories[pid].append(step)

        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        needs_interrupt = True

        # ---- interrupt window (mirrors orchestrator.interrupt_node) ----
        last_discarder = pid
        candidates, cand_steps = [], []
        for offset in range(1, 4):
            other = (last_discarder + offset) % 4
            options = table.get_interrupt_actions(other)
            if len(options) == 1:        # skip-only: no decision to make
                continue
            s, a_str = _choose(seat_nets.get(other, net), table, other, options,
                               seat_temp[other], device, cmode=critic_feats)
            if shaping:
                s.phi = potential(table, other)
            m = ACTION_RE.search(a_str)
            candidates.append({"player_id": other, "parsed": a_str,
                               "type": m.group(1) if m else None,
                               "reward": 0.0})
            cand_steps.append(s)

        executed, done = _resolve_claims(table, candidates)
        for cand, s in zip(candidates, cand_steps):
            s.reward = cand["reward"]
            s.is_terminal = done and cand in executed
            game.trajectories[cand["player_id"]].append(s)
        if done:
            break
        # Nothing was claimed: the engine still has to close the window —
        # a pending kan completes, otherwise the turn passes and the next
        # player draws. Skipping this froze the game on one seat until its
        # hand ran out (caught by the step-count smoke check).
        if not executed:
            if table.pending_kan:
                table.resolve_pending_kan()
            else:
                _, r_done = table.advance_turn()
                if r_done:
                    break

    # settlement is distributed to every seat's last recorded step
    if table.final_rewards:
        for p in range(4):
            if game.trajectories[p]:
                last = game.trajectories[p][-1]
                last.reward += table.final_rewards[p]
                last.is_terminal = True
    game.result = table.result_summary
    game.points = list(table.points)
    game.riichi = [bool(table.riichi[p]) for p in range(4)]
    game.n_melds = [len(table.melds[p]) for p in range(4)]
    return game


def apply_shaping(steps: List[DnnStep], gamma: float) -> None:
    """In-place PBRS: F_t = gamma*Phi_{t+1} - Phi_t, terminal Phi := 0.

    Potential-based, so the discounted shaping telescopes to -Phi(s_0) and
    the optimal policy is unchanged (Ng et al. 1999). exp2 showed this adds
    nothing once a teacher prior exists; the point HERE is the opposite
    case — from scratch the sparse settlement carries no gradient at all
    until the agent wins by accident, and this supplies one.
    """
    for i, s in enumerate(steps):
        nxt = 0.0 if i + 1 >= len(steps) else steps[i + 1].phi
        s.reward += gamma * nxt - s.phi


def returns_to_go(steps: List[DnnStep], gamma: float) -> List[float]:
    out, R = [], 0.0
    for s in reversed(steps):
        R = s.reward + gamma * R
        out.insert(0, R)
    return out


# ----------------------------------------------------------------------
# Generator form of play_game (vectorized rollout, perf 2026-08-23): the
# driver yields decision requests and receives (DnnStep, action_str) pairs,
# so one worker process can interleave K games and batch their requests
# into a single inference RPC. Game logic is identical to play_game.
# ----------------------------------------------------------------------
def play_game_gen(deal_seed: Optional[int] = None, randomize_round: bool = True,
                  shaping: bool = False, seat_model: Optional[dict] = None):
    """Yields (table, [(pid, actions), ...]); `.send()` the list of
    (DnnStep, action_str) back in the same order. Returns the DnnGame.
    seat_model: {pid: model_id} (league); the caller routes each request.
    Note: random.seed(deal_seed) must be applied by the caller right before
    the table is built (done here) — K interleaved games share the global
    RNG, so the engine's forced-discard fallback is the only later use."""
    if deal_seed is not None:
        random.seed(deal_seed)
    table = PyMahjongTable(randomize_round=randomize_round)
    table.text_obs = False
    game = DnnGame()
    guard = 0
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        ((step, action_str),) = yield (table, [(pid, actions)])
        if shaping:
            step.phi = potential(table, pid)
        _, rewards, done, info = table.step(pid, action_str)
        step.reward = rewards[pid]
        step.is_terminal = done
        game.trajectories[pid].append(step)
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        reqs = []
        for offset in range(1, 4):
            other = (pid + offset) % 4
            options = table.get_interrupt_actions(other)
            if len(options) == 1:
                continue
            reqs.append((other, options))
        candidates, cand_steps = [], []
        if reqs:
            replies = yield (table, reqs)
            for (other, _opts), (s, a_str) in zip(reqs, replies):
                if shaping:
                    s.phi = potential(table, other)
                m = ACTION_RE.search(a_str)
                candidates.append({"player_id": other, "parsed": a_str,
                                   "type": m.group(1) if m else None, "reward": 0.0})
                cand_steps.append(s)
        executed, done = _resolve_claims(table, candidates)
        for cand, s in zip(candidates, cand_steps):
            s.reward = cand["reward"]
            s.is_terminal = done and cand in executed
            game.trajectories[cand["player_id"]].append(s)
        if done:
            break
        if not executed:
            if table.pending_kan:
                table.resolve_pending_kan()
            else:
                _, r_done = table.advance_turn()
                if r_done:
                    break
    if table.final_rewards:
        for p in range(4):
            if game.trajectories[p]:
                last = game.trajectories[p][-1]
                last.reward += table.final_rewards[p]
                last.is_terminal = True
    game.result = table.result_summary
    game.points = list(table.points)
    game.riichi = [bool(table.riichi[p]) for p in range(4)]
    game.n_melds = [len(table.melds[p]) for p in range(4)]
    return game


def make_step(table, pid, actions, variant, idx, lp, cmode="none") -> "DnnStep":
    """DnnStep for a decision whose action index/logprob were produced
    elsewhere (the batched RPC path)."""
    planes, scalars = encode_state(table, pid, variant=variant)
    mask, lookup = legal_mask(actions)
    return DnnStep(planes=planes, scalars=scalars, mask=mask, action_idx=idx,
                   logprob=lp, cfeats=critic_features(table, pid, cmode)), lookup
