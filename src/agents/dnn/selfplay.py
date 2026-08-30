"""Self-play rollout driver for the conventional-DNN baseline.

Mirrors the LLM orchestrator's game loop exactly — same engine, same turn
/interrupt structure, and the SAME claim resolution helper
(`orchestrator._resolve_claims`) so multi-ron, meld priority and riichi
sticks behave identically. Only the policy differs, which is the point of
the comparison.
"""

import os
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
    # multi-step action spaces: the batched rollout contract returns ONE step
    # per request, so an earlier query of the same decision rides here and is
    # unpacked into the trajectory by _package_game. None for single-step.
    extra_steps: Optional[list] = None
    # compact write-log form of `planes` for wide observations (Mortal's 934
    # planes). When set, _package_game ships THIS instead of the dense tensor
    # -- 98x less pickle traffic, and the trainer densifies per minibatch.
    sparse_planes: Optional[object] = None


@dataclass
class DnnGame:
    trajectories: dict = field(default_factory=lambda: {i: [] for i in range(4)})
    result: Optional[str] = None
    points: Optional[List[int]] = None
    # end-of-game style facts (eval_style_profile): riichi declared per seat,
    # meld count per seat (ankan included — noted as a proxy for 副露)
    riichi: Optional[List[bool]] = None
    n_melds: Optional[List[int]] = None
    n_discards: Optional[int] = None      # table-wide discards at the end (hand length, ~4 per 巡)
    # first-tenpai 巡目 per seat (None = never tenpai); filled only when the
    # caller asks for it (play_game(track_tenpai=True) — eval-only, the
    # training rollout path never pays the extra shanten calls)
    tenpai_turns: Optional[List[Optional[int]]] = None
    start_points: Optional[List[int]] = None  # context-randomized starting scores


def _choose(net, table, pid, actions, temperature, device, cmode="none"):
    """Query the policy for one engine action.

    Returns ``(steps, action_str)`` where `steps` is a LIST because some action
    spaces are multi-step: Mortal declares riichi (or the intent to kan) and
    only then picks the tile, so one engine action can consume two policy
    queries -- each a genuine decision that PPO must see, hence two DnnSteps.
    The native 374-slot space never asks for a follow-up, so it always returns
    exactly one step and its behaviour is unchanged.
    """
    from src.agents.dnn.action_space import get_space
    space = get_space(net)
    variant = getattr(net, "encoder_variant", "v1")
    planes, scalars = encode_state(table, pid, variant=variant)
    cf = critic_features(table, pid, cmode)

    steps, mode = [], None
    for _ in range(2):                      # at most one follow-up
        mask, lookup = space.mask(actions, mode=mode)
        if os.environ.get("INFER_DEBUG") and not bool(mask.any()):
            with open("/tmp/choose_debug.txt", "a") as _f:
                _f.write(f"EMPTY mask mode={mode} pid={pid} actions={actions}\n")
        idx, lp = net.act(planes[None].to(device), scalars[None].to(device),
                          mask[None].to(device), temperature=temperature)
        i = int(idx)
        steps.append(DnnStep(planes=planes, scalars=scalars, mask=mask,
                             action_idx=i, logprob=float(lp), cfeats=cf))
        nxt = space.follow_up(i, actions, mode=mode)
        if nxt is None:
            return steps, space.resolve(i, lookup)
        mode = nxt
    raise RuntimeError(f"{space.name}: follow-up did not terminate")


def play_game(net, temperature: float = 1.0, device="cpu",
              deal_seed: Optional[int] = None,
              randomize_round: bool = True,
              shaping: bool = False,
              critic_feats: str = "none",
              seat_nets: Optional[dict] = None,
              track_tenpai: bool = False) -> DnnGame:
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
    tenpai_turns: List[Optional[int]] = [None, None, None, None]

    guard = 0
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        steps, action_str = _choose(seat_nets.get(pid, net), table, pid, actions,
                                    seat_temp[pid], device, cmode=critic_feats)
        if shaping:
            for st in steps:
                st.phi = potential(table, pid)
        _, rewards, done, info = table.step(pid, action_str)
        # a multi-step decision shares one engine transition: the reward and
        # terminal flag land on the LAST step, earlier ones carry 0 so the
        # return-to-go accumulates over them exactly once
        steps[-1].reward = rewards[pid]
        steps[-1].is_terminal = done
        game.trajectories[pid].extend(steps)

        if (track_tenpai and info.get("discarded")
                and tenpai_turns[pid] is None
                and table._shanten(table.hands[pid], len(table.melds[pid])) <= 0):
            tenpai_turns[pid] = table.discard_count[pid]

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
            s_list, a_str = _choose(seat_nets.get(other, net), table, other,
                                    options, seat_temp[other], device,
                                    cmode=critic_feats)
            if shaping:
                for st in s_list:
                    st.phi = potential(table, other)
            m = ACTION_RE.search(a_str)
            candidates.append({"player_id": other, "parsed": a_str,
                               "type": m.group(1) if m else None,
                               "reward": 0.0})
            # keep the whole list: a multi-step space must not have its first
            # query dropped from the trajectory (it is a real decision)
            cand_steps.append(s_list)

        executed, done = _resolve_claims(table, candidates)
        for cand, s_group in zip(candidates, cand_steps):
            s_group[-1].reward = cand["reward"]
            s_group[-1].is_terminal = done and cand in executed
            game.trajectories[cand["player_id"]].extend(s_group)
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
    game.n_discards = sum(table.discard_count)
    game.start_points = list(getattr(table, "start_points", [25000] * 4))
    if track_tenpai:
        game.tenpai_turns = tenpai_turns
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
                  shaping: bool = False, seat_model: Optional[dict] = None,
                  table: Optional[PyMahjongTable] = None):
    """Yields (table, [(pid, actions), ...]); `.send()` the list of
    (DnnStep, action_str) back in the same order. Returns the DnnGame.
    seat_model: {pid: model_id} (league); the caller routes each request.
    table: pre-built table (hanchan rollout injects a HanchanTable that
    carries match context); when given, deal_seed/randomize_round are
    ignored — the caller owns the RNG and the table.
    Note: random.seed(deal_seed) must be applied by the caller right before
    the table is built (done here) — K interleaved games share the global
    RNG, so the engine's forced-discard fallback is the only later use."""
    if table is None:
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
    game.n_discards = sum(table.discard_count)
    game.start_points = list(getattr(table, "start_points", [25000] * 4))
    return game


def make_step(table, pid, actions, variant, idx, lp, cmode="none") -> "DnnStep":
    """DnnStep for a decision whose action index/logprob were produced
    elsewhere (the batched RPC path)."""
    planes, scalars = encode_state(table, pid, variant=variant)
    mask, lookup = legal_mask(actions)
    return DnnStep(planes=planes, scalars=scalars, mask=mask, action_idx=idx,
                   logprob=lp, cfeats=critic_features(table, pid, cmode)), lookup
