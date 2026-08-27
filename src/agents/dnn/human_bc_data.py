"""Streaming human-BC dataset (exp45): replay tenhou mjlogs on the fly.

Tensors are NOT materialised to disk. A v3r state is ~2KB and the mortal
obs ~32KB per decision; at ~12M decisions that is 20-400GB per variant.
Replay through the verified MJAI bridge instead runs at ~8 games/s per
worker process (~640 decisions/game across the 4 seat views), so a
DataLoader with a dozen workers regenerates the whole dataset faster
than the GPU consumes it — one implementation, every encoder variant.

Split discipline (SKILLS:106): holdout is cut per GAME (all four seat
views of a game land on the same side), decided by a stable hash of the
log id — no state-level leakage of shared outcomes, stable across runs.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.agents.dnn.encoder import action_to_index, encode_state, legal_mask
from src.agents.dnn.mjai_bridge import MjaiDnnBot


def _derive():
    # scripts/ is not importable at module load in every entrypoint; the
    # harvester owns the lookahead logic (single source of truth).
    from scripts.harvest_human_bc import Ambiguous, derive_reaction
    from tests.test_mjai_bridge import _norm, reaction_to_action
    return Ambiguous, derive_reaction, _norm, reaction_to_action


def is_holdout(path: str, holdout_pct: int = 10) -> bool:
    """Stable per-game split on the log id (never on decision index)."""
    h = hashlib.md5(os.path.basename(path).encode()).hexdigest()
    return int(h[:8], 16) % 100 < holdout_pct


def game_decisions(path: str, seat: int, variant: str,
                   action_space: str = "native") -> List[dict]:
    """Replay one seat view of one game; returns decision rows.

    Row: planes, scalars (torch), mask (bool, space-dim), label (int),
    phase, vs_riichi (defensive context: an opponent riichi is live).

    `action_space="mortal46"` expands a fused engine action into the SAME
    decision sequence the rollout runs at play time (selfplay._choose):
    riichi/kan first hit their declare slot, then — with the identical
    observation tensors and only the mask changed — a second row picks the
    tile. Both rows are genuine training decisions, exactly as PPO sees.
    """
    Ambiguous, derive_reaction, _norm, reaction_to_action = _derive()
    from tools.tenhou.mjlog_to_mjai import mask_for_seat, parse_mjlog
    from src.agents.dnn.action_space import get_space
    from src.agents.dnn.mortal_action import action_to_slot

    space = get_space(action_space)
    events = parse_mjlog(open(path).read())["events"]
    rows: List[dict] = []
    ctx = {"i": 0}

    def emit(planes, scalars, mask, label, phase, vs_riichi):
        if label is None or not bool(mask[label]):
            return                       # unmappable / outside mask: drop
        rows.append({"planes": planes, "scalars": scalars, "mask": mask,
                     "label": int(label), "phase": phase,
                     "vs_riichi": vs_riichi})

    def recorder(table, pid, actions):
        try:
            reaction = derive_reaction(events, ctx["i"], pid, bot.phase)
        except Ambiguous:
            return '<action type="skip" />', {}, None
        a_xml = reaction_to_action(reaction, bot)
        if _norm([a_xml]) <= _norm(actions):
            vs_r = any(table.riichi[p] for p in range(4) if p != pid)
            planes, scalars = encode_state(table, pid, variant=variant)
            if space.name == "mortal46":
                slot1 = action_to_slot(a_xml)
                m1, _ = space.mask(actions)
                emit(planes, scalars, m1, slot1, bot.phase, vs_r)
                fu = space.follow_up(slot1, actions) if slot1 is not None else None
                if fu:                   # same obs, second mask — as at play time
                    m2, _ = space.mask(actions, mode=fu)
                    slot2 = action_to_slot(
                        a_xml, at_riichi_select=(fu == "riichi"),
                        at_kan_select=(fu == "kan"))
                    emit(planes, scalars, m2, slot2, bot.phase, vs_r)
            else:
                m, _ = legal_mask(actions)
                emit(planes, scalars, m, action_to_index(a_xml),
                     bot.phase, vs_r)
        return a_xml, {}, None

    bot = MjaiDnnBot(recorder, seat=seat)
    for i, ev in enumerate(mask_for_seat(events, seat)):
        ctx["i"] = i
        bot.react(ev)
    return rows


class HumanBCDataset(IterableDataset):
    """Iterable over (planes, scalars, mask, label) with buffer shuffling.

    `files` should already be filtered to the desired split via
    `is_holdout`. Each DataLoader worker replays a disjoint slice of the
    (file, seat) list; ordering reshuffles from `seed + epoch`.
    """

    def __init__(self, files: List[str], variant: str = "v3r",
                 shuffle_buffer: int = 20000, seed: int = 0,
                 seats: Tuple[int, ...] = (0, 1, 2, 3),
                 action_space: str = "native"):
        self.units = [(f, s) for f in files for s in seats]
        self.variant = variant
        self.action_space = action_space
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _iter_rows(self, units) -> Iterator[dict]:
        for path, seat in units:
            try:
                yield from game_decisions(path, seat, self.variant,
                                          self.action_space)
            except Exception:                              # noqa: BLE001
                # fidelity is the harvester's job; training just skips
                continue

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 100003 + self.epoch)
        units = list(self.units)
        rng.shuffle(units)                    # same order in every worker,
        units = units[wid::nw]                # then a disjoint slice of it
        buf: List[dict] = []
        rloc = random.Random(rng.random() + wid)
        for row in self._iter_rows(units):
            buf.append(row)
            if len(buf) >= self.shuffle_buffer:
                j = rloc.randrange(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                yield self._pack(buf.pop())
        rloc.shuffle(buf)
        for row in buf:
            yield self._pack(row)

    @staticmethod
    def _pack(row: dict):
        return (row["planes"], row["scalars"], row["mask"],
                torch.tensor(row["label"], dtype=torch.long),
                {"turn": 0, "claim": 1, "chankan": 2}[row["phase"]],
                int(row["vs_riichi"]))


def list_games(raw_dir: str, holdout: Optional[bool] = None,
               holdout_pct: int = 10, limit: int = 0) -> List[str]:
    import glob
    files = sorted(glob.glob(os.path.join(raw_dir, "*", "*.mjlog")))
    if holdout is not None:
        files = [f for f in files if is_holdout(f, holdout_pct) == holdout]
    return files[:limit] if limit else files
