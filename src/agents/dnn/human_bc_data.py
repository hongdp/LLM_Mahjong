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


def game_decisions(path: str, seat: int, variant: str) -> List[dict]:
    """Replay one seat view of one game; returns decision rows.

    Row: planes, scalars (torch), mask [374] bool, label (int), phase,
    vs_riichi (defensive context: an opponent riichi is live).
    """
    Ambiguous, derive_reaction, _norm, reaction_to_action = _derive()
    from tools.tenhou.mjlog_to_mjai import mask_for_seat, parse_mjlog

    events = parse_mjlog(open(path).read())["events"]
    rows: List[dict] = []
    ctx = {"i": 0}

    def recorder(table, pid, actions):
        try:
            reaction = derive_reaction(events, ctx["i"], pid, bot.phase)
        except Ambiguous:
            return '<action type="skip" />', {}, None
        a_xml = reaction_to_action(reaction, bot)
        if _norm([a_xml]) <= _norm(actions):
            idx = action_to_index(a_xml)
            if idx is not None:
                planes, scalars = encode_state(table, pid, variant=variant)
                m, _ = legal_mask(actions)
                rows.append({
                    "planes": planes, "scalars": scalars, "mask": m,
                    "label": idx, "phase": bot.phase,
                    "vs_riichi": any(table.riichi[p] for p in range(4)
                                     if p != pid),
                })
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
                 seats: Tuple[int, ...] = (0, 1, 2, 3)):
        self.units = [(f, s) for f in files for s in seats]
        self.variant = variant
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _iter_rows(self, units) -> Iterator[dict]:
        for path, seat in units:
            try:
                yield from game_decisions(path, seat, self.variant)
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
