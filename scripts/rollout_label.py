"""exp66 pilot: rollout-based action-value labels for contested decisions.

Pipeline (all on CPU, multiprocess):
  1. play greedy self-play with the reference policy; at TURN-phase decisions
     whose top-2 logit margin is below --margin, snapshot the table (deepcopy)
     together with the policy's top-k actions and logits;
  2. for every snapshot x action x M determinizations: resample the hidden
     information from the deciding seat's point of view (non-riichi
     opponents' concealed hands, the live wall, the unrevealed dead-wall
     slots), force the action, finish the hand with the reference policy
     greedy in all four seats, record the deciding seat's point delta;
  3. Q(s,a) = mean over M, with its standard error.

Determinization caveats (pilot): riichi opponents keep their real hand
(they are tenpai and a random hand would not be); furiten flags are not
recomputed for resampled hands.

  python scripts/rollout_label.py --ckpt experiments/_anchors_epoch6/bc49.pt \
      --states 500 --M 64 --topk 3 --workers 20 --out experiments/probes/exp66_labels.json
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                           # noqa: E402

from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE, is_red, norm_tile, sort_key   # noqa: E402

_NET = {}


def _policy():
    """Per-process lazy loader (CPU, single thread)."""
    if "net" not in _NET:
        torch.set_num_threads(1)
        from scripts.run_arena_dnn import load_dnn
        from src.agents.dnn.action_space import get_space
        net = load_dnn(_NET["ckpt"], "cpu")
        _NET["net"] = net
        _NET["space"] = get_space(net)
        _NET["variant"] = getattr(net, "encoder_variant", "v1")
    return _NET["net"], _NET["space"], _NET["variant"]


@torch.no_grad()
def greedy(table, pid, actions, mode=None):
    """Greedy action (with follow-up handling) plus the masked logits."""
    from src.agents.dnn.encoder import encode_state
    net, space, variant = _policy()
    mask, lookup = space.mask(actions, mode)
    if int(mask.sum()) == 1:
        slot = int(mask.nonzero()[0])
        logits = None
    else:
        planes, sc = encode_state(table, pid, variant=variant)
        logits = net(planes[None], sc[None], mask[None])[0].float()
        logits = logits.masked_fill(~mask, float("-inf"))
        slot = int(logits.argmax())
    fu = space.follow_up(slot, actions, mode)
    if fu is not None:
        return greedy(table, pid, actions, mode=fu)[0], logits, lookup, mask
    return space.resolve(slot, lookup), logits, lookup, mask


def play_hand(table, first=None, on_turn_decision=None):
    """Drive one hand to its end with the greedy policy in all seats.
    `first=(pid, action_xml)` forces the first turn action for `pid`.
    `on_turn_decision(table, pid, actions, logits, lookup, mask)` may return
    a snapshot hook value (used by the collector); ignored otherwise."""
    from src.agents.dnn.selfplay import _resolve_claims
    guard = 0
    forced = first
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        if forced is not None and forced[0] == pid:
            chosen = forced[1]
            forced = None
        else:
            chosen, logits, lookup, mask = greedy(table, pid, actions)
            if on_turn_decision is not None and logits is not None:
                on_turn_decision(table, pid, actions, logits, lookup, mask)
        _, _, done, info = table.step(pid, chosen)
        if done:
            break
        if not (info.get("discarded") or info.get("chankan")):
            continue
        candidates = []
        for offset in range(1, 4):
            other = (pid + offset) % 4
            options = table.get_interrupt_actions(other)
            if len(options) == 1:
                continue
            chosen, _, _, _ = greedy(table, other, options)
            mm = ACTION_RE.search(chosen)
            candidates.append({"player_id": other, "parsed": chosen,
                               "type": mm.group(1) if mm else None, "reward": 0.0})
        executed, done = _resolve_claims(table, candidates)
        if done:
            break
        if not executed:
            if table.pending_kan:
                table.resolve_pending_kan()
            else:
                _, r_done = table.advance_turn()
                if r_done:
                    break
    return table


def resample_hidden(table, me, rng):
    """Redistribute everything `me` cannot see among the places it could be."""
    pool = []
    targets = []                                   # (kind, key, count)
    for pid in range(4):
        if pid == me or table.riichi[pid]:
            continue
        hand = table.display_hand(pid)             # red spelled '0x'
        pool += hand
        targets.append(("hand", pid, len(hand)))
    pool += table.wall
    targets.append(("wall", None, len(table.wall)))
    hidden_slots = list(range(table._rinshan_idx, 4)) \
        + list(range(4 + len(table.dora_indicators), 9)) + list(range(9, 14))
    pool += [table.dead_wall[i] for i in hidden_slots]
    targets.append(("dead", hidden_slots, len(hidden_slots)))
    rng.shuffle(pool)
    k = 0
    for kind, key, n in targets:
        chunk = pool[k:k + n]
        k += n
        if kind == "hand":
            table.hands[key] = []
            table.red[key] = {"m": 0, "p": 0, "s": 0}
            for raw in chunk:
                if is_red(raw):
                    table.red[key][raw[1]] += 1
                table.hands[key].append(norm_tile(raw))
            table.hands[key].sort(key=sort_key)
        elif kind == "wall":
            table.wall = list(chunk)
        else:
            for i, raw in zip(key, chunk):
                table.dead_wall[i] = norm_tile(raw) if i >= 4 else raw
    table._waits_cache = {}
    return table


def all_tiles_multiset(table):
    """Every tile in the game as a sorted list (invariant under resampling)."""
    out = []
    for pid in range(4):
        out += table.display_hand(pid)
        for m in table.melds[pid]:
            tiles = list(m["tiles"])
            for _ in range(m.get("red", 0)):
                j = next((i for i, t in enumerate(tiles) if t[0] == "5" and t[-1] in "mps"), None)
                if j is not None:
                    tiles[j] = "0" + tiles[j][-1]
            out += tiles
        out += table.discards[pid]
    out += table.wall + table.dead_wall
    return sorted(out)


def collect_states(seed0, n_games, margin, topk):
    """Self-play; snapshot contested turn-phase decisions."""
    snaps = []

    def hook(table, pid, actions, logits, lookup, mask):
        fin = logits[torch.isfinite(logits)]
        if fin.numel() < 2:
            return
        top = torch.topk(logits, min(topk, fin.numel()))
        m = float(top.values[0] - top.values[1])
        acts = [lookup.get(int(s)) for s in top.indices]
        if any(a is None for a in acts):
            return
        snaps.append({"table": copy.deepcopy(table), "pid": pid, "margin": m,
                      "actions": acts, "logits": [float(v) for v in top.values],
                      "seed": table._seed, "contested": m < margin})

    for g in range(n_games):
        random.seed(seed0 + g)
        t = PyMahjongTable(randomize_round=True)
        t.text_obs = False
        t._seed = seed0 + g
        play_hand(t, on_turn_decision=hook)
    return snaps


def label_one(args):
    """Worker: (snapshot, M, seed) -> per-action rollout returns."""
    snap, M, seed = args
    rng = random.Random(seed)
    me = snap["pid"]
    out = []
    for a in snap["actions"]:
        rets = []
        for k in range(M):
            t = copy.deepcopy(snap["table"])
            resample_hidden(t, me, rng)
            start = t.points[me]
            play_hand(t, first=(me, a))
            rets.append(t.points[me] - start)
        out.append(rets)
    return out


def _init(ckpt):
    _NET["ckpt"] = ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/_anchors_epoch6/bc49.pt")
    ap.add_argument("--states", type=int, default=500, help="contested states to label")
    ap.add_argument("--control_states", type=int, default=50, help="stable (margin>=2) states as sanity control")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=73_000_000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _init(a.ckpt)
    t0 = time.time()
    snaps = collect_states(a.seed0, a.games, a.margin, a.topk)
    contested = [s for s in snaps if s["contested"]][:a.states]
    stable = [s for s in snaps if s["margin"] >= 2.0][:a.control_states]
    todo = contested + stable
    print(f"collected {len(snaps)} turn decisions from {a.games} games: {len(contested)} contested "
          f"+ {len(stable)} stable selected [{time.time() - t0:.0f}s]", flush=True)
    jobs = [(s, a.M, a.seed0 + 7919 * i) for i, s in enumerate(todo)]
    t1 = time.time()
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.ckpt,)) as pool:
        results = []
        for i, r in enumerate(pool.imap(label_one, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 25 == 0:
                print(f"  labelled {i + 1}/{len(jobs)} [{time.time() - t1:.0f}s]", flush=True)
    rows = []
    for s, rets in zip(todo, results):
        rows.append({"seed": s["seed"], "pid": s["pid"], "margin": round(s["margin"], 4),
                     "contested": s["contested"], "actions": s["actions"],
                     "logits": [round(x, 4) for x in s["logits"]], "returns": rets})
    meta = {"ckpt": a.ckpt, "M": a.M, "topk": a.topk, "margin": a.margin, "games": a.games,
            "seed0": a.seed0, "label_seconds": round(time.time() - t1, 1),
            "labels": len(rows), "recorded": time.strftime("%F %T")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(a.out, "w"))
    print(f"✅ {len(rows)} states x {a.topk} actions x M={a.M} in {meta['label_seconds']}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
