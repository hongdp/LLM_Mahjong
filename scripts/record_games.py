"""Record fully-instrumented self-play games for the arena dashboard.

For each checkpoint (a training stage), play the SAME deal seeds and log,
at every decision of every seat: the observer-view table state, the legal
actions with the policy's probabilities, the sampled action, and V(s).
Output is one compact JSON the dashboard embeds inline.

Usage:
  python scripts/record_games.py --ckpt 80k=path.pt 700k=path.pt \
      --games 6 --seed0 77000000 --out experiments/arena_dashboard.json
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                           # noqa: E402
from src.agents.dnn.encoder import encode_state, legal_mask          # noqa: E402
from src.agents.dnn.selfplay import _resolve_claims                  # noqa: E402
from src.tasks.mahjong.shanten import dora_from_indicator            # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE        # noqa: E402


def snapshot(table, pid):
    return {
        "pid": pid,
        "hands": [sorted(table.hands[p]) for p in range(4)],
        "drawn": table.last_drawn[pid],
        "rivers": [list(table.discards[p]) for p in range(4)],
        "melds": [[{"t": m["type"], "tiles": m["tiles"]} for m in table.melds[p]] for p in range(4)],
        "dora": [dora_from_indicator(i) for i in table.dora_indicators],
        "riichi": [bool(x) for x in table.riichi],
        "points": list(table.points),
        "wall": len(table.wall),
        "last": table.last_discard,
    }


def decide(net, table, pid, actions, temperature, rng):
    planes, scalars = encode_state(table, pid,
                                   variant=getattr(net, "encoder_variant", "v1"))
    mask, lookup = legal_mask(actions)
    with torch.no_grad():
        logits, v = net.forward_with_value(planes[None], scalars[None], mask[None])
        probs = torch.softmax(logits[0] / max(temperature, 1e-6), 0)
    legal = [(lookup[i], float(probs[i])) for i in lookup]
    legal.sort(key=lambda x: -x[1])
    idx = int(torch.multinomial(probs, 1, generator=rng))
    chosen = lookup.get(idx, legal[0][0])
    short = [(ACTION_RE.search(a).group(1) + (":" + ACTION_RE.search(a).group(2) if ACTION_RE.search(a).group(2) else ""), round(p, 4))
             for a, p in legal]
    m = ACTION_RE.search(chosen)
    return chosen, short, (m.group(1) + (":" + m.group(2) if m.group(2) else "")), round(float(v[0]), 3)


def play_recorded(net, seed, temperature, rng):
    random.seed(seed)
    table = PyMahjongTable(randomize_round=True)
    table.text_obs = False
    steps = []
    guard = 0
    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        state = snapshot(table, pid)
        chosen, dist, short, v = decide(net, table, pid, actions, temperature, rng)
        steps.append({"phase": "turn", "s": state, "acts": dist[:10], "pick": short, "v": v})
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
            state = snapshot(table, other)
            chosen, dist, short, v = decide(net, table, other, options, temperature, rng)
            steps.append({"phase": "claim", "s": state, "acts": dist[:10], "pick": short, "v": v})
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
    return {"seed": seed, "result": table.result_summary, "points": list(table.points),
            "dealer": table.dealer, "round_wind": table.round_wind_idx, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="STAGE=path pairs")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=77000000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = {"seed0": a.seed0, "games": a.games, "temperature": a.temperature, "models": []}
    for spec in a.ckpt:
        stage, path = spec.split("=", 1)
        net = load_dnn(path, "cpu")
        rng = torch.Generator().manual_seed(1234)
        games = [play_recorded(net, a.seed0 + g, a.temperature, rng) for g in range(a.games)]
        out["models"].append({"stage": stage, "path": path, "games": games})
        print(stage, "recorded", len(games), "games,",
              sum(len(g["steps"]) for g in games), "decisions", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
    print("saved", a.out, round(os.path.getsize(a.out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
