"""Record self-play games for the Mortal-style review page (tools/webui, 复盘 tab).

Unlike record_games.py (top-10 probs of one model), every decision here carries
EVERY legal action with, for each of N models side by side: policy probability,
raw logit (= Q(s,a) for DQN checkpoints, whose head is reinterpreted as Q),
and V(s). The first model drives play (greedy by default) so all models are
scored on the same states; decisions where the models' greedy picks differ are
flagged so the page can jump between disagreements.

  python scripts/record_review.py --models bc49=experiments/_anchors_epoch6/bc49.pt \
      exp60=experiments/exp60_pool/candidate.pt --games 20 --seed0 70000000 \
      --out experiments/review/review_bc49_vs_exp60.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.record_games import snapshot                              # noqa: E402
from scripts.run_arena_dnn import load_dnn                             # noqa: E402
from src.agents.dnn.action_space import get_space                      # noqa: E402
from src.agents.dnn.encoder import encode_state                        # noqa: E402
from src.agents.dnn.selfplay import _resolve_claims                    # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE          # noqa: E402


def short(a):
    m = ACTION_RE.search(a)
    return m.group(1) + (":" + m.group(2) if m.group(2) else "")


def score_all(models, table, pid, actions):
    """Per model: {probs, q, v, pick} over the SAME ordered legal-action list."""
    out = []
    for m in models:
        net = m["net"]
        space = m["space"]
        planes, scalars = encode_state(table, pid, variant=getattr(net, "encoder_variant", "v1"))
        mask, lookup = space.mask(actions)
        with torch.no_grad():
            logits, v = net.forward_with_value(planes[None].to(m["dev"]), scalars[None].to(m["dev"]),
                                               mask[None].to(m["dev"]))
            logits = logits[0].float().cpu()
            probs = torch.softmax(logits, 0)
        # action -> slot (a slot may map to a follow-up mode; keep the first slot per action)
        slot_of = {}
        for i, a in lookup.items():
            slot_of.setdefault(a, i)
        q = [float(logits[slot_of[a]]) if a in slot_of else None for a in actions]
        p = [float(probs[slot_of[a]]) if a in slot_of else 0.0 for a in actions]
        best = max(range(len(actions)), key=lambda k: (q[k] if q[k] is not None else -1e9))
        out.append({"probs": [round(x, 4) for x in p], "q": [None if x is None else round(x, 4) for x in q],
                    "v": round(float(v[0]), 3), "pick": best})
    return out


def play_reviewed(models, seed, driver=0, temperature=0.0):
    random.seed(seed)
    table = PyMahjongTable(randomize_round=True)
    table.text_obs = False
    rng = torch.Generator().manual_seed(seed)
    steps, guard = [], 0

    def decide(pid, actions):
        scores = score_all(models, table, pid, actions)
        d = scores[driver]
        if temperature > 0:
            probs = torch.tensor(d["probs"]) + 1e-9
            k = int(torch.multinomial(probs / probs.sum(), 1, generator=rng))
        else:
            k = d["pick"]
        picks = {m["pick"] for m in scores}
        return actions[k], {"legal": [short(a) for a in actions], "models": scores,
                            "taken": k, "disagree": len(picks) > 1}

    while not table.finished and guard < 600:
        guard += 1
        pid = table.turn
        actions = table.get_legal_actions(pid)
        if not actions:
            break
        state = snapshot(table, pid)
        chosen, rec = decide(pid, actions)
        if len(actions) > 1:
            steps.append(dict(phase="turn", s=state, **rec))
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
            chosen, rec = decide(other, options)
            steps.append(dict(phase="claim", s=state, **rec))
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
            "dealer": table.dealer, "round_wind": table.round_wind_idx, "steps": steps,
            "n_disagree": sum(1 for st in steps if st["disagree"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="NAME=ckpt (first drives play)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=70_000_000)
    ap.add_argument("--temperature", type=float, default=0.0, help="driver sampling T (0 = greedy)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = []
    for spec in a.models:
        # NAME=path[:q]  — ":q" marks a DQN checkpoint (its logits are Q(s,a));
        # exp59/exp60 paths are recognised automatically
        name, path = spec.split("=", 1)
        kind = "policy"
        if path.endswith(":q"):
            path, kind = path[:-2], "q"
        elif any(k in path.lower() for k in ("dqn", "exp59", "exp60")):
            kind = "q"
        net = load_dnn(path, a.device)
        net.eval()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        models.append({"name": name, "path": path, "net": net, "space": get_space(net),
                       "dev": a.device, "kind": kind,
                       "arch": blob.get("arch"), "games_trained": blob.get("games")})
    t0 = time.time()
    games = [play_reviewed(models, a.seed0 + g, temperature=a.temperature) for g in range(a.games)]
    out = {"seed0": a.seed0, "games": len(games), "temperature": a.temperature,
           "models": [{k: v for k, v in m.items() if k not in ("net", "space", "dev")} for m in models],
           "records": games, "recorded": time.strftime("%F %T")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    nd = sum(g["n_disagree"] for g in games)
    ns = sum(len(g["steps"]) for g in games)
    print(f"✅ {len(games)} games, {ns} decisions, {nd} disagreements "
          f"({nd / max(ns, 1):.1%}) -> {a.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
