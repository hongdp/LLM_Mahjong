"""exp50: rate a REAL Mortal checkpoint in our Elo pool (black-box route).

The model runs on its own stack — libriichi (Rust) does the v4 obs
encoding and `libriichi.mjai.Bot` the acting — while OUR engine drives
the game and emits the full-information MJAI stream
(`play_game_mjai(observer=None)`); this runner fans out per-seat masked
views to two Mortal bots seated as side A of the league's standard 2v2
duplicate match. Scoring and Elo fitting are imported verbatim from
run_elo_league, so the number lands on the same scale as every other
rating in history.jsonl.

Riichi is two-step in mjai and fused in our engine: when the bot answers
`reach`, we feed the reach event back immediately, read the follow-up
dahai, emit the fused `<riichi tile>` action, and swallow the duplicated
reach event when the game stream later echoes it.

Usage:
  PYTHONPATH=. conda run -n rlhf_mahjong python scripts/arena_mortal_mjai.py \
      --state data/mortal_ext/mortal_298k.pth --deals 100 --label mortal298k_ext
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MORTAL_DIR = "/home/hongdp/Workspace/mortal_upstream/mortal"

from src.agents.dnn.action_space import get_space                      # noqa: E402
from src.agents.dnn.encoder import encode_state                        # noqa: E402
from src.agents.dnn.mjai_bridge import mjai_to_engine, mjai_to_engine_spelled  # noqa: E402
from src.agents.dnn.mjai_export import play_game_mjai                  # noqa: E402
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable          # noqa: E402
from scripts.run_arena_dnn import load_dnn                             # noqa: E402
from scripts.run_elo_league import (LEAGUE_DIR, deal_scores,           # noqa: E402
                                    engine_fingerprint, fit_ratings,
                                    rating_se, residuals)
from tests.test_mjai_bridge import _norm                               # noqa: E402


def build_mortal_engine(state_path: str, device: str):
    sys.path.insert(0, MORTAL_DIR)
    from model import Brain, DQN                                       # noqa: E402
    from engine import MortalEngine                                    # noqa: E402
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    brain = Brain(conv_channels=cfg["resnet"]["conv_channels"],
                  num_blocks=cfg["resnet"]["num_blocks"], version=version).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(brain, dqn, is_oracle=False, version=version,
                        device=torch.device(device), enable_amp=False,
                        enable_quick_eval=True, name="mortal298k"), version


def mask_for_seat(ev: dict, seat: int) -> dict:
    t = ev.get("type")
    if t == "start_kyoku":
        ev = dict(ev, tehais=[h if i == seat else ["?"] * len(h)
                              for i, h in enumerate(ev["tehais"])])
    elif t == "tsumo" and ev.get("actor") != seat:
        ev = dict(ev, pai="?")
    return ev


class MortalSeat:
    def __init__(self, bot_cls, engine, seat: int, stats):
        self.bot = bot_cls(engine, seat)
        self.seat = seat
        self.stats = stats
        self.pending = None
        self._swallow_reach = False
        self.bot.react(json.dumps({"type": "start_game", "id": seat}))

    def feed(self, ev: dict):
        if (self._swallow_reach and ev.get("type") == "reach"
                and ev.get("actor") == self.seat):
            self._swallow_reach = False        # our manual feed already did this
            return
        r = self.bot.react(json.dumps(mask_for_seat(ev, self.seat)))
        if r:
            self.pending = json.loads(r)

    # ---- reaction -> engine action xml --------------------------------
    def _to_xml(self, r: dict, table) -> str:
        t = r["type"]
        if t == "none":
            return '<action type="skip" />'
        if t == "hora":
            return ('<action type="tsumo" />' if r.get("target") == self.seat
                    else '<action type="ron" />')
        if t == "dahai":
            return f'<action type="discard" tile="{mjai_to_engine_spelled(r["pai"])}" />'
        if t == "reach":
            # two-step -> fused: acknowledge the declare, read the tile
            self._swallow_reach = True
            rr = self.bot.react(json.dumps({"type": "reach", "actor": self.seat}))
            tile = mjai_to_engine_spelled(json.loads(rr)["pai"]) if rr else None
            return (f'<action type="riichi" tile="{tile}" />' if tile
                    else '<action type="skip" />')
        if t == "chi":
            a, b = [mjai_to_engine(c) for c in r["consumed"]]
            return (f'<action type="chi" tile="{mjai_to_engine(r["pai"])}" '
                    f'with="{a} {b}" />')
        if t == "pon":
            return f'<action type="pon" tile="{mjai_to_engine(r["pai"])}" />'
        if t in ("daiminkan", "kakan"):
            return f'<action type="kan" tile="{mjai_to_engine(r["pai"])}" />'
        if t == "ankan":
            return f'<action type="kan" tile="{mjai_to_engine(r["consumed"][0])}" />'
        if t == "ryukyoku":
            return '<action type="kyuushu" />'
        return '<action type="skip" />'

    def policy(self, table, pid, actions):
        r, self.pending = self.pending, None
        if r is None:
            xml = '<action type="skip" />'
        else:
            xml = self._to_xml(r, table)
        if _norm([xml]) <= _norm(actions):
            self.stats["ok"] += 1
            return xml
        self.stats["fallback"] += 1
        if self.stats["fallback"] <= 10:
            print(f"[fallback] seat{pid} wanted {xml} legal={actions[:6]}",
                  flush=True)
        drawn = table.last_drawn[pid]
        cand = f'<action type="discard" tile="{drawn}" />' if drawn else None
        if cand and _norm([cand]) <= _norm(actions):
            return cand
        skip = '<action type="skip" />'
        return skip if _norm([skip]) <= _norm(actions) else actions[0]


def net_policy(net, device, temperature=1.0):
    space = get_space(net)
    variant = getattr(net, "encoder_variant", "v1")

    def pol(table, pid, actions):
        mask, lookup = space.mask(actions)
        if not lookup:
            return actions[0]
        planes, scalars = encode_state(table, pid, variant=variant)
        idx, _ = net.act(planes[None].to(device), scalars[None].to(device),
                         mask[None].to(device), temperature=temperature)
        mode = space.follow_up(int(idx), actions)
        if mode is not None:
            m2, lk2 = space.mask(actions, mode=mode)
            idx2, _ = net.act(planes[None].to(device), scalars[None].to(device),
                              m2[None].to(device), temperature=temperature)
            return lk2.get(int(idx2)) or actions[0]
        return space.resolve(int(idx), lookup) or actions[0]
    return pol


def play_match(engine, bot_cls, anchor_net, deals, seed0, device, stats):
    """League-identical 2v2 duplicate: Mortal on A seats, anchor on B."""
    rows = []
    anchor_pol = net_policy(anchor_net, device)
    for i in range(deals):
        seed = seed0 + i
        diff = 0.0
        orients = []
        for orient in (0, 1):
            a_seats = (0, 2) if orient == 0 else (1, 3)
            random.seed(seed)
            table = PyMahjongTable()
            table.text_obs = False
            seats = {s: MortalSeat(bot_cls, engine, s, stats) for s in a_seats}
            policies = {}
            for s in range(4):
                policies[s] = (seats[s].policy if s in a_seats else anchor_pol)

            def sink(ev, _seats=seats):
                for ms in _seats.values():
                    ms.feed(ev)
            play_game_mjai(table, policies, observer=None, sink=sink)
            a_pts = sum(table.points[s] for s in a_seats)
            b_pts = sum(table.points[s] for s in range(4) if s not in a_seats)
            diff += (a_pts - b_pts)
            orients.append({"a_seats": list(a_seats),
                            "points": list(table.points)})
        rows.append({"seed": seed, "diff": diff, "orient": orients})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/mortal_ext/mortal_298k.pth")
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--label", default="mortal298k_ext")
    ap.add_argument("--anchors", default=None, help="comma subset; default all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    engine, version = build_mortal_engine(a.state, a.device)
    sys.path.insert(0, MORTAL_DIR)
    from libriichi.mjai import Bot                                     # noqa: E402
    print(f"[mortal] v{version} engine ready on {a.device}", flush=True)

    league = json.load(open(f"{LEAGUE_DIR}/anchors.json"))
    anchors = league["anchors"]
    use = a.anchors.split(",") if a.anchors else list(anchors)
    stats = {"ok": 0, "fallback": 0}
    games = []
    for name in use:
        t0 = time.time()
        net = load_dnn(anchors[name]["path"], a.device)
        rows = play_match(engine, Bot, net, a.deals, a.seed0, a.device, stats)
        scores = deal_scores(rows)
        games += [("cand", name, s) for s in scores]
        print(f"[match] mortal vs {name}: score {sum(scores):.1f}/{a.deals} "
              f"mean_diff {sum(r['diff'] for r in rows)/len(rows):.0f} "
              f"({time.time()-t0:.0f}s, fallback {stats['fallback']})", flush=True)

    ratings = {n: anchors[n]["rating"] for n in use}
    ratings["cand"] = 1100.0
    fit_ratings(games, ratings, ["cand"])
    total = stats["ok"] + stats["fallback"]
    rec = {"ckpt": a.state, "label": a.label,
           "elo": round(ratings["cand"], 1),
           "se": round(rating_se(games, ratings, "cand"), 1),
           "anchors": use, "deals_per_anchor": a.deals, "seed0": a.seed0,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "engine": engine_fingerprint(), "engine_mismatch": True,
           "temperature": "mortal-native-greedy",
           "bridge_fallback_rate": round(stats["fallback"] / max(total, 1), 5),
           "residuals": residuals(games, ratings, "cand")}
    with open(f"{LEAGUE_DIR}/history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"ELO {a.label}: {rec['elo']} ± {rec['se']} "
          f"fallback_rate {rec['bridge_fallback_rate']} "
          f"residuals {rec['residuals']}", flush=True)


if __name__ == "__main__":
    main()
