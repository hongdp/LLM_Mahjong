"""exp120: one-ply value-head probe — does the critic separate the
tie-break situations the policy flips coins on?

For every turn-phase discard decision we score each legal discard d by:
  V1(d)   = E_draw[ V(after discarding d, everyone passes, we draw t) ]
            (own-draw expectation over unseen tile types; blind to the
            deal-in risk of d itself, since a ron ends the trajectory)
  Vcf(d)  = self-play only: clone the TRUE table, discard d, let the
            opponents respond with the same policy (ron -> exact final
            reward), run to our next turn, then take the own-draw
            expectation there. The determinized one-ply search target.
  V0(d)   = V(right after discarding d) — out of distribution for the
            critic (we never get asked in our own interrupt window);
            recorded as a diagnostic only.
and compare the argmax of pi (T=0), V1, Vcf against
  * tile efficiency: TileEfficiency.evaluate_discards_ranked (shanten
    first, then ukeire with visible tiles subtracted)
  * safety: genbutsu vs every riichi opponent (else vs >=2-meld opponents)

Decision sources:
  A. Majsoul session logs (scripts/serve_mjai_bot.py JSONL) replayed through
     MjaiDnnBot -> ShadowTable (hidden hands unknown: V1/V0 only)
  B. self-play (4 copies, T=0, random round) via play_game_gen (V1, Vcf)

Usage (model code lives in the runpod worktree):
  PYTHONPATH=<worktree-with-cnn_l_v3r> python scripts/probe_value_1ply.py \
      --ckpt experiments/exp113_C/games_384000000.pt \
      --mjai-log experiments/majsoul_exp113C_20261025_000702/mjai_session.jsonl \
      --games 300 --sample-p 0.25 --workers 12 \
      --out experiments/probes/exp120_value1ply.json
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter

import torch

# LLM_MAHJONG_ROOT lets the probe import the engine/model code from another
# checkout (cnn_l_v3r lives in the runpod worktree only)
sys.path.insert(0, os.environ.get("LLM_MAHJONG_ROOT")
                or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_arena_dnn import load_dnn                             # noqa: E402
from src.agents.dnn.action_space import get_space                      # noqa: E402
from src.agents.dnn.encoder import encode_state                        # noqa: E402
from src.agents.dnn.mjai_bridge import DnnPolicy, MjaiDnnBot           # noqa: E402
from src.agents.dnn.selfplay import DnnStep, play_game_gen             # noqa: E402
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds    # noqa: E402
from src.tasks.mahjong.table import ACTION_RE, norm_tile, sort_key     # noqa: E402

_te = TileEfficiency()
TILES34 = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [f"{n}z" for n in range(1, 8)]
_TILE_RE = re.compile(r'tile="([^"]+)"')


def _tile(action_xml):
    m = _TILE_RE.search(action_xml)
    return norm_tile(m.group(1)) if m else None


def _is_discard(a):
    return 'type="discard"' in a


def visible_counts(table, pid):
    c = Counter(norm_tile(t) for t in table.hands[pid])
    for q in range(4):
        c.update(norm_tile(t.rstrip("*")) for t in table.discards[q])
        for m in table.melds[q]:
            c.update(norm_tile(t) for t in m["tiles"])
    c.update(norm_tile(t) for t in table.dora_indicators)
    return c


def efficiency(table, pid):
    """{tile: (post_shanten, ukeire_count)} for the 14-tile hand."""
    hand = [norm_tile(t) for t in table.hands[pid]]
    nm = max(0, min(len(table.melds[pid]), (14 - len(hand)) // 3))
    try:
        rk = _te.evaluate_discards_ranked(pad_for_melds(hand, nm))
    except ValueError:
        return {}
    vis = visible_counts(table, pid)
    return {t: (sh, sum(max(0, 4 - vis[u]) for u in uk)) for t, (sh, uk) in rk.items()}


def threats(table, pid):
    r = [q for q in range(4) if q != pid and table.riichi[q]]
    if r:
        return r, "riichi"
    m = [q for q in range(4) if q != pid and len(table.melds[q]) >= 2]
    return m, ("melds" if m else "none")


def genbutsu_set(table, opps):
    if not opps:
        return set()
    sets = [{norm_tile(t.rstrip("*")) for t in table.discards[q]} for q in opps]
    return set.intersection(*sets)


class Critic:
    def __init__(self, net, device):
        self.net = net.to(device).eval()
        self.device = device
        self.variant = getattr(net, "encoder_variant", "v1")
        self.space = get_space(net)

    @torch.no_grad()
    def values(self, states, mask):
        """states: list of (table, pid). One batched forward."""
        if not states:
            return []
        enc = [encode_state(t, p, variant=self.variant) for t, p in states]
        planes = torch.stack([e[0] for e in enc]).to(self.device)
        scalars = torch.stack([e[1] for e in enc]).to(self.device)
        masks = mask[None].expand(len(enc), -1).to(self.device)
        _, v = self.net.forward_with_value(planes, scalars, masks)
        return v.float().cpu().tolist()

    def draw_expectation(self, table, pid, mask):
        """E over unseen tile types of V(table with t drawn by pid);
        the table is mutated and restored."""
        vis = visible_counts(table, pid)
        draws = [(t, 4 - vis[t]) for t in TILES34 if vis[t] < 4]
        saved = (list(table.hands[pid]), table.last_drawn[pid], table.turn)
        states = []
        for t, _w in draws:                         # mutate, encode now, restore
            table.hands[pid] = sorted(saved[0] + [t], key=sort_key)
            table.last_drawn[pid] = t
            table.turn = pid
            states.append(encode_state(table, pid, variant=self.variant))
        table.hands[pid], table.last_drawn[pid], table.turn = saved
        planes = torch.stack([e[0] for e in states]).to(self.device)
        scalars = torch.stack([e[1] for e in states]).to(self.device)
        masks = mask[None].expand(len(states), -1).to(self.device)
        with torch.no_grad():
            _, v = self.net.forward_with_value(planes, scalars, masks)
        v = v.float().cpu().tolist()
        wsum = sum(w for _, w in draws)
        return sum(vi * w for vi, (_, w) in zip(v, draws)) / wsum


# ----------------------------------------------------------------------
# one decision -> record
# ----------------------------------------------------------------------
def probe_decision(critic, table, pid, actions, probs, chosen, src, apply_discard,
                   counterfactual=None):
    """apply_discard(clone, action_xml) mutates a deep copy; counterfactual
    (self-play only) maps action_xml -> value (float) or None."""
    cands = {}
    for a in actions:
        if _is_discard(a):
            t = _tile(a)
            if t not in cands or '"0' not in a:          # prefer the plain copy
                cands[t] = a
    if len(cands) < 2 or not _is_discard(chosen):
        return None
    mask, _ = critic.space.mask(actions)
    eff = efficiency(table, pid)
    opps, kind = threats(table, pid)
    gb = genbutsu_set(table, opps)
    rec = {"src": src, "pid": pid, "threat": kind, "opps": opps,
           "shanten": min((v[0] for v in eff.values()), default=None),
           "chosen": _tile(chosen), "cands": {}}
    v0_states = []
    for t, a in cands.items():
        clone = copy.deepcopy(table)
        apply_discard(clone, a)
        v1 = critic.draw_expectation(clone, pid, mask)
        v0_states.append((clone, pid))
        rec["cands"][t] = {"p": float(probs.get(a, 0.0)), "v1": v1,
                           "sh": eff.get(t, (None, None))[0], "uk": eff.get(t, (None, None))[1],
                           "genbutsu": t in gb}
    for (t, _a), v0 in zip(cands.items(), critic.values(v0_states, mask)):
        rec["cands"][t]["v0"] = v0
    if counterfactual is not None:
        for t, a in cands.items():
            rec["cands"][t]["vcf"] = counterfactual(a)
    return rec


# ----------------------------------------------------------------------
# A. Majsoul log replay
# ----------------------------------------------------------------------
def replay_mjai_log(critic, path, tag):
    policy = DnnPolicy(critic.net, critic.device, 0.0)
    bot = MjaiDnnBot(policy=None)
    out = []
    n_match = [0, 0]

    def recording_policy(table, pid, actions):
        chosen, dist, v = policy(table, pid, actions)
        if bot.phase == "turn":
            def apply(clone, a):
                tile = _TILE_RE.search(a).group(1)
                pai = clone.physical(tile)
                clone.dahai(pid, pai, pai == clone.my_drawn_mjai)
            rec = probe_decision(critic, table, pid, actions, dist, chosen,
                                 f"{tag}#{bot.n_decisions + 1}", apply)
            if rec:
                rec["n"] = bot.n_decisions + 1
                out.append(rec)
        return chosen, dist, v

    bot.policy = recording_policy
    logged = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k, d = r["kind"], r["data"]
            if k == "start":
                bot.start_game(int(d["seat"]))
            elif k == "in":
                bot.react(d)
            elif k == "in_batch":
                bot.react_batch(d)
            elif k == "decision":
                logged[d["n"]] = _tile(d["chosen"]) if _is_discard(d["chosen"]) else None
    for rec in out:
        if rec["n"] in logged and logged[rec["n"]] is not None:
            n_match[1] += 1
            n_match[0] += int(logged[rec["n"]] == rec["chosen"])
    print(f"[A:{tag}] {len(out)} discard decisions; replay==logged pick "
          f"{n_match[0]}/{n_match[1]}", flush=True)
    return out


# ----------------------------------------------------------------------
# B. self-play with determinized counterfactual
# ----------------------------------------------------------------------
def _placeholder_step(idx=0):
    return DnnStep(planes=torch.zeros(1), scalars=torch.zeros(1),
                   mask=torch.zeros(1, dtype=torch.bool), action_idx=idx, logprob=0.0)


def counterfactual_value(critic, policy, table, pid, action_xml, mask):
    """Clone the true table, force `action_xml`, play opponents with the
    policy until our next turn-phase decision (-> own-draw expectation)
    or the end of the deal (-> exact final reward)."""
    clone = copy.deepcopy(table)
    gen = play_game_gen(table=clone)
    try:
        _t, reqs = next(gen)
    except StopIteration:
        return None
    if not reqs or reqs[0][0] != pid:
        return None
    try:
        t, reqs = gen.send([(_placeholder_step(), action_xml)])
        guard = 0
        while guard < 40:
            guard += 1
            if len(reqs) == 1 and reqs[0][0] == pid and any(_is_discard(a) for a in reqs[0][1]):
                drawn = t.last_drawn[pid]
                if drawn is None or drawn not in t.hands[pid]:
                    return critic.values([(t, pid)], mask)[0]
                t.hands[pid].remove(drawn)
                v = critic.draw_expectation(t, pid, mask)
                t.hands[pid].append(drawn)
                return v
            replies = []
            for q, opts in reqs:
                a, _d, _v = policy(t, q, opts)
                replies.append((_placeholder_step(), a))
            t, reqs = gen.send(replies)
    except StopIteration:
        pass
    fr = clone.final_rewards
    return float(fr[pid]) if fr else None


def selfplay_chunk(args):
    ckpt, seeds, sample_p, tag = args
    torch.set_num_threads(1)
    net = load_dnn(ckpt, "cpu")
    critic = Critic(net, "cpu")
    policy = DnnPolicy(net, "cpu", 0.0)
    out = []
    for seed in seeds:
        rng = random.Random(seed * 7 + 1)
        random.seed(seed)
        gen = play_game_gen(deal_seed=None, randomize_round=True)
        try:
            table, reqs = next(gen)
        except StopIteration:
            continue
        while True:
            replies = []
            for pid, actions in reqs:
                chosen, dist, _v = policy(table, pid, actions)
                if (len(reqs) == 1 and any(_is_discard(a) for a in actions)
                        and _is_discard(chosen) and rng.random() < sample_p):
                    mask, _ = critic.space.mask(actions)
                    cf = lambda a, _m=mask: counterfactual_value(critic, policy, table, pid, a, _m)
                    rec = probe_decision(critic, table, pid, actions, dist, chosen,
                                         f"{tag}{seed}", lambda c, a: c.step(pid, a), cf)
                    if rec:
                        out.append(rec)
                replies.append((_placeholder_step(), chosen))
            try:
                table, reqs = gen.send(replies)
            except StopIteration:
                break
    return out


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
def _argmax(c, key):
    xs = [(v[key], t) for t, v in c.items() if v.get(key) is not None]
    return max(xs)[1] if xs else None


def _eff_best(c):
    xs = [((-v["sh"], v["uk"]), t) for t, v in c.items() if v["sh"] is not None]
    return max(xs)[1] if xs else None


def _eff_key(c, t):
    v = c[t]
    return (v["sh"], v["uk"])


def summarize(recs):
    rankers = ["p", "v1", "vcf", "v0"]
    S = {}
    def add(stratum, ranker, hit):
        d = S.setdefault(stratum, {}).setdefault(ranker, [0, 0])
        d[0] += int(hit); d[1] += 1
    for r in recs:
        c = r["cands"]
        best = _eff_best(c)
        if best is None:
            continue
        bsh, buk = _eff_key(c, best)
        ps = sorted((v["p"] for v in c.values()), reverse=True)
        coin = r["threat"] != "riichi" and len(ps) >= 2 and ps[1] >= 0.30
        exposed = r["threat"] != "none"
        picks = {k: _argmax(c, k) for k in rankers}
        for k, t in picks.items():
            if t is None:
                continue
            sh, uk = _eff_key(c, t)
            hit_full = (sh == bsh and uk == buk)
            hit_sh = (sh == bsh)
            for st in (["all"] + (["coinflip"] if coin else []) + (["noriichi"] if r["threat"] != "riichi" else [])):
                add(st + "/eff_full", k, hit_full)
                add(st + "/eff_shanten", k, hit_sh)
            if exposed:
                add(f"exposed_{r['threat']}/genbutsu_top", k, c[t]["genbutsu"])
                # safe-tie stratum: a genbutsu candidate within (same shanten, uk±2) of the policy's pick
                pt = picks["p"]
                psh, puk = _eff_key(c, pt)
                tie = [u for u, v in c.items() if v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2]
                if tie:
                    add(f"exposed_{r['threat']}/safe_tie/genbutsu_top", k, c[t]["genbutsu"])
                    add(f"exposed_{r['threat']}/safe_tie/genbutsu_and_eff", k,
                        c[t]["genbutsu"] and sh == psh and abs(uk - puk) <= 2)
    table = {}
    for st, d in sorted(S.items()):
        table[st] = {k: {"hit": h, "n": n, "rate": round(h / n, 3) if n else None} for k, (h, n) in d.items()}
    table["paired_gaps"] = paired_gaps(recs)
    return table


def paired_gaps(recs):
    """Paired value gaps that need no external ground truth for the
    ranker: value(alternative) - value(pi's pick), per critic read.
      coinflip:  alternative = efficiency-best (when pi's pick is not it)
      safe_tie:  alternative = best-efficiency genbutsu within the tie band
    A positive mean says the critic prefers the alternative pi passed on."""
    G = {}
    def add(name, key, gap):
        G.setdefault(name, {}).setdefault(key, []).append(gap)
    for r in recs:
        c = r["cands"]
        best = _eff_best(c)
        pt = _argmax(c, "p")
        if best is None or pt is None:
            continue
        ps = sorted((v["p"] for v in c.values()), reverse=True)
        coin = r["threat"] != "riichi" and len(ps) >= 2 and ps[1] >= 0.30
        if coin and best != pt:
            for k in ("v1", "vcf", "v0"):
                if c[best].get(k) is not None and c[pt].get(k) is not None:
                    add("coinflip/effbest_minus_pick", k, c[best][k] - c[pt][k])
        if r["threat"] != "none" and not c[pt]["genbutsu"]:
            psh, puk = _eff_key(c, pt)
            tie = [(v["uk"], u) for u, v in c.items()
                   if v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2]
            if tie:
                alt = max(tie)[1]
                for k in ("v1", "vcf", "v0"):
                    if c[alt].get(k) is not None and c[pt].get(k) is not None:
                        add(f"exposed_{r['threat']}/safe_tie/genbutsu_minus_pick", k, c[alt][k] - c[pt][k])
    out = {}
    for name, d in G.items():
        out[name] = {}
        for k, xs in d.items():
            n = len(xs); m = sum(xs) / n
            sd = (sum((x - m) ** 2 for x in xs) / max(1, n - 1)) ** 0.5
            out[name][k] = {"n": n, "mean": round(m, 4), "se": round(sd / n ** 0.5, 4) if n > 1 else None,
                            "frac_positive": round(sum(x > 0 for x in xs) / n, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mjai-log", nargs="*", default=[])
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--sample-p", type=float, default=0.25)
    ap.add_argument("--seed0", type=int, default=70000000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = []
    if a.mjai_log:
        net = load_dnn(a.ckpt, a.device)
        critic = Critic(net, a.device)
        for i, p in enumerate(a.mjai_log):
            recs += replay_mjai_log(critic, p, f"mjai{i}")
    if a.games > 0:
        seeds = [a.seed0 + i for i in range(a.games)]
        n = max(1, min(a.workers, a.games))
        chunks = [(a.ckpt, seeds[i::n], a.sample_p, "sp") for i in range(n)]
        ctx = mp.get_context("fork")
        with ctx.Pool(n) as pool:
            for part in pool.imap_unordered(selfplay_chunk, chunks):
                recs += part
                print(f"[B] +{len(part)} decisions (total {len(recs)})", flush=True)
    summary = summarize(recs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "n_decisions": len(recs), "summary": summary,
                   "decisions": recs}, f, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {a.out} ({len(recs)} decisions)")


if __name__ == "__main__":
    main()
