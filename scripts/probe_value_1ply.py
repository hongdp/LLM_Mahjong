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
                   counterfactual=None, with_critic=True):
    """apply_discard(clone, action_xml) mutates a deep copy; counterfactual
    (self-play only) maps action_xml -> (vcf, vend). with_critic=False skips
    the V1/V0 reads (cheap record: p / efficiency / genbutsu only)."""
    cands = {}
    for a in actions:
        if _is_discard(a):
            t = _tile(a)
            if t not in cands or '"0' not in a:          # prefer the plain copy
                cands[t] = a
    if len(cands) < 2 or not _is_discard(chosen):
        return None
    eff = efficiency(table, pid)
    opps, kind = threats(table, pid)
    gb = genbutsu_set(table, opps)
    rec = {"src": src, "pid": pid, "threat": kind, "opps": opps,
           "shanten": min((v[0] for v in eff.values()), default=None),
           "chosen": _tile(chosen), "cands": {}, "_actions": {}}
    for t, a in cands.items():
        rec["cands"][t] = {"p": float(probs.get(a, 0.0)),
                           "sh": eff.get(t, (None, None))[0], "uk": eff.get(t, (None, None))[1],
                           "genbutsu": t in gb}
        rec["_actions"][t] = a
    if with_critic:
        mask, _ = critic.space.mask(actions)
        v0_states = []
        for t, a in cands.items():
            clone = copy.deepcopy(table)
            apply_discard(clone, a)
            rec["cands"][t]["v1"] = critic.draw_expectation(clone, pid, mask)
            v0_states.append((clone, pid))
        for (t, _a), v0 in zip(cands.items(), critic.values(v0_states, mask)):
            rec["cands"][t]["v0"] = v0
    if counterfactual is not None:
        for t, a in cands.items():
            vcf, vend = counterfactual(a)
            rec["cands"][t]["vcf"] = vcf
            if vend is not None:
                rec["cands"][t]["vend"] = vend
    return rec


# ----------------------------------------------------------------------
# exp121 Stage 0 v2: K reshuffled walls (PIMC over the hidden wall)
# ----------------------------------------------------------------------
def reshuffle_hidden(table, rng):
    """Reshuffle everything in the wall that nobody has seen yet: the live
    wall plus the dead-wall slots that are neither drawn rinshan tiles
    (0.._rinshan_idx-1) nor revealed dora indicators (4..4+n_dora-1).
    Hidden hands are kept as they are (oracle on hands, expectation over
    the wall). ura indicators are re-derived from the new dead wall."""
    revealed = set(range(table._rinshan_idx)) | set(range(4, 4 + len(table.dora_indicators)))
    hidden_idx = [i for i in range(14) if i not in revealed]
    pool = list(table.wall) + [table.dead_wall[i] for i in hidden_idx]
    rng.shuffle(pool)
    n = len(table.wall)
    table.wall = pool[:n]
    for i, t in zip(hidden_idx, pool[n:]):
        table.dead_wall[i] = t
    table.ura_indicators = [table.dead_wall[9 + k] for k in range(len(table.ura_indicators))]


def rollout_end(policy, clone, pid, action_xml):
    """Force `action_xml` on `clone` (already copied / reshuffled) and play
    the deal to its end with the policy; exact final reward for pid."""
    gen = play_game_gen(table=clone)
    try:
        _t, reqs = next(gen)
        if not reqs or reqs[0][0] != pid:
            return None
        t, reqs = gen.send([(_placeholder_step(), action_xml)])
        guard = 0
        while guard < 600:
            guard += 1
            replies = [(_placeholder_step(), policy(t, q, opts)[0]) for q, opts in reqs]
            t, reqs = gen.send(replies)
    except StopIteration:
        pass
    fr = clone.final_rewards
    return float(fr[pid]) if fr else None


def classify(rec):
    """Strata of interest for the K-wall measurement."""
    c = rec["cands"]
    pt = _argmax(c, "p")
    ps = sorted((v["p"] for v in c.values()), reverse=True)
    coin = rec["threat"] != "riichi" and len(ps) >= 2 and ps[1] >= 0.30
    safe_tie = False
    if rec["threat"] != "none" and not c[pt]["genbutsu"]:
        psh, puk = _eff_key(c, pt)
        safe_tie = any(v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2 for v in c.values())
    return coin, safe_tie


def kwall_candidates(rec, max_cands):
    """pi's pick, the efficiency-best, the best-efficiency genbutsu tie
    alternative, then the most probable others, up to max_cands."""
    c = rec["cands"]
    pt = _argmax(c, "p")
    keep = [pt]
    best = _eff_best(c)
    if best and best not in keep:
        keep.append(best)
    psh, puk = _eff_key(c, pt)
    tie = [(v["uk"], u) for u, v in c.items() if v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2 and u != pt]
    if tie:
        alt = max(tie)[1]
        if alt not in keep:
            keep.append(alt)
    for u in sorted(c, key=lambda u: -c[u]["p"]):
        if len(keep) >= max_cands:
            break
        if u not in keep:
            keep.append(u)
    return keep


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
                rec.pop("_actions", None)
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


def counterfactual_value(critic, policy, table, pid, action_xml, mask, to_end=False):
    """Clone the true table, force `action_xml`, play everyone else with the
    policy. Returns (vcf, vend):
      vcf  = critic bootstrap at our next turn-phase decision (own-draw
             expectation), or the exact final reward if the deal ends first
      vend = exact final reward for `pid` when the deal is played to its end
             (to_end=True; deterministic at T=0 on the true wall — exp121
             Stage 0 gold standard), else None."""
    clone = copy.deepcopy(table)
    gen = play_game_gen(table=clone)
    try:
        _t, reqs = next(gen)
    except StopIteration:
        return None, None
    if not reqs or reqs[0][0] != pid:
        return None, None
    vcf = None
    try:
        t, reqs = gen.send([(_placeholder_step(), action_xml)])
        guard = 0
        while guard < 600:
            guard += 1
            if (vcf is None and len(reqs) == 1 and reqs[0][0] == pid
                    and any(_is_discard(a) for a in reqs[0][1])):
                drawn = t.last_drawn[pid]
                if drawn is None or drawn not in t.hands[pid]:
                    vcf = critic.values([(t, pid)], mask)[0]
                else:
                    t.hands[pid].remove(drawn)
                    vcf = critic.draw_expectation(t, pid, mask)
                    t.hands[pid].append(drawn)
                    t.hands[pid].sort(key=sort_key)
                if not to_end:
                    return vcf, None
            replies = []
            for q, opts in reqs:
                a, _d, _v = policy(t, q, opts)
                replies.append((_placeholder_step(), a))
            t, reqs = gen.send(replies)
    except StopIteration:
        pass
    fr = clone.final_rewards
    vend = float(fr[pid]) if fr else None
    if vcf is None:                      # deal ended before our next turn
        vcf = vend
    return vcf, (vend if to_end else None)


def selfplay_chunk(args):
    ckpt, seeds, sample_p, tag, to_end, kw = args
    k_coin, k_def, max_cands = kw if kw else (0, 0, 0)
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
        nstep = Counter()                 # per-seat turn-decision index (replay key)
        while True:
            replies = []
            for pid, actions in reqs:
                chosen, dist, _v = policy(table, pid, actions)
                is_turn = len(reqs) == 1 and any(_is_discard(a) for a in actions)
                if is_turn:
                    nstep[pid] += 1
                if is_turn and _is_discard(chosen) and rng.random() < sample_p:
                    if kw:                                   # K-wall mode: strata of interest only
                        rec = probe_decision(critic, table, pid, actions, dist, chosen,
                                             f"{tag}{seed}", None, None, with_critic=False)
                        if rec:
                            coin, safe_tie = classify(rec)
                            K = max(k_coin if coin else 0, k_def if safe_tie else 0)
                            if K > 0:
                                rec["seed"] = seed; rec["step"] = nstep[pid]
                                rec["coinflip"] = coin; rec["safe_tie"] = safe_tie; rec["K"] = K
                                for t in kwall_candidates(rec, max_cands):
                                    vals = []
                                    for k in range(K):
                                        clone = copy.deepcopy(table)
                                        # same k -> same reshuffle for every candidate (common random numbers)
                                        reshuffle_hidden(clone, random.Random(seed * 1000003 + nstep[pid] * 1009 + k))
                                        vals.append(rollout_end(policy, clone, pid, rec["_actions"][t]))
                                    vals = [v for v in vals if v is not None]
                                    m = sum(vals) / len(vals) if vals else None
                                    sd = (sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
                                    rec["cands"][t]["vw"] = vals
                                    rec["cands"][t]["vw_mean"] = m
                                    rec["cands"][t]["vw_se"] = sd / len(vals) ** 0.5 if sd is not None else None
                                out.append(rec)
                    else:
                        mask, _ = critic.space.mask(actions)
                        cf = lambda a, _m=mask: counterfactual_value(critic, policy, table, pid, a, _m, to_end)
                        rec = probe_decision(critic, table, pid, actions, dist, chosen,
                                             f"{tag}{seed}", lambda c, a: c.step(pid, a), cf)
                        if rec:
                            rec["seed"] = seed
                            rec["step"] = nstep[pid]
                            out.append(rec)
                    if rec:
                        rec.pop("_actions", None)
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


def stage0_gate(recs):
    """exp121 Stage 0: exact same-wall counterfactual to the end of the deal.
    Per stratum: Vend_gain = mean(max_alt vend - vend[pi pick]) (hindsight
    upper bound of what any critic-driven search could recover), pi_end_opt =
    fraction of decisions where pi's pick is already vend-optimal."""
    G = {}
    for r in recs:
        c = r["cands"]
        if not all(v.get("vend") is not None for v in c.values()):
            continue
        pt = _argmax(c, "p")
        best = _eff_best(c)
        if pt is None or best is None:
            continue
        vmax = max(v["vend"] for v in c.values())
        gain = vmax - c[pt]["vend"]
        opt = c[pt]["vend"] >= vmax - 1e-9
        # chaos diagnostic: alternatives EQUIVALENT to the pick by efficiency
        # and safety (same shanten, ukeire within 2, same genbutsu flag) can
        # only differ in vend through the butterfly effect of a different
        # tile hitting the river; their spread is the hindsight noise floor
        psh, puk = _eff_key(c, pt)
        equiv = [v["vend"] for u, v in c.items()
                 if u != pt and v["sh"] == psh and abs(v["uk"] - puk) <= 2
                 and v["genbutsu"] == c[pt]["genbutsu"]]
        gain_equiv = (max(equiv) - c[pt]["vend"]) if equiv else None
        if equiv:
            grp = equiv + [c[pt]["vend"]]
            mu = sum(grp) / len(grp)
            chaos_sd = (sum((x - mu) ** 2 for x in grp) / (len(grp) - 1)) ** 0.5
        else:
            chaos_sd = None
        ps = sorted((v["p"] for v in c.values()), reverse=True)
        coin = r["threat"] != "riichi" and len(ps) >= 2 and ps[1] >= 0.30
        strata = ["all"] + (["coinflip"] if coin else [])
        if r["threat"] != "none":
            strata.append(f"exposed_{r['threat']}")
            if not c[pt]["genbutsu"] and any(v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2
                                             for v in c.values()):
                strata.append(f"exposed_{r['threat']}/safe_tie")
        for st in strata:
            d = G.setdefault(st, {"gain": [], "opt": [], "gain_equiv": [], "chaos_sd": []})
            d["gain"].append(gain); d["opt"].append(opt)
            if gain_equiv is not None:
                d["gain_equiv"].append(gain_equiv); d["chaos_sd"].append(chaos_sd)
    out = {}
    def _ms(xs):
        n = len(xs)
        if not n:
            return None, None
        m = sum(xs) / n
        sd = (sum((x - m) ** 2 for x in xs) / max(1, n - 1)) ** 0.5
        return round(m, 4), (round(sd / n ** 0.5, 4) if n > 1 else None)
    for st, d in G.items():
        xs = d["gain"]; n = len(xs)
        m, se = _ms(xs)
        me, _ = _ms(d["gain_equiv"])
        cs, _ = _ms(d["chaos_sd"])
        out[st] = {"n": n, "Vend_gain": m, "se": se,
                   "pi_end_opt": round(sum(d["opt"]) / n, 3),
                   "gain_ge_0.5": round(sum(x >= 0.5 for x in xs) / n, 3),
                   "gain_ge_2": round(sum(x >= 2.0 for x in xs) / n, 3),
                   "n_with_equiv": len(d["gain_equiv"]),
                   "gain_over_equivalents": me,       # hindsight gain from alternatives that cannot be better
                   "chaos_sd_within_equivalents": cs}
    return out


def summarize(recs):
    rankers = ["p", "v1", "vcf", "vend", "v0"]
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
    s0 = stage0_gate(recs)
    if s0:
        table["stage0_gate"] = s0
    kw = kwall_gate(recs)
    if kw:
        table["kwall_gate"] = kw
    return table


def kwall_gate(recs):
    """exp121 Stage 0 v2: per stratum, paired difference of K-wall means
    (alternative - pi's pick); SE across decisions (contains the per-decision
    MC error, so it is conservative), mean per-decision MC SE, fraction of
    decisions where the alternative beats the pick by > 2 pooled MC SE, and
    how often pi's pick is the best of the kept candidates."""
    G = {}
    def put(st, name, diff, mc_se, sig):
        d = G.setdefault(st, {}).setdefault(name, {"diff": [], "mc": [], "sig": []})
        d["diff"].append(diff); d["mc"].append(mc_se); d["sig"].append(sig)
    for r in recs:
        c = r["cands"]
        kept = {u: v for u, v in c.items() if v.get("vw_mean") is not None}
        if len(kept) < 2:
            continue
        pt = _argmax(c, "p")
        if pt not in kept:
            continue
        pm, pse = kept[pt]["vw_mean"], kept[pt]["vw_se"] or 0.0
        vmax = max(v["vw_mean"] for v in kept.values())
        best_kept = pt if pm >= vmax - 1e-9 else max(kept, key=lambda u: kept[u]["vw_mean"])
        strata = []
        if r.get("coinflip"):
            strata.append("coinflip")
        if r.get("safe_tie"):
            strata.append(f"safe_tie_{r['threat']}")
        for st in strata:
            put(st, "pick_is_best", float(best_kept == pt), 0.0, False)
            eb = _eff_best(c)
            if eb in kept and eb != pt:
                mc = ((kept[eb]["vw_se"] or 0.0) ** 2 + pse ** 2) ** 0.5
                diff = kept[eb]["vw_mean"] - pm
                put(st, "effbest_minus_pick", diff, mc, diff > 2 * mc)
            psh, puk = _eff_key(c, pt)
            tie = [(v["uk"], u) for u, v in c.items()
                   if v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2 and u != pt and u in kept]
            if tie:
                alt = max(tie)[1]
                mc = ((kept[alt]["vw_se"] or 0.0) ** 2 + pse ** 2) ** 0.5
                diff = kept[alt]["vw_mean"] - pm
                put(st, "genbutsu_minus_pick", diff, mc, diff > 2 * mc)
            # best kept alternative vs pick (hindsight over means, but means are over K walls now)
            alts = [u for u in kept if u != pt]
            if alts:
                ba = max(alts, key=lambda u: kept[u]["vw_mean"])
                mc = ((kept[ba]["vw_se"] or 0.0) ** 2 + pse ** 2) ** 0.5
                diff = kept[ba]["vw_mean"] - pm
                put(st, "bestalt_minus_pick", diff, mc, diff > 2 * mc)
    out = {}
    for st, d in G.items():
        out[st] = {}
        for name, x in d.items():
            xs = x["diff"]; n = len(xs); m = sum(xs) / n
            sd = (sum((v - m) ** 2 for v in xs) / max(1, n - 1)) ** 0.5
            out[st][name] = {"n": n, "mean": round(m, 4), "se": round(sd / n ** 0.5, 4) if n > 1 else None,
                             "mean_mc_se": round(sum(x["mc"]) / n, 4),
                             "frac_sig_better": round(sum(x["sig"]) / n, 3)}
    return out


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
            for k in ("v1", "vcf", "vend", "v0"):
                if c[best].get(k) is not None and c[pt].get(k) is not None:
                    add("coinflip/effbest_minus_pick", k, c[best][k] - c[pt][k])
        if r["threat"] != "none" and not c[pt]["genbutsu"]:
            psh, puk = _eff_key(c, pt)
            tie = [(v["uk"], u) for u, v in c.items()
                   if v["genbutsu"] and v["sh"] == psh and abs(v["uk"] - puk) <= 2]
            if tie:
                alt = max(tie)[1]
                for k in ("v1", "vcf", "vend", "v0"):
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
    ap.add_argument("--cf-to-end", action="store_true",
                    help="self-play: also play each counterfactual to the end of the deal "
                         "(exact same-wall value `vend`; exp121 Stage 0 gate)")
    ap.add_argument("--cf-walls", type=int, default=0,
                    help="exp121 Stage 0 v2: K reshuffled hidden walls per candidate on COIN-FLIP decisions "
                         "(hands kept, wall reshuffled, same K shuffles for every candidate); 0 = off")
    ap.add_argument("--cf-walls-defense", type=int, default=0,
                    help="K for safe-tie (exposed) decisions; 0 = skip them")
    ap.add_argument("--cf-walls-max-cands", type=int, default=6,
                    help="candidates evaluated per decision: pick, efficiency-best, genbutsu tie alt, then top-p")
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
        kw = (a.cf_walls, a.cf_walls_defense, a.cf_walls_max_cands) if (a.cf_walls or a.cf_walls_defense) else None
        chunks = [(a.ckpt, seeds[i::n], a.sample_p, "sp", a.cf_to_end, kw) for i in range(n)]
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
