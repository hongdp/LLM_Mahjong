"""Replay tenhou houou mjlogs through the MJAI bridge and harvest BC labels.

For every game and every seat, a `MjaiDnnBot` (the verified ShadowTable
consumer) replays the masked event stream. At each decision the bridge
raises, the human's actual choice is derived by lookahead in the
full-information stream and asserted to be inside the engine's legal set
— that assertion IS the exp45 fidelity gate. Interrupt windows where the
human did nothing become explicit `skip` labels (the mjlog records no
declined calls), mirroring arch_sweep_bc's teacher collection.

Ambiguity rule (preregistered in exp45): when a different seat claimed
the discard we had a window on, the human's intent is unrecorded — the
sample is DROPPED and counted (`ambiguous_claim`), never labelled.

Output: one JSONL line per kept decision (file, kyoku, seat, phase,
action xml, action index, n_legal) + a summary block. Tensors are NOT
materialised here; they are regenerated per-arch at BC-training time by
replaying the verified streams.

Usage:
  PYTHONPATH=. python scripts/harvest_human_bc.py --raw data/tenhou/raw \
      --out data/tenhou/decisions.jsonl [--limit 100]
"""

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.encoder import action_to_index                     # noqa: E402
from src.agents.dnn.mjai_bridge import MjaiDnnBot                      # noqa: E402
from tests.test_mjai_bridge import _norm, reaction_to_action           # noqa: E402
from tools.tenhou.mjlog_to_mjai import mask_for_seat, parse_mjlog      # noqa: E402

_SKIP_TURN = {"dora"}
_SKIP_CLAIM = {"reach_accepted", "dora"}


class Ambiguous(Exception):
    """Another seat claimed the discard: our pass/claim intent unrecorded."""


def derive_reaction(events, i, me, phase):
    """Human's MJAI reaction for the decision the bridge raised at event i."""
    if phase == "turn":
        j = i + 1
        while events[j]["type"] in _SKIP_TURN:
            j += 1
        ev = events[j]
        if ev["type"] == "reach" and ev["actor"] == me:
            k = j + 1
            while events[k]["type"] in _SKIP_TURN:
                k += 1
            return {"type": "reach", "actor": me, "reach_dahai": events[k]}
        if ev.get("actor") == me and ev["type"] in ("dahai", "ankan", "kakan"):
            return ev
        if ev["type"] == "hora" and ev["actor"] == me and ev["target"] == me:
            return {"type": "hora", "actor": me, "target": me, "pai": ev["pai"]}
        if ev["type"] == "ryukyoku" and ev.get("reason") == "yao9":
            return {"type": "ryukyoku", "actor": me}
        raise AssertionError(f"turn lookahead hit {ev}")

    if phase in ("claim", "chankan"):
        target = events[i]["actor"]
        j = i + 1
        while events[j]["type"] in _SKIP_CLAIM:
            j += 1
        # double ron: scan through consecutive hora events
        while events[j]["type"] == "hora":
            if events[j]["actor"] == me:
                return {"type": "hora", "actor": me, "target": target,
                        "pai": events[j]["pai"]}
            j += 1
        ev = events[j]
        if phase == "claim" and ev.get("actor") == me and \
                ev["type"] in ("chi", "pon", "daiminkan"):
            return ev
        if ev["type"] in ("chi", "pon", "daiminkan") and ev.get("actor") != me:
            raise Ambiguous()          # someone else took the tile
        if events[i + 1]["type"] == "hora" or events[j]["type"] == "hora":
            raise Ambiguous()          # someone else's ron closed the window
        return {"type": "none"}
    raise AssertionError(phase)


def harvest_game(path, out, stats):
    xml = open(path).read()
    full = parse_mjlog(xml)
    events = full["events"]
    fname = os.path.basename(path)

    for me in range(4):
        stream = mask_for_seat(events, me)
        rows, ctx = [], {"i": 0, "kyoku": -1}

        def recorder(table, seat, actions):
            try:
                reaction = derive_reaction(events, ctx["i"], seat, bot.phase)
            except Ambiguous:
                stats["ambiguous_claim"] += 1
                return '<action type="skip" />', {}, None
            a_xml = reaction_to_action(reaction, bot)
            if _norm([a_xml]) <= _norm(actions):
                stats[f"ok_{bot.phase}"] += 1
                idx = action_to_index(a_xml)
                rows.append({"file": fname, "kyoku": ctx["kyoku"], "seat": seat,
                             "phase": bot.phase, "action": a_xml, "idx": idx,
                             "n_legal": len(actions)})
            else:
                stats["illegal"] += 1
                stats[f"illegal_{bot.phase}"] += 1
                if len(stats["illegal_examples"]) < 20:
                    stats["illegal_examples"].append(
                        {"file": fname, "seat": seat, "phase": bot.phase,
                         "kyoku": ctx["kyoku"], "human": a_xml,
                         "legal": actions[:12]})
            return a_xml, {}, None

        bot = MjaiDnnBot(recorder, seat=me)
        try:
            for i, ev in enumerate(stream):
                ctx["i"] = i
                if ev["type"] == "start_kyoku":
                    ctx["kyoku"] += 1
                bot.react(ev)
        except Exception as e:                                # noqa: BLE001
            stats["replay_crash"] += 1
            if len(stats["crash_examples"]) < 10:
                stats["crash_examples"].append(
                    {"file": fname, "seat": me, "err": repr(e)[:200]})
            continue
        for r in rows:
            out.write(json.dumps(r) + "\n")
        stats["seat_replays"] += 1
    stats["games"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/tenhou/raw")
    ap.add_argument("--out", default="data/tenhou/decisions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "*", "*.mjlog")))
    if a.limit:
        files = files[: a.limit]
    stats = collections.Counter()
    stats["illegal_examples"] = []
    stats["crash_examples"] = []

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as out:
        for n, path in enumerate(files, 1):
            try:
                harvest_game(path, out, stats)
            except Exception as e:                            # noqa: BLE001
                stats["parse_fail"] += 1
                if len(stats["crash_examples"]) < 10:
                    stats["crash_examples"].append(
                        {"file": os.path.basename(path), "err": repr(e)[:200]})
            if n % 200 == 0:
                print(f"[{n}/{len(files)}] {dict((k, v) for k, v in stats.items() if isinstance(v, int))}",
                      flush=True)

    ok = sum(v for k, v in stats.items() if isinstance(k, str) and k.startswith("ok_"))
    bad = stats["illegal"]
    print("\n=== fidelity summary ===")
    for k in sorted(k for k in stats if isinstance(stats[k], int)):
        print(f"  {k}: {stats[k]}")
    total = ok + bad
    print(f"  decisions kept: {ok}   illegal: {bad} "
          f"({(bad / total * 100) if total else 0:.4f}%)")
    for ex in stats["illegal_examples"]:
        print("  ILLEGAL", json.dumps(ex, ensure_ascii=False))
    for ex in stats["crash_examples"]:
        print("  CRASH  ", json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
