"""Fidelity check of the MJAI bridge with a REAL policy checkpoint.

Plays engine self-play with the checkpoint on all four seats; one seat is
driven through the MJAI bridge (engine -> MJAI events -> ShadowTable ->
policy -> MJAI reaction -> engine action). At every bridge decision the
shadow table's encoder tensors (v1 + v3) and legal-action set must equal
the engine's, and the reaction must round-trip to the chosen action.
Prints coverage (how many riichi / ron / chankan / kan decisions the bot
actually made) so a green run is known to be non-vacuous.

Usage:
  PYTHONPATH=. python scripts/verify_mjai_bridge.py --ckpt <path.pt> --games 50
"""

import argparse
import collections
import random
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.encoder import encode_state                       # noqa: E402
from src.agents.dnn.mjai_bridge import MjaiDnnBot, load_policy         # noqa: E402
from src.agents.dnn.mjai_export import play_game_mjai                  # noqa: E402
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable          # noqa: E402
from tests.test_mjai_bridge import _norm, reaction_to_action           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=424242)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    policy = load_policy(a.ckpt, "cpu", a.temperature)
    stats = collections.Counter()

    for g in range(a.games):
        seed, me = a.seed0 + g, g % 4
        random.seed(seed)
        torch.manual_seed(seed)
        table = PyMahjongTable(randomize_round=True)
        table.text_obs = False

        def checking(shadow, pid, actions):
            real = (table.get_legal_actions(me) if bot.phase == "turn"
                    else table.get_interrupt_actions(me))
            assert _norm(actions) == _norm(real), (seed, bot.phase, actions, real)
            assert len(shadow.wall) == len(table.wall)
            for variant in ("v1", "v3"):
                p0, s0 = encode_state(table, me, variant=variant)
                p1, s1 = encode_state(shadow, me, variant=variant)
                assert torch.equal(p0, p1) and torch.equal(s0, s1), (seed, variant)
            stats["decision:" + bot.phase] += 1
            return policy(shadow, pid, actions)

        bot = MjaiDnnBot(checking, seat=me)
        last = {}

        def sink(ev):
            r = bot.react(ev)
            if r is not None:
                last["r"] = r

        def bot_policy(_t, pid, actions):
            r = last.pop("r")
            xml = reaction_to_action(r, bot)
            assert _norm([xml]) == _norm([bot.last_decision["chosen"]]), (r, bot.last_decision)
            assert _norm([xml]).pop() in _norm(actions)
            stats["react:" + r["type"]] += 1
            return xml

        def net_policy(t, pid, actions):
            return policy(t, pid, actions)[0]

        play_game_mjai(table, {p: (bot_policy if p == me else net_policy) for p in range(4)}, me, sink)
        assert "r" not in last or last["r"]["type"] == "none"
        stats["games"] += 1
        stats["result:" + ("hora" if "和" in table.result_summary or "tsumo" in table.result_summary.lower()
                           or "ron" in table.result_summary.lower() else "other")] += 1
    print("OK", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
