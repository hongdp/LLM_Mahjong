"""Shadow-table fidelity: engine self-play replayed as an MJAI stream must
reproduce, at every decision of the observer seat, the exact encoder
tensors (v1 and v3) and the exact legal-action set the engine itself
produces — and the MJAI reaction must round-trip to the chosen action.

Seat `me` is played BY THE BOT (its reaction is converted back into the
engine action), the other three seats by a seeded random policy, so the
stream exercises our own chi/pon/kan/riichi/ron paths as well as theirs.
"""

import random
import unittest

import torch

from src.agents.dnn.encoder import encode_state
from src.agents.dnn.mjai_bridge import MjaiDnnBot, mjai_to_engine
from src.agents.dnn.mjai_export import play_game_mjai
from src.tasks.mahjong.table import ACTION_RE, PyMahjongTable


def reaction_to_action(reaction, bot):
    """MJAI reaction -> engine action xml for the bot's seat."""
    tb, t = bot.table, reaction["type"]
    if t == "none":
        return '<action type="skip" />'
    if t == "hora":
        return '<action type="tsumo" />' if reaction["target"] == bot.seat else '<action type="ron" />'
    if t == "reach":
        return f'<action type="riichi" tile="{mjai_to_engine(reaction["reach_dahai"]["pai"])}" />'
    if t == "dahai":
        return f'<action type="discard" tile="{mjai_to_engine(reaction["pai"])}" />'
    if t == "chi":
        a, b = [mjai_to_engine(c) for c in reaction["consumed"]]
        return f'<action type="chi" tile="{mjai_to_engine(reaction["pai"])}" with="{a} {b}" />'
    if t == "pon":
        return f'<action type="pon" tile="{mjai_to_engine(reaction["pai"])}" />'
    if t == "daiminkan":
        return f'<action type="kan" tile="{mjai_to_engine(reaction["pai"])}" />'
    if t == "ankan":
        return f'<action type="kan" tile="{mjai_to_engine(reaction["consumed"][0])}" />'
    if t == "kakan":
        return f'<action type="kan" tile="{mjai_to_engine(reaction["pai"])}" />'
    raise AssertionError(reaction)


def _norm(actions):
    out = set()
    for a in actions:
        m = ACTION_RE.search(a)
        w = tuple(sorted((m.group(3) or "").split()))
        out.add((m.group(1), m.group(2), w))
    return out


class ShadowFidelityTest(unittest.TestCase):
    def _run_game(self, seed, me, prefer=None):
        rng = random.Random(seed)
        random.seed(seed)
        table = PyMahjongTable(randomize_round=True)
        table.text_obs = False
        checks = {"n": 0, "claims": 0, "calls": 0}

        def pick(actions, r):
            if prefer:
                liked = [a for a in actions if ACTION_RE.search(a).group(1) in prefer]
                if liked and r.random() < 0.8:
                    return r.choice(liked)
            return r.choice(actions)

        # The bot's policy: verify the shadow against the real table here.
        def checking_policy(shadow, pid, actions):
            self.assertEqual(pid, me)
            phase = bot.phase
            real_actions = (table.get_legal_actions(me) if phase == "turn"
                            else table.get_interrupt_actions(me))
            self.assertEqual(_norm(actions), _norm(real_actions),
                             f"seed={seed} legal set mismatch ({phase})")
            self.assertEqual(sorted(shadow.hands[me]), sorted(table.hands[me]))
            self.assertEqual(len(shadow.wall), len(table.wall), "wall count")
            for variant in ("v1", "v3"):
                p0, s0 = encode_state(table, me, variant=variant)
                p1, s1 = encode_state(shadow, me, variant=variant)
                self.assertTrue(torch.equal(p0, p1), f"planes differ ({variant}) seed={seed}")
                self.assertTrue(torch.equal(s0, s1), f"scalars differ ({variant}) seed={seed}\n{s0}\n{s1}")
            checks["n"] += 1
            chosen = pick(actions, rng)
            probs = {a: 1.0 / len(actions) for a in actions}
            return chosen, probs, 0.0

        bot = MjaiDnnBot(checking_policy, seat=me)
        bot.react({"type": "start_game", "id": me})
        last = {"reaction": None}

        def sink(ev):
            r = bot.react(ev)
            if r is not None:
                last["reaction"] = r

        def bot_policy(_table, pid, actions):
            r = last["reaction"]
            self.assertIsNotNone(r, "engine asked seat me to act but the bot produced no reaction")
            last["reaction"] = None
            xml = reaction_to_action(r, bot)
            self.assertEqual(_norm([xml]), _norm([bot.last_decision["chosen"]]),
                             "reaction does not round-trip to the chosen action")
            self.assertIn(_norm([xml]).pop(), _norm(actions))
            if r["type"] in ("chi", "pon", "daiminkan", "ankan", "kakan"):
                checks["calls"] += 1
            return xml

        def rand_policy(_table, pid, actions):
            return pick(actions, rng)

        policies = {p: (bot_policy if p == me else rand_policy) for p in range(4)}
        play_game_mjai(table, policies, me, sink)
        # an interrupt with only "skip" never reaches the bot; otherwise the
        # engine must not have been left holding an unconsumed reaction
        self.assertTrue(last["reaction"] is None or last["reaction"]["type"] == "none")
        return checks

    def test_many_seeds_random_play(self):
        total = 0
        for seed in range(40):
            total += self._run_game(1000 + seed, seed % 4)["n"]
        self.assertGreater(total, 400)

    def test_call_heavy_play(self):
        """Bias everyone towards calls/riichi/kan so melds, kuikae, rinshan
        draws, kan dora and chankan windows are all exercised."""
        calls = 0
        for seed in range(40):
            c = self._run_game(5000 + seed, seed % 4, prefer={"chi", "pon", "kan", "riichi", "ron", "tsumo"})
            calls += c["calls"]
        self.assertGreater(calls, 20)

    def test_red_five_bookkeeping(self):
        bot = MjaiDnnBot(lambda tb, pid, acts: (acts[0], {acts[0]: 1.0}, 0.0), seat=1)
        bot.react({"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 0,
                   "kyoku": 1, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
                   "tehais": [["?"] * 13, ["1m", "5mr", "5m", "2p", "3p", "4p", "6s", "7s", "8s", "E", "E", "S", "W"],
                              ["?"] * 13, ["?"] * 13]})
        bot.react({"type": "tsumo", "actor": 0, "pai": "?"})
        bot.react({"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": True})
        bot.react({"type": "tsumo", "actor": 1, "pai": "5m"})
        self.assertEqual(bot.table.hands[1].count("5m"), 3)
        # discarding an engine 5m must name a plain copy, not the red one
        self.assertEqual(bot.table.physical("5m"), "5m")
        bot.react({"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True})
        self.assertEqual(sorted(bot.table.my_tiles_mjai).count("5mr"), 1)
        bot.react({"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False})
        self.assertEqual(bot.table.physical("5m"), "5mr")   # only the red one left


if __name__ == "__main__":
    unittest.main()


class ChankanTest(unittest.TestCase):
    def test_rob_added_kan(self):
        """Handcrafted (never arises in 400 random games): we are tenpai on
        3m, an opponent adds 3m to their pon -> the bot must offer ron with
        target = the kan's actor, and the kakan must still mutate the table."""
        seen = {}

        def policy(tb, pid, acts):
            seen["phase"] = tb_bot.phase
            seen["acts"] = list(acts)
            ron = next((a for a in acts if 'type="ron"' in a), acts[0])
            return ron, {a: 1.0 / len(acts) for a in acts}, 0.0

        tb_bot = MjaiDnnBot(policy, seat=2)
        # 1m2m + 4m5m6m + 7p8p9p + 1s2s3s + WW: penchan wait on 3m
        hand = ["1m", "2m", "4m", "5m", "6m", "7p", "8p", "9p", "1s", "2s", "3s", "W", "W"]
        tb_bot.react({"type": "start_kyoku", "bakaze": "E", "dora_marker": "9s", "honba": 0,
                      "kyoku": 1, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
                      "tehais": [["?"] * 13, ["?"] * 13, hand, ["?"] * 13]})
        # opponent 1 pons 3m from seat 0, later draws and adds the 4th 3m
        tb_bot.react({"type": "tsumo", "actor": 0, "pai": "?"})
        tb_bot.react({"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True})
        # (we could ron this 3m too — but a closed hand needs a yaku; pinfu-ish? the
        #  penchan wait has no pinfu, no riichi -> no yaku -> engine offers no ron)
        tb_bot.react({"type": "pon", "actor": 1, "target": 0, "pai": "3m", "consumed": ["3m", "3m"]})
        tb_bot.react({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False})
        tb_bot.react({"type": "tsumo", "actor": 2, "pai": "N"})
        # declare riichi so we have a yaku for the later chankan
        tb_bot.react({"type": "reach", "actor": 2})
        tb_bot.react({"type": "dahai", "actor": 2, "pai": "N", "tsumogiri": True})
        tb_bot.react({"type": "reach_accepted", "actor": 2})
        tb_bot.react({"type": "tsumo", "actor": 3, "pai": "?"})
        tb_bot.react({"type": "dahai", "actor": 3, "pai": "P", "tsumogiri": True})
        tb_bot.react({"type": "tsumo", "actor": 0, "pai": "?"})
        tb_bot.react({"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": True})
        tb_bot.react({"type": "tsumo", "actor": 1, "pai": "?"})
        seen.clear()
        reaction = tb_bot.react({"type": "kakan", "actor": 1, "pai": "3m", "consumed": ["3m"] * 3})
        self.assertEqual(seen.get("phase"), "chankan")
        self.assertEqual(reaction["type"], "hora")
        self.assertEqual(reaction["target"], 1)
        self.assertEqual(reaction["pai"], "3m")
        self.assertEqual(tb_bot.table.melds[1][0]["type"], "shouminkan")
        self.assertEqual(tb_bot.table.kan_count, 1)
        self.assertEqual(tb_bot.table.riichi[2], True)
        self.assertEqual(tb_bot.table.kyotaku, 1000)
        self.assertEqual(tb_bot.table.points[2], 24000)
