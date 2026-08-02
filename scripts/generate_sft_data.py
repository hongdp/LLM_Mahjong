"""
Generate SFT training data from complete simulated Mahjong games.

Each sample is one player decision (turn or interrupt phase):
  - prompt  : ChatML system+user message built from the SAME prompt module
              the live orchestrator uses (src/tasks/mahjong/prompts.py)
  - response: <think>...</think> + one action copied verbatim from the
              engine-provided legal action list

FAITHFUL CoT: the <think> content is derived from the SAME computation
that selects the action — real shanten counts, ukeire comparisons, wait
tiles, and yaku from the scorer — never post-hoc template phrases. The
model is taught the decision procedure (enumerate -> compare -> pick),
not a rationalization habit.

Teacher policy: always take validated tsumo/ron; declare riichi often
(picking the widest wait); claim melds only when they REDUCE shanten,
and only sometimes; explicit skip decisions (with the shanten reasoning
for skipping) are also sampled so the model learns to pass.

Usage:
    python -m scripts.generate_sft_data --num_games 300 --out data/sft_mahjong.jsonl
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE
from src.tasks.mahjong.prompts import SYSTEM_PROMPT, build_user_content
from src.tasks.mahjong.shanten import (TileEfficiency, pad_for_melds,
                                       dora_from_indicator)

_te = TileEfficiency()

RIICHI_PROB = 0.8
MELD_PROB = 0.25     # gate applies only to shanten-REDUCING melds
SKIP_SAMPLE_PROB = 0.3
KAN_PROB = 0.3
# Match the training default (MahjongTask.randomize_round) so the teacher
# covers every 场风/自风 combination the policy will meet at rollout time.
RANDOMIZE_ROUND = True


def _shanten(tiles: list, n_melds: int) -> int:
    n_melds = max(0, min(n_melds, (14 - len(tiles)) // 3))
    try:
        return _te.calculate_shanten(pad_for_melds(tiles, n_melds))
    except ValueError:
        return 8


def ranked_discards(hand: list, n_melds: int) -> dict:
    """{discard_tile: (post_shanten, ukeire_list)} for tiles in hand."""
    try:
        ranked = _te.evaluate_discards_ranked(pad_for_melds(hand, n_melds))
        return {t: v for t, v in ranked.items() if t in hand}
    except Exception:
        return {}


def waits_after_discard(table, pid: int, discard_tile: str) -> list:
    rest = list(table.hands[pid])
    rest.remove(discard_tile)
    n = len(table.melds[pid])
    waits = []
    for i34 in range(34):
        from src.tasks.mahjong.table import str_from_34
        t = str_from_34(i34)
        if rest.count(t) < 4 and _shanten(rest + [t], n) == -1:
            waits.append(t)
    return waits


VALUE_AWARE = False  # set by --value_facts: tie-break keeps dora


def discard_decision(table, pid: int, hand: list):
    """Shanten-first, ukeire-second discard + faithful think from the
    same computation. With VALUE_AWARE, ties on (shanten, ukeire) are
    broken by keeping dora — and the think says so (faithful CoT)."""
    n_melds = len(table.melds[pid])
    ranked = ranked_discards(hand, n_melds)
    if not ranked:
        tile = random.choice(hand)
        return tile, f"打{tile}调整手牌结构。"
    key = lambda t: (ranked[t][0], -len(ranked[t][1]))
    tile = min(ranked, key=key)
    value_note = ""
    if VALUE_AWARE:
        dora_tiles = {dora_from_indicator(i) for i in table.dora_indicators}
        tied = [t for t in ranked if key(t) == key(tile)]
        non_dora = [t for t in tied if t not in dora_tiles]
        kept_dora = [t for t in tied if t in dora_tiles]
        if kept_dora and non_dora and tile in dora_tiles:
            tile = non_dora[0]
            value_note = (
                f"打{tile}与打{kept_dora[0]}同向听同受入，"
                f"弃{tile}保留宝牌{kept_dora[0]}。"
            )
    top3 = sorted(ranked.items(), key=lambda kv: (kv[1][0], -len(kv[1][1])))[:3]
    comparison = "，".join(
        f"打{t}→{sh}向听/受入{len(uk)}种" for t, (sh, uk) in top3
    )
    sh_best = ranked[tile][0]
    return tile, (
        f"候选对比：{comparison}。打{tile}保持{sh_best}向听且受入最大。"
        + value_note
    )


def win_think(table, pid: int, win_tile: str, is_tsumo: bool) -> str:
    result = table._win_result(pid, win_tile, is_tsumo=is_tsumo)
    how = "自摸" if is_tsumo else "荣和"
    if result is None:  # defensive: should not happen for offered wins
        return f"{win_tile}补全和牌形，{how}。"
    yaku = "、".join(str(y) for y in (result.yaku or []))
    return f"{win_tile}补全和牌形：{yaku}，{result.han}番{result.fu}符，{how}。"


def riichi_think(table, pid: int, tile: str) -> str:
    waits = waits_after_discard(table, pid, tile)
    visible = table.discards
    remain = {
        w: 4 - table.hands[pid].count(w)
        - sum(d.replace('*', '') == w for r in visible.values() for d in r)
        for w in waits
    }
    wait_str = " ".join(f"{w}(剩{max(0, remain[w])}张)" for w in waits)
    return f"打{tile}后门清听牌，等 {wait_str}。宣告立直施压。"


def meld_shanten_delta(table, pid: int, action_xml: str):
    """(before, after) shanten if this claim were executed."""
    m = ACTION_RE.search(action_xml)
    a_type, tile, with_attr = m.group(1), m.group(2), m.group(3)
    hand = list(table.hands[pid])
    n = len(table.melds[pid])
    before = _shanten(hand, n)
    removed = []
    if a_type == "pon":
        removed = [tile, tile]
    elif a_type == "kan":
        removed = [tile, tile, tile]
    elif a_type == "chi" and with_attr:
        removed = with_attr.split()
    rest = list(hand)
    for t in removed:
        if t in rest:
            rest.remove(t)
    after = _shanten(rest, n + 1)
    return before, after


MELD_NAME = {"pon": "碰", "kan": "杠", "chi": "吃"}


def make_response(think: str, action_xml: str) -> str:
    return f"<think>\n{think}\n</think>\n{action_xml}"


def make_sample(game_id: int, player_id: int, obs: str,
                legal_actions: list, action_xml: str, think: str) -> dict:
    user_content = build_user_content(obs, legal_actions)
    response = make_response(think, action_xml)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    flat_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return {"game_id": game_id, "player_id": player_id,
            "messages": messages, "text": flat_text}


def pick_turn_action(table, pid: int, hand: list, legal_actions: list):
    """Teacher decision for the turn phase. Returns (action_xml, think)."""
    by_type = {}
    for a in legal_actions:
        by_type.setdefault(ACTION_RE.search(a).group(1), []).append(a)

    if "tsumo" in by_type:
        return by_type["tsumo"][0], win_think(
            table, pid, table.last_drawn[pid], is_tsumo=True)

    if table.riichi[pid]:
        a = by_type["discard"][0]
        tile = ACTION_RE.search(a).group(2)
        return a, f"已立直锁手，只能摸切{tile}，等待听牌命中。"

    if "riichi" in by_type and random.random() < RIICHI_PROB:
        best_a, best_waits = None, []
        for a in by_type["riichi"]:
            tile = ACTION_RE.search(a).group(2)
            waits = waits_after_discard(table, pid, tile)
            if len(waits) > len(best_waits):
                best_a, best_waits = a, waits
        tile = ACTION_RE.search(best_a).group(2)
        return best_a, riichi_think(table, pid, tile)

    if "kan" in by_type and random.random() < KAN_PROB:
        a = by_type["kan"][0]
        tile = ACTION_RE.search(a).group(2)
        return a, f"{tile}已集齐四张，开杠翻新宝牌并补摸岭上牌。"

    tile, think = discard_decision(table, pid, hand)
    a = f'<action type="discard" tile="{tile}" />'
    if a not in legal_actions:
        a = next((x for x in legal_actions if 'discard' in x), legal_actions[0])
    return a, think


def pick_interrupt_action(table, i_id: int, options: list):
    """Teacher decision for the interrupt phase.
    Returns (action_xml, think, is_claim) or None to emit no sample."""
    # During a chankan window the contested tile is the added kan's tile,
    # not the stale last discard (RCR 4.2.1.12).
    if table.pending_kan:
        claim_tile = table.pending_kan["tile"]
    else:
        claim_tile = table.last_discard.replace('*', '') if table.last_discard else ""
    ron = next((a for a in options if 'ron' in a), None)
    if ron is not None:
        return ron, win_think(table, i_id, claim_tile, is_tsumo=False), True

    melds = [a for a in options if 'skip' not in a]
    if not melds:
        return None
    # Evaluate every claim by its real shanten delta.
    scored = []
    for a in melds:
        before, after = meld_shanten_delta(table, i_id, a)
        scored.append((after, before, a))
    scored.sort()
    after, before, best = scored[0]
    m = ACTION_RE.search(best)
    name = MELD_NAME.get(m.group(1), m.group(1))

    if after < before and random.random() < MELD_PROB:
        think = (f"{name}{claim_tile}后向听{before}→{after}，"
                 f"鸣牌加速听牌。")
        return best, think, True
    if random.random() < SKIP_SAMPLE_PROB:
        if after >= before:
            think = (f"鸣牌不降向听（{before}→{after}），"
                     f"破坏门清不值得，跳过。")
        else:
            think = (f"虽然{name}{claim_tile}可降向听（{before}→{after}），"
                     f"但保留门清和立直机会价值更高，跳过。")
        return '<action type="skip" />', think, False
    return None


def simulate_game(game_id: int) -> list:
    table = PyMahjongTable(value_facts=VALUE_AWARE,
                           randomize_round=RANDOMIZE_ROUND)
    samples = []

    for _ in range(400):  # safety cap; games end naturally well before
        player_id = table.turn
        hand = list(table.hands[player_id])
        obs = table._format_state(player_id)
        legal_actions = table.get_legal_actions(player_id)

        action_xml, think = pick_turn_action(
            table, player_id, hand, legal_actions)
        samples.append(make_sample(
            game_id, player_id, obs, legal_actions, action_xml, think))

        _, _, done, info = table.step(player_id, action_xml)
        if done:
            break
        if not (info.get("discarded", False) or info.get("chankan")):
            continue  # ankan: same player keeps the turn

        # --- INTERRUPT PHASE (ron first, then pon/kan over chi) ---
        # An added kan opens a ron-only window here (chankan, RCR 4.2.1.12).
        claims = []
        for offset in range(1, 4):
            i_id = (player_id + offset) % 4
            options = table.get_interrupt_actions(i_id)
            if len(options) == 1:
                continue
            decision = pick_interrupt_action(table, i_id, options)
            if decision is None:
                continue
            action, i_think, is_claim = decision
            i_obs = table._format_state(i_id)
            if is_claim:
                claims.append((0 if 'ron' in action else 1,
                               i_id, i_obs, options, action, i_think))
            else:
                samples.append(make_sample(
                    game_id, i_id, i_obs, options, action, i_think))

        interrupted = False
        for _prio, i_id, i_obs, options, action, i_think in sorted(
                claims, key=lambda c: c[0]):
            if interrupted:
                break
            samples.append(make_sample(
                game_id, i_id, i_obs, options, action, i_think))
            _, _, done, info = table.step_interrupt(i_id, action)
            interrupted = info.get("interrupt", False)
            if done:
                return samples

        if not interrupted:
            if table.pending_kan:
                # Nobody robbed the kan: it completes and the same player
                # carries on to their post-rinshan discard.
                table.resolve_pending_kan()
            else:
                _, done = table.advance_turn()
                if done:
                    break

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Mahjong SFT data")
    parser.add_argument("--num_games", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/sft_mahjong.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value_facts", action="store_true",
                        help="Value-aware template + teacher: 自家宝牌 line "
                             "in prompts, dora-keeping tie-break in CoT.")
    args = parser.parse_args()

    global VALUE_AWARE
    VALUE_AWARE = args.value_facts
    random.seed(args.seed)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for game_id in range(args.num_games):
            samples = simulate_game(game_id)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total += len(samples)
            if (game_id + 1) % 20 == 0:
                print(f"  {game_id + 1}/{args.num_games} games "
                      f"({total} samples so far)")

    print(f"\n✅ Done! {total} samples written to {args.out}")
    print(f"   Avg {total / args.num_games:.1f} samples/game")


if __name__ == "__main__":
    main()
