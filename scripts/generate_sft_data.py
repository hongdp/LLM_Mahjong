"""
Generate SFT training data from complete simulated Mahjong games.

Each sample is one player turn:
  - prompt  : ChatML-formatted system+user message (game state)
  - response: <think>...</think><action type="discard" tile="X" />

The discard chosen is either the best Ukeire move (if shanten module
is available) or a random legal tile, giving the model a realistic
but always-legal teacher signal.

Usage:
    python -m scripts.generate_sft_data --num_games 200 --out data/sft_mahjong.jsonl
"""

import argparse
import json
import os
import random
import sys
import re

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tasks.mahjong.table import PyMahjongTable

try:
    from src.tasks.mahjong.shanten import TileEfficiency
    _te = TileEfficiency()
    USE_SHANTEN = True
except Exception:
    _te = None
    USE_SHANTEN = False

# ─────────────────────────────────────────────
# Prompt template (mirrors orchestrator.py)
# ─────────────────────────────────────────────
SYSTEM_CONTENT = (
    "你是一个专业的日本麻将AI。你的最终目标是胡牌。\n"
    "### 麻将基础知识：\n"
    "- 胡牌：当你的手牌加上摸到的一张牌，刚好凑成4个面子（顺子/刻子）加1个雀头（对子），总计14张牌时，即为胡牌，这是游戏的最终获胜目标。\n"
    "- 顺子：同花色相连的3张牌（例如 1m 2m 3m）。\n"
    "- 刻子：相同的3张牌（例如 5p 5p 5p）。\n"
    "- 对子：相同的2张牌（例如 7z 7z）。\n"
    "### 状态说明：\n"
    "- 场况 (Global)：包含场风、局数和宝牌指示牌。\n"
    "- 私有 (Private)：包含你的自风、点数和手牌。注：【点数】是你的游戏得分/筹码（初始25000），与凑齐胡牌牌型无关。"
    "牌名使用天凤拼音：m=万，p=筒，s=索，z=字牌（1z-4z为东南西北，5z-7z为白发中）。\n"
    "- 公共 (Public)：包含其他所有玩家的牌河和副露。\n"
    "### 规则与输出格式要求：\n"
    "1. 每轮只打一张牌：你每次行动只能从手牌中选择【一张】牌打出，而不是多张。\n"
    "2. 必须合法：你【只能】打出目前存在于你【手牌】中的牌。打出没有的牌将受到严厉惩罚。\n"
    "3. 思考过程：所有的思考分析必须全部写在 <think> 和 </think> 标签内部。"
    "禁止使用 Thought:、### 解答、discard X Y Z 等无关格式。\n"
    "4. 动作格式：思考结束后，在 </think> 标签外部只输出唯一的单行XML动作。\n"
    "   - type 属性必须且只能填写 discard，严禁使用 cut/play/hit。\n"
    "   - tile 属性只能填写【一个】牌名（如 1m），不能填多个。\n"
    "### 输出示例（必须严格遵循，思考过程必须简短）：\n"
    "<think>\n"
    "手牌中1m是多余的孤张，且无法凑成顺子或刻子，打出1m。\n"
    "</think>\n"
    "<action type=\"discard\" tile=\"1m\" />\n"
)


def choose_discard(hand: list[str]) -> str:
    """Pick the best tile to discard (highest-Ukeire or random)."""
    if USE_SHANTEN and _te is not None:
        try:
            candidates = _te.evaluate_discards(hand)
            if candidates:
                best = max(candidates, key=lambda t: len(candidates[t]))
                return best
        except Exception:
            pass
    return random.choice(hand)


def make_think(tile: str, hand: list[str]) -> str:
    """Generate a short, plausible thinking rationale for the chosen discard."""
    # Check if it forms any pair / partial sequence context
    count = hand.count(tile)
    suit = tile[-1]
    num = tile[:-1] if tile[:-1].isdigit() else ""

    if suit == "z":
        reason = f"{tile}是字牌（孤张），无法凑成顺子，优先打出。"
    elif count == 1:
        # Check adjacency
        neighbors = []
        if num:
            n = int(num)
            for delta in [-2, -1, 1, 2]:
                neighbor = f"{n+delta}{suit}"
                if neighbor in hand and neighbor != tile:
                    neighbors.append(neighbor)
        if neighbors:
            reason = f"{tile}附近有搭子，但整体而言是多余牌，打出{tile}。"
        else:
            reason = f"{tile}是孤张，无法与其他牌凑成顺子或刻子，打出{tile}。"
    else:
        reason = f"{tile}在手牌中多余，打出{tile}以优化手牌结构。"
    return reason


def format_messages(obs: str) -> tuple[str, str]:
    """Return (system_content, user_content) for a given observation."""
    user_content = f"### 当前状态：\nState:\n{obs}\n\n请输出你的动作："
    return SYSTEM_CONTENT, user_content


def simulate_game(game_id: int) -> list[dict]:
    """Run one complete game and collect (prompt, response) pairs."""
    table = PyMahjongTable()
    table.reset()
    samples = []

    max_turns = 200  # safety cap
    for _ in range(max_turns):
        player_id = table.turn
        hand = list(table.hands[player_id])
        obs = table._format_state(player_id)

        tile = choose_discard(hand)
        think_text = make_think(tile, hand)
        response = (
            f"<think>\n{think_text}\n</think>\n"
            f'<action type="discard" tile="{tile}" />'
        )

        system_content, user_content = format_messages(obs)

        # Store in both raw-message and flat-text form
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": response},
        ]
        # Flat text for trainers that expect a single "text" field
        flat_text = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

        samples.append({
            "game_id": game_id,
            "player_id": player_id,
            "messages": messages,
            "text": flat_text,
        })

        # Step env with the chosen tile
        action_xml = f'<action type="discard" tile="{tile}" />'
        _, _, done, _ = table.step(player_id, action_xml)
        if done:
            break

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Mahjong SFT data")
    parser.add_argument("--num_games", type=int, default=200,
                        help="Number of complete games to simulate")
    parser.add_argument("--out", type=str, default="data/sft_mahjong.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    total_samples = 0
    strategy = "Ukeire-optimal" if USE_SHANTEN else "random-legal"
    print(f"Generating SFT data: {args.num_games} games | discard strategy: {strategy}")

    with open(args.out, "w", encoding="utf-8") as f:
        for game_id in range(args.num_games):
            samples = simulate_game(game_id)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_samples += len(samples)
            if (game_id + 1) % 20 == 0:
                print(f"  {game_id+1}/{args.num_games} games  ({total_samples} samples so far)")

    print(f"\n✅ Done! {total_samples} samples written to {args.out}")
    print(f"   Avg {total_samples / args.num_games:.1f} turns/game")


if __name__ == "__main__":
    main()
