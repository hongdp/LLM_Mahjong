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
    "你是一个专业的日本麻将AI。\n"
    "### 状态说明：\n"
    "- 场况 (Global)：包含场风、局数和宝牌指示牌。\n"
    "- 私有 (Private)：包含你的自风、点数和手牌。\n"
    "- 公共 (Public)：包含其他所有玩家的牌河和副露。\n"
    "- 合法动作 (Legal Actions)：包含当前你可以执行的合法动作列表。\n"
    "### 输出格式要求：\n"
    "你必须且只能从当前状态的【合法动作】列表中选择一个动作输出。\n"
    "所有的思考必须写在 <think> 和 </think> 标签内。\n"
    "最后在标签外部输出唯一的单行XML动作（如 <action type=\"discard\" tile=\"1m\" />）。\n"
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


def format_messages(obs: str, legal_actions: list[str]) -> tuple[str, str]:
    """Return (system_content, user_content) for a given observation."""
    legal_actions_str = "\n".join([f"  - {act}" for act in legal_actions])
    user_content = f"### 当前状态：\nState:\n{obs}合法动作 (Legal Actions)：\n{legal_actions_str}\n\n请输出你的动作："
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
        legal_actions = table.get_legal_actions(player_id)

        # Allow testing tsumo/riichi directly if available
        non_discard = [a for a in legal_actions if 'discard' not in a]
        if non_discard and random.random() < 0.5:
            action_xml = random.choice(non_discard)
            tile = re.search(r'tile="([^"]+)"', action_xml)
            tile = tile.group(1) if tile else ""
            think_text = "条件满足，执行特殊动作以推进胜利进度。"
        else:
            tile = choose_discard(hand)
            think_text = make_think(tile, hand)
            action_xml = f'<action type="discard" tile="{tile}" />'
            
        response = f"<think>\n{think_text}\n</think>\n{action_xml}"
        system_content, user_content = format_messages(obs, legal_actions)

        # Store in both raw-message and flat-text form
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": response},
        ]
        flat_text = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

        samples.append({"game_id": game_id, "player_id": player_id, "messages": messages, "text": flat_text})

        # Step env with the chosen action
        _, _, done, _ = table.step(player_id, action_xml)
        if done: break
        
        # --- INTERRUPT PHASE ---
        interrupted = False
        for offset in range(1, 4):
            i_id = (player_id + offset) % 4
            interrupt_actions = table.get_interrupt_actions(i_id)
            non_skip = [a for a in interrupt_actions if 'skip' not in a]
            
            if non_skip and random.random() < 0.3:  # 30% chance to claim
                action = random.choice(non_skip)
                i_obs = table._format_state(i_id)
                i_think = f"机会来了，可以执行鸣牌或和牌动作，执行 {action}。"
                i_response = f"<think>\n{i_think}\n</think>\n{action}"
                i_sys, i_user = format_messages(i_obs, interrupt_actions)
                
                i_msg = [
                    {"role": "system", "content": i_sys},
                    {"role": "user", "content": i_user},
                    {"role": "assistant", "content": i_response}
                ]
                i_flat = f"<|im_start|>system\n{i_sys}<|im_end|>\n<|im_start|>user\n{i_user}<|im_end|>\n<|im_start|>assistant\n{i_response}<|im_end|>"
                samples.append({"game_id": game_id, "player_id": i_id, "messages": i_msg, "text": i_flat})
                
                _, _, done, _ = table.step_interrupt(i_id, action)
                interrupted = True
                break
                
        if done: break
        
        if not interrupted:
            _, done = table.advance_turn()
            if done: break

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
