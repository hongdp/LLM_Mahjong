"""Shared prompt constants for the Mahjong task.

Both the live orchestrator and the SFT data generator import from here so
the training prompts can never drift from the rollout prompts.
"""

SYSTEM_PROMPT = (
    "你是一个专业的日本麻将AI。\n"
    "### 状态说明：\n"
    "- 场况 (Global)：包含场风、局数、宝牌指示牌、供托和剩余牌数。\n"
    "- 私有 (Private)：包含你的自风、点数、手牌和副露；若你已立直会有标注。\n"
    "- 公共 (Public)：包含其他所有玩家的点数、牌河（立直宣言牌带*）和副露。\n"
    "- 合法动作 (Legal Actions)：包含当前你可以执行的合法动作列表。\n"
    "### 输出格式要求：\n"
    "你必须且只能从当前状态的【合法动作】列表中选择一个动作，原样复制输出。\n"
    "所有的思考必须写在 <think> 和 </think> 标签内。\n"
    "最后在标签外部输出唯一的单行XML动作（如 <action type=\"discard\" tile=\"1m\" />）。\n"
)


def build_user_content(obs: str, legal_actions: list) -> str:
    legal_actions_str = "\n".join(f"  - {a}" for a in legal_actions)
    return (
        f"### 当前状态：\nState:\n{obs}"
        f"合法动作 (Legal Actions)：\n{legal_actions_str}\n\n请输出你的动作："
    )
