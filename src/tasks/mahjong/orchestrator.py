from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
import random
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.tasks.mahjong.table import PyMahjongTable
from src.core.rollout import TrajectoryStep

class MahjongState(TypedDict):
    table: PyMahjongTable
    trajectories: Dict[int, List[TrajectoryStep]]
    model: Any
    tokenizer: Any
    done: bool
    last_action: str
    last_player: int
    exp_dir: str

def turn_node(state: MahjongState):
    """
    Wakes up the active player to draw and discard.
    In a real rollout, we query the LLM.
    """
    table = state['table']
    model = state.get('model')
    tokenizer = state.get('tokenizer')
    player_id = table.turn
    obs = table._format_state(player_id)
    
    system_content = (
        "你是一个专业的日本麻将AI。你的最终目标是胡牌。\n"
        "### 麻将基础知识：\n"
        "- 胡牌：当你的手牌加上摸到的一张牌，刚好凑成4个面子（顺子/刻子）加1个雀头（对子），总计14张牌时，即为胡牌，这是游戏的最终获胜目标。\n"
        "- 顺子：同花色相连的3张牌（例如 1m 2m 3m）。\n"
        "- 刻子：相同的3张牌（例如 5p 5p 5p）。\n"
        "- 对子：相同的2张牌（例如 7z 7z）。\n"
        "### 状态说明：\n"
        "- 场况 (Global)：包含场风、局数和宝牌指示牌。\n"
        "- 私有 (Private)：包含你的自风、点数和手牌。注：【点数】是你的游戏得分/筹码（初始25000），与凑齐胡牌牌型无关。牌名使用天凤拼音：m=万，p=筒，s=索，z=字牌（1z-4z为东南西北，5z-7z为白发中）。\n"
        "- 公共 (Public)：包含其他所有玩家的牌河和副露。\n"
        "### 规则与输出格式要求：\n"
        "1. 每轮只打一张牌：你每次行动只能从手牌中选择【一张】牌打出，而不是多张。\n"
        "2. 必须合法：你【只能】打出目前存在于你【手牌】中的牌。打出没有的牌将受到严厉惩罚。\n"
        "3. 思考过程：所有的思考分析必须全部写在 <think> 和 </think> 标签内部。禁止使用 Thought:、### 解答、discard X Y Z 等无关格式。\n"
        "4. 动作格式：思考结束后，在 </think> 标签外部只输出唯一的单行XML动作。\n"
        "   - type 属性必须且只能填写 discard，严禁使用 cut/play/hit。\n"
        "   - tile 属性只能填写【一个】牌名（如 1m），不能填多个。\n"
        "### 输出示例（必须严格遵循，思考过程必须简短）：\n"
        "<think>\n"
        "手牌中1m是多余的孤张，且无法凑成顺子或刻子，打出1m。\n"
        "</think>\n"
        "<action type=\"discard\" tile=\"1m\" />\n"
    )
    
    user_content = f"### 当前状态：\nState:\n{obs}\n\n请输出你的动作："
    
    if model and tokenizer:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = chat_prompt
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        # Decode only the newly generated text
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        action_text_for_training = raw_output
        
        # Extract the action XML, ignoring the <think> part
        import re
        action_match = re.search(r'<action\s+.*?/>', raw_output)
        action_xml = action_match.group(0) if action_match else '<action type="skip" />'
    else:
        prompt = f"System: {system_content}\nUser: {user_content}"
        # Fallback to random legal action (for testing pipeline quickly)
        legal_actions = table.get_legal_actions(player_id)
        action_xml = random.choice(legal_actions) if legal_actions else '<action type="skip" />'
        raw_output = action_xml
        action_text_for_training = action_xml
        
    state['last_action'] = action_xml
    state['last_player'] = player_id
    
    # --- LIVE LOGGING ---
    exp_dir = state.get('exp_dir') or "./logs"
    import os
    os.makedirs(exp_dir, exist_ok=True)
    live_log_path = os.path.join(exp_dir, "live_rollout.txt")
    with open(live_log_path, "a", encoding="utf-8") as f:
        # If we have raw_output from LLM, log it completely (including <think>)
        if model and tokenizer:
            f.write(f"=== [Player {player_id}] ===\n[INPUT PROMPT]:\n{prompt}\n[MODEL OUTPUT]:\n{raw_output}\n[PARSED ACTION]: {action_xml}\n{'-'*60}\n")
        else:
            f.write(f"=== [Player {player_id}] ===\n[ACTION]: {action_xml}\n{'-'*60}\n")
    # --------------------
    
    # Apply step to environment
    obs_dict, rewards, done, _ = table.step(player_id, action_xml)
    state['done'] = done
    
    # Record trajectory
    step = TrajectoryStep(
        prompt_text=prompt,
        action_text=action_text_for_training,
        reward=rewards[player_id], # Local step reward (e.g. Ukeire)
        is_terminal=done
    )
    state['trajectories'][player_id].append(step)
    
    return state

def interrupt_node(state: MahjongState):
    """
    Parallel check for other 3 players to declare Pon/Ron.
    (Simplified for Phase 0)
    """
    return state

def should_continue(state: MahjongState) -> str:
    if state.get('done', False):
        return END
    
    action = state.get('last_action', '')
    if 'discard' in action:
        return "interrupt"
    return "turn"

def build_mahjong_graph():
    builder = StateGraph(MahjongState)
    builder.add_node("turn", turn_node)
    builder.add_node("interrupt", interrupt_node)
    
    builder.set_entry_point("turn")
    builder.add_conditional_edges("turn", should_continue, {"interrupt": "interrupt", "turn": "turn", END: END})
    builder.add_edge("interrupt", "turn")
    
    return builder.compile()

def run_rollout(num_games: int, model=None, tokenizer=None, exp_dir: str=None) -> List[List[TrajectoryStep]]:
    """
    Runs self-play games and returns a list of episodes.
    Each episode is a list of TrajectorySteps.
    """
    graph = build_mahjong_graph()
    all_episodes = []
    
    for _ in range(num_games):
        table = PyMahjongTable()
        table.reset()
        
        # 4 independent trajectory tracks for the 4 players
        trajectories = {i: [] for i in range(4)}
        
        # Clear live log and write header
        import os
        live_log_dir = exp_dir or "./logs"
        os.makedirs(live_log_dir, exist_ok=True)
        live_log_path = os.path.join(live_log_dir, "live_rollout.txt")
        with open(live_log_path, "w", encoding="utf-8") as f:
            f.write("=== NEW MAHJONG GAME ROLLOUT ===\n")
            
        initial_state = MahjongState({
            "table": table,
            "trajectories": trajectories,
            "model": model,
            "tokenizer": tokenizer,
            "done": False,
            "last_action": "",
            "last_player": -1,
            "exp_dir": exp_dir
        })
        
        # Run graph until END
        final_state = graph.invoke(initial_state)
        
        # Add all 4 player's trajectories to the replay buffer
        for p_id in range(4):
            if len(final_state['trajectories'][p_id]) > 0:
                all_episodes.append(final_state['trajectories'][p_id])
                
    return all_episodes
