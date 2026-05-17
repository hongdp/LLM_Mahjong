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
    
    prompt = obs + "\nAction: "
    
    if model and tokenizer:
        # Actually use the LLM to generate the action
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        # Decode only the newly generated text
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        action_xml = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        # Fallback to random legal action (for testing pipeline quickly)
        legal_actions = table.get_legal_actions(player_id)
        action_xml = random.choice(legal_actions) if legal_actions else '<action type="skip" />'
        
    state['last_action'] = action_xml
    state['last_player'] = player_id
    
    # Apply step to environment
    obs_dict, rewards, done, _ = table.step(player_id, action_xml)
    state['done'] = done
    
    # Record trajectory
    step = TrajectoryStep(
        prompt_text=prompt,
        action_text=action_xml,
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

def run_rollout(num_games: int, model=None, tokenizer=None) -> List[List[TrajectoryStep]]:
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
        
        initial_state = MahjongState({
            "table": table,
            "trajectories": trajectories,
            "model": model,
            "tokenizer": tokenizer,
            "done": False,
            "last_action": "",
            "last_player": -1
        })
        
        # Run graph until END
        final_state = graph.invoke(initial_state)
        
        # Add all 4 player's trajectories to the replay buffer
        for p_id in range(4):
            if len(final_state['trajectories'][p_id]) > 0:
                all_episodes.append(final_state['trajectories'][p_id])
                
    return all_episodes
