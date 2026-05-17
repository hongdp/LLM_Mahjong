from typing import Dict, Any
from langgraph.graph import StateGraph, END
import re

class MahjongState(Dict[str, Any]):
    """
    State passed between nodes in the LangGraph orchestration.
    Contains: table_engine, current_player, action_history, etc.
    """
    pass

def turn_node(state: MahjongState):
    """
    Wakes up the active player to draw and discard a tile.
    (In actual training, this yields control back to the RL batch generator)
    """
    table = state['table']
    player_id = table.turn
    obs = table._format_state(player_id)
    
    # In a full simulation, we'd query the LLM here.
    # For Phase 0 training loops, the graph serves to manage the state flow,
    # but the actual batched generation happens inside train_rlhf.py.
    
    # Let's mock an LLM response randomly from legal actions for demonstration
    import random
    legal_actions = table.get_legal_actions(player_id)
    action_xml = random.choice(legal_actions) if legal_actions else '<action type="skip" />'
    
    state['last_action'] = action_xml
    state['last_player'] = player_id
    
    # Apply step
    obs_dict, rewards, done, _ = table.step(player_id, action_xml)
    state['done'] = done
    
    return state

def interrupt_node(state: MahjongState):
    """
    Parallel check for other 3 players to declare Pon/Ron.
    """
    # If a discard happened, ask other 3 bots (or LLMs) if they want to Pon/Ron.
    # In Phase 0 local testing, we assume no interruptions to keep the basic loop simple.
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
    builder.add_conditional_edges("turn", should_continue, {"interrupt": "interrupt", END: END})
    builder.add_edge("interrupt", "turn")
    
    return builder.compile()
