import os
import re
import random
from typing import Any, Dict, List, Optional, TypedDict

import torch
from langgraph.graph import StateGraph, END

from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE
from src.tasks.mahjong.prompts import SYSTEM_PROMPT, build_user_content
from src.core.chat_format import visible_text, render_generation_prompt
from src.core.rollout import TrajectoryStep

# Interrupt resolution priority: ron beats kan/pon beats chi (atamahane:
# earlier seat order wins ties because collection order is preserved).
_PRIORITY = {"ron": 0, "kan": 1, "pon": 2, "chi": 3}


class MahjongState(TypedDict):
    table: PyMahjongTable
    trajectories: Dict[int, List[TrajectoryStep]]
    model: Any
    tokenizer: Any
    done: bool
    last_player: int
    needs_interrupt: bool
    exp_dir: str
    capture_logprobs: bool


def _extract_action(raw_output: str) -> Optional[str]:
    """Extracts the action tag from OUTSIDE the think block only —
    actions merely mentioned while reasoning must not be executed."""
    m = ACTION_RE.search(visible_text(raw_output))
    return m.group(0) if m else None


def _query(state: MahjongState, player_id: int, legal_actions: List[str]):
    """Asks the LLM (or a random fallback policy) for an action.
    Returns (prompt_text, raw_output, parsed_action_or_None, gen_ids, old_lp).
    gen_ids/old_lp are None unless state['capture_logprobs'] is set (PPO):
    then they hold the sampled token ids and the behavior policy's
    raw-logit logprobs for them."""
    table = state['table']
    model, tokenizer = state.get('model'), state.get('tokenizer')
    capture = state.get('capture_logprobs', False)
    obs = table._format_state(player_id)
    user_content = build_user_content(obs, legal_actions)
    gen_ids, old_lp = None, None

    if model and tokenizer:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = render_generation_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=256,
            do_sample=True, temperature=0.9, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            # Sampling (not greedy) is essential: policy-gradient RL can only
            # reinforce actions the policy actually explores.
            if capture:
                out = model.generate(
                    **inputs, **gen_kwargs,
                    return_dict_in_generate=True, output_logits=True,
                )
                generated = out.sequences[0][inputs.input_ids.shape[-1]:]
                # output_logits = RAW pre-warp logits: old/new logprobs share
                # the same (untempered) convention, as PPO requires.
                step_logits = torch.stack(out.logits, dim=0).squeeze(1).float()
                logprobs = torch.log_softmax(step_logits, dim=-1)
                tok_lp = logprobs[torch.arange(generated.shape[0],
                                               device=generated.device),
                                  generated]
                gen_ids = generated.tolist()
                old_lp = tok_lp.cpu().tolist()
            else:
                outputs = model.generate(**inputs, **gen_kwargs)
                generated = outputs[0][inputs.input_ids.shape[-1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = _extract_action(raw_output)
    else:
        prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_content}"
        parsed = random.choice(legal_actions)
        raw_output = parsed
    return prompt, raw_output, parsed, gen_ids, old_lp


def _log_live(state: MahjongState, header: str, prompt: str, raw: str, parsed: str):
    exp_dir = state.get('exp_dir') or "./logs"
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "live_rollout.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"=== {header} ===\n[INPUT PROMPT]:\n{prompt}\n"
            f"[MODEL OUTPUT]:\n{raw}\n[PARSED ACTION]: {parsed}\n{'-' * 60}\n"
        )


def _log_result(state: MahjongState, table: PyMahjongTable):
    if not table.result_summary:
        return
    exp_dir = state.get('exp_dir') or "./logs"
    path = os.path.join(exp_dir, "live_rollout.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"=== 对局结束: {table.result_summary} ===\n")


def turn_node(state: MahjongState):
    table = state['table']
    player_id = table.turn
    legal_actions = table.get_legal_actions(player_id)

    prompt, raw_output, parsed, gen_ids, old_lp = _query(
        state, player_id, legal_actions)
    action_for_engine = parsed or ""
    _log_live(state, f"[Player {player_id}]", prompt, raw_output,
              parsed or "(no action tag)")

    obs_dict, rewards, done, info = table.step(player_id, action_for_engine)
    state['done'] = done
    state['last_player'] = player_id
    state['needs_interrupt'] = info.get('discarded', False)

    state['trajectories'][player_id].append(TrajectoryStep(
        prompt_text=prompt,
        action_text=raw_output,
        reward=rewards[player_id],
        is_terminal=done,
        gen_token_ids=gen_ids,
        old_logprobs=old_lp,
    ))
    if done:
        _log_result(state, table)
    return state


def interrupt_node(state: MahjongState):
    table = state['table']
    last_discarder = state['last_player']

    # Phase 1: collect every interested player's decision.
    candidates = []
    for offset in range(1, 4):
        player_id = (last_discarder + offset) % 4
        options = table.get_interrupt_actions(player_id)
        if len(options) == 1:  # skip-only: don't bother the LLM
            continue
        prompt, raw_output, parsed, gen_ids, old_lp = _query(
            state, player_id, options)
        _log_live(state, f"[Player {player_id} (Interrupt)]", prompt,
                  raw_output, parsed or "(no action tag)")
        a_type = None
        if parsed:
            m = ACTION_RE.search(parsed)
            a_type = m.group(1) if m else None
        candidates.append({
            "player_id": player_id, "prompt": prompt, "raw": raw_output,
            "parsed": parsed, "type": a_type, "reward": 0.0,
            "gen_ids": gen_ids, "old_lp": old_lp,
        })

    # Phase 2: resolve by priority (ron > kan > pon > chi); collection
    # order breaks ties, giving atamahane for double ron.
    executed = None
    done = False
    for cand in sorted(candidates, key=lambda c: _PRIORITY.get(c["type"], 9)):
        if cand["parsed"] is None:
            cand["reward"] = table.FORMAT_PENALTY
            continue
        if cand["type"] == "skip" or cand["type"] is None:
            continue
        if executed is not None:
            continue  # lost the priority race: action not applied, no penalty
        _, rewards, i_done, info = table.step_interrupt(
            cand["player_id"], cand["parsed"]
        )
        cand["reward"] = rewards[cand["player_id"]]
        if info.get("interrupt", False):
            executed = cand
            done = i_done

    # Phase 3: record every queried player's trajectory step.
    for cand in candidates:
        state['trajectories'][cand["player_id"]].append(TrajectoryStep(
            prompt_text=cand["prompt"],
            action_text=cand["raw"],
            reward=cand["reward"],
            is_terminal=done and executed is cand,
            gen_token_ids=cand["gen_ids"],
            old_logprobs=cand["old_lp"],
        ))

    if executed is not None:
        state['done'] = done
        state['last_player'] = executed["player_id"]
        if done:
            _log_result(state, table)
    else:
        _, r_done = table.advance_turn()
        state['done'] = r_done
        if r_done:
            _log_result(state, table)

    state['needs_interrupt'] = False
    return state


def should_continue(state: MahjongState) -> str:
    if state.get('done', False):
        return END
    if state.get('needs_interrupt', False):
        return "interrupt"
    return "turn"


def build_mahjong_graph():
    builder = StateGraph(MahjongState)
    builder.add_node("turn", turn_node)
    builder.add_node("interrupt", interrupt_node)
    builder.set_entry_point("turn")
    builder.add_conditional_edges(
        "turn", should_continue,
        {"interrupt": "interrupt", "turn": "turn", END: END},
    )
    builder.add_conditional_edges(
        "interrupt", should_continue,
        {"interrupt": "interrupt", "turn": "turn", END: END},
    )
    return builder.compile()


def run_rollout(num_games: int, model=None, tokenizer=None,
                exp_dir: str = None,
                capture_logprobs: bool = False,
                value_facts: bool = False) -> List[List[TrajectoryStep]]:
    """Runs self-play games and returns one trajectory per player per game,
    with terminal settlement rewards distributed to all four players.
    capture_logprobs=True additionally records sampled token ids + behavior
    logprobs on every step (required for PPO)."""
    graph = build_mahjong_graph()
    all_episodes = []

    for game_idx in range(num_games):
        table = PyMahjongTable(value_facts=value_facts)
        trajectories = {i: [] for i in range(4)}

        live_log_dir = exp_dir or "./logs"
        os.makedirs(live_log_dir, exist_ok=True)
        mode = "w" if game_idx == 0 else "a"
        with open(os.path.join(live_log_dir, "live_rollout.txt"), mode,
                  encoding="utf-8") as f:
            f.write(f"=== NEW MAHJONG GAME ROLLOUT (game {game_idx}) ===\n")

        initial_state = MahjongState({
            "table": table,
            "trajectories": trajectories,
            "model": model,
            "tokenizer": tokenizer,
            "done": False,
            "last_player": -1,
            "needs_interrupt": False,
            "exp_dir": exp_dir,
            "capture_logprobs": capture_logprobs,
        })
        final_state = graph.invoke(
            initial_state, config={"recursion_limit": 1000}
        )

        # Distribute end-of-game settlement to EVERY player's trajectory —
        # deal-in penalties and opponent losses must reach the gradient.
        table = final_state['table']
        trajectories = final_state['trajectories']
        if table.final_rewards:
            for pid in range(4):
                if trajectories[pid]:
                    trajectories[pid][-1].reward += table.final_rewards[pid]
                    trajectories[pid][-1].is_terminal = True

        for pid in range(4):
            if trajectories[pid]:
                all_episodes.append(trajectories[pid])

    return all_episodes
