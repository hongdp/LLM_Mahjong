"""Parallel-game rollout: N tables progress concurrently, and every table's
pending decision is folded into ONE batched generate() call per round.

Motivation (measured, Aug 2026): single-game rollout is host-launch-bound —
~11 tok/s decode on an A100 at 18% GPU util. Batching B concurrent sequences
amortizes the per-token host overhead across games for a near-linear rollout
speedup, without touching any game semantics.

Game logic is a 1:1 port of orchestrator.turn_node / interrupt_node driven as
a generator: each game yields a list of decision requests, the scheduler
answers them (batched LLM call or random fallback), and the generator applies
engine transitions exactly like the sequential path.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.tasks.mahjong.table import PyMahjongTable, ACTION_RE
from src.tasks.mahjong.prompts import SYSTEM_PROMPT, build_user_content, get_system_prompt
from src.core.chat_format import visible_text, render_generation_prompt
from src.core.rollout import TrajectoryStep
from src.tasks.mahjong.orchestrator import _resolve_claims, _extract_action


@dataclass
class _Req:
    player_id: int
    prompt: str          # fully rendered generation prompt (or debug pseudo)
    legal: List[str]
    # filled by the scheduler:
    raw: str = ""
    parsed: Optional[str] = None
    gen_ids: Optional[list] = None
    old_lp: Optional[list] = None


def _mk_request(table, player_id, legal, model, tokenizer,
                sys_prompt=SYSTEM_PROMPT) -> _Req:
    obs = table._format_state(player_id)
    user_content = build_user_content(obs, legal)
    if model and tokenizer:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = render_generation_prompt(tokenizer, messages)
    else:
        prompt = f"System: {sys_prompt}\nUser: {user_content}"
    return _Req(player_id=player_id, prompt=prompt, legal=legal)


def _drive_game(table: PyMahjongTable, trajectories: Dict[int, list],
                model, tokenizer, sys_prompt=SYSTEM_PROMPT):
    """Generator: yields [_Req, ...]; expects the same list back with
    raw/parsed/gen_ids/old_lp filled. Mirrors orchestrator node logic."""
    while True:
        # ---- turn phase (orchestrator.turn_node) ----
        player_id = table.turn
        legal = table.get_legal_actions(player_id)
        req = _mk_request(table, player_id, legal, model, tokenizer, sys_prompt)
        (req,) = yield [req]

        _, rewards, done, info = table.step(player_id, req.parsed or "")
        trajectories[player_id].append(TrajectoryStep(
            prompt_text=req.prompt, action_text=req.raw,
            reward=rewards[player_id], is_terminal=done,
            gen_token_ids=req.gen_ids, old_logprobs=req.old_lp,
        ))
        if done:
            return
        # A discard opens the call window; an added kan opens a
        # chankan-only window (RCR 4.2.1.12).
        if not (info.get('discarded', False) or info.get('chankan')):
            continue

        # ---- interrupt phase (orchestrator.interrupt_node) ----
        reqs = []
        for offset in range(1, 4):
            pid = (player_id + offset) % 4
            options = table.get_interrupt_actions(pid)
            if len(options) == 1:   # skip-only: don't bother the LLM
                continue
            reqs.append(_mk_request(table, pid, options, model, tokenizer, sys_prompt))

        candidates = []
        if reqs:
            reqs = yield reqs
            for req in reqs:
                a_type = None
                if req.parsed:
                    m = ACTION_RE.search(req.parsed)
                    a_type = m.group(1) if m else None
                candidates.append({
                    "player_id": req.player_id, "prompt": req.prompt,
                    "raw": req.raw, "parsed": req.parsed, "type": a_type,
                    "reward": 0.0, "gen_ids": req.gen_ids, "old_lp": req.old_lp,
                })

        executed, done = _resolve_claims(table, candidates)

        for cand in candidates:
            trajectories[cand["player_id"]].append(TrajectoryStep(
                prompt_text=cand["prompt"], action_text=cand["raw"],
                reward=cand["reward"],
                is_terminal=done and cand in executed,
                gen_token_ids=cand["gen_ids"], old_logprobs=cand["old_lp"],
            ))

        if executed:
            if done:
                return
        elif table.pending_kan:
            table.resolve_pending_kan()
        else:
            _, r_done = table.advance_turn()
            if r_done:
                return


def _batch_generate(model, tokenizer, reqs: List[_Req],
                    capture: bool, max_batch: int = 24):
    """Fills raw/parsed(/gen_ids/old_lp) on every request, batching the
    LLM calls. Falls back to the random policy without a model."""
    if model is None or tokenizer is None:
        for r in reqs:
            r.parsed = random.choice(r.legal)
            r.raw = r.parsed
        return

    eos = tokenizer.eos_token_id
    for lo in range(0, len(reqs), max_batch):
        chunk = reqs[lo:lo + max_batch]
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer([r.prompt for r in chunk], return_tensors="pt",
                        padding=True).to(model.device)
        tokenizer.padding_side = old_side
        gen_kwargs = dict(max_new_tokens=256, do_sample=True,
                          temperature=0.9, top_p=0.95, pad_token_id=eos)
        with torch.no_grad():
            if capture:
                out = model.generate(**enc, **gen_kwargs,
                                     return_dict_in_generate=True,
                                     output_logits=True)
                seqs = out.sequences
                step_logits = out.logits   # tuple[gen_len] of [B, vocab]
            else:
                seqs = model.generate(**enc, **gen_kwargs)
                step_logits = None
        in_len = enc.input_ids.shape[1]
        for b, r in enumerate(chunk):
            ids = seqs[b, in_len:].tolist()
            if eos in ids:                       # cut at first eos, inclusive
                ids = ids[:ids.index(eos) + 1]
            r.raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
            r.parsed = _extract_action(r.raw)
            if capture and step_logits is not None and ids:
                # per-sequence slice keeps the fp32 log_softmax bounded
                sl = torch.stack([step_logits[t][b] for t in range(len(ids))])
                lp = torch.log_softmax(sl.float(), dim=-1)
                tok_lp = lp[torch.arange(len(ids)),
                            torch.tensor(ids, device=lp.device)]
                r.gen_ids = ids
                r.old_lp = tok_lp.cpu().tolist()


def _log_batch(exp_dir, game_idx, reqs: List[_Req]):
    path = os.path.join(exp_dir or "./logs", "live_rollout.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in reqs:
            f.write(
                f"=== [game {game_idx} | Player {r.player_id}] ===\n"
                f"[INPUT PROMPT]:\n{r.prompt}\n[MODEL OUTPUT]:\n{r.raw}\n"
                f"[PARSED ACTION]: {r.parsed or '(no action tag)'}\n{'-'*60}\n"
            )


def run_rollout_batched(num_games: int, model=None, tokenizer=None,
                        exp_dir: str = None, capture_logprobs: bool = False,
                        value_facts: bool = False,
                        no_think: bool = False,
                        parallel: int = 4,
                        randomize_round: bool = False
                        ) -> List[List[TrajectoryStep]]:
    """Drop-in alternative to orchestrator.run_rollout with concurrent games.
    Semantics per game are identical; only scheduling differs."""
    log_dir = exp_dir or "./logs"
    os.makedirs(log_dir, exist_ok=True)
    open(os.path.join(log_dir, "live_rollout.txt"), "w", encoding="utf-8").write(
        f"=== BATCHED ROLLOUT: {num_games} games, parallel={parallel} ===\n")

    all_episodes: List[List[TrajectoryStep]] = []
    next_game = 0
    active: Dict[int, dict] = {}

    def finalize(gid):
        st = active.pop(gid)
        table, trajectories = st["table"], st["traj"]
        if table.final_rewards:
            for pid in range(4):
                if trajectories[pid]:
                    last = trajectories[pid][-1]
                    last.reward += table.final_rewards[pid]
                    last.is_terminal = True
                    last.settlement = table.final_rewards[pid]
                    last.final_points = table.points[pid]
                    last.rank_bonus = (table.final_rewards[pid]
                                       - (table.points[pid] - 25000)
                                       * table.REWARD_SCALE)
                    last.game_result = table.result_summary
        for pid in range(4):
            if trajectories[pid]:
                all_episodes.append(trajectories[pid])
        if table.result_summary:
            with open(os.path.join(log_dir, "live_rollout.txt"), "a",
                      encoding="utf-8") as f:
                f.write(f"=== 对局结束 (game {gid}): {table.result_summary} ===\n")

    while next_game < num_games or active:
        # top up the pool
        while next_game < num_games and len(active) < parallel:
            table = PyMahjongTable(value_facts=value_facts,
                                   randomize_round=randomize_round)
            trajectories = {i: [] for i in range(4)}
            gen = _drive_game(table, trajectories, model, tokenizer,
                              get_system_prompt(no_think))
            gid = next_game
            next_game += 1
            try:
                pending = next(gen)
                active[gid] = {"gen": gen, "table": table,
                               "traj": trajectories, "pending": pending}
            except StopIteration:      # pathological: instant end
                active[gid] = {"gen": gen, "table": table,
                               "traj": trajectories, "pending": []}
                finalize(gid)

        if not active:
            break

        # one batched decision round across all active games
        flat: List[_Req] = []
        for st in active.values():
            flat.extend(st["pending"])
        _batch_generate(model, tokenizer, flat, capture_logprobs,
                        max_batch=max(24, parallel))

        for gid in list(active.keys()):
            st = active[gid]
            _log_batch(exp_dir, gid, st["pending"])
            try:
                st["pending"] = st["gen"].send(st["pending"])
            except StopIteration:
                finalize(gid)

    return all_episodes
