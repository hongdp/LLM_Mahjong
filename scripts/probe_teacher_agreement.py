"""How faithfully did each architecture absorb the SAME teacher?

The DNN reports a held-out teacher-agreement rate for free (it is trained
with cross-entropy on teacher actions). The LLM's SFT never measured the
equivalent number, so the two were not directly comparable. This probe
fixes that: it replays teacher games on HELD-OUT seeds, asks the LLM (or
a DNN) for an action in exactly the states the teacher decided in, and
reports the agreement rate on identical states for both.

Every agent sees the same state and the same legal-action list, so the
number is an apples-to-apples measure of imitation quality.
"""

import argparse
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.encoder import action_to_index, encode_state, legal_mask  # noqa: E402
from src.core.chat_format import render_generation_prompt, visible_text        # noqa: E402
from src.tasks.mahjong.orchestrator import _resolve_claims                     # noqa: E402
from src.tasks.mahjong.prompts import SYSTEM_PROMPT, build_user_content        # noqa: E402
from src.tasks.mahjong.table import PyMahjongTable                             # noqa: E402
import scripts.generate_sft_data as teacher                                    # noqa: E402

ACTION_TAG = re.compile(r'<action[^>]*/>')


def collect_states(n_games, seed0):
    """Replay teacher games; yield (table_snapshot_fn, pid, legal, teacher_action).

    States are captured as (deepcopy of table, pid) so both agents can be
    queried on exactly the same position.
    """
    import copy
    out = []
    for g in range(n_games):
        random.seed(seed0 + g)
        table = PyMahjongTable(randomize_round=True)
        guard = 0
        while not table.finished and guard < 600:
            guard += 1
            pid = table.turn
            legal = table.get_legal_actions(pid)
            if not legal:
                break
            a_xml, _ = teacher.pick_turn_action(table, pid, table.hands[pid], legal)
            if a_xml in legal and len(legal) > 1:
                out.append((copy.deepcopy(table), pid, list(legal), a_xml))
            if a_xml not in legal:
                a_xml = legal[0]
            _, _, done, info = table.step(pid, a_xml)
            if done:
                break
            if not (info.get("discarded") or info.get("chankan")):
                continue
            candidates = []
            for off in range(1, 4):
                other = (pid + off) % 4
                options = table.get_interrupt_actions(other)
                if len(options) == 1:
                    continue
                picked = teacher.pick_interrupt_action(table, other, options)
                a2 = picked[0] if picked else '<action type="skip" />'
                if a2 not in options:
                    a2 = '<action type="skip" />'
                out.append((copy.deepcopy(table), other, list(options), a2))
                m = re.search(r'type="(\w+)"', a2)
                candidates.append({"player_id": other, "parsed": a2,
                                   "type": m.group(1) if m else None, "reward": 0.0})
            executed, done = _resolve_claims(table, candidates)
            if done:
                break
            if not executed:
                if table.pending_kan:
                    table.resolve_pending_kan()
                else:
                    _, r_done = table.advance_turn()
                    if r_done:
                        break
    return out


def ask_llm(model, tokenizer, table, pid, legal, temperature):
    obs = table._format_state(pid)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(obs, legal)}]
    prompt = render_generation_prompt(tokenizer, messages)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.95,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][enc.input_ids.shape[-1]:], skip_special_tokens=True)
    vis = visible_text(text)                      # action must be OUTSIDE <think>
    tags = ACTION_TAG.findall(vis)
    return (tags[-1] if tags else None), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LLM adapter to probe")
    ap.add_argument("--dnn", default=None, help="DNN checkpoint to probe")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=900000, help="HELD-OUT seeds")
    ap.add_argument("--max_states", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="LLM sampling temperature (0 = greedy)")
    ap.add_argument("--dnn_temperature", type=float, default=0.0,
                    help="DNN sampling temperature (0 = greedy). Must match "
                         "the LLM's setting for a fair fidelity comparison.")
    ap.add_argument("--out", default="teacher_agreement.json")
    args = ap.parse_args()

    print(f"[probe] replaying {args.games} teacher games (held-out seeds)...")
    states = collect_states(args.games, args.seed0)
    random.seed(0)
    random.shuffle(states)
    states = states[:args.max_states]
    print(f"[probe] {len(states)} decision points with >1 legal action")

    result = {"n_states": len(states)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dnn:
        from src.agents.dnn.net import MahjongPolicyNet
        blob = torch.load(args.dnn, map_location=device)
        net = MahjongPolicyNet(channels=blob.get("channels", 64),
                               blocks=blob.get("blocks", 3)).to(device).eval()
        net.load_state_dict(blob["state_dict"])
        hit = 0
        for table, pid, legal, truth in states:
            mask, lookup = legal_mask(legal)
            planes, scal = encode_state(table, pid)
            idx, _ = net.act(planes[None].to(device), scal[None].to(device),
                             mask[None].to(device),
                             temperature=args.dnn_temperature)
            hit += int(lookup.get(int(idx)) == truth)
        result["dnn_agreement"] = hit / max(len(states), 1)
        print(f"[probe] DNN greedy agreement: {result['dnn_agreement']:.1%}")

    if args.adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                    device_map={"": 0})
        model = PeftModel.from_pretrained(base, args.adapter).eval()
        hit = illegal = nofmt = 0
        for i, (table, pid, legal, truth) in enumerate(states):
            act, _ = ask_llm(model, tok, table, pid, legal, args.temperature)
            if act is None:
                nofmt += 1
            elif act not in legal:
                illegal += 1
            elif act == truth:
                hit += 1
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(states)} agree={hit/(i+1):.1%}", flush=True)
        n = max(len(states), 1)
        result.update(llm_agreement=hit / n, llm_illegal_rate=illegal / n,
                      llm_no_action_rate=nofmt / n)
        print(f"[probe] LLM agreement: {result['llm_agreement']:.1%} "
              f"(illegal {result['llm_illegal_rate']:.1%}, "
              f"no-tag {result['llm_no_action_rate']:.1%})")

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
