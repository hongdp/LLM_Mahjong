"""Is mid-game value decodable from the policy's prompt hidden state?

exp4 verdict: the trained value head reached only ~0.02 out-of-sample
explained variance while its in-training MSE looked much better. That is
consistent with two very different diagnoses:

  (A) the FEATURES carry no generalizable value signal  -> need richer
      context (v3 threaded prompt), a bigger head won't help;
  (B) the HEAD/optimization overfit 2048-dim features   -> regularization,
      lower lr, or more data fixes it.

This probe separates them: on a FRESH rollout it measures (1) the trained
head as-is, and (2) a ridge regression fit on a train split and scored on
a held-out split. Splits are BY EPISODE — steps inside one game share the
terminal settlement, so a step-level split leaks the label and inflates R2.

If ridge test-R2 is also ~0, the features are the problem (diagnosis A).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import get_task                     # noqa: E402
from src.core.value_head import (explained_variance,       # noqa: E402
                                 last_prompt_hidden, load_value_head)
import src.tasks.mahjong.task                              # noqa: E402,F401
# ^ importing the module is what REGISTERS "mahjong" in the task registry;
#   without it get_task() raises "Task 'mahjong' not found" (trainer.py does
#   the same import for the same reason).


def collect(model, tokenizer, task, games, exp_dir, gamma):
    buffer = task.collect_rollouts(num_episodes=games, model=model,
                                   tokenizer=tokenizer, exp_dir=exp_dir,
                                   capture_logprobs=False)
    prompts, returns, ep_ids = [], [], []
    for ep_id, episode in enumerate(buffer.episodes):
        R = 0.0
        rets = []
        for step in reversed(episode):
            R = step.reward + gamma * R
            rets.insert(0, R)
        for step, r in zip(episode, rets):
            prompts.append(step.prompt_text)
            returns.append(r)
            ep_ids.append(ep_id)
    return prompts, torch.tensor(returns), torch.tensor(ep_ids)


def collect_from_logs(paths, gamma):
    """Parse episodes out of saved rollout logs (free, and 10x the sample
    size of a fresh rollout). Format written by trainer.save_trajectory_log:
        --- Episode N (Total Steps: K) ---
        [Step i] Reward: R | Terminal: T
        PROMPT:
        <prompt text ...>
        ACTION: <action text ...>
    Caveat: states come from the policy at that epoch, not exactly the
    checkpoint being probed — fine for a decodability question when the
    policies are near-identical (exp4 ref_kl 0.0066), noted in the report.
    """
    import glob as _glob
    import re as _re
    prompts, returns, ep_ids = [], [], []
    ep_counter = 0
    for path in sorted(sum([_glob.glob(p) for p in paths], [])):
        txt = open(path, encoding="utf-8", errors="replace").read()
        for ep_block in _re.split(r"^--- Episode \d+ .*?---$", txt, flags=_re.M)[1:]:
            steps = _re.findall(
                r"^\[Step \d+\] Reward: (-?[\d.]+) \| Terminal: \w+\nPROMPT:\n(.*?)\nACTION:",
                ep_block, flags=_re.M | _re.S)
            if not steps:
                continue
            rewards = [float(r) for r, _ in steps]
            R, rets = 0.0, []
            for r in reversed(rewards):
                R = r + gamma * R
                rets.insert(0, R)
            for (_, pr), ret in zip(steps, rets):
                prompts.append(pr.strip())
                returns.append(ret)
                ep_ids.append(ep_counter)
            ep_counter += 1
    return prompts, torch.tensor(returns), torch.tensor(ep_ids)


def hidden_states(model, tokenizer, prompts, device, batch_size=4):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = [tokenizer(p)["input_ids"] for p in chunk]
            m = max(len(e) for e in enc)
            ids = torch.full((len(enc), m), pad_id, dtype=torch.long)
            attn = torch.zeros((len(enc), m), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e)
                attn[j, :len(e)] = 1
            res = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                        output_hidden_states=True)
            h = last_prompt_hidden(res.hidden_states[-1],
                                   torch.tensor([len(e) for e in enc], device=device))
            out.append(h.float().cpu())
    return torch.cat(out)


def ridge_fit_score(h_tr, y_tr, h_te, y_te, alpha):
    """Closed-form ridge on centered features; returns test explained variance."""
    mu, sd = h_tr.mean(0, keepdim=True), h_tr.std(0, keepdim=True) + 1e-6
    xtr, xte = (h_tr - mu) / sd, (h_te - mu) / sd
    ymu = y_tr.mean()
    ytr = y_tr - ymu
    d = xtr.shape[1]
    A = xtr.T @ xtr + alpha * torch.eye(d, dtype=xtr.dtype)
    w = torch.linalg.solve(A, xtr.T @ ytr)
    pred_te = xte @ w + ymu
    return explained_variance(pred_te, y_te), explained_variance(xtr @ w + ymu, y_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="critic checkpoint dir (must contain value_head.pt)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--out", default="value_probe.json")
    ap.add_argument("--exp_dir", default="/tmp/value_probe")
    ap.add_argument("--control", action="store_true",
                    help="POSITIVE CONTROL: also decode a quantity that is "
                         "literally written in the prompt (own points). If "
                         "this scores high while the return scores ~0, the "
                         "probe works and the RETURN is what is unpredictable.")
    ap.add_argument("--from_logs", nargs="*", default=None,
                    help="glob(s) of saved rollout logs to parse instead of "
                         "rolling out fresh games (free, larger sample)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter)
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.from_logs:
        print(f"[probe] parsing episodes from logs: {args.from_logs}")
        prompts, returns, ep_ids = collect_from_logs(args.from_logs, args.gamma)
    else:
        task = get_task("mahjong", device=device, reward_model="settlement",
                        gamma=args.gamma, parallel_games=args.parallel)
        print(f"[probe] rolling out {args.games} fresh games...")
        prompts, returns, ep_ids = collect(model, tokenizer, task, args.games,
                                           args.exp_dir, args.gamma)
    print(f"[probe] {len(prompts)} states from {int(ep_ids.max()) + 1} episodes; "
          f"return std {returns.std():.3f}")

    print("[probe] forwarding prompts for hidden states...")
    H = hidden_states(model, tokenizer, prompts, device)

    head_path = os.path.join(args.adapter, "value_head.pt")
    result = {"n_states": len(prompts), "n_episodes": int(ep_ids.max()) + 1,
              "return_std": returns.std().item()}
    if os.path.exists(head_path):
        head = load_value_head(head_path, device="cpu")
        with torch.no_grad():
            v = head(H)
        result["trained_head_ev_all"] = explained_variance(v, returns)
        print(f"[probe] trained head EV (fresh data): {result['trained_head_ev_all']:.4f}")

    # Episode-level split: steps in one game share the settlement label.
    n_ep = int(ep_ids.max()) + 1
    n_te = max(1, int(n_ep * args.test_frac))
    te_eps = set(range(n_ep - n_te, n_ep))
    te_mask = torch.tensor([int(e) in te_eps for e in ep_ids])
    tr_mask = ~te_mask
    print(f"[probe] split: {int(tr_mask.sum())} train / {int(te_mask.sum())} test "
          f"states ({n_ep - n_te}/{n_te} episodes)")

    # alpha selected by K-fold CV *inside the train split* (episode-level
    # folds). Picking alpha by test EV would be test-set peeking and would
    # bias the diagnosis optimistically.
    ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    tr_eps = sorted({int(e) for e in ep_ids[tr_mask]})
    K = min(5, len(tr_eps))
    folds = [set(tr_eps[i::K]) for i in range(K)]
    cv_scores = {}
    for alpha in ALPHAS:
        evs = []
        for f in folds:
            va = torch.tensor([int(e) in f for e in ep_ids]) & tr_mask
            tr = tr_mask & ~va
            if va.sum() < 5 or tr.sum() < 20:
                continue
            ev, _ = ridge_fit_score(H[tr], returns[tr], H[va], returns[va], alpha)
            evs.append(ev)
        cv_scores[alpha] = sum(evs) / max(len(evs), 1) if evs else float("-inf")
        print(f"[probe] ridge alpha={alpha:<9g} CV_EV(train folds)={cv_scores[alpha]:+.4f}")
    best_alpha = max(cv_scores, key=cv_scores.get)
    te_ev, tr_ev = ridge_fit_score(H[tr_mask], returns[tr_mask],
                                   H[te_mask], returns[te_mask], best_alpha)
    result["ridge"] = {"cv_scores": {str(k): v for k, v in cv_scores.items()},
                       "selected_alpha": best_alpha,
                       "train_ev": tr_ev, "test_ev": te_ev}
    print(f"[probe] selected alpha={best_alpha:g} (by train-fold CV) -> "
          f"train_EV={tr_ev:+.4f}  HELD-OUT test_EV={te_ev:+.4f}")
    best = te_ev
    result["best_ridge_test_ev"] = best

    if args.control:
        import re as _re
        pts_re = _re.compile(r'私有[^\n]*?点数: (\d+)')
        y_ctrl, keep = [], []
        for i, pr in enumerate(prompts):
            m = pts_re.search(pr)
            if m:
                y_ctrl.append(float(m.group(1)) / 1000.0)
                keep.append(i)
        if len(keep) > 100:
            k = torch.tensor(keep)
            Hc, yc, ec = H[k], torch.tensor(y_ctrl), ep_ids[k]
            tec = torch.tensor([int(e) in te_eps for e in ec])
            ctrl_ev, ctrl_tr = ridge_fit_score(Hc[~tec], yc[~tec],
                                               Hc[tec], yc[tec], best_alpha)
            result["control_own_points"] = {"test_ev": ctrl_ev,
                                            "train_ev": ctrl_tr,
                                            "n": len(keep)}
            print(f"[probe] CONTROL decode own-points from the same hidden "
                  f"states (n={len(keep)}): train_EV={ctrl_tr:+.4f} "
                  f"test_EV={ctrl_ev:+.4f}")
            print("[probe]   -> high control EV + near-zero return EV means "
                  "the probe is sound and the RETURN is the unpredictable part")
    result["diagnosis"] = ("features_insufficient (need richer context)"
                           if best < 0.10 else
                           "head_or_optimization (features do carry signal)")
    print(f"[probe] BEST ridge test EV = {best:.4f} -> {result['diagnosis']}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
