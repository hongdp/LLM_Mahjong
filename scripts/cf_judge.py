"""exp109 judge: who ranks exposed discards better — the policy, the blind Q, or the oracle Q — against realised
same-wall counterfactual returns (Python-engine branches: at each riichi-exposed discard decision, cf_k alternative
discards are played to the end of the deal with the same policy).

Per decision: candidates = {executed a0: R0} ∪ {alt b: R_b}.  For each ranker (policy log-pi, BQ, OQ):
  gain_vs_executed  = R(ranker's favourite among candidates) - R0          (paired; 0 when it keeps a0)
  pair_acc          = P[ sign(score(a0)-score(b)) == sign(R0-R_b) ]        (over pairs with |R0-R_b| > 0)
  pair_acc_w        = same, weighted by |R0 - R_b|
usage: cf_judge.py <run_dir_with_critic_ckpts> <policy.pt> <n_games> <workers> <out.json>
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.agents.dnn.parallel_rollout import collect_parallel
    from src.agents.dnn.oracle_critic import SeparateCritic
    from scripts.run_elo_league import load_dnn
    run, pol, n_games, workers, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    blob = torch.load(pol, map_location="cpu"); net = load_dnn(pol, "cpu")
    cfg = dict(channels=blob.get("channels", 64), blocks=blob.get("blocks", 3), arch=blob.get("arch"), temperature=1.0, gamma=0.995,
               games_per_worker=16, rollout_temps=None, shaping=False, seed=4242, hanchan=False, hanchan_w_path=None,
               critic_feats="oracle_ron", gpu_infer=True, gpu_infer_opponents=True, infer_max_batch=128, infer_wait_ms=0.0,
               infer_device="cuda", bf16_infer=False, no_episodes=False, league_frac=0.0, league=None,
               encoder_variant=getattr(net, "encoder_variant", "v3r"), action_space=getattr(net, "action_space", "native"),
               symmetrize=blob.get("symmetrize"), cf_p=1.0, cf_k=3, cf_only_exposed=True, cf_branch_slots=96)
    t0 = time.time()
    eps, _ = collect_parallel(net, n_games, cfg, workers, [5_100_000 + i for i in range(n_games)])
    dt = time.time() - t0
    netg = load_dnn(pol, "cuda").eval()
    crits = {}
    for name in ("BQ", "OQ", "ODQ"):
        f = os.path.join(run, f"critic_{name}_latest.pt")
        if os.path.exists(f):
            cb = torch.load(f, map_location="cuda")
            c = SeparateCritic("q", cb["channels"], cb["blocks"], blind=cb["blind"]).cuda(); c.load_state_dict(cb["state_dict"]); c.eval()
            crits[name] = (c, bool(cb.get("decomposed", False)))
    from scripts.critic_stage0 import ron_cost_full
    rankers = ["policy"] + list(crits)
    gain = {r: [] for r in rankers}; pairs = {r: [] for r in rankers}; wpairs = {r: [] for r in rankers}; n_dec = 0
    with torch.no_grad():
        for e in eps:
            alts = e.get("cf_alts") or {}
            if not alts:
                continue
            steps = sorted(alts)
            P = torch.from_numpy(np.asarray(e["planes"][steps])).cuda().float(); S = torch.from_numpy(np.asarray(e["scalars"][steps])).cuda().float()
            M = torch.from_numpy(np.asarray(e["mask"][steps])).cuda(); CF = torch.from_numpy(np.asarray(e["cfeats"][steps])).cuda().float()
            C, RC = CF[:, :319], CF[:, 319:]
            cfull = ron_cost_full(RC, M)
            scores = {"policy": torch.log_softmax(netg(P, S, M), 1)}
            for name, (c, dec) in crits.items():
                q = c(P, S, C)
                if dec:
                    q = q - cfull
                scores[name] = q.masked_fill(~M, float("-inf"))
            for j, st in enumerate(steps):
                a0 = int(e["actions"][st]); R0 = float(e["returns"][st]); cand = {a0: R0, **{int(b): float(r) for b, r in alts[st].items() if int(b) != a0}}
                if len(cand) < 2:
                    continue
                n_dec += 1
                slots = list(cand); R = np.array([cand[s] for s in slots])
                for r in rankers:
                    sc = scores[r][j, slots].float().cpu().numpy()
                    pick = slots[int(np.argmax(sc))]
                    gain[r].append(cand[pick] - R0)
                    for b in slots:
                        if b == a0 or cand[b] == R0:
                            continue
                        agree = float(np.sign(scores[r][j, a0].item() - scores[r][j, b].item()) == np.sign(R0 - cand[b]))
                        pairs[r].append(agree); wpairs[r].append((agree, abs(R0 - cand[b])))
    res = {"games": n_games, "collect_s": round(dt), "games_per_s": round(n_games / dt, 1), "decisions": n_dec, "critic_games": None}
    for r in rankers:
        g = np.array(gain[r]); p = np.array(pairs[r]); w = np.array(wpairs[r])
        res[r] = {"gain_vs_executed_pts": round(float(g.mean()) * 1000, 1), "gain_se_pts": round(float(g.std(ddof=1) / np.sqrt(len(g))) * 1000, 1),
                  "frac_keeps_executed": round(float((g == 0).mean()), 3),
                  "pair_acc": round(float(p.mean()), 4), "pair_acc_se": round(float(p.std(ddof=1) / np.sqrt(len(p))), 4),
                  "pair_acc_weighted": round(float((w[:, 0] * w[:, 1]).sum() / w[:, 1].sum()), 4), "n_pairs": int(len(p))}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
