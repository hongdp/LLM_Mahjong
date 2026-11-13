import glob, json, numpy as np
P = "/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/r6_mortal"
rows = sorted((r for f in glob.glob(f"{P}/mortal_shard*.json") for r in json.load(open(f))["rows"]), key=lambda r: r["d"])
assert [r["d"] for r in rows] == list(range(len(rows)))
st = {}
for f in glob.glob(f"{P}/mortal_shard*.json"):
    for k, v in json.load(open(f))["stats"].items():
        st[k] = st.get(k, 0) + v
s = np.array([r["share"] for r in rows]); d = np.array([r["diff"] for r in rows])
json.dump({"share": s.tolist(), "diff": d.tolist()}, open("/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/pairdiff_R6_mortal298k_101152000.json", "w"))
se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
print(f"mortal n={len(s)} share={s.mean():.4f} ± {se(s):.4f} pts/wall={d.mean():+.0f} ± {se(d):.0f} bridge {st}")
for k, f in (("192M", "games_192000000"), ("160M", "games_160000000"), ("133M", "snap_exp124_A_games_132710400"), ("112M", "games_112000000")):
    t = json.load(open(f"/home/hongdp/Workspace/LLM_Mahjong/experiments/probes/pairdiff_R1tost_{f}.pt_101152000.json"))
    ts, td = np.array(t["share"]), np.array(t["diff"])
    print(f"  {k} - Mortal: {(ts - s).mean()*100:+.2f} ± {se(ts - s)*100:.2f} pp; pts {(td - d).mean():+.0f} ± {se(td - d):.0f}")
