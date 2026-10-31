import os, sys, torch
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev).eval(); net.load_state_dict(d["state"])
P, S = d["P"].to(dev), d["S"].to(dev)
print("nbad", d["nbad"], "of", d["B"])
sc = S.abs().max(0).values
print("scalar idx with |s|>5:", [(i, round(float(v), 1)) for i, v in enumerate(sc) if v > 5])
with torch.no_grad():
    # residual magnitude per block (eager fp32)
    Bn, C, _ = P.shape
    s = P[:, :, :27].reshape(Bn, C, 3, 9).permute(0, 2, 1, 3).reshape(Bn * 3, C, 9)
    s = net.suit_conv(s); dd = s.shape[1]
    suits = s.reshape(Bn, 3, dd, 9).permute(0, 1, 3, 2).reshape(Bn, 27, dd)
    tok = torch.cat([suits, net.honor_proj(P[:, :, 27:].transpose(1, 2))], 1) + net.tile_embed.weight[None]
    g = net.global_proj(S)[:, None, :]
    print(f"tile token absmax {float(tok.abs().max()):.1f}  global token absmax {float(g.abs().max()):.1f}")
    x = torch.cat([g, tok], 1)
    for i, blk in enumerate(net.blocks):
        x = fc._block_fast(blk, x, blk.bias[:, net.bucket])
        xg = x[:, 0]
        print(f" after block {i}: residual absmax {float(x.abs().max()):.1f}, global-token row std/mean ratio {float((xg.std(-1) / xg.mean(-1).abs().clamp(min=1e-6)).min()):.4f}")
    ref32 = fc.trunk_fast(net, P, S)
    with torch.autocast("cuda", torch.bfloat16):
        eag16 = fc.trunk_fast(net, P, S).float()
        comp = torch.compile(lambda p, s: fc.trunk_fast(net, p, s), dynamic=True)(P, S).float()
    def rep(name, t):
        fin = torch.isfinite(t)
        print(f"{name}: finite {bool(fin.all())}, maxdiff vs fp32 {float((t[fin] - ref32[fin]).abs().max()):.4g}")
    rep("eager bf16 autocast", eag16); rep("compiled bf16 autocast", comp)
