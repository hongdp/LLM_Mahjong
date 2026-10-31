import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
torch._dynamo.config.cache_size_limit = 64
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev).eval(); net.load_state_dict(d["state"])
P, S = d["P"].to(dev).repeat(64, 1, 1)[:925], d["S"].to(dev).repeat(64, 1)[:925]

def stem(p, s, explicit=False):
    B, C, _ = p.shape
    t = p[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
    t = net.suit_conv(t); dd = t.shape[1]
    suits = t.reshape(B, 3, dd, 9).permute(0, 1, 3, 2).reshape(B, 27, dd)
    hon = net.honor_proj(p[:, :, 27:].transpose(1, 2))
    if explicit:
        tok = torch.cat([suits.float(), hon.float()], 1) + net.tile_embed.weight[None].float()
        return torch.cat([net.global_proj(s).float()[:, None, :], tok], 1)
    tok = torch.cat([suits, hon], 1) + net.tile_embed.weight[None]
    return torch.cat([net.global_proj(s)[:, None, :], tok], 1)

def blocks(t):
    for b in net.blocks:
        t = fc._block_fast(b, t, b.bias[:, net.bucket])
    return net.norm_f(t)

cb = torch.compile(blocks, dynamic=True)
variants = {
    "A full graph (current)": torch.compile(lambda p, s: blocks(stem(p, s)), dynamic=True),
    "B explicit fp32 casts in stem": torch.compile(lambda p, s: blocks(stem(p, s, True)), dynamic=True),
    "C eager stem + compiled blocks": (lambda p, s: cb(stem(p, s))),
}
with torch.no_grad():
    ref = blocks(stem(P, S)).float()
    with torch.autocast("cuda", torch.bfloat16):
        e = blocks(stem(P, S)).float()
    print(f"eager bf16 err {float((e-ref).abs().max()):.4g}")
    for name, fn in variants.items():
        with torch.autocast("cuda", torch.bfloat16):
            errs = []
            for B in (925, 1, 7, 2048, 4096, 925):
                pp = torch.rand(B, P.shape[1], 34, device=dev).round(); ss = S[torch.randint(0, 925, (B,), device=dev)]
                r = blocks(stem(pp, ss)).float()
                with torch.autocast("cuda", enabled=False):
                    r = blocks(stem(pp, ss)).float()
                o = fn(pp, ss).float()
                errs.append(float((o - r).abs().max()) if torch.isfinite(o).all() else float("nan"))
            o = fn(P, S).float()
        print(f"{name:32s} dump err {float((o-ref).abs().max()):.4g}  random-batch errs {[round(x,3) for x in errs]}", flush=True)
