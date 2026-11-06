import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
torch._dynamo.config.cache_size_limit = 64
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev).eval(); net.load_state_dict(d["state"])
P, S = d["P"].to(dev), d["S"].to(dev)
P = P.repeat(64, 1, 1)[:925]; S = S.repeat(64, 1)[:925]        # same batch size as the failing call
P = P + 0.0; S = S + 0.0

def blk_manual(blk, x, bias):
    B, L, D = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).reshape(B, L, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
    att = (q @ k.transpose(-2, -1)) / blk.dh ** 0.5 + bias[None]
    y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
    x = x + blk.proj(y)
    return x + blk.ffn(blk.ln2(x))

def trunk_with(block_fn):
    def f(p, s):
        B, C, _ = p.shape
        t = p[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
        t = net.suit_conv(t); dd = t.shape[1]
        suits = t.reshape(B, 3, dd, 9).permute(0, 1, 3, 2).reshape(B, 27, dd)
        tok = torch.cat([suits, net.honor_proj(p[:, :, 27:].transpose(1, 2))], 1) + net.tile_embed.weight[None]
        x = torch.cat([net.global_proj(s)[:, None, :], tok], 1)
        for blk in net.blocks:
            x = block_fn(blk, x, blk.bias[:, net.bucket])
        return net.norm_f(x)
    return f

with torch.no_grad():
    ref = fc.trunk_fast(net, P, S)
    def run(name, fn, amp):
        torch._dynamo.reset()
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            o = fn(P, S).float()
        fin = torch.isfinite(o)
        print(f"{name:42s} finite={bool(fin.all())} maxdiff={float((o[fin]-ref[fin]).abs().max()):.4g}", flush=True)
    run("eager SDPA bf16", trunk_with(fc._block_fast), True)
    run("compiled SDPA fp32 (no autocast)", torch.compile(trunk_with(fc._block_fast), dynamic=True), False)
    run("compiled SDPA bf16", torch.compile(trunk_with(fc._block_fast), dynamic=True), True)
    run("compiled SDPA bf16 static", torch.compile(trunk_with(fc._block_fast), dynamic=False), True)
    run("compiled manual-attn bf16", torch.compile(trunk_with(blk_manual), dynamic=True), True)
    run("eager manual-attn bf16", trunk_with(blk_manual), True)
