import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
torch._dynamo.config.cache_size_limit = 64
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev).eval(); net.load_state_dict(d["state"])
P, S = d["P"].to(dev).repeat(64, 1, 1)[:925], d["S"].to(dev).repeat(64, 1)[:925]

def stem(p, s):
    B, C, _ = p.shape
    t = p[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
    t = net.suit_conv(t); dd = t.shape[1]
    suits = t.reshape(B, 3, dd, 9).permute(0, 1, 3, 2).reshape(B, 27, dd)
    tok = torch.cat([suits, net.honor_proj(p[:, :, 27:].transpose(1, 2))], 1) + net.tile_embed.weight[None]
    return torch.cat([net.global_proj(s)[:, None, :], tok], 1)

def cmp(name, fn, *args):
    with torch.no_grad():
        ref = fn(*args).float()
        with torch.autocast("cuda", torch.bfloat16):
            e = fn(*args); torch._dynamo.reset(); c = torch.compile(fn, dynamic=True)(*args)
        print(f"{name:28s} dtype eager={e.dtype} comp={c.dtype} | eager err {float((e.float()-ref).abs().max()):.4g} comp err {float((c.float()-ref).abs().max()):.4g}", flush=True)
cmp("stem", stem, P, S)
cmp("suit_conv only", lambda p: net.suit_conv(p[:, :, :9]), P)
with torch.no_grad():
    x = stem(P, S)
blk = net.blocks[0]
cmp("block0 SDPA", lambda t: fc._block_fast(blk, t, blk.bias[:, net.bucket]), x)
def attn_only(t):
    B, L, D = t.shape
    q, k, v = blk.qkv(blk.ln1(t)).reshape(B, L, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=blk.bias[:, net.bucket].to(q.dtype)[None])
cmp("block0 attention core", attn_only, x)
def blocks_only(t):
    for b in net.blocks:
        t = fc._block_fast(b, t, b.bias[:, net.bucket])
    return net.norm_f(t)
cmp("blocks only (fp32 x in)", blocks_only, x)
cmp("stem+block0", lambda p, s: fc._block_fast(blk, stem(p, s), blk.bias[:, net.bucket]), P, S)
def stem_dtype_probe(p, s):
    g = net.global_proj(s)[:, None, :]
    tok = torch.randn(p.shape[0], 34, 160, device=p.device)
    x = torch.cat([g, tok], 1)
    return x
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    torch._dynamo.reset()
    print("cat(bf16 g, fp32 tok) dtype eager:", stem_dtype_probe(P, S).dtype, " compiled:", torch.compile(stem_dtype_probe, dynamic=True)(P, S).dtype)
    def resid(t):
        b = net.blocks[0]
        return t + b.proj(t)   # bf16 linear output added to fp32 residual
    torch._dynamo.reset()
    r_e = resid(x); r_c = torch.compile(resid, dynamic=True)(x)
    print("fp32 + bf16-linear residual dtype eager:", r_e.dtype, "compiled:", r_c.dtype, "err", float((r_e.float()-r_c.float()).abs().max()))
print("---- stem + first k blocks + norm_f, error measured after norm_f ----")
for k in range(0, 7):
    def f(p, s, k=k):
        t = stem(p, s)
        for b in net.blocks[:k]:
            t = fc._block_fast(b, t, b.bias[:, net.bucket])
        return net.norm_f(t)
    cmp(f"k={k}", f, P, S)
print("---- same but norm_f input forced fp32 ----")
for k in (0, 6):
    def f2(p, s, k=k):
        t = stem(p, s)
        for b in net.blocks[:k]:
            t = fc._block_fast(b, t, b.bias[:, net.bucket])
        return F.layer_norm(t.float(), (t.shape[-1],), net.norm_f.weight.float(), net.norm_f.bias.float(), net.norm_f.eps)
    cmp(f"k={k} explicit fp32 LN", f2, P, S)
