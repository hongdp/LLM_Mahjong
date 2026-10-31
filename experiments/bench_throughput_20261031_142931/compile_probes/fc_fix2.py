import os, sys, time, random, torch
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
torch._dynamo.config.cache_size_limit = 64
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev); net.load_state_dict(d["state"])
S0 = d["S"].to(dev)
def stem(p, s):
    B, C, _ = p.shape
    t = p[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
    t = net.suit_conv(t); dd = t.shape[1]
    suits = t.reshape(B, 3, dd, 9).permute(0, 1, 3, 2).reshape(B, 27, dd)
    tok = torch.cat([suits, net.honor_proj(p[:, :, 27:].transpose(1, 2))], 1) + net.tile_embed.weight[None]
    return torch.cat([net.global_proj(s)[:, None, :], tok], 1)
def blocks(t):
    for b in net.blocks:
        t = fc._block_fast(b, t, b.bias[:, net.bucket])
    return net.norm_f(t)
cb = torch.compile(blocks, dynamic=True)
random.seed(1); sizes = sorted(set([random.randint(1, 8192) for _ in range(40)] + [1, 2, 925, 2048, 4096, 8192]))
worst_c = worst_e = 0.0; nonfinite = 0
net.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    for B in sizes:
        pp = torch.rand(B, 56 if hasattr(net, "in_planes") is False else net.in_planes, 34, device=dev).round()
        ss = S0[torch.randint(0, S0.shape[0], (B,), device=dev)]
        with torch.autocast("cuda", enabled=False):
            r = blocks(stem(pp, ss)).float()
        e = blocks(stem(pp, ss)).float(); o = cb(stem(pp, ss)).float()
        if not torch.isfinite(o).all(): nonfinite += 1; continue
        worst_c = max(worst_c, float((o - r).abs().max())); worst_e = max(worst_e, float((e - r).abs().max()))
print(f"{len(sizes)} batch sizes 1..8192: nonfinite {nonfinite}; worst err compiled-blocks {worst_c:.4f} vs eager-bf16 {worst_e:.4f}")
# gradient check (train mode, autocast like --amp_update)
net.train()
B = 2048; pp = torch.rand(B, net.in_planes, 34, device=dev).round(); ss = S0[torch.randint(0, S0.shape[0], (B,), device=dev)]
w = torch.randn(B, 35, 160, device=dev)
def grads(fn):
    net.zero_grad()
    with torch.autocast("cuda", torch.bfloat16):
        loss = (fn(stem(pp, ss)).float() * w).mean()
    loss.backward()
    return [p.grad.detach().clone() for p in net.parameters() if p.grad is not None]
g32 = None
with torch.autocast("cuda", enabled=False):
    net.zero_grad(); (blocks(stem(pp, ss)) * w).mean().backward()
    g32 = [p.grad.detach().clone() for p in net.parameters() if p.grad is not None]
ge, gc = grads(blocks), grads(cb)
rel = lambda a, b: max(float((x - y).norm() / (y.norm() + 1e-12)) for x, y in zip(a, b))
print(f"grad max relative err vs fp32: eager-bf16 {rel(ge, g32):.4f}, compiled-blocks {rel(gc, g32):.4f}")
