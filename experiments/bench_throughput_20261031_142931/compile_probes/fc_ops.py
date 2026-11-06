import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.environ["LLM_MAHJONG_ROOT"])
from src.agents.dnn.arch_zoo import ZOO
from src.agents.dnn import fast_convformer as fc
torch._dynamo.config.cache_size_limit = 64
d = torch.load("/tmp/claude-1000/fc_nan_dump.pt", weights_only=False)
dev = torch.device("cuda")
net = ZOO["convformer_m_v3r"][0]().to(dev).eval(); net.load_state_dict(d["state"])
P, S = d["P"].to(dev).repeat(64, 1, 1)[:925], d["S"].to(dev).repeat(64, 1)[:925]
blk = net.blocks[0]
with torch.no_grad():
    g = net.global_proj(S)                         # fp32 [B, d]
    x = torch.cat([g[:, None], torch.randn(925, 34, 160, device=dev)], 1)
    tests = {
        "global_proj(S) linear": (lambda s: net.global_proj(s), S),
        "ln1(x)": (lambda t: blk.ln1(t), x),
        "qkv(ln1(x))": (lambda t: blk.qkv(blk.ln1(t)), x),
        "ffn(ln2(x))": (lambda t: blk.ffn(blk.ln2(t)), x),
        "x + ffn(ln2(x))": (lambda t: t + blk.ffn(blk.ln2(t)), x),
    }
    for name, (fn, inp) in tests.items():
        ref = fn(inp).float()
        with torch.autocast("cuda", torch.bfloat16):
            e = fn(inp).float()
            torch._dynamo.reset(); c = torch.compile(fn, dynamic=True)(inp).float()
        scale = float(ref.abs().max())
        print(f"{name:22s} |ref|max {scale:8.2f}  eager-bf16 err {float((e-ref).abs().max()):.4g}  compiled-bf16 err {float((c-ref).abs().max()):.4g}  out dtypes eager/comp {e.dtype}", flush=True)
    print("S col 9 values:", sorted(set(S[:, 9].tolist()))[:10])
