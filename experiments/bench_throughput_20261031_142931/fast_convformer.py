"""Fast ConvFormer trunk: same parameters / state_dict / outputs as arch_zoo.ConvFormer.trunk.

Changes (all math-preserving):
  1. the three suit convs run as ONE conv over a [3B, C, 9] batch instead of three calls;
  2. attention uses F.scaled_dot_product_attention with the relative bias as an additive float mask
     (fused kernel, no [B,H,L,L] softmax materialised by hand);
  3. optional torch.compile(dynamic=True) of the 6-block stack + final norm ONLY. Compiling the whole trunk
     (stem + >=2 blocks in one graph) under bf16 autocast is numerically wrong on torch 2.12/cu130: logit error
     0.02-0.84 depending on batch size (eager bf16: 0.02) and NaN rows inside the trainer (bench 2026-10-31,
     fc_ops2/fc_fix probes). fp32 compile of the same graph is exact, and stem-eager + compiled-blocks matches
     eager bf16 on 46 batch sizes (1..8192) and in gradients.
The compiled callable lives in the instance __dict__, so checkpoints are untouched (same trick as
net.enable_fast_trunk for the CNN)."""
import types
import torch
import torch.nn.functional as F


def _block_fast(blk, x, bias):                       # bias: [H, L, L] (already gathered by bucket)
    B, L, D = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).reshape(B, L, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype)[None])
    x = x + blk.proj(y.transpose(1, 2).reshape(B, L, D))
    return x + blk.ffn(blk.ln2(x))


def _stem_fast(net, planes, scalars):
    B, C, _ = planes.shape
    s = planes[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
    s = net.suit_conv(s)                                             # [3B, d, 9]
    d = s.shape[1]
    suits = s.reshape(B, 3, d, 9).permute(0, 1, 3, 2).reshape(B, 27, d)
    honors = net.honor_proj(planes[:, :, 27:].transpose(1, 2))     # [B, 7, d]
    tok = torch.cat([suits, honors], dim=1) + net.tile_embed.weight[None]
    return torch.cat([net.global_proj(scalars)[:, None, :], tok], dim=1)


def _blocks_fast(net, x):
    for blk in net.blocks:
        x = _block_fast(blk, x, blk.bias[:, net.bucket])
    return net.norm_f(x)


def trunk_fast(net, planes, scalars):
    return _blocks_fast(net, _stem_fast(net, planes, scalars))


def enable_fast_convformer(net, compile: bool = False):
    if compile:
        cblocks = torch.compile(lambda x: _blocks_fast(net, x), dynamic=True)   # stem stays eager (see docstring)
        fn = (lambda p, s: cblocks(_stem_fast(net, p, s)))
    else:
        fn = (lambda p, s: trunk_fast(net, p, s))
    net.__dict__["_fast_trunk_fn"] = fn
    net.trunk = types.MethodType(lambda self, p, s: self.__dict__["_fast_trunk_fn"](p, s), net)
    return net


# ---- debug harness (bench 2026-10-31): FC_DEBUG=1 compares every compiled call against the eager fast trunk ----
import os as _os
if _os.environ.get("FC_DEBUG"):
    _orig_enable = enable_fast_convformer
    _stats = {"calls": 0, "maxdiff": 0.0}

    def enable_fast_convformer(net, compile: bool = False):          # noqa: F811
        _orig_enable(net, compile=compile)
        cfn = net.__dict__["_fast_trunk_fn"]

        def dbg(p, s):
            out = cfn(p, s)
            with torch.no_grad():
                ref = trunk_fast(net, p, s)
                fin_o, fin_r = torch.isfinite(out).all().item(), torch.isfinite(ref).all().item()
                d = float((out.float() - ref.float()).abs().max()) if fin_o and fin_r else float("nan")
            _stats["calls"] += 1
            _stats["maxdiff"] = max(_stats["maxdiff"], d) if d == d else _stats["maxdiff"]
            if not fin_o or _stats["calls"] <= 3 or _stats["calls"] % 50 == 0:
                print(f"[FC_DEBUG] call {_stats['calls']} B={p.shape[0]} dtype_in={p.dtype} out={out.dtype} "
                      f"autocast={torch.is_autocast_enabled()} grad={torch.is_grad_enabled()} "
                      f"finite out/ref={fin_o}/{fin_r} maxdiff={d:.4g} running={_stats['maxdiff']:.4g} "
                      f"P[min,max]=[{float(p.min()):.3g},{float(p.max()):.3g}] S[min,max]=[{float(s.min()):.3g},{float(s.max()):.3g}]", flush=True)
            if not fin_o:
                bad = (~torch.isfinite(out)).any(-1).any(-1)
                torch.save({"P": p[bad][:16].cpu(), "S": s[bad][:16].cpu(), "B": p.shape[0], "nbad": int(bad.sum()),
                            "state": {k: v.cpu() for k, v in net.state_dict().items()}}, "/tmp/claude-1000/fc_nan_dump.pt")
                raise RuntimeError(f"FC_DEBUG: compiled trunk non-finite on {int(bad.sum())}/{p.shape[0]} rows (eager finite={fin_r})")
            return out
        net.__dict__["_fast_trunk_fn"] = dbg
        return net
