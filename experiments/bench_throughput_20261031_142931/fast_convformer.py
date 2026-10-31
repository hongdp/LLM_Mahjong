"""Fast ConvFormer trunk: same parameters / state_dict / outputs as arch_zoo.ConvFormer.trunk.

Changes (all math-preserving):
  1. the three suit convs run as ONE conv over a [3B, C, 9] batch instead of three calls;
  2. attention uses F.scaled_dot_product_attention with the relative bias as an additive float mask
     (fused kernel, no [B,H,L,L] softmax materialised by hand);
  3. optional torch.compile(dynamic=True) of the whole trunk (fuses LN/GELU/residual adds).
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


def trunk_fast(net, planes, scalars):
    B, C, _ = planes.shape
    s = planes[:, :, :27].reshape(B, C, 3, 9).permute(0, 2, 1, 3).reshape(B * 3, C, 9)
    s = net.suit_conv(s)                                             # [3B, d, 9]
    d = s.shape[1]
    suits = s.reshape(B, 3, d, 9).permute(0, 1, 3, 2).reshape(B, 27, d)
    honors = net.honor_proj(planes[:, :, 27:].transpose(1, 2))     # [B, 7, d]
    tok = torch.cat([suits, honors], dim=1) + net.tile_embed.weight[None]
    x = torch.cat([net.global_proj(scalars)[:, None, :], tok], dim=1)
    for blk in net.blocks:
        x = _block_fast(blk, x, blk.bias[:, net.bucket])
    return net.norm_f(x)


def enable_fast_convformer(net, compile: bool = False):
    fn = (lambda p, s: trunk_fast(net, p, s))
    if compile:
        fn = torch.compile(fn, dynamic=True)
    net.__dict__["_fast_trunk_fn"] = fn
    net.trunk = types.MethodType(lambda self, p, s: self.__dict__["_fast_trunk_fn"](p, s), net)
    return net
