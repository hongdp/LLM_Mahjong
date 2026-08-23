"""Architecture zoo for the mahjong policy — exp10's search space.

All models share the same contract as MahjongPolicyNet.forward
(planes, scalars, mask -> masked logits [B, ACTION_DIM]) so any of them
can drop into self-play, BC, and the arena unchanged.

Two axes explored:
  * capacity (CNN-S/M/L; ViT tiny/small)
  * structure  (1-D CNN over the tile axis vs tiles-as-tokens transformer;
                binary rivers vs order-aware rivers, encoder v2)

The transformer factorizes the action head naturally: token t scores the
8 action types for tile t, giving exactly the (type, key_tile) = 272-way
action space used everywhere else.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.dnn.encoder import (ACTION_DIM, ACTION_TYPES, N_PLANES,
                                    N_PLANES_V1R, N_PLANES_V3R,
                                    N_PLANES_V2, N_PLANES_V3, N_SCALARS,
                                    N_SCALARS_V3, TILE_TYPES)
from src.agents.dnn.net import MahjongPolicyNet, ResBlock


class CnnPolicy(MahjongPolicyNet):
    """The incumbent, parameterized by input planes for encoder v2."""

    def __init__(self, channels=64, blocks=3, in_planes=N_PLANES,
                 in_scalars=N_SCALARS, encoder_variant="v1"):
        nn.Module.__init__(self)
        self.encoder_variant = encoder_variant
        self.in_planes = in_planes
        # bypasses MahjongPolicyNet.__init__, so the exp11 critic-variant
        # attributes its inherited forward_with_value reads must exist here
        self.critic_feat_dim = 0
        self.hazard = False
        self.hazard_head = None
        self.stem = nn.Conv1d(in_planes, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.scalar_fc = nn.Sequential(nn.Linear(in_scalars, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(channels * TILE_TYPES + 64, 512), nn.ReLU(),
            nn.Linear(512, ACTION_DIM),
        )
        self.value = nn.Sequential(
            nn.Linear(channels * TILE_TYPES + 64, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )


class TilesTransformer(nn.Module):
    """Tiles-as-tokens: 34 tile tokens + 1 global token carrying scalars.

    Per-token output head scores the 8 action types for that tile, which
    matches the (type, key_tile) action indexing exactly.
    """

    def __init__(self, d=64, layers=2, heads=4, in_planes=N_PLANES):
        super().__init__()
        self.in_planes = in_planes
        self.tile_embed = nn.Embedding(TILE_TYPES, d)
        self.feat_proj = nn.Linear(in_planes, d)
        self.global_proj = nn.Linear(N_SCALARS, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            batch_first=True, dropout=0.0, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.type_head = nn.Linear(d, len(ACTION_TYPES))
        self.value_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                        nn.Linear(128, 1))

    def trunk(self, planes, scalars):
        B = planes.shape[0]
        idx = torch.arange(TILE_TYPES, device=planes.device)
        tok = self.feat_proj(planes.transpose(1, 2)) + self.tile_embed(idx)[None]
        g = self.global_proj(scalars)[:, None, :]
        return self.encoder(torch.cat([g, tok], dim=1))   # [B, 1+34, d]

    def forward(self, planes, scalars, mask):
        h = self.trunk(planes, scalars)
        per_tile = self.type_head(h[:, 1:, :])            # [B, 34, 8]
        logits = per_tile.permute(0, 2, 1).reshape(-1, ACTION_DIM)
        return logits.masked_fill(~mask, float("-inf"))

    def forward_with_value(self, planes, scalars, mask, cfeats=None):
        # cfeats accepted-and-ignored for trainer call compatibility: the PPO
        # trainer passes the kwarg unconditionally (None when --critic_feats
        # none); zoo nets don't implement critic variants and the trainer
        # refuses --arch + --critic_feats up front. Missing this parameter
        # killed the vit-r3 cloud run at startup (TypeError).
        h = self.trunk(planes, scalars)
        per_tile = self.type_head(h[:, 1:, :])
        logits = per_tile.permute(0, 2, 1).reshape(-1, ACTION_DIM)
        return (logits.masked_fill(~mask, float("-inf")),
                self.value_head(h[:, 0, :]).squeeze(-1))

    @torch.no_grad()
    def act(self, planes, scalars, mask, temperature: float = 1.0):
        if not bool(mask.any(dim=1).all()):
            raise ValueError("act() got a row with no legal actions")
        logits = self.forward(planes, scalars, mask)
        if temperature <= 0:
            idx = logits.argmax(dim=1)
            lp = torch.log_softmax(logits, dim=1).gather(1, idx[:, None]).squeeze(1)
        else:                                   # behaviour logprob (see net.act)
            logb = torch.log_softmax(logits / temperature, dim=1)
            idx = torch.multinomial(logb.exp(), 1).squeeze(1)
            lp = logb.gather(1, idx[:, None]).squeeze(1)
        return idx, lp


ZOO = {
    # name: (factory, needs_order_planes)
    "cnn_s":        (lambda: CnnPolicy(32, 2), False),
    "cnn_m":        (lambda: CnnPolicy(64, 3), False),            # incumbent
    "cnn_l":        (lambda: CnnPolicy(128, 4), False),
    "cnn_m_order":  (lambda: CnnPolicy(64, 3, in_planes=N_PLANES_V2), True),
    "vit_tiny":     (lambda: TilesTransformer(64, 2, 4), False),
    "vit_small":    (lambda: TilesTransformer(128, 4, 8), False),
    "vit_small_order": (lambda: TilesTransformer(128, 4, 8,
                                                 in_planes=N_PLANES_V2), True),
    "convformer_m": (lambda: ConvFormer(160, 6, 5), False),   # exp19
    # encoder v3 (complete public record) variants — exp23
    "cnn_m_v3": (lambda: CnnPolicy(64, 3, in_planes=N_PLANES_V3,
                                   in_scalars=N_SCALARS_V3, encoder_variant="v3"), False),
    "convformer_m_v3": (lambda: ConvFormer(160, 6, 5, in_planes=N_PLANES_V3,
                                           in_scalars=N_SCALARS_V3, encoder_variant="v3"), False),
    # red-dora variants (Majsoul rules, 2026-08-23): +5 red planes
    "cnn_m_r": (lambda: CnnPolicy(64, 3, in_planes=N_PLANES_V1R, encoder_variant="v1r"), False),
    "convformer_m_r": (lambda: ConvFormer(160, 6, 5, in_planes=N_PLANES_V1R,
                                          encoder_variant="v1r"), False),
    "cnn_m_v3r": (lambda: CnnPolicy(64, 3, in_planes=N_PLANES_V3R,
                                    in_scalars=N_SCALARS_V3, encoder_variant="v3r"), False),
    "convformer_m_v3r": (lambda: ConvFormer(160, 6, 5, in_planes=N_PLANES_V3R,
                                            in_scalars=N_SCALARS_V3, encoder_variant="v3r"), False),
}


class RelBiasBlock(nn.Module):
    """Pre-LN attention block with a learned per-head bias indexed by a
    static (rank-distance, suit-relation) bucket matrix — Swin/T5-style
    relative bias specialized to the 34-tile layout, so adjacency (the
    run/taatsu prior a k=3 conv gets for free) is a lookup, not a thing
    attention must spend capacity rediscovering."""

    def __init__(self, d, heads, n_buckets):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(heads, n_buckets))
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, bucket, key_mask=None):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(
            B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / self.dh ** 0.5
        att = att + self.bias[:, bucket][None]          # [B, H, L, L]
        if key_mask is not None:                        # set models: absent slots
            att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1)
        x = x + self.proj(y)
        return x + self.ffn(self.ln2(x))


def _tile_buckets():
    """[35, 35] bucket ids. 0 global-pair, 1 cross-suit, 2 honor-pair,
    3..19 same-suit rank delta -8..+8."""
    n = TILE_TYPES + 1
    b = torch.ones(n, n, dtype=torch.long)
    b[0, :] = 0
    b[:, 0] = 0
    for i in range(TILE_TYPES):
        for j in range(TILE_TYPES):
            if i >= 27 and j >= 27:
                b[i + 1, j + 1] = 2
            elif i < 27 and j < 27 and i // 9 == j // 9:
                b[i + 1, j + 1] = 3 + 8 + max(-8, min(8, (j % 9) - (i % 9)))
    return b


class ConvFormer(TilesTransformer):
    """exp19 design: attention that concedes nothing to the CNN.

    Fixes for vit_small's three RL losses (exp18 post-mortem):
      1. locality prior — suit-local shared conv stem (k=3 over rank,
         never crossing suit boundaries, honors via 1x1) builds tokens
         that already see runs/taatsu; plus rank-relative attention bias.
      2. capacity — d=160 x 6 layers ~= 1.97M, matched to cnn_m's 1.94M.
      3. RL optimization — pre-LN throughout, zero-init policy head
         (uniform logits at step 0), pairs with trainer --warmup_updates.
    Keeps vit's wins: per-tile 8-type action head, global token, and
    attention available for the relational skills CNNs can't route.
    """

    def __init__(self, d=160, layers=6, heads=5, in_planes=N_PLANES,
                 in_scalars=N_SCALARS, encoder_variant="v1"):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.encoder_variant = encoder_variant
        self.tile_embed = nn.Embedding(TILE_TYPES, d)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(in_planes, d, 3, padding=1), nn.GELU(),
            nn.Conv1d(d, d, 3, padding=1))
        self.honor_proj = nn.Linear(in_planes, d)
        self.global_proj = nn.Linear(in_scalars, d)
        self.blocks = nn.ModuleList(
            [RelBiasBlock(d, heads, 20) for _ in range(layers)])
        self.norm_f = nn.LayerNorm(d)
        self.type_head = nn.Linear(d, len(ACTION_TYPES))
        nn.init.zeros_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)
        self.value_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                        nn.Linear(128, 1))
        self.register_buffer("bucket", _tile_buckets(), persistent=False)

    def trunk(self, planes, scalars):
        suits = [self.suit_conv(planes[:, :, s:s + 9]).transpose(1, 2)
                 for s in (0, 9, 18)]
        honors = self.honor_proj(planes[:, :, 27:].transpose(1, 2))
        idx = torch.arange(TILE_TYPES, device=planes.device)
        tok = torch.cat(suits + [honors], dim=1) + self.tile_embed(idx)[None]
        x = torch.cat([self.global_proj(scalars)[:, None, :], tok], dim=1)
        for blk in self.blocks:
            x = blk(x, self.bucket)
        return self.norm_f(x)


# ----------------------------------------------------------------------
# exp27: hand-as-a-SET encoder (tile instances, no positional encoding)
# ----------------------------------------------------------------------
class SetAttnBlock(nn.Module):
    """Pre-LN attention block over a compact token set with a per-sample
    bias [B, L, L] (bucket ids -> learned per-head bias) and a key padding
    mask, via fused scaled_dot_product_attention."""

    def __init__(self, d, heads, n_buckets):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(heads, n_buckets))
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bucket, keep):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        # [B, H, L, L] additive mask: rank bias + -inf on padded keys.
        # F.embedding, not bias[:, bucket]: advanced-indexing backward
        # scatter-adds B*L*L values into 20 slots with atomics — on the
        # 20M arm one minibatch took ~30 s (2026-08-23 g4-2 stall).
        bias = F.embedding(bucket, self.bias.t()).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~keep[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, -1))
        return x + self.ffn(self.ln2(x))


class HandSetEncoder(nn.Module):
    """Permutation-invariant encoder over the player's own tile INSTANCES.

    Tokens are built from the hand count planes: slot (copy k, tile t) is a
    token iff the hand holds >= k+1 copies of t (so two copies of 3s are
    two tokens, unlike the 34-type axis where they are one channel). Own
    meld tiles are tokens too (flagged). No positional encoding: the only
    structure is content — tile identity (suit + rank embedding + rank as
    a number), copy id, red flag, meld flag — plus an optional RANK-RELATIVE
    attention bias (same-suit rank distance / honor / cross-suit buckets),
    which is content-based and keeps permutation equivariance. With
    rel_bias=False this is a pure set transformer (the ablation).

    Perf (2026-08-23 rollout review): the 136 candidate slots are compacted
    to the <= MAX_TOKENS present ones before attention (a hand has <= 14
    tiles + <= 16 meld tiles), and attention runs through fused SDPA —
    the 137-slot fp32 version was GPU-bound at 15 games/s.
    Output: [CLS] embedding (d) after `layers` attention blocks.
    """
    MAX_TOKENS = 32

    def __init__(self, d=128, layers=4, heads=4, rel_bias=True, max_copies=4):
        super().__init__()
        self.d, self.rel_bias = d, rel_bias
        self.tile_embed = nn.Embedding(TILE_TYPES, d)
        self.copy_embed = nn.Embedding(max_copies, d)
        self.flag_proj = nn.Linear(3, d)            # [rank/9 number, red, meld]
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        n_buckets = 3 + 17
        self.blocks = nn.ModuleList([SetAttnBlock(d, heads, n_buckets) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        tiles = torch.arange(TILE_TYPES).repeat(max_copies)
        copies = torch.arange(max_copies).repeat_interleave(TILE_TYPES)
        rank = torch.where(tiles < 27, (tiles % 9).float() / 8.0, torch.full_like(tiles, 0.5, dtype=torch.float))
        self.register_buffer("slot_tile", tiles, persistent=False)
        self.register_buffer("slot_copy", copies, persistent=False)
        self.register_buffer("slot_rank", rank, persistent=False)
        self.register_buffer("tile_bucket", _tile_buckets(), persistent=False)   # [35, 35], 0 = CLS

    def forward(self, planes):
        """planes [B, P, 34]: planes 0-3 = hand count>=k, plane 4 = own meld
        presence (v1 / v1r / v3 / v3r); own-red plane when present (v1r:
        plane 15, v3r: plane 50)."""
        B = planes.shape[0]
        dev = planes.device
        hand = planes[:, :4, :].reshape(B, -1) > 0.5                  # [B, 136]
        meld = planes[:, 4, :] > 0.5
        P = planes.shape[1]
        red_plane = planes[:, N_PLANES, :] if P == N_PLANES + 6 else (
            planes[:, N_PLANES_V3, :] if P == N_PLANES_V3 + 6 else torch.zeros_like(planes[:, 0, :]))
        meld_slot = torch.cat([meld, torch.zeros(B, 3 * TILE_TYPES, dtype=torch.bool, device=dev)], 1)
        present = hand | meld_slot
        is_meld = meld_slot & ~hand
        red = torch.zeros(B, 4 * TILE_TYPES, dtype=planes.dtype, device=dev)
        red[:, :TILE_TYPES] = red_plane * hand[:, :TILE_TYPES].to(planes.dtype)
        # compact: present slots first (stable), keep the first MAX_TOKENS
        order = torch.argsort((~present).to(torch.int8), dim=1, stable=True)[:, :self.MAX_TOKENS]
        keep_t = torch.gather(present, 1, order)                      # [B, T]
        tile = self.slot_tile[order]                                  # [B, T]
        x = (self.tile_embed(tile) + self.copy_embed(self.slot_copy[order])
             + self.flag_proj(torch.stack([self.slot_rank[order],
                                           torch.gather(red, 1, order),
                                           torch.gather(is_meld, 1, order).to(planes.dtype)], -1)))
        x = torch.cat([self.cls.expand(B, 1, -1), x], 1)              # [B, 1+T, d]
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), keep_t], 1)
        ids = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), tile + 1], 1)
        if self.rel_bias:
            bucket = self.tile_bucket[ids[:, :, None], ids[:, None, :]]   # [B, 1+T, 1+T]
        else:
            bucket = torch.zeros(B, ids.shape[1], ids.shape[1], dtype=torch.long, device=dev)
        for blk in self.blocks:
            x = blk(x, bucket, keep)
        return self.ln(x[:, 0])


class HandSetCnn(CnnPolicy):
    """exp27: CNN board trunk (unchanged) + HandSetEncoder branch over the
    player's tile instances, fused before the policy/value heads. The only
    variable vs cnn_m_r is the set-based hand branch."""

    def __init__(self, channels=64, blocks=3, d=128, layers=4, heads=4, rel_bias=True,
                 in_planes=N_PLANES_V1R, in_scalars=N_SCALARS, encoder_variant="v1r"):
        super().__init__(channels, blocks, in_planes=in_planes, in_scalars=in_scalars,
                         encoder_variant=encoder_variant)
        self.hand_set = HandSetEncoder(d, layers, heads, rel_bias)
        feat = channels * TILE_TYPES + 64 + d
        self.head = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, ACTION_DIM))
        self.value = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))

    def trunk(self, planes, scalars):
        h = torch.relu(self.stem(planes))
        h = self.blocks(h).flatten(1)
        return torch.cat([h, self.scalar_fc(scalars), self.hand_set(planes)], dim=1)


ZOO.update({
    "handset_cnn_m_r": (lambda: HandSetCnn(64, 3, 128, 4, 4, True), False),         # 2.9M
    "handset_pure_cnn_m_r": (lambda: HandSetCnn(64, 3, 128, 4, 4, False), False),   # no rank bias (ablation)
    # scaled attention branch (user 2026-08-23: attention has to scale): d256 x 8 layers x 8 heads
    "handset_l_cnn_m_r": (lambda: HandSetCnn(64, 3, 256, 8, 8, True), False),
    "handset_l_pure_cnn_m_r": (lambda: HandSetCnn(64, 3, 256, 8, 8, False), False),
    "handset_xl_cnn_m_r": (lambda: HandSetCnn(64, 3, 384, 10, 12, True), False),
    # parameter-matched CNN controls for the scaled arms (same red/yakuhai planes)
    "cnn_l_r": (lambda: CnnPolicy(128, 4, in_planes=N_PLANES_V1R, encoder_variant="v1r"), False),
    "cnn_xl_r": (lambda: CnnPolicy(192, 6, in_planes=N_PLANES_V1R, encoder_variant="v1r"), False),
})


# ----------------------------------------------------------------------
# exp30: HandRiverFormer — hand instance tokens CROSS-ATTENDING on the
# river EVENT sequence (user design 2026-08-23). Spec highlights:
#   * hand tokens (<=18 real + CLS, 20 slots): concealed tiles + own meld
#     tiles; learnable embeddings summed, rank via shared sinusoid;
#   * event tokens (<=128 + 1 global): per-discard events with absolute
#     junme PE and RECENCY PE through SEPARATE learned projections (the
#     two sinusoids must not share frequency dims — user 2026-08-23),
#     seat/kind embeddings, flag attributes, is-last-discard;
#   * per block: hand self-attn (rank rel-bias) -> cross-attn(Q=hand,
#     KV=events) -> FFN; context pre-encoded by 2 plain self-attn layers;
#   * ablations: no_cross (pooled-context concat instead), no_temporal_pe,
#     free_rank_emb (learned table instead of sinusoid).
# ----------------------------------------------------------------------
from src.agents.dnn.encoder import EV_MAX, EV_F, EV_PLANES, N_PLANES_V4, N_PLANES_V1R as _NP1R


def _sinusoid(n_pos, d):
    import math
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(n_pos, dtype=torch.float)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(200.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CrossBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.lnq = nn.LayerNorm(d)
        self.lnk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx, ctx_keep):
        B, L, _ = x.shape
        S = ctx.shape[1]
        q = self.q(self.lnq(x)).reshape(B, L, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.lnk(ctx)).reshape(B, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = torch.zeros(B, 1, 1, S, device=x.device, dtype=q.dtype)
        bias = bias.masked_fill(~ctx_keep[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return x + self.proj(y.transpose(1, 2).reshape(B, L, -1))


class HRBlock(nn.Module):
    """hand self-attn (rel-bias) -> cross-attn on events -> FFN."""

    def __init__(self, d, heads, n_buckets, cross=True):
        super().__init__()
        self.self_attn = SetAttnBlock(d, heads, n_buckets)   # includes its own FFN
        self.cross = CrossBlock(d, heads) if cross else None
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bucket, keep, ctx, ctx_keep):
        x = self.self_attn(x, bucket, keep)
        if self.cross is not None:
            x = self.cross(x, ctx, ctx_keep)
        return x + self.ffn(self.ln(x))


class HandRiverFormer(nn.Module):
    HAND_SLOTS = 20            # <=18 real tokens (2 concealed + 4 kans worst case) + CLS

    def __init__(self, d=320, layers=8, heads=8, ctx_layers=2,
                 no_cross=False, no_temporal_pe=False, free_rank_emb=False,
                 encoder_variant="v4"):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.in_planes = N_PLANES_V4
        self.d = d
        self.no_cross, self.no_temporal_pe = no_cross, no_temporal_pe
        self.critic_feat_dim, self.hazard, self.hazard_head = 0, False, None
        # --- shared tile factorization (hand + events)
        if free_rank_emb:
            self.rank_pe = nn.Parameter(torch.randn(10, 64) * 0.02)     # 0=honor,1..9
        else:
            self.register_buffer("rank_pe_const", _sinusoid(10, 64), persistent=False)
            self.rank_pe = None
        self.rank_proj = nn.Linear(64, d)
        self.suit_emb = nn.Embedding(4, d)
        self.honor_emb = nn.Embedding(8, d)          # 0 = suited, 1..7 honors
        # --- hand-side
        self.copy_emb = nn.Embedding(4, d)
        self.hand_attr = nn.Linear(2, d)             # [red, in-meld]
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        # --- event-side
        self.kind_emb = nn.Embedding(3, d)           # discard / opp meld / dora
        self.seat_emb = nn.Embedding(4, d)
        self.junme_proj = nn.Linear(64, d)           # separate projections: the two
        self.recency_proj = nn.Linear(64, d)         # sinusoids must not share dims
        self.register_buffer("time_pe", _sinusoid(EV_MAX + 1, 64), persistent=False)
        self.event_attr = nn.Linear(6, d)            # tsumogiri/riichi/called/red/last/from-me
        self.global_proj = nn.Linear(N_SCALARS_V3, d)
        n_buckets = 3 + 17
        self.ctx_blocks = nn.ModuleList([SetAttnBlock(d, heads, n_buckets) for _ in range(ctx_layers)])
        self.blocks = nn.ModuleList([HRBlock(d, heads, n_buckets, cross=not no_cross)
                                     for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        feat = 3 * d
        self.head = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, ACTION_DIM))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.value = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 1))
        tiles = torch.arange(TILE_TYPES).repeat(4)
        self.register_buffer("slot_tile", tiles, persistent=False)
        self.register_buffer("slot_copy", torch.arange(4).repeat_interleave(TILE_TYPES), persistent=False)
        tb = _tile_buckets()
        self.register_buffer("tile_bucket", tb, persistent=False)

    # ---- token builders ------------------------------------------------
    def _rank_vec(self, rank_idx):
        table = self.rank_pe if self.rank_pe is not None else self.rank_pe_const
        return self.rank_proj(table[rank_idx])

    def _tile_vec(self, i34):
        suit = torch.where(i34 < 27, i34 // 9, torch.full_like(i34, 3))
        rank = torch.where(i34 < 27, i34 % 9 + 1, torch.zeros_like(i34))
        hon = torch.where(i34 >= 27, i34 - 27 + 1, torch.zeros_like(i34))
        return self._rank_vec(rank) + self.suit_emb(suit) + self.honor_emb(hon)

    def _hand_tokens(self, planes):
        B = planes.shape[0]
        dev = planes.device
        hand = planes[:, :4, :].reshape(B, -1) > 0.5
        meld = planes[:, 4, :] > 0.5
        red_plane = planes[:, N_PLANES, :]
        meld_slot = torch.cat([meld, torch.zeros(B, 3 * TILE_TYPES, dtype=torch.bool, device=dev)], 1)
        present = hand | meld_slot
        is_meld = (meld_slot & ~hand).to(planes.dtype)
        red = torch.zeros_like(present, dtype=planes.dtype)
        red[:, :TILE_TYPES] = red_plane * hand[:, :TILE_TYPES].to(planes.dtype)
        order = torch.argsort((~present).to(torch.int8), dim=1, stable=True)[:, :self.HAND_SLOTS - 1]
        keep = torch.gather(present, 1, order)
        tile = self.slot_tile[order]
        x = (self._tile_vec(tile) + self.copy_emb(self.slot_copy[order])
             + self.hand_attr(torch.stack([torch.gather(red, 1, order),
                                           torch.gather(is_meld, 1, order)], -1)))
        x = torch.cat([self.cls.expand(B, 1, -1), x], 1)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), keep], 1)
        ids = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), tile + 1], 1)
        bucket = self.tile_bucket[ids[:, :, None], ids[:, None, :]]
        return x, keep, bucket

    def _event_tokens(self, planes, scalars):
        B = planes.shape[0]
        dev = planes.device
        buf = planes[:, _NP1R:, :].reshape(B, -1)
        ev = buf[:, :EV_MAX * EV_F].reshape(B, EV_MAX, EV_F)
        n = buf[:, -1].long().clamp(0, EV_MAX)
        idx = torch.arange(EV_MAX, device=dev)[None]
        keep = idx < n[:, None]
        rank = ev[..., 0].long().clamp(0, 9)
        suit = ev[..., 1].long().clamp(0, 3)
        hon = ev[..., 2].long().clamp(0, 7)
        kind = ev[..., 3].long().clamp(0, 2)
        seat = ev[..., 4].long().clamp(0, 3)
        junme = ev[..., 5].long().clamp(0, EV_MAX)
        rec = ev[..., 6].long().clamp(0, EV_MAX)
        flags = ev[..., 7].long()
        attrs = torch.stack([(flags >> b) & 1 for b in range(6)], -1).to(planes.dtype)
        x = (self._rank_vec(rank) + self.suit_emb(suit) + self.honor_emb(hon)
             + self.kind_emb(kind) + self.seat_emb(seat) + self.event_attr(attrs))
        if not self.no_temporal_pe:
            x = x + self.junme_proj(self.time_pe[junme]) + self.recency_proj(self.time_pe[rec])
        g = self.global_proj(scalars)[:, None, :]
        x = torch.cat([g, x], 1)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), keep], 1)
        zero_bucket = torch.zeros(B, x.shape[1], x.shape[1], dtype=torch.long, device=dev)
        return x, keep, zero_bucket

    # ---- api -----------------------------------------------------------
    def trunk(self, planes, scalars):
        hx, hkeep, hbucket = self._hand_tokens(planes)
        cx, ckeep, cbucket = self._event_tokens(planes, scalars)
        for blk in self.ctx_blocks:
            cx = blk(cx, cbucket, ckeep)
        if self.no_cross:
            pooled_ctx = (cx * ckeep[..., None]).sum(1) / ckeep.sum(1, keepdim=True).clamp(min=1)
            hx = hx + pooled_ctx[:, None, :]
        for blk in self.blocks:
            hx = blk(hx, hbucket, hkeep, cx, ckeep)
        hx = self.ln_out(hx)
        hand_pool = (hx * hkeep[..., None]).sum(1) / hkeep.sum(1, keepdim=True).clamp(min=1)
        ctx_pool = (cx * ckeep[..., None]).sum(1) / ckeep.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([hx[:, 0], hand_pool, ctx_pool], dim=1)

    def forward(self, planes, scalars, mask):
        logits = self.head(self.trunk(planes, scalars))
        return logits.masked_fill(~mask, float("-inf"))

    def forward_with_value(self, planes, scalars, mask, cfeats=None):
        h = self.trunk(planes, scalars)
        return (self.head(h).masked_fill(~mask, float("-inf")),
                self.value(h).squeeze(-1))

    @torch.no_grad()
    def act(self, planes, scalars, mask, temperature: float = 1.0):
        if not bool(mask.any(dim=1).all()):
            raise ValueError("act() got a row with no legal actions")
        logits = self.forward(planes, scalars, mask)
        if temperature <= 0:
            idx = logits.argmax(dim=1)
            lp = torch.log_softmax(logits, dim=1).gather(1, idx[:, None]).squeeze(1)
        else:
            logb = torch.log_softmax(logits / temperature, dim=1)
            idx = torch.multinomial(logb.exp(), 1).squeeze(1)
            lp = logb.gather(1, idx[:, None]).squeeze(1)
        return idx, lp


ZOO.update({
    "hrf_xl_v4": (lambda: HandRiverFormer(320, 8, 8, 2), False),
    "hrf_xl_nocross_v4": (lambda: HandRiverFormer(320, 8, 8, 2, no_cross=True), False),
    "hrf_xl_notime_v4": (lambda: HandRiverFormer(320, 8, 8, 2, no_temporal_pe=True), False),
    "hrf_xl_freerank_v4": (lambda: HandRiverFormer(320, 8, 8, 2, free_rank_emb=True), False),
})
