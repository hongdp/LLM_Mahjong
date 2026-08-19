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

from src.agents.dnn.encoder import (ACTION_DIM, ACTION_TYPES, N_PLANES,
                                    N_PLANES_V2, N_SCALARS, TILE_TYPES)
from src.agents.dnn.net import MahjongPolicyNet, ResBlock


class CnnPolicy(MahjongPolicyNet):
    """The incumbent, parameterized by input planes for encoder v2."""

    def __init__(self, channels=64, blocks=3, in_planes=N_PLANES):
        nn.Module.__init__(self)
        # bypasses MahjongPolicyNet.__init__, so the exp11 critic-variant
        # attributes its inherited forward_with_value reads must exist here
        self.critic_feat_dim = 0
        self.hazard = False
        self.hazard_head = None
        self.stem = nn.Conv1d(in_planes, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.scalar_fc = nn.Sequential(nn.Linear(N_SCALARS, 64), nn.ReLU())
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
        else:
            probs = torch.softmax(logits / temperature, dim=1)
            idx = torch.multinomial(probs, 1).squeeze(1)
        lp = torch.log_softmax(logits, dim=1).gather(1, idx[:, None]).squeeze(1)
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

    def forward(self, x, bucket):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(
            B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / self.dh ** 0.5
        att = att + self.bias[:, bucket][None]          # [B, H, L, L]
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

    def __init__(self, d=160, layers=6, heads=5, in_planes=N_PLANES):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.tile_embed = nn.Embedding(TILE_TYPES, d)
        self.suit_conv = nn.Sequential(
            nn.Conv1d(in_planes, d, 3, padding=1), nn.GELU(),
            nn.Conv1d(d, d, 3, padding=1))
        self.honor_proj = nn.Linear(in_planes, d)
        self.global_proj = nn.Linear(N_SCALARS, d)
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
