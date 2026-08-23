"""Diagnostic rule overrides wrapped around a trained policy (arena only).

Not a training prior: these wrappers exist to *measure* whether the policy
has learned a specific piece of value. If `KanOverride(net)` beats `net` in
a duplicate-deal arena, the net has not learned the value of kan.

Interface: `_fill_with_dnn` (arena) calls `net.override(table, pid, legal,
chosen)` when the attribute exists and uses the returned action string.
"""

import re
from collections import Counter

_TILE = re.compile(r'tile="(\w+)"')


class KanOverride:
    """Human-heuristic kan policy on top of a net:
    - ankan (closed quad) is always declared when the hand is tenpai after
      the kan (riichi included) — the dora / rinshan / fu bonus stays ours;
    - daiminkan (calling an opponent's 4th tile) is never declared — it
      opens a closed hand and hands everyone a new dora;
    - shouminkan is left to the net.
    """

    def __init__(self, net, ankan=True, no_daiminkan=True):
        self.net = net
        self.ankan, self.no_daiminkan = ankan, no_daiminkan
        self.stats = Counter()

    def __getattr__(self, name):            # proxy everything else
        return getattr(self.net, name)

    def override(self, table, pid, legal, chosen):
        kans = [a for a in legal if 'type="kan"' in a]
        if not kans:
            return chosen
        own_turn = table.turn == pid and table.last_drawn[pid] is not None
        if not own_turn:                     # interrupt window: daiminkan
            if self.no_daiminkan and 'type="kan"' in chosen:
                self.stats["daiminkan_blocked"] += 1
                skip = next((a for a in legal if 'type="skip"' in a), None)
                return skip or chosen
            return chosen
        if not self.ankan or 'type="kan"' in chosen:
            return chosen
        for a in kans:
            tile = _TILE.search(a).group(1)
            if table.hands[pid].count(tile) != 4:      # shouminkan: net decides
                continue
            if not table.riichi[pid] and not table._can_ankan(pid, tile):
                continue
            rest = [t for t in table.hands[pid] if t != tile]
            if table._shanten(rest, len(table.melds[pid]) + 1) <= 0:
                self.stats["ankan_forced"] += 1
                return a
        return chosen
