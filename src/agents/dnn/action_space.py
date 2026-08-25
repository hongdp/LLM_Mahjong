"""Pluggable action spaces, so models with different output layouts can be
trained, swapped and compared without the game loop knowing which is in play.

Motivation (2026-08-24): our native space is 11 action types x 34 tiles = 374,
of which only ~29% is ever legal. Mortal compresses the same decisions into 46
slots by exploiting the fact that a claimed tile is context-determined. To
compare the two fairly we need both to run through the *same* engine, rollout
loop and PPO code -- hence this adapter.

Design constraints this satisfies:

1. **Existing models are untouched.** A model that declares no action space (or
   ``"native"``) goes through exactly the code path it always has;
   `test_action_space.py` asserts byte-identical masks and lookups against the
   raw `encoder.legal_mask`, so no historical checkpoint changes behaviour.
2. **Multi-step protocols stay contained.** Mortal declares riichi and kan and
   *then* chooses a tile, where our engine bundles both into one action string.
   That difference lives entirely inside `MortalActionSpace`: it answers
   `follow_up()` with a mode, and the caller re-queries the policy with a second
   mask. The native space always answers ``None``, so single-step callers never
   pay for the mechanism.
3. **Models carry their own space**, mirroring the existing `encoder_variant`
   convention -- `space_of_arch("...")` resolves it from the zoo name, and
   `net.action_space` overrides when set.

Adding a third space later means implementing `ActionSpace` and registering it;
nothing in the loop changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.agents.dnn import encoder as _enc
from src.agents.dnn import mortal_action as _ma

# A follow-up mode is an opaque string the space hands back to itself.
FollowUp = Optional[str]


class ActionSpace:
    """Interface. `mask()` turns the engine's legal-action strings into a
    boolean vector over this space's slots plus a slot -> action-string lookup;
    `follow_up()` says whether the chosen slot still needs a second query."""

    name: str = "abstract"
    dim: int = 0

    def mask(self, actions: List[str], mode: FollowUp = None
             ) -> Tuple[np.ndarray, Dict[int, str]]:
        raise NotImplementedError

    def follow_up(self, slot: int, actions: List[str],
                  mode: FollowUp = None) -> FollowUp:
        """Mode string for a required second query, else None."""
        return None

    def resolve(self, slot: int, lookup: Dict[int, str]) -> Optional[str]:
        """Chosen slot -> engine action string (None when a follow-up is due)."""
        return lookup.get(slot)


class NativeActionSpace(ActionSpace):
    """Our incumbent 374-slot space: 11 action types x 34 tiles, single step.

    This is a thin pass-through to `encoder.legal_mask` on purpose -- it must
    stay behaviourally identical to the pre-adapter code path so that every
    checkpoint trained before this module keeps its exact semantics.
    """

    name = "native"
    dim = _enc.ACTION_DIM

    def mask(self, actions, mode=None):
        return _enc.legal_mask(actions)


class MortalActionSpace(ActionSpace):
    """Mortal's 46-slot space (see `mortal_action` for the layout).

    Two decisions are two-step here and one-step in our engine:
      * slot 37 declares riichi, then a second query picks the discard;
      * slot 42 decides to kan, then a second query picks which tile.
    `follow_up()` reports those modes; `mask(mode=...)` builds the second-step
    mask restricted to the matching engine actions.
    """

    name = "mortal46"
    dim = _ma.MORTAL_ACTION_DIM

    _KINDS = {"riichi": ("riichi", "riichi0"),
              "kan": ("kan", "ankan", "kakan", "daiminkan")}

    def mask(self, actions, mode=None):
        if mode in self._KINDS:
            actions = [a for a in actions
                       if _kind_of(a) in self._KINDS[mode]]
        m, lookup = _ma.legal_mask_46(
            actions,
            at_riichi_select=(mode == "riichi"),
            at_kan_select=(mode == "kan"),
        )
        return np.asarray(m, dtype=bool), lookup

    def follow_up(self, slot, actions, mode=None):
        if mode is not None:
            return None                     # already the second step
        if slot == _ma.IDX_RIICHI:
            return "riichi"
        if slot == _ma.IDX_KAN:
            # only ambiguous when the engine offers more than one kan tile
            kans = [a for a in actions if _kind_of(a) in self._KINDS["kan"]]
            tiles = {_ma.TILE_RE.search(a).group(1) for a in kans
                     if _ma.TILE_RE.search(a)}
            return "kan" if len(tiles) > 1 else None
        return None


def _kind_of(action_xml: str) -> str:
    m = _ma.ACTION_RE.search(action_xml)
    return m.group(1) if m else ""


REGISTRY: Dict[str, ActionSpace] = {
    NativeActionSpace.name: NativeActionSpace(),
    MortalActionSpace.name: MortalActionSpace(),
}


def space_of_arch(arch: str) -> str:
    """Action space implied by a zoo arch name, mirroring `variant_of_arch`.

    Anything without an explicit marker is native, which keeps every existing
    arch (and every checkpoint carrying its name) on the incumbent space.
    """
    arch = arch or ""
    if "_m46" in arch or arch.startswith("mortal_full"):
        return MortalActionSpace.name
    return NativeActionSpace.name


def get_space(net_or_name) -> ActionSpace:
    """Resolve a space from a net (reads `.action_space`, else its arch) or a
    plain name. Unknown values fall back to native rather than raising, so a
    stale checkpoint can never be locked out."""
    if isinstance(net_or_name, str):
        return REGISTRY.get(net_or_name, REGISTRY["native"])
    name = getattr(net_or_name, "action_space", None)
    if name is None:
        name = space_of_arch(getattr(net_or_name, "arch_name", "") or "")
    return REGISTRY.get(name, REGISTRY["native"])
