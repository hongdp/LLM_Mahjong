import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.constants import EAST, SOUTH, WEST, NORTH

from src.tasks.mahjong.wrapper import MahjongEngineAPI
from src.tasks.mahjong.shanten import (TileEfficiency, pad_for_melds,
                                       dora_from_indicator)

WINDS_ZH = ["东", "南", "西", "北"]
WIND_CONST = [EAST, SOUTH, WEST, NORTH]
SUIT_ORDER = {'p': 0, 's': 1, 'm': 2, 'z': 3}
SUIT_BASE_34 = {'m': 0, 'p': 9, 's': 18, 'z': 27}

DRAGONS_34 = frozenset({31, 32, 33})        # 5z 6z 7z
WINDS_34 = frozenset({27, 28, 29, 30})      # 1z 2z 3z 4z
PAO_YAKU = frozenset({"Daisangen", "Dai Suushii"})   # RCR 4.2.5.10

ACTION_RE = re.compile(
    r'<action\s+type="([^"]+)"(?:\s+tile="([^"]+)")?(?:\s+with="([^"]+)")?\s*/>'
)


def sort_key(tile: str):
    return (SUIT_ORDER.get(tile[-1], 4), int(tile[:-1]))


def str_to_34(tile: str) -> int:
    return SUIT_BASE_34[tile[-1]] + (int(tile[:-1]) or 5) - 1


RED_TILES = ("0m", "0p", "0s")        # 赤宝牌 spelling in wall / rivers / actions
RED_136 = {4: 16, 13: 52, 22: 88}     # 34-index of 5m/5p/5s -> the library's red copy id


def norm_tile(t: str) -> str:
    """'0m' (red five) -> '5m'; anything else unchanged. Hands hold only
    normalized tiles; redness lives in PyMahjongTable.red counts."""
    return "5" + t[1] if t[0] == "0" else t


def is_red(t: str) -> bool:
    return t[0] == "0"


def str_from_34(idx: int) -> str:
    if idx < 9:
        return f"{idx + 1}m"
    if idx < 18:
        return f"{idx - 9 + 1}p"
    if idx < 27:
        return f"{idx - 18 + 1}s"
    return f"{idx - 27 + 1}z"



# ---- rollout perf (2026-08-22): shanten LRU + wait-candidate pruning ----
import functools as _functools

_ORPHANS_34 = frozenset([0, 8, 9, 17, 18, 26] + list(range(27, 34)))


def _tile34(t: str) -> int:
    val, suit = int(t[:-1]) or 5, t[-1]
    return 27 + val - 1 if suit == "z" else "mps".index(suit) * 9 + val - 1


def _wait_candidates(tiles: List[str]):
    """34-indices that could possibly complete `tiles`: a winning tile must
    pair/triplet with a tile in hand (same tile) or extend a sequence
    (same suit, rank within +-2). Kokushi-shaped hands (all orphans) are the
    one case where the missing tile need not be adjacent -> full scan."""
    idx = [_tile34(t) for t in tiles]
    if all(i in _ORPHANS_34 for i in idx):
        return range(34)
    cand = set()
    for i in idx:
        cand.add(i)
        if i < 27:
            r, base = i % 9, i - i % 9
            for d in (-2, -1, 1, 2):
                if 0 <= r + d <= 8:
                    cand.add(base + r + d)
    return sorted(cand)

def _decompositions(counts: List[int], n_sets: int):
    """Yield every standard reading of a 34-count hand as `n_sets` sets
    (koutsu / shuntsu) plus one pair. Each reading is a list of
    ("set"|"seq"|"pair", first_tile_index)."""
    def rec(i, sets_left, pair_used, acc):
        while i < 34 and counts[i] == 0:
            i += 1
        if i == 34:
            if sets_left == 0 and pair_used:
                yield list(acc)
            return
        c = counts[i]
        if not pair_used and c >= 2:
            counts[i] -= 2
            acc.append(("pair", i))
            yield from rec(i, sets_left, True, acc)
            acc.pop(); counts[i] += 2
        if sets_left == 0:
            return
        if c >= 3:
            counts[i] -= 3
            acc.append(("set", i))
            yield from rec(i, sets_left - 1, pair_used, acc)
            acc.pop(); counts[i] += 3
        if i < 27 and i % 9 <= 6 and counts[i + 1] > 0 and counts[i + 2] > 0:
            counts[i] -= 1; counts[i + 1] -= 1; counts[i + 2] -= 1
            acc.append(("seq", i))
            yield from rec(i, sets_left - 1, pair_used, acc)
            acc.pop(); counts[i] += 1; counts[i + 1] += 1; counts[i + 2] += 1
    yield from rec(0, n_sets, False, [])


def _tile_only_as_triplet(tiles: List[str], tile: str, n_sets: int) -> bool:
    """RCR 3.12 (2): in every winning reading of `tiles`, `tile` is a koutsu
    (never part of a run, never the pair). False if no standard reading
    exists (chiitoitsu/kokushi readings can't hold a triplet anyway)."""
    counts = [0] * 34
    for t in tiles:
        counts[_tile34(t)] += 1
    k = _tile34(tile)
    found = False
    for reading in _decompositions(counts, n_sets):
        found = True
        if ("pair", k) in reading:
            return False
        # A run holding the tile is only fine when the reading also has the
        # koutsu (then the run uses the 4th copy, i.e. the tile is the wait
        # itself: 1111m23m -> 111m + 123m keeps the in-hand triplet intact).
        if (("set", k) not in reading
                and any(kind == "seq" and i <= k <= i + 2 for kind, i in reading)):
            return False
    return found



@_functools.lru_cache(maxsize=262144)
def _shanten_cached(key: tuple, num_melds: int) -> int:
    return PyMahjongTable._efficiency.calculate_shanten(
        pad_for_melds(list(key), num_melds))

class PyMahjongTable(MahjongEngineAPI):
    """
    Single-round four-player riichi mahjong table, implementing the EMA
    "Riichi — Rules for Japanese Mahjong" (2016 revision, RCR).

    Conformance notes (section numbers refer to the RCR):
    - 2.7/3.7.1 dead wall stays 14 tiles: every kan moves the live wall's
      tail tile into it, so a round always yields exactly 70 draws.
    - 3.7.2 at most four kans per ROUND across all players.
    - 3.8 kuikae (swap calling) is rejected after chi/pon.
    - 3.11 multiple ron: every declaring player wins (see step_ron).
    - 3.12 riichi needs a closed tenpai hand, >=1000 points and >=4 tiles
      left in the wall; the 1000 stick is only paid once the declaration
      tile survives the interrupt window; ankan while riichi is allowed
      under the three listed conditions; ippatsu / double riichi / ura
      dora are scored.
    - 3.13 furiten: own-discard (permanent record, survives the tile being
      called), same-turn missed ron, and permanent furiten after a riichi
      player passes a ron.
    - 3.14 no kan on the haitei tile; the houtei discard can only be ronned;
      tenpai payments 3000 split by tenpai count.
    - 4.2 situational yaku (ippatsu, double riichi, chankan, rinshan,
      haitei, houtei, tenhou, chiihou, renhou) and ura dora are wired into
      the HandCalculator; 4.2.5.10 pao liability is settled.

    Deliberate divergences (see docs/engine_known_issues.md):
    - Starting points are 25000, not the RCR's 30000 (project decision).
    - Match structure (renchan / honba / multi-round hanchan / uma) is out
      of scope: one episode == one round. `randomize_round=True` samples
      the round wind and dealer so the policy sees every seat/wind combo.
    - Kazoe yakuman (13+ han counts as a yakuman) is kept, where the RCR
      caps counted han at sanbaiman (project decision).

    Rewards: per-step rewards only carry format/legality penalties. All
    game-outcome rewards land in `final_rewards` (point deltas x
    REWARD_SCALE plus a placement bonus), which the orchestrator
    distributes to every player's trajectory at game end.
    """

    REWARD_SCALE = 0.001
    # No flat deal-in surcharge: the real point transfer plus the placement
    # bonus already price the loss — an extra flat penalty would double-count
    # and push the policy toward over-folding relative to true point EV.
    HOUJUU_EXTRA = 0.0
    ILLEGAL_PENALTY = -5.0
    FORMAT_PENALTY = -10.0
    MAX_MELDS = 4
    MAX_KANS = 4            # RCR 3.7.2, counted over the whole round
    RIICHI_MIN_WALL = 4     # RCR 3.12
    # Per-round placement bonus (design doc 4.2 Uma, single-round scale):
    # sharpens the zero-sum ranking pressure that motivates defense. The
    # RCR's real uma (+-15000/+-5000) applies at hanchan end and belongs
    # with the deferred multi-round work.
    RANK_BONUS = [2.0, 0.5, -0.5, -2.0]

    _efficiency = TileEfficiency()
    _calculator = HandCalculator()

    def __init__(self, value_facts: bool = False,
                 randomize_round: bool = False):
        # value_facts=True appends computed value information (own dora
        # tiles) to the private state line. This CHANGES the prompt
        # template — SFT data and rollouts must agree on the flag, or
        # format compliance collapses (template-consistency rule).
        self.value_facts = value_facts
        # randomize_round=True samples round wind (东/南) and dealer seat
        # so a single-round episode is not always 东1局 with player 0 as
        # dealer. Only the VALUES in the 场况 line change, not the
        # template's shape.
        self.randomize_round = randomize_round
        self.reset()

    # ------------------------------------------------------------------
    # Setup / state
    # ------------------------------------------------------------------
    def reset(self) -> Dict[int, str]:
        if self.randomize_round:
            self.dealer = random.randrange(4)
            # 东 45% / 南 45% / 西 10% (西入 exists in Majsoul; the encoder
            # spells West as both round-wind bits set)
            r = random.random()
            self.round_wind_idx = 0 if r < 0.45 else (1 if r < 0.9 else 2)
        else:
            self.dealer = 0
            self.round_wind_idx = 0
        self.round_wind = WIND_CONST[self.round_wind_idx]
        # Seat 0 is the starting dealer, so 东N局 <=> dealer sits at N-1.
        self.round_number = self.dealer + 1
        self.turn = self.dealer

        # Context randomization (user 2026-08-23): a single hand is played
        # under a random MATCH context — unequal starting scores and carried
        # riichi sticks — so placement pressure (protect a lead / push when
        # behind) is learnable without the multi-hand structure. Rewards use
        # the point DELTA from these starts; the placement bonus stays on
        # final points. Deterministic in the deal seed (dup_k replicas share
        # the context, so the group baseline removes its level effect).
        # progression index k = hands notionally played (东1=0 .. 西4=11):
        # 东1 starts dead equal; the spread grows like a random walk
        # (sigma ~4500/hand -> std ~4500*sqrt(k)); sticks only for k>=1.
        # Scores are multiples of 100, sum exactly 100000, floor 0
        # (Majsoul: 0 is alive, negative busts -> no hand STARTS negative;
        # in-hand payments may go negative = the bust ending itself).
        k = self.round_wind_idx * 4 + self.dealer
        if self.randomize_round and k > 0:
            spread = random.uniform(0.5, 1.5) * 4500.0 * (k ** 0.5)
            z = [max(-2.2, min(2.2, random.gauss(0.0, 1.0))) for _ in range(4)]
            zm = sum(z) / 4.0
            d = [spread * (x - zm) for x in z]          # zero-sum by construction
            pts = [25000 + int(round(x / 100.0)) * 100 for x in d]
            pts[pts.index(max(pts))] += 100000 - sum(pts)   # rounding remainder (x100)
            while min(pts) < 0:                              # floor 0, paid by the leader
                i, j = pts.index(min(pts)), pts.index(max(pts))
                pts[j] += pts[i]
                pts[i] = 0
            self.points = pts
            self.kyotaku = 1000 * random.choices(
                (0, 1, 2, 3), weights=(70, 20, 8, 2))[0]
        else:
            self.points = [25000, 25000, 25000, 25000]
            self.kyotaku = 0
        self.start_points = list(self.points)
        self.start_kyotaku = self.kyotaku
        # Visible river: a called tile leaves it (it now sits in a meld).
        self.discards = {i: [] for i in range(4)}
        # Permanent discard record for furiten (RCR 3.13.1): a tile you
        # discarded keeps making you furiten even after someone calls it.
        self.furiten_river = {i: [] for i in range(4)}
        self.melds = {i: [] for i in range(4)}  # {"type","tiles","opened"}

        self.riichi = [False, False, False, False]
        self.riichi_pending: Optional[int] = None   # declared, stick unpaid
        # --- public-fact record for encoder v3 (2026-08-22): per-seat river
        # events [tile, tsumogiri, riichi_decl, called, discard_idx]; additive,
        # never read by game logic. riichi_turn = discard index at declaration.
        self.river_events: Dict[int, list] = {i: [] for i in range(4)}
        self.riichi_turn: List[Optional[int]] = [None, None, None, None]
        self.ippatsu = [False, False, False, False]
        self.daburu = [False, False, False, False]
        self.temp_furiten = [False, False, False, False]   # RCR 3.13.2
        self.perm_furiten = [False, False, False, False]   # RCR 3.13.3
        self.rinshan = [False, False, False, False]

        self.discard_count = [0, 0, 0, 0]
        self.any_call = False        # any chi/pon/kan happened this round
        self.kan_count = 0
        self.kuikae: Optional[Tuple[int, set]] = None
        self.pending_kan: Optional[dict] = None    # open chankan window
        self.pao: Dict[int, int] = {}              # winner -> liable player
        # Majsoul kan-dora timing: daiminkan / shouminkan reveal after the
        # kan player's next discard (or on a rinshan win); ankan reveals now.
        self.pending_dora_reveal = 0
        self.kan_players: set = set()              # for 四杠散了
        self._pending_abort: Optional[str] = None  # 途中流局 applied once the window passes
        self._ron_chance: set = set()
        self._waits_cache: Dict[tuple, List[str]] = {}

        self.last_discard: Optional[str] = None
        self.last_discarder: Optional[int] = None
        self.last_drawn: List[Optional[str]] = [None, None, None, None]
        self.finished = False
        self.final_rewards: Optional[List[float]] = None
        self.result_summary = ""

        all_tiles = [f"{i}{s}" for s in "mps" for i in range(1, 10)]
        all_tiles += [f"{i}z" for i in range(1, 8)]
        self.wall = all_tiles * 4
        # 赤宝牌 (Majsoul): one red five per suit replaces a plain five.
        for suit in "mps":
            self.wall[self.wall.index(f"5{suit}")] = f"0{suit}"
        random.shuffle(self.wall)
        # red fives held per seat (hands store them as plain '5x')
        self.red = {i: {"m": 0, "p": 0, "s": 0} for i in range(4)}
        self.last_drawn_red = [False, False, False, False]
        self.last_discard_red = False
        # Dead wall layout (fixed slots, never popped so indices are
        # stable): [0:4] rinshan draws, [4:9] dora indicators,
        # [9:14] the ura indicator under each of them.
        self.dead_wall = [self.wall.pop() for _ in range(14)]
        self._rinshan_idx = 0
        self.dead_wall = [norm_tile(t) if i >= 4 else t
                          for i, t in enumerate(self.dead_wall)]   # indicators: red == plain
        self.dora_indicators = [self.dead_wall[4]]
        self.ura_indicators = [self.dead_wall[9]]

        self.hands = {i: [] for i in range(4)}
        for pid in range(4):
            for _ in range(13):
                self._give(pid, self.wall.pop())
            self.hands[pid].sort(key=sort_key)
        first = self._give(self.dealer, self.wall.pop())
        self.hands[self.dealer].sort(key=sort_key)
        self.last_drawn[self.dealer] = first
        return self._obs()

    # ---- red-five bookkeeping ---------------------------------------
    def _give(self, pid: int, raw: str) -> str:
        """Move a wall tile into a hand; returns the normalized tile."""
        if is_red(raw):
            self.red[pid][raw[1]] += 1
        t = norm_tile(raw)
        self.hands[pid].append(t)
        self.last_drawn_red[pid] = is_red(raw)
        return t

    def _plain_copies(self, pid: int, tile: str) -> int:
        """Non-red copies of `tile` in hand."""
        n = self.hands[pid].count(tile)
        if tile[0] == "5" and tile[-1] in "mps":
            n -= self.red[pid][tile[-1]]
        return n

    def _take_from_hand(self, pid: int, tile: str, n: int = 1, prefer_red: bool = False) -> int:
        """Remove n copies of `tile` from the hand, plain copies first
        (red last) unless prefer_red; returns how many reds were used."""
        used_red = 0
        for _ in range(n):
            self.hands[pid].remove(tile)
        if tile[0] == "5" and tile[-1] in "mps":
            suit = tile[-1]
            plain = self.hands[pid].count(tile) + n - self.red[pid][suit]   # plain copies before removal
            used_red = n if prefer_red else max(0, n - plain)
            used_red = min(used_red, self.red[pid][suit])
            self.red[pid][suit] -= used_red
        return used_red

    def display_hand(self, pid: int) -> List[str]:
        """Hand with red fives spelled '0x' (for text obs / records)."""
        out, left = [], dict(self.red[pid])
        for t in self.hands[pid]:
            if t[0] == "5" and t[-1] in "mps" and left[t[-1]] > 0:
                left[t[-1]] -= 1
                out.append("0" + t[-1])
            else:
                out.append(t)
        return out

    def _meld_str(self, meld: dict) -> str:
        tiles = list(meld['tiles'])
        for _ in range(meld.get("red", 0)):          # spell the red copies
            k = next((i for i, t in enumerate(tiles) if t[0] == "5" and t[-1] in "mps"), None)
            if k is not None:
                tiles[k] = "0" + tiles[k][-1]
        return f"{meld['type']}({' '.join(tiles)})"

    def _obs(self) -> Dict[int, str]:
        """Per-seat text observations for the LLM path; DNN drivers set
        text_obs=False because they never read them (perf 2026-08-22)."""
        if not getattr(self, "text_obs", True):
            return {}
        return {i: self._format_state(i) for i in range(4)}

    def _format_state(self, player_id: int) -> str:
        dora = ' '.join(self.dora_indicators)
        wind_zh = WINDS_ZH[self.round_wind_idx]
        state = (
            f"场况 (Global)： 场风: {wind_zh}, 局数: {wind_zh}{self.round_number}局, "
            f"宝牌指示牌: {dora}, "
            f"供托: {self.kyotaku // 1000}, 剩余牌数: {len(self.wall)}\n"
        )
        own_melds = ' '.join(self._meld_str(m) for m in self.melds[player_id]) or '无'
        riichi_tag = ", 已立直" if self.riichi[player_id] else ""
        value_tag = ""
        if self.value_facts:
            # Computed, not rule text: list this player's dora tiles
            # (hand + own melds) so a small model can use value directly.
            dora_tiles = {dora_from_indicator(i) for i in self.dora_indicators}
            held = [t for t in self.hands[player_id] if t in dora_tiles]
            held += [t for m in self.melds[player_id] for t in m['tiles']
                     if t in dora_tiles]
            value_tag = f", 自家宝牌: {' '.join(held) if held else '无'}"
        state += (
            f"私有 (Private)： 自风: {WINDS_ZH[(player_id - self.dealer) % 4]}, "
            f"点数: {self.points[player_id]}, "
            f"手牌: {' '.join(self.display_hand(player_id))}, 副露: {own_melds}{riichi_tag}{value_tag}\n"
        )
        state += "公共 (Public)：\n"
        for i in range(4):
            if i == player_id:
                continue
            melds = ' '.join(self._meld_str(m) for m in self.melds[i]) or '无'
            river = ' '.join(self.discards[i]) or '无'
            r_tag = ", 已立直" if self.riichi[i] else ""
            state += (
                f"  玩家{i} ({WINDS_ZH[(i - self.dealer) % 4]}): "
                f"点数: {self.points[i]}, 牌河: {river}, 副露: {melds}{r_tag}\n"
            )
        return state

    # ------------------------------------------------------------------
    # Shanten / win evaluation helpers
    # ------------------------------------------------------------------
    def _shanten(self, tiles: List[str], num_melds: int) -> int:
        """Shanten of a (possibly melded) hand via dummy-triplet padding."""
        # Clamp so padding can never exceed a legal 14-tile hand.
        num_melds = max(0, min(num_melds, (14 - len(tiles)) // 3))
        try:
            return _shanten_cached(tuple(sorted(tiles)), num_melds)
        except ValueError:
            # Degenerate tile counts (broken mid-action states) are never
            # winning/tenpai — report "far from tenpai" instead of crashing.
            return 8

    def _alloc_136(self, tiles: List[str], counter: dict,
                   reds: Optional[dict] = None) -> List[int]:
        """136-ids for tiles; fives draw their red copy (id copy 0) while
        `reds[suit]` > 0, plain copies use 1..3."""
        ids = []
        for t in tiles:
            i34 = str_to_34(t)
            if i34 in RED_136:
                suit = t[-1]
                if reds and reds.get(suit, 0) > 0:
                    reds[suit] -= 1
                    ids.append(RED_136[i34])
                    continue
                copy = min(counter.get(i34, 1), 3)
                ids.append(i34 * 4 + copy)
                counter[i34] = copy + 1
                continue
            copy = min(counter.get(i34, 0), 3)
            ids.append(i34 * 4 + copy)
            counter[i34] = copy + 1
        return ids

    _MELD_TYPE_MAP = {
        "chi": (Meld.CHI, True),
        "pon": (Meld.PON, True),
        "kan": (Meld.KAN, True),
        "ankan": (Meld.KAN, False),
        "shouminkan": (Meld.SHOUMINKAN, True),
    }

    def _situational_config(self, player_id: int, is_tsumo: bool,
                            chankan: bool) -> HandConfig:
        """HandConfig with every RCR 4.2 situational yaku wired up."""
        # 天和/地和/人和 and 双立直 all require an untouched first go-around.
        virgin = not self.any_call
        no_discard_yet = self.discard_count[player_id] == 0
        is_dealer = player_id == self.dealer
        tenhou = (is_tsumo and is_dealer and virgin
                  and sum(self.discard_count) == 0)
        chiihou = (is_tsumo and not is_dealer and virgin and no_discard_yet)
        # Majsoul rules (2026-08-23): no renhou; double yakuman on.
        rinshan = is_tsumo and self.rinshan[player_id]
        # The wall being empty means the tile just drawn was the haitei
        # tile / the discard is the houtei discard (RCR 3.14).
        last_tile = len(self.wall) == 0
        return HandConfig(
            is_tsumo=is_tsumo,
            is_riichi=self.riichi[player_id],
            is_daburu_riichi=self.daburu[player_id],
            is_ippatsu=self.ippatsu[player_id],
            is_rinshan=rinshan,
            is_chankan=chankan,
            is_haitei=is_tsumo and last_tile and not rinshan,
            is_houtei=(not is_tsumo) and last_tile and not chankan,
            is_tenhou=tenhou,
            is_chiihou=chiihou,
            is_renhou=False,
            player_wind=WIND_CONST[(player_id - self.dealer) % 4],
            round_wind=self.round_wind,
            options=OptionalRules(
                has_open_tanyao=True,        # RCR 4.2.1.7 kuitan
                has_aka_dora=True,           # Majsoul 赤宝牌
                has_double_yakuman=True,     # Majsoul: 四暗刻单骑 etc. ×2
                # kazoe_limit left at the library default (13+ han counts
                # as a yakuman) — a deliberate project divergence.
            ),
        )

    def _win_result(self, player_id: int, win_tile: str, is_tsumo: bool,
                    chankan: bool = False):
        """Returns a HandResponse if (hand + win_tile) is a legal, yaku-
        bearing win for this player, else None."""
        hand = list(self.hands[player_id])
        if is_tsumo:
            if win_tile not in hand:
                return None
            concealed = hand
        else:
            # memoized waits == {t : shanten(hand+t) == -1}; same gate,
            # but cached per hand instead of one shanten per ron check
            if win_tile not in self._waits(player_id):
                return None
            concealed = hand + [win_tile]

        if self._shanten(concealed, len(self.melds[player_id])) != -1:
            return None

        counter: dict = {}
        rest = list(concealed)
        rest.remove(win_tile)
        reds = dict(self.red[player_id])          # reds in the concealed part
        if is_tsumo:
            # the drawn tile is already in the hand / red counts
            win_red = (self.last_drawn_red[player_id]
                       and win_tile == self.last_drawn[player_id])
        else:
            win_red = (self.pending_kan.get("red", False) if chankan and self.pending_kan
                       else self.last_discard_red)
        if win_red:
            if is_tsumo:
                reds[win_tile[-1]] -= 1           # reserve the red copy for the win tile
            win_id = RED_136[str_to_34(win_tile)]
            tile_ids = self._alloc_136(rest, counter, reds)
        else:
            tile_ids = self._alloc_136(rest, counter, reds)
            win_id = self._alloc_136([win_tile], counter)[0]
        tile_ids.append(win_id)
        meld_objs = []
        for meld in self.melds[player_id]:
            mtype, opened = self._MELD_TYPE_MAP[meld["type"]]
            m_reds = {"m": 0, "p": 0, "s": 0}
            if meld.get("red") and meld["tiles"][0][-1] in "mps":
                m_reds[meld["tiles"][0][-1]] = meld["red"]
            m_ids = self._alloc_136(meld["tiles"], counter, m_reds)
            tile_ids.extend(m_ids)
            meld_objs.append(Meld(meld_type=mtype, tiles=m_ids, opened=opened))

        # RCR 3.12: a riichi winner also turns over the ura indicators.
        indicators = list(self.dora_indicators)
        if is_tsumo and self.pending_dora_reveal:
            k = len(indicators)
            indicators += self.dead_wall[4 + k:4 + min(5, k + self.pending_dora_reveal)]
        if self.riichi[player_id]:
            indicators += self.ura_indicators[:len(indicators)]
        dora_ids = self._alloc_136(indicators, {})

        result = self._calculator.estimate_hand_value(
            sorted(tile_ids), win_id,
            melds=meld_objs or None,
            dora_indicators=dora_ids,
            config=self._situational_config(player_id, is_tsumo, chankan),
        )
        return None if result.error else result

    def _waits(self, player_id: int) -> List[str]:
        """Tiles that would complete this player's 13-tile-state hand.

        Memoized on (hand, meld count): furiten checks, the missed-ron
        snapshot and the riichi-ankan test all query the same hand many
        times per discard, and each miss costs 34 shanten evaluations.
        """
        hand = self.hands[player_id]
        n_melds = len(self.melds[player_id])
        key = (tuple(hand), n_melds)
        cached = self._waits_cache.get(key)
        if cached is not None:
            return cached
        waits = []
        # a 13-tile-state hand has waits only if it is tenpai: one shanten
        # call replaces 34 when it is not (perf 2026-08-22)
        if self._shanten(hand, n_melds) == 0:
            for i34 in _wait_candidates(hand):
                t = str_from_34(i34)
                if hand.count(t) >= 4:
                    continue
                if self._shanten(hand + [t], n_melds) == -1:
                    waits.append(t)
        if len(self._waits_cache) > 4096:
            self._waits_cache.clear()
        self._waits_cache[key] = waits
        return waits

    def _is_furiten(self, player_id: int) -> bool:
        """RCR 3.13: own-discard furiten (permanent record), plus the
        same-turn / riichi-permanent flags set by a passed-up ron."""
        if self.perm_furiten[player_id] or self.temp_furiten[player_id]:
            return True
        own_river = {t.replace('*', '') for t in self.furiten_river[player_id]}
        return any(w in own_river for w in self._waits(player_id))

    def _can_ron(self, player_id: int, tile: str, chankan: bool = False) -> bool:
        if self._is_furiten(player_id):
            return False
        return self._win_result(
            player_id, tile, is_tsumo=False, chankan=chankan) is not None

    def _is_closed(self, player_id: int) -> bool:
        return all(m["type"] == "ankan" for m in self.melds[player_id])

    # ------------------------------------------------------------------
    # Kan legality (RCR 3.4 / 3.5 / 3.6 / 3.7.2 / 3.12 / 3.14)
    # ------------------------------------------------------------------
    def _kan_allowed(self, player_id: int, new_meld: bool = True) -> bool:
        """Shared preconditions for any kan.

        `new_meld=False` for shouminkan: it upgrades an existing pon in
        place, so the meld-count cap does not apply (found 2026-08-26 by
        exp45 human-log replay: a toitoi hand with 4 pons kakans its pair
        tile — tenhou-legal, we never offered it)."""
        return (self.kan_count < self.MAX_KANS
                and (not new_meld
                     or len(self.melds[player_id]) < self.MAX_MELDS)
                # RCR 3.14: the haitei drawer may not kan (and there would
                # be no live tile left to move into the dead wall).
                and len(self.wall) > 0)

    def _can_ankan(self, player_id: int, tile: str) -> bool:
        if not self._kan_allowed(player_id):
            return False
        if self.last_drawn[player_id] is None:      # RCR 3.6: after a draw
            return False
        if self.hands[player_id].count(tile) != 4:
            return False
        if not self.riichi[player_id]:
            return True
        # RCR 3.12: ankan during riichi needs all three conditions.
        # (1) the drawn tile is the fourth copy
        if self.last_drawn[player_id] != tile:
            return False
        before = list(self.hands[player_id])
        before.remove(self.last_drawn[player_id])
        n_melds = len(self.melds[player_id])
        waits_before = self._waits_of(before, n_melds)
        if not waits_before:
            return False
        # (2) the tile may only be read as a triplet: in every winning
        # decomposition of the riichi hand (for every wait) it must sit in
        # a koutsu, never in a run or as the pair. Exact enumeration —
        # before 2026-08-23 this was approximated by "refuse if any
        # neighbouring tile is in hand", which also refused legal kans
        # such as 2345555s (234s + 555s + tanki) on the 4th 5s.
        if not all(_tile_only_as_triplet(before + [w], tile, 4 - n_melds)
                   for w in waits_before):
            return False
        # (3) the wait may not change
        after = [t for t in self.hands[player_id] if t != tile]
        waits_after = self._waits_of(after, n_melds + 1)
        return waits_before == waits_after

    def _waits_of(self, tiles: List[str], n_melds: int) -> set:
        if self._shanten(tiles, n_melds) != 0:
            return set()
        return {str_from_34(i) for i in _wait_candidates(tiles)
                if tiles.count(str_from_34(i)) < 4
                and self._shanten(tiles + [str_from_34(i)], n_melds) == -1}

    def _can_shouminkan(self, player_id: int, tile: str) -> bool:
        if not self._kan_allowed(player_id, new_meld=False):
            return False
        # RCR 3.12: a riichi hand may only ankan.
        if self.riichi[player_id] or self.last_drawn[player_id] is None:
            return False
        if tile not in self.hands[player_id]:
            return False
        return any(m["type"] == "pon" and m["tiles"][0] == tile
                   for m in self.melds[player_id])

    # ------------------------------------------------------------------
    # Kuikae (RCR 3.8)
    # ------------------------------------------------------------------
    def _kuikae_tiles(self, called: str, used: Sequence[str]) -> set:
        """Tiles the caller may not discard right after claiming `called`
        with the hand tiles `used`."""
        forbidden = {called}
        if len(used) == 2 and used[0][-1] == used[1][-1] != 'z':
            a, b = sorted(int(t[:-1]) for t in used)
            suit = used[0][-1]
            if b == a + 1:      # ryanmen: the other end is the swap tile
                for end in (a - 1, b + 1):
                    if 1 <= end <= 9 and f"{end}{suit}" != called:
                        forbidden.add(f"{end}{suit}")
        return forbidden

    def _forbidden_discards(self, player_id: int) -> set:
        if self.kuikae and self.kuikae[0] == player_id:
            return self.kuikae[1]
        return set()

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def get_legal_actions(self, player_id: int) -> List[str]:
        hand = self.hands[player_id]
        drawn = self.last_drawn[player_id]
        banned = self._forbidden_discards(player_id)

        # Riichi lock: tsumogiri (or tsumo/ankan) only.
        if self.riichi[player_id]:
            actions = []
            if drawn and self._win_result(player_id, drawn, is_tsumo=True):
                actions.append('<action type="tsumo" />')
            if drawn and self._can_ankan(player_id, drawn):
                actions.append(f'<action type="kan" tile="{drawn}" />')
            if drawn:
                spelled = "0" + drawn[-1] if self.last_drawn_red[player_id] else drawn
                actions.append(f'<action type="discard" tile="{spelled}" />')
            return actions or [
                f'<action type="discard" tile="{t}" />'
                for t in self._discardable(player_id, banned)
            ]

        uniq = self._discardable(player_id, banned)
        actions = [f'<action type="discard" tile="{t}" />' for t in uniq]

        n_melds = len(self.melds[player_id])
        # 九种九牌 (Majsoul): first draw, no call yet, >=9 distinct terminals/honors.
        if (drawn and not self.any_call and self.discard_count[player_id] == 0
                and self._kyuushu_kinds(hand) >= 9):
            actions.append('<action type="kyuushu" />')
        # Riichi declaration: closed hand, >=1000 points, >=4 tiles left in
        # the wall (RCR 3.12), tenpai after the discard.
        if (self._is_closed(player_id) and self.points[player_id] >= 1000
                and drawn and len(self.wall) >= self.RIICHI_MIN_WALL):
            # 14-tile shanten is the best post-discard shanten: if it is
            # worse than 0, no discard reaches tenpai -> skip the scan.
            # -1 (completed hand) must pass too: declining the win and
            # declaring riichi is legal (RCR 3.12 only requires tenpai
            # after the discard) — 凤凰卓 players do it for ura/ippatsu;
            # found 2026-08-26 by exp45 human-log replay (5/974k decisions).
            if self._shanten(hand, n_melds) <= 0:
                for t in uniq:
                    rest = list(hand)
                    rest.remove(norm_tile(t))
                    if self._shanten(rest, n_melds) == 0:
                        actions.append(f'<action type="riichi" tile="{t}" />')

        # Kan from own turn: ankan (4 in hand) / shouminkan (4th tile of own pon).
        for t in sorted(set(hand), key=sort_key):
            if self._can_ankan(player_id, t) or self._can_shouminkan(player_id, t):
                actions.append(f'<action type="kan" tile="{t}" />')

        if drawn and self._win_result(player_id, drawn, is_tsumo=True):
            actions.append('<action type="tsumo" />')
        return actions

    def _discardable(self, player_id: int, banned: set) -> List[str]:
        """Distinct discard spellings: plain tiles, plus '0x' for a held red
        five (listed right after its plain '5x' when both exist)."""
        out = []
        for t in sorted(set(self.hands[player_id]), key=sort_key):
            if t in banned:
                continue
            if t[0] == "5" and t[-1] in "mps" and self.red[player_id][t[-1]] > 0:
                if self._plain_copies(player_id, t) > 0:
                    out.append(t)
                out.append("0" + t[-1])
            else:
                out.append(t)
        return out

    @staticmethod
    def _kyuushu_kinds(hand: List[str]) -> int:
        return len({t for t in hand
                    if t[-1] == 'z' or t[0] in '19'})

    def get_interrupt_actions(self, player_id: int) -> List[str]:
        # Chankan window (RCR 4.2.1.12): only a ron may interrupt an
        # added kan, and only from the other three players.
        if self.pending_kan:
            actions = ['<action type="skip" />']
            if player_id != self.pending_kan["player"]:
                if self.pending_kan.get("ankan"):
                    if player_id in self._kokushi_robbers(self.pending_kan["player"],
                                                          self.pending_kan["tile"]):
                        actions.append('<action type="ron" />')
                elif self._can_ron(player_id, self.pending_kan["tile"], chankan=True):
                    actions.append('<action type="ron" />')
            return actions

        if self.finished or not self.last_discard:
            return ['<action type="skip" />']

        tile = self.last_discard.replace('*', '')
        actions = ['<action type="skip" />']

        if self._can_ron(player_id, tile):
            actions.append('<action type="ron" />')

        # Riichi players may only ron or pass. So may everyone once the
        # wall is empty: the houtei discard cannot be called (RCR 3.14).
        if (self.riichi[player_id] or len(self.wall) == 0
                or len(self.melds[player_id]) >= self.MAX_MELDS):
            return actions

        hand = self.hands[player_id]
        if hand.count(tile) >= 2:
            actions.append(f'<action type="pon" tile="{tile}" />')
        if hand.count(tile) >= 3 and self.kan_count < self.MAX_KANS:
            actions.append(f'<action type="kan" tile="{tile}" />')

        if (self.last_discarder + 1) % 4 == player_id and tile[-1] != 'z':
            for pair in self._chi_pairs(player_id, tile):
                actions.append(
                    f'<action type="chi" tile="{tile}" with="{pair[0]} {pair[1]}" />'
                )
        return actions

    def _chi_pairs(self, player_id: int, tile: str) -> List[List[str]]:
        val, suit = int(tile[:-1]), tile[-1]
        hand = self.hands[player_id]
        pairs = []
        for a, b in [(val - 2, val - 1), (val - 1, val + 1), (val + 1, val + 2)]:
            if 1 <= a <= 9 and 1 <= b <= 9:
                ta, tb = f"{a}{suit}", f"{b}{suit}"
                if ta in hand and tb in hand:
                    pairs.append([ta, tb])
        return pairs

    # ------------------------------------------------------------------
    # Turn phase
    # ------------------------------------------------------------------
    def step(self, player_id: int, action_xml: str):
        match = ACTION_RE.search(action_xml or "")
        rewards = {i: 0.0 for i in range(4)}
        discarded = False

        def do_discard(tile: str, riichi_mark: bool = False, red: bool = False):
            nonlocal discarded
            spelled = "0" + tile[-1] if red else tile
            tsumogiri = (self.last_drawn[player_id] == tile
                         and self.last_drawn_red[player_id] == red)
            self.river_events[player_id].append(
                [spelled, tsumogiri, riichi_mark, False,
                 self.discard_count[player_id]])
            if riichi_mark:
                self.riichi_turn[player_id] = self.discard_count[player_id]
            self._take_from_hand(player_id, tile, 1, prefer_red=red)
            self.discards[player_id].append(spelled + ('*' if riichi_mark else ''))
            self.furiten_river[player_id].append(tile)
            self.last_discard = tile
            self.last_discard_red = red
            self.last_discarder = player_id
            self.last_drawn[player_id] = None
            self.rinshan[player_id] = False
            # A riichi player's own next discard ends their ippatsu window.
            self.ippatsu[player_id] = False
            self.discard_count[player_id] += 1
            self.kuikae = None
            # Majsoul: open-kan dora is turned over after the discard.
            while self.pending_dora_reveal > 0:
                self.pending_dora_reveal -= 1
                self._reveal_kan_dora()
            # 四风连打: four first discards, same wind, no call in between.
            if (not self.any_call and sum(self.discard_count) == 4
                    and all(self.discard_count[i] == 1 for i in range(4))):
                firsts = {self.furiten_river[i][0] for i in range(4)}
                if len(firsts) == 1 and next(iter(firsts)) in ("1z", "2z", "3z", "4z"):
                    self._pending_abort = "四风连打"
            # 四杠散了: four kans by two or more players, after the discard.
            if self.kan_count >= 4 and len(self.kan_players) >= 2:
                self._pending_abort = "四杠散了"
            # RCR 3.13.2: remember who could have ronned this tile; the
            # flags are applied once the interrupt window closes.
            self._ron_chance = {p for p in range(4) if p != player_id
                                and tile in self._waits(p)}
            discarded = True

        def forced_discard():
            # Auto-correct so the game always progresses.
            if not self.hands[player_id]:
                return
            if self.riichi[player_id] and self.last_drawn[player_id] in self.hands[player_id]:
                do_discard(self.last_drawn[player_id], red=self.last_drawn_red[player_id])
            else:
                banned = self._forbidden_discards(player_id)
                pool = [t for t in self.hands[player_id] if t not in banned]
                t = random.choice(pool or self.hands[player_id])
                do_discard(t, red=self._plain_copies(player_id, t) == 0)

        if not match:
            rewards[player_id] = self.FORMAT_PENALTY
            forced_discard()
        else:
            action_type, tile, _with = match.groups()
            hand = self.hands[player_id]
            drawn = self.last_drawn[player_id]
            banned = self._forbidden_discards(player_id)
            red = bool(tile) and is_red(tile)
            if tile:
                tile = norm_tile(tile)
            # the spelled copy must exist: red needs a red five, plain a plain one
            has_copy = bool(tile) and tile in hand and (
                self.red[player_id][tile[-1]] > 0 if red else self._plain_copies(player_id, tile) > 0)

            if (action_type == "discard" and has_copy and tile not in banned
                    and (not self.riichi[player_id]
                         or (tile == drawn and red == self.last_drawn_red[player_id]))):
                do_discard(tile, red=red)

            elif (
                action_type == "riichi"
                and has_copy
                and tile not in banned
                and not self.riichi[player_id]
                and self._is_closed(player_id)
                and self.points[player_id] >= 1000
                and len(self.wall) >= self.RIICHI_MIN_WALL
                and self._riichi_tenpai(player_id, tile)
            ):
                # RCR 3.12: the stick is only paid once the declaration
                # tile passes the interrupt window (a ron voids the riichi).
                self.riichi[player_id] = True
                self.riichi_pending = player_id
                self.daburu[player_id] = (self.discard_count[player_id] == 0
                                          and not self.any_call)
                do_discard(tile, riichi_mark=True, red=red)
                self.ippatsu[player_id] = True

            elif (action_type == "kyuushu" and drawn and not self.any_call
                    and self.discard_count[player_id] == 0
                    and self._kyuushu_kinds(hand) >= 9):
                self._abort("九种九牌")
                return (self._obs(), rewards, True, {"discarded": False})

            elif action_type == "tsumo" and drawn and (
                result := self._win_result(player_id, drawn, is_tsumo=True)
            ):
                self._settle_tsumo(player_id, result)
                return (
                    self._obs(),
                    rewards, True, {"discarded": False},
                )

            elif action_type == "kan" and tile and self._can_ankan(player_id, tile):
                if self._kokushi_robbers(player_id, tile):
                    # Majsoul: 国士无双 can rob a concealed kan. Defer the
                    # kan exactly like an added kan; resolve_pending_kan
                    # completes it if nobody rons.
                    self.pending_kan = {"player": player_id, "tile": tile, "ankan": True,
                                        "red": False}
                    self._ron_chance = set(self._kokushi_robbers(player_id, tile))
                    return (self._obs(), rewards, False,
                            {"discarded": False, "chankan": tile})
                self._do_ankan(player_id, tile)
                # Turn continues: player discards after the rinshan draw.

            elif action_type == "kan" and tile and self._can_shouminkan(player_id, tile):
                # RCR 4.2.1.12: an added kan can be robbed. Nothing is
                # mutated until the chankan window closes.
                self.pending_kan = {"player": player_id, "tile": tile,
                                    "red": self._plain_copies(player_id, tile) == 0
                                    and self.red[player_id].get(tile[-1], 0) > 0}
                # a passed-up chankan is a missed win too (同巡/立直振听)
                self._ron_chance = {p for p in range(4) if p != player_id
                                    and tile in self._waits(p)}
                return (
                    self._obs(),
                    rewards, False, {"discarded": False, "chankan": tile},
                )

            else:
                # Illegal action for the turn phase (incl. false tsumo,
                # "skip" on one's own turn, riichi without tenpai, kuikae).
                rewards[player_id] = self.ILLEGAL_PENALTY
                forced_discard()

        self.hands[player_id].sort(key=sort_key)
        obs = self._obs()
        return obs, rewards, self.finished, {"discarded": discarded}

    def _riichi_tenpai(self, player_id: int, discard_tile: str) -> bool:
        rest = list(self.hands[player_id])
        rest.remove(discard_tile)
        return self._shanten(rest, len(self.melds[player_id])) == 0

    def _do_ankan(self, player_id: int, tile: str):
        used_red = self._take_from_hand(player_id, tile, 4)
        self.melds[player_id].append(
            {"type": "ankan", "tiles": [tile] * 4, "opened": False, "red": used_red}
        )
        self._after_kan(player_id, reveal_now=True)

    def _kokushi_robbers(self, player_id: int, tile: str) -> List[int]:
        """Opponents whose 13-orphan hand completes with `tile` (and who
        are not furiten): the only hands allowed to rob an ankan."""
        if tile[-1] != 'z' and tile[0] not in '19':
            return []
        out = []
        for p in range(4):
            if p == player_id or self._is_furiten(p):
                continue
            hand = self.hands[p]
            if len(hand) != 13 or self.melds[p]:
                continue
            if not all(t[-1] == 'z' or t[0] in '19' for t in hand):
                continue
            res = self._win_result(p, tile, is_tsumo=False, chankan=True)
            if res is not None and any("Kokushi" in str(y) for y in (res.yaku or [])):
                out.append(p)
        return out

    def resolve_pending_kan(self):
        """Nobody robbed the kan: complete it (RCR 3.5 / 3.7.1)."""
        if not self.pending_kan:
            return
        player_id, tile = self.pending_kan["player"], self.pending_kan["tile"]
        was_ankan = self.pending_kan.get("ankan", False)
        self.pending_kan = None
        self._apply_missed_ron()
        if was_ankan:
            self._do_ankan(player_id, tile)
            return
        pon = next((m for m in self.melds[player_id]
                    if m["type"] == "pon" and m["tiles"][0] == tile), None)
        if pon is None or tile not in self.hands[player_id]:
            return
        pon["red"] = pon.get("red", 0) + self._take_from_hand(player_id, tile, 1)
        pon["type"] = "shouminkan"
        pon["tiles"] = [tile] * 4
        self._after_kan(player_id)

    def _after_kan(self, player_id: int, reveal_now: bool = False):
        self.kan_count += 1
        self.kan_players.add(player_id)
        self.any_call = True
        self.ippatsu = [False, False, False, False]   # RCR 4.2.1.3
        if reveal_now:                 # ankan: immediately
            self._reveal_kan_dora()
        else:                          # daiminkan / shouminkan: after the discard
            self.pending_dora_reveal += 1
        self._rinshan_draw(player_id)
        self.hands[player_id].sort(key=sort_key)

    def _reveal_kan_dora(self):
        k = len(self.dora_indicators)
        if k < 5:
            self.dora_indicators.append(self.dead_wall[4 + k])
            self.ura_indicators.append(self.dead_wall[9 + k])

    def _rinshan_draw(self, player_id: int):
        if self._rinshan_idx >= 4 or not self.wall:
            return
        t = self.dead_wall[self._rinshan_idx]
        self._rinshan_idx += 1
        # RCR 3.7.1: the live wall's tail tile joins the dead wall, so the
        # round still yields exactly 70 draws in total.
        self.wall.pop(0)
        t = self._give(player_id, t)
        self.hands[player_id].sort(key=sort_key)
        self.last_drawn[player_id] = t
        self.rinshan[player_id] = True
        self.temp_furiten[player_id] = False

    def advance_turn(self):
        self._confirm_riichi()
        self._apply_missed_ron()
        if self.finished:                          # 四家立直 aborted in _confirm_riichi
            return self._obs(), True
        if self._pending_abort:
            self._abort(self._pending_abort)
            return self._obs(), True
        self.turn = (self.turn + 1) % 4
        if not self.wall:
            self._ryuukyoku()
            return self._obs(), True
        t = self._give(self.turn, self.wall.pop())
        self.hands[self.turn].sort(key=sort_key)
        self.last_drawn[self.turn] = t
        self.rinshan[self.turn] = False
        # RCR 3.13.2: same-turn furiten lifts once the player draws again.
        self.temp_furiten[self.turn] = False
        return self._obs(), False

    # ------------------------------------------------------------------
    # Riichi confirmation / missed-ron furiten
    # ------------------------------------------------------------------
    def _confirm_riichi(self):
        """The declaration tile survived: the 1000 stick is now paid."""
        if self.riichi_pending is None:
            return
        pid = self.riichi_pending
        self.points[pid] -= 1000
        self.kyotaku += 1000
        self.riichi_pending = None
        if all(self.riichi):
            self._abort("四家立直")

    def _void_riichi(self):
        """RCR 3.12: the declaration tile was ronned — riichi never was."""
        if self.riichi_pending is None:
            return
        pid = self.riichi_pending
        self.riichi[pid] = False
        self.ippatsu[pid] = False
        self.daburu[pid] = False
        self.riichi_pending = None
        if self.discards[pid] and self.discards[pid][-1].endswith('*'):
            self.discards[pid][-1] = self.discards[pid][-1][:-1]
        if self.river_events[pid]:
            self.river_events[pid][-1][2] = False
        self.riichi_turn[pid] = None

    def _apply_missed_ron(self):
        """RCR 3.13.2 / 3.13.3: passing up a ron makes you furiten —
        until your next draw/call, or permanently if you had declared."""
        for pid in self._ron_chance:
            if self.riichi[pid]:
                self.perm_furiten[pid] = True
            else:
                self.temp_furiten[pid] = True
        self._ron_chance = set()

    # ------------------------------------------------------------------
    # Interrupt phase
    # ------------------------------------------------------------------
    def step_ron(self, player_ids: Sequence[int]):
        """RCR 3.11: every player who declares a legal ron wins.

        `player_ids` must be ordered by seat distance from the discarder
        (the orchestrator collects them that way), which decides who takes
        the riichi sticks.
        """
        rewards = {i: 0.0 for i in range(4)}
        obs = lambda: self._obs()
        if self.pending_kan:
            tile = self.pending_kan["tile"]
            discarder = self.pending_kan["player"]
            chankan = True
        elif self.last_discard:
            tile = self.last_discard.replace('*', '')
            discarder = self.last_discarder
            chankan = False
        else:
            for pid in player_ids:
                rewards[pid] = self.ILLEGAL_PENALTY
            return obs(), rewards, False, {"interrupt": False, "winners": []}

        winners = []
        robbers = (self._kokushi_robbers(discarder, tile)
                   if self.pending_kan and self.pending_kan.get("ankan") else None)
        for pid in player_ids:
            result = (self._win_result(pid, tile, is_tsumo=False,
                                       chankan=chankan)
                      if not self._is_furiten(pid) else None)
            if robbers is not None and pid not in robbers:
                result = None                       # only 国士 may rob an ankan
            if result is None:
                rewards[pid] = self.ILLEGAL_PENALTY
            else:
                winners.append((pid, result))

        if not winners:
            return obs(), rewards, False, {"interrupt": False, "winners": []}

        self.pending_kan = None
        if len(winners) >= 3:                       # Majsoul: 三家和了 -> draw
            if self.riichi_pending == discarder:
                self._void_riichi()
            self._abort("三家和了")
            return obs(), rewards, True, {
                "interrupt": True, "winners": [w for w, _ in winners],
                "abort": "三家和了"}
        self._settle_ron(winners, discarder, chankan=chankan)
        return obs(), rewards, True, {
            "interrupt": True, "winners": [w for w, _ in winners],
        }

    def step_interrupt(self, player_id: int, action_xml: str):
        match = ACTION_RE.search(action_xml or "")
        rewards = {i: 0.0 for i in range(4)}
        obs = lambda: self._obs()

        if not match:
            rewards[player_id] = self.FORMAT_PENALTY
            return obs(), rewards, False, {"interrupt": False}

        action_type = match.group(1)
        if action_type == "ron":
            _, r, done, info = self.step_ron([player_id])
            return obs(), r, done, {"interrupt": info["interrupt"]}

        # Only a ron may interrupt an added kan (RCR 4.2.1.12).
        if self.pending_kan:
            if action_type != "skip":
                rewards[player_id] = self.ILLEGAL_PENALTY
            return obs(), rewards, False, {"interrupt": False}

        # SECURITY: the claimed tile is ALWAYS the actual last discard —
        # model-supplied tile attributes are never trusted here.
        tile = self.last_discard.replace('*', '') if self.last_discard else None
        if action_type == "skip" or tile is None:
            return obs(), rewards, False, {"interrupt": False}

        interrupted = False
        hand = self.hands[player_id]
        can_meld = (
            not self.riichi[player_id]
            and len(self.melds[player_id]) < self.MAX_MELDS
            # RCR 3.14: the houtei discard may only be ronned.
            and len(self.wall) > 0
        )

        if action_type == "pon" and can_meld and hand.count(tile) >= 2:
            used_red = self._take_from_hand(player_id, tile, 2)
            self.melds[player_id].append(
                {"type": "pon", "tiles": [tile] * 3, "opened": True,
                 "from": self.last_discarder,
                 "red": used_red + int(self.last_discard_red)}
            )
            self._record_pao(player_id, tile, self.last_discarder)
            self.kuikae = (player_id, self._kuikae_tiles(tile, []))
            self._claim_discard(player_id)
            interrupted = True

        elif (action_type == "kan" and can_meld and hand.count(tile) >= 3
                and self.kan_count < self.MAX_KANS):
            used_red = self._take_from_hand(player_id, tile, 3)
            self.melds[player_id].append(
                {"type": "kan", "tiles": [tile] * 4, "opened": True,
                 "from": self.last_discarder,
                 "red": used_red + int(self.last_discard_red)}
            )
            self._record_pao(player_id, tile, self.last_discarder)
            self._claim_discard(player_id)
            self._after_kan(player_id)
            interrupted = True

        elif action_type == "chi" and can_meld and tile[-1] != 'z' and (
            (self.last_discarder + 1) % 4 == player_id
        ):
            pairs = self._chi_pairs(player_id, tile)
            if pairs:
                wanted = (match.group(3) or "").split()
                chosen = next((p for p in pairs if sorted(p) == sorted(wanted)), pairs[0])
                used_red = 0
                for t in chosen:
                    used_red += self._take_from_hand(player_id, t, 1)
                seq = sorted(chosen + [tile], key=sort_key)
                self.melds[player_id].append(
                    {"type": "chi", "tiles": seq, "opened": True,
                     "from": self.last_discarder,
                     "red": used_red + int(self.last_discard_red)}
                )
                self.kuikae = (player_id, self._kuikae_tiles(tile, chosen))
                self._claim_discard(player_id)
                interrupted = True

        if not interrupted:
            rewards[player_id] = self.ILLEGAL_PENALTY

        hand.sort(key=sort_key)
        return obs(), rewards, False, {"interrupt": interrupted}

    def _claim_discard(self, player_id: int):
        self._pending_abort = None
        # A call does not void a riichi declaration (only a ron does).
        self._confirm_riichi()
        self._apply_missed_ron()
        if self.discards[self.last_discarder]:
            # Visible river only — furiten_river keeps the permanent record.
            self.discards[self.last_discarder].pop()
        if self.river_events[self.last_discarder]:
            self.river_events[self.last_discarder][-1][3] = True
        self.turn = player_id
        self.last_discard = None
        self.last_drawn[player_id] = None
        self.rinshan[player_id] = False
        self.any_call = True
        self.ippatsu = [False, False, False, False]   # RCR 4.2.1.3
        self.temp_furiten[player_id] = False

    def _record_pao(self, player_id: int, tile: str, feeder: Optional[int]):
        """RCR 4.2.5.10: the player who completes an opponent's third
        dragon set / fourth wind set by discard is liable for the yakuman."""
        if feeder is None or feeder == player_id:
            return
        i34 = str_to_34(tile)
        group = DRAGONS_34 if i34 in DRAGONS_34 else (
            WINDS_34 if i34 in WINDS_34 else None)
        if group is None:
            return
        owned = {str_to_34(m["tiles"][0]) for m in self.melds[player_id]
                 if str_to_34(m["tiles"][0]) in group}
        if len(owned) == len(group):
            self.pao[player_id] = feeder

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------
    def _pao_liable(self, winner: int, result) -> Optional[int]:
        if winner not in self.pao:
            return None
        names = {str(y) for y in (result.yaku or [])}
        return self.pao[winner] if names & PAO_YAKU else None

    def _settle_tsumo(self, winner: int, result):
        cost = result.cost
        liable = self._pao_liable(winner, result)
        if winner == self.dealer:
            total = cost['main'] * 3
        else:
            total = cost['main'] + cost['additional'] * 2
        if liable is not None:
            self.points[liable] -= total
            self.points[winner] += total
        elif winner == self.dealer:
            for i in range(4):
                if i != winner:
                    self.points[i] -= cost['main']
                    self.points[winner] += cost['main']
        else:
            for i in range(4):
                if i == winner:
                    continue
                pay = cost['main'] if i == self.dealer else cost['additional']
                self.points[i] -= pay
                self.points[winner] += pay

        self.points[winner] += self.kyotaku
        self.kyotaku = 0
        self.finished = True
        self.result_summary = (
            f"玩家{winner} 自摸 | {result.han}番{result.fu}符 | "
            f"{', '.join(str(y) for y in (result.yaku or []))}"
            f"{f' | 包牌:玩家{liable}' if liable is not None else ''} | "
            f"点数: {self.points}"
        )
        self._compute_final_rewards(houjuu_player=None)

    def _settle_ron(self, winners: List[Tuple[int, object]], discarder: int,
                    chankan: bool = False):
        # RCR 3.12: a riichi voided by this very ron never paid its stick.
        if self.riichi_pending == discarder:
            self._void_riichi()
        else:
            self._confirm_riichi()

        parts = []
        for pid, result in winners:
            cost = result.cost['main']
            liable = self._pao_liable(pid, result)
            if liable is not None and liable != discarder:
                # Liability is shared half/half with the discarder.
                half = cost // 2
                self.points[liable] -= half
                self.points[discarder] -= cost - half
            else:
                self.points[discarder] -= cost
            self.points[pid] += cost
            how = "抢杠" if chankan else "荣和"
            parts.append(
                f"玩家{pid} {how}(放铳:玩家{discarder}) | "
                f"{result.han}番{result.fu}符 | "
                f"{', '.join(str(y) for y in (result.yaku or []))}"
                f"{f' | 包牌:玩家{liable}' if liable is not None else ''}"
            )

        # RCR 3.12: the riichi sticks go to the winner sitting closest
        # counter-clockwise from the discarder (first in `winners`).
        self.points[winners[0][0]] += self.kyotaku
        self.kyotaku = 0
        self.finished = True
        self.result_summary = (
            (" ; ".join(parts))
            + (" | 双响" if len(winners) > 1 else "")
            + f" | 点数: {self.points}"
        )
        self._compute_final_rewards(houjuu_player=discarder)

    def _abort(self, reason: str):
        """途中流局 (Majsoul): no tenpai payments; sticks stay on the table."""
        if self.finished:
            return
        self._pending_abort = None
        self.finished = True
        self.result_summary = f"途中流局({reason}) | 点数: {self.points}"
        self._compute_final_rewards(houjuu_player=None)

    def _nagashi_mangan(self) -> List[int]:
        """Players whose every discard was a terminal/honor and none was called."""
        out = []
        for i in range(4):
            ev = self.river_events[i]
            if ev and all((e[0][-1] == 'z' or e[0][0] in '19') and not e[3] for e in ev):
                out.append(i)
        return out

    def _ryuukyoku(self):
        if self.finished:
            return
        nagashi = self._nagashi_mangan()
        if nagashi:
            for w in nagashi:
                for i in range(4):
                    if i == w:
                        continue
                    pay = 4000 if (w == self.dealer or i == self.dealer) else 2000
                    self.points[i] -= pay
                    self.points[w] += pay
            self.finished = True
            self.result_summary = (
                f"流局满贯 | 玩家{nagashi} | 点数: {self.points}")
            self._compute_final_rewards(houjuu_player=None)
            return
        tenpai = [
            self._shanten(self.hands[i], len(self.melds[i])) <= 0 for i in range(4)
        ]
        n = sum(tenpai)
        if 0 < n < 4:
            gain = 3000 // n
            loss = 3000 // (4 - n)
            for i in range(4):
                self.points[i] += gain if tenpai[i] else -loss
        self.finished = True
        self.result_summary = (
            f"流局 | 听牌: {[i for i in range(4) if tenpai[i]]} | 点数: {self.points}"
        )
        self._compute_final_rewards(houjuu_player=None)

    def _compute_final_rewards(self, houjuu_player):
        start = getattr(self, "start_points", None) or [25000] * 4
        self.final_rewards = [
            (self.points[i] - start[i]) * self.REWARD_SCALE for i in range(4)
        ]
        if houjuu_player is not None:
            self.final_rewards[houjuu_player] += self.HOUJUU_EXTRA

        # Placement bonus: players tied on points share the average bonus
        # of the positions they span (an all-25000 draw gives everyone 0).
        order = sorted(range(4), key=lambda i: -self.points[i])
        pos = 0
        while pos < 4:
            tied = [i for i in order if self.points[i] == self.points[order[pos]]]
            shared = sum(self.RANK_BONUS[pos:pos + len(tied)]) / len(tied)
            for i in tied:
                self.final_rewards[i] += shared
            pos += len(tied)
