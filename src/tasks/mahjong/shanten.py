from typing import List, Set
from mahjong.shanten import Shanten

_SUIT_BASE_34 = {'m': 0, 'p': 9, 's': 18, 'z': 27}


def _tile_to_34(tile: str) -> int:
    return _SUIT_BASE_34[tile[-1]] + int(tile[:-1]) - 1


def _tile_from_34(idx: int) -> str:
    if idx < 9:
        return f"{idx + 1}m"
    if idx < 18:
        return f"{idx - 9 + 1}p"
    if idx < 27:
        return f"{idx - 18 + 1}s"
    return f"{idx - 27 + 1}z"


def pad_for_melds(tiles: List[str], num_melds: int) -> List[str]:
    """Pad one dummy completed triplet per meld (using tile types absent
    from the hand) so shanten math treats an open hand as full-size."""
    if num_melds == 0:
        return list(tiles)
    used = {_tile_to_34(t) for t in tiles}
    pad_types = [i for i in range(33, -1, -1) if i not in used]
    padded = list(tiles)
    for m in range(num_melds):
        padded.extend([_tile_from_34(pad_types[m])] * 3)
    return padded


class TileEfficiency:
    def __init__(self):
        self.shanten_calc = Shanten()
        # 34 indices for all mahjong tiles (0-8: man, 9-17: pin, 18-26: sou, 27-33: honors)
        self.all_tiles = list(range(34))

    def _string_to_34_array(self, tiles: List[str]) -> List[int]:
        """
        Converts a list of Tenhou strings (e.g., ['1m', '2p', '1z']) to a 34-array.
        """
        tiles_34 = [0] * 34
        for t in tiles:
            if not t or len(t) != 2:
                continue
            val = int(t[0]) - 1
            suit = t[1]
            if suit == 'm':
                idx = val
            elif suit == 'p':
                idx = 9 + val
            elif suit == 's':
                idx = 18 + val
            elif suit == 'z':
                idx = 27 + val
            else:
                continue
            tiles_34[idx] += 1
        return tiles_34

    def _34_index_to_string(self, idx: int) -> str:
        if idx < 9:
            return f"{idx+1}m"
        elif idx < 18:
            return f"{idx-9+1}p"
        elif idx < 27:
            return f"{idx-18+1}s"
        else:
            return f"{idx-27+1}z"

    def calculate_shanten(self, tiles: List[str]) -> int:
        tiles_34 = self._string_to_34_array(tiles)
        return self.shanten_calc.calculate_shanten(tiles_34)

    def calculate_ukeire(self, tiles: List[str]) -> List[str]:
        """
        Returns a list of tiles that will reduce the shanten.
        """
        tiles_34 = self._string_to_34_array(tiles)
        current_shanten = self.shanten_calc.calculate_shanten(tiles_34)
        
        # Already win
        if current_shanten < 0:
            return []

        ukeire_tiles = []
        for i in range(34):
            if tiles_34[i] < 4:
                tiles_34[i] += 1
                new_shanten = self.shanten_calc.calculate_shanten(tiles_34)
                if new_shanten < current_shanten:
                    ukeire_tiles.append(self._34_index_to_string(i))
                tiles_34[i] -= 1
                
        return ukeire_tiles

    def evaluate_discards_ranked(self, tiles: List[str]) -> dict:
        """
        For a 14-tile-equivalent hand: { discard_tile: (post_shanten, ukeire) }.

        Correct tile efficiency ranks by (lowest post-discard shanten,
        then widest ukeire). Ranking by ukeire count alone is WRONG:
        looser (higher-shanten) hands accept more tile types, so a pure
        ukeire-greedy policy systematically discards joint tiles and
        regresses the hand.
        """
        tiles_34 = self._string_to_34_array(tiles)
        result = {}
        for i in range(34):
            if tiles_34[i] > 0:
                tiles_34[i] -= 1
                base = self.shanten_calc.calculate_shanten(tiles_34)
                ukeire = []
                for j in range(34):
                    if tiles_34[j] < 4:
                        tiles_34[j] += 1
                        if self.shanten_calc.calculate_shanten(tiles_34) < base:
                            ukeire.append(self._34_index_to_string(j))
                        tiles_34[j] -= 1
                result[self._34_index_to_string(i)] = (base, ukeire)
                tiles_34[i] += 1
        return result

    def evaluate_discards(self, tiles: List[str]) -> dict:
        """
        Evaluates the ukeire for all possible legal discards.
        Returns: { 'discard_tile': ['ukeire_tile1', 'ukeire_tile2', ...] }
        """
        tiles_34 = self._string_to_34_array(tiles)
        current_shanten = self.shanten_calc.calculate_shanten(tiles_34)
        
        candidates = {}
        for i in range(34):
            if tiles_34[i] > 0:
                # Try discarding tile i
                tiles_34[i] -= 1
                
                # After discard, we must calculate the base shanten for 13 tiles
                base_shanten = self.shanten_calc.calculate_shanten(tiles_34)
                
                ukeire = []
                # Find all tiles that would improve this base_shanten
                for j in range(34):
                    if tiles_34[j] < 4:
                        tiles_34[j] += 1
                        new_shanten = self.shanten_calc.calculate_shanten(tiles_34)
                        if new_shanten < base_shanten:
                            ukeire.append(self._34_index_to_string(j))
                        tiles_34[j] -= 1
                        
                candidates[self._34_index_to_string(i)] = ukeire
                tiles_34[i] += 1
                
        return candidates
