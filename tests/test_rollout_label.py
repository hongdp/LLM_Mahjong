import copy
import random

from scripts.rollout_label import all_tiles_multiset, resample_hidden
from src.tasks.mahjong.table import PyMahjongTable


def _table(seed):
    random.seed(seed)
    t = PyMahjongTable(randomize_round=True)
    t.text_obs = False
    return t


def test_resample_preserves_tiles_and_visible_state():
    for seed in (1, 2, 3):
        t = _table(seed)
        # advance a few plain discards so rivers are non-empty
        for _ in range(6):
            pid = t.turn
            acts = [a for a in t.get_legal_actions(pid) if 'type="discard"' in a]
            _, _, done, info = t.step(pid, acts[0])
            if done:
                break
            for off in range(1, 4):
                other = (pid + off) % 4
                t.get_interrupt_actions(other)
            t.advance_turn()
        me = t.turn
        before = all_tiles_multiset(t)
        my_hand = list(t.hands[me]); my_red = dict(t.red[me])
        rivers = {p: list(t.discards[p]) for p in range(4)}
        dora = list(t.dora_indicators)
        u = resample_hidden(copy.deepcopy(t), me, random.Random(99))
        assert all_tiles_multiset(u) == before            # 136-tile multiset invariant
        assert u.hands[me] == my_hand and u.red[me] == my_red
        assert {p: u.discards[p] for p in range(4)} == rivers
        assert u.dora_indicators == dora and u.dead_wall[4] == t.dead_wall[4]
        assert len(u.wall) == len(t.wall)
        for p in range(4):
            assert len(u.hands[p]) == len(t.hands[p])
            assert sum(u.red[p].values()) == sum(1 for x in u.display_hand(p) if x[0] == "0")
