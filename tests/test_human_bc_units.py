import os
import tempfile

from src.agents.dnn.human_bc_data import list_games, list_units, seat_ratings


def test_list_games_frozen_holdout():
    with tempfile.TemporaryDirectory() as d:
        names = [f"2026010{i}/g{i}.mjlog" for i in range(1, 6)]
        for n in names:
            os.makedirs(os.path.join(d, os.path.dirname(n)), exist_ok=True)
            open(os.path.join(d, n), "w").write(HEAD)
        lst = os.path.join(d, "hold.txt")
        open(lst, "w").write(names[1] + "\n" + names[3] + "\n")
        hold = list_games(d, holdout=True, holdout_list=lst)
        train = list_games(d, holdout=False, holdout_list=lst)
        assert [os.path.relpath(f, d) for f in hold] == [names[1], names[3]]
        assert [os.path.relpath(f, d) for f in train] == [names[0], names[2], names[4]]
        assert len(list_games(d)) == 5

HEAD = ('<mjloggm ver="2.3"><SHUFFLE seed="x"/><GO type="169" lobby="0"/>'
        '<UN n0="%41" n1="%42" n2="%43" n3="%44" dan="16,17,18,16" '
        'rate="2161.15,2137.39,2288.71,2118.66" sx="M,M,M,M"/><TAIKYOKU oya="0"/>')


def test_seat_ratings_and_units():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.mjlog")
        open(p, "w").write(HEAD)
        dans, rates = seat_ratings(p)
        assert dans == [16, 17, 18, 16]
        assert rates == [2161.15, 2137.39, 2288.71, 2118.66]
        assert list_units([p]) == [(p, 0), (p, 1), (p, 2), (p, 3)]
        assert list_units([p], min_rate=2150) == [(p, 0), (p, 2)]
        assert list_units([p], min_rate=2300) == []
        q = os.path.join(d, "noheader.mjlog")
        open(q, "w").write("<mjloggm ver=\"2.3\"/>")
        assert list_units([q], min_rate=2000) == []            # no header -> skipped
        assert list_units([q]) == [(q, s) for s in range(4)]   # unfiltered keeps it
