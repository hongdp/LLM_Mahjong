"""Locks the Mortal observation encoder against a golden checksum set.

The encoder was optimised on 2026-08-25 (dict tile tables, hoisted dora set,
buffered vectorised writes). Every one of those is supposed to be a pure
speed change, so this test pins the exact bytes: the golden file was captured
from the pre-optimisation implementation, and any future optimisation that
perturbs a single value fails here rather than silently invalidating a
running comparison.
"""

import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.mahjong.table import PyMahjongTable          # noqa: E402
from src.agents.dnn.mortal_obs import encode_mortal_obs     # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data", "mortal_obs_golden.json")


def test_encoder_output_is_bit_identical_to_golden():
    expected = {(a, b, c, d): h for a, b, c, d, h in json.load(open(GOLDEN))}
    checked = 0
    for seed in range(12):
        random.seed(9000 + seed)
        t = PyMahjongTable(randomize_round=True)
        t.text_obs = False
        for step_i in range(25):
            if t.finished:
                break
            acts = t.get_legal_actions(t.turn)
            if not acts:
                break
            for pid in range(4):
                for der in (True, False):
                    key = (9000 + seed, step_i, pid, int(der))
                    if key not in expected:
                        continue
                    arr = encode_mortal_obs(t, pid, derived=der)
                    got = hashlib.sha1(
                        np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
                    assert got == expected[key], f"encoder output changed at {key}"
                    checked += 1
            _, _, done, info = t.step(t.turn, random.choice(acts))
            if done:
                break
            if info.get("discarded"):
                try:
                    if t.pending_kan:
                        t.resolve_pending_kan()
                    else:
                        _, rd = t.advance_turn()
                        if rd:
                            break
                except Exception:
                    break
    assert checked > 1500, f"golden set barely exercised ({checked} states)"
