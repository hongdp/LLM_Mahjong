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


def test_sparse_write_log_densifies_bit_identically():
    """The compact transport format must reproduce the dense observation
    exactly, on every device, deterministically.

    Two real bugs this pins, both found on 2026-08-25:
      * float16 values lost the rescaled-score / RBF / river-decay planes;
      * a GPU scatter with duplicate indices picked a race-dependent winner,
        so the log is deduplicated to last-write-wins at construction.
    """
    import torch
    from src.agents.dnn.mortal_obs import encode_mortal_obs_sparse, densify

    batch, expect = [], []
    for seed in range(6):
        random.seed(500 + seed)
        t = PyMahjongTable(randomize_round=True)
        t.text_obs = False
        for _ in range(15):
            if t.finished:
                break
            acts = t.get_legal_actions(t.turn)
            if not acts:
                break
            for pid in range(4):
                for der in (True, False):
                    batch.append(encode_mortal_obs_sparse(t, pid, derived=der))
                    expect.append(encode_mortal_obs(t, pid, derived=der))
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
    assert len(batch) > 200
    exp = np.stack(expect)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        for _ in range(2):          # repeat: catches non-deterministic scatter
            got = densify(batch, device=dev).cpu().numpy()
            assert np.array_equal(got, exp), f"densify mismatch on {dev}"

    # and it must actually be small, or it is not worth the machinery
    sparse_bytes = sum(o.nbytes() for o in batch)
    dense_bytes = exp.nbytes
    assert dense_bytes / sparse_bytes > 50, (
        f"compression only {dense_bytes / sparse_bytes:.0f}x")
