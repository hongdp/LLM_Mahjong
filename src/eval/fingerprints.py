"""Content fingerprints for every surface that can change what a match means.

v1 stamped only the five engine files, so two rule changes to the hanchan
driver (renchan on abortive draws, nagashi mangan) silently voided a day
of published hanchan numbers with no guard firing (exp56, 2026-08-30).
v2 stamps every surface and lets the FIT filter on them (design D4), so a
change excludes rows instead of nuking history.
"""

import hashlib
import os
import subprocess

SURFACES = {
    "engine": ["src/tasks/mahjong/table.py", "src/tasks/mahjong/shanten.py",
               "src/tasks/mahjong/claims.py", "src/tasks/mahjong/wrapper.py",
               "src/tasks/mahjong/arena.py"],
    # match-level rules: renchan/honba/kyotaku/nagashi/bust/uma
    "hanchan": ["src/tasks/mahjong/hanchan.py"],
    # what a model is allowed to say, and what it sees
    "action_space": ["src/agents/dnn/action_space.py"],
    "encoder": ["src/agents/dnn/encoder.py"],
}


def _digest(paths, rev=None):
    h = hashlib.sha256()
    for f in paths:
        if rev:
            blob = subprocess.run(["git", "show", f"{rev}:{f}"],
                                  capture_output=True).stdout
        else:
            blob = open(f, "rb").read() if os.path.exists(f) else b""
        h.update(f.encode() + b"\0" + blob + b"\0")
    return h.digest().hex()[:16]


def fingerprint(surface, rev=None):
    return _digest(SURFACES[surface], rev)


def all_fingerprints(rev=None):
    return {k: fingerprint(k, rev) for k in SURFACES}


def file_sha(path):
    """Content id for a checkpoint — the identity that survives renames."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
