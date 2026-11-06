"""Entity registry (design D1/D8).

The rated thing is an ENTITY = (checkpoint, condition), not a "model":
bc49 playing greedy and bc49 sampling at T=1 are different opponents and
get different ratings. That turns v1's "sampling tax" from a permanent
footnote into a measured quantity — the gap between two entities.

Registering COPIES the checkpoint into experiments/_registry/ keyed by
its content hash. v1's epoch-6 board had five rows pointing at another
session's /tmp scratchpad; a board you cannot replay is not a board.
"""

import json
import os
import shutil
import subprocess
import time

from src.eval.fingerprints import file_sha

ROOT = "experiments/rating"
STORE = "experiments/_registry"
REG = f"{ROOT}/registry.json"


def _load():
    if os.path.exists(REG):
        return json.load(open(REG))
    return {"models": {}, "entities": {}}


def _save(d):
    os.makedirs(ROOT, exist_ok=True)
    tmp = REG + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, REG)


def entity_id(model_id, temperature):
    """Canonical, content-derived, and readable enough to grep."""
    return f"{model_id}.t{int(round(float(temperature) * 100)):03d}"


def describe_ckpt(path):
    """Architecture facts that decide who can share a table."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        return {}
    from src.agents.dnn.action_space import space_of_arch
    from src.agents.dnn.encoder import variant_of_arch
    arch = blob.get("arch") or ""
    sd = blob.get("model") or blob.get("state_dict") or {}
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) or None
    return {"arch": arch,
            "action_space": space_of_arch(arch),
            "encoder_variant": (blob.get("encoder_variant")
                                or variant_of_arch(arch)),
            "params": n,
            "channels": blob.get("channels"), "blocks": blob.get("blocks")}


def register_model(path, name, kind="dnn", lineage=None, provenance=None):
    """Idempotent by content: re-registering the same bytes is a no-op."""
    d = _load()
    mid = file_sha(path)
    stored = f"{STORE}/{mid}{os.path.splitext(path)[1] or '.pt'}"
    if mid in d["models"]:
        return mid
    os.makedirs(STORE, exist_ok=True)
    if not os.path.exists(stored):
        shutil.copy2(path, stored)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rec = {"id": mid, "name": name, "kind": kind, "path": stored,
           "lineage": lineage, "registered": time.strftime("%Y-%m-%d %H:%M:%S"),
           "provenance": dict(provenance or {}, src_path=path, git=git)}
    if kind == "dnn":
        rec.update(describe_ckpt(path))
    d["models"][mid] = rec
    _save(d)
    return mid


def register_entity(model_id, temperature, label=None):
    d = _load()
    if model_id not in d["models"]:
        raise KeyError(f"unregistered model {model_id}")
    eid = entity_id(model_id, temperature)
    if eid in d["entities"]:
        return eid
    nm = d["models"][model_id]["name"]
    label = label or f"{nm}@T{('%g' % float(temperature))}"
    taken = {e["label"] for e in d["entities"].values()}
    if label in taken:
        raise ValueError(f"duplicate entity label {label}")
    d["entities"][eid] = {"id": eid, "model": model_id,
                          "temperature": float(temperature), "label": label}
    _save(d)
    return eid


def add(path, name, temperatures=(0.0,), kind="dnn", lineage=None,
        provenance=None):
    mid = register_model(path, name, kind, lineage, provenance)
    return [register_entity(mid, t) for t in temperatures]


def entities():
    return _load()["entities"]


def models():
    return _load()["models"]


def resolve(label_or_id):
    d = _load()
    if label_or_id in d["entities"]:
        return d["entities"][label_or_id]
    for e in d["entities"].values():
        if e["label"] == label_or_id:
            return e
    raise KeyError(label_or_id)


def path_of(eid):
    d = _load()
    return d["models"][d["entities"][eid]["model"]]["path"]


def compatible(eids):
    """Every entity at one table must be hostable in one batch. Mixed
    action spaces ARE hosted natively (server pads to the pool max), so
    this only rejects what the worker genuinely cannot co-host."""
    d = _load()
    kinds = {d["models"][d["entities"][e]["model"]]["kind"] for e in eids}
    return kinds == {"dnn"}
