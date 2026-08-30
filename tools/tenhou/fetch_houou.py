"""Tenhou houou (凤凰卓) log downloader for exp45 human-BC.

Index: daily archives https://tenhou.net/sc/raw/dat/scc{YYYYMMDD}.html.gz
(list.cgi?old currently reaches back to 2026-01-01; older years ship as
scraw{year}.zip which we integrate only if the recent window proves too
small). Rooms are filtered to 四鳳南喰赤 (4p houou hanchan, calls + red
fives, game-type 00a9) — the distribution Suphx/Mortal trained on and the
closest to our engine rules.

Logs: https://tenhou.net/0/log/?{id} (XML mjlog) -> data/tenhou/raw/{date}/{id}.mjlog
Idempotent: existing files are skipped, so the script can be re-run to
resume. Politeness: one request per --throttle seconds (default 1.0).

Usage:
  python tools/tenhou/fetch_houou.py --start 20260601 --end 20260825 \
      --max_games 20000 [--index_only]
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.request

BASE = "https://tenhou.net"
UA = {"User-Agent": "Mozilla/5.0 (research; LLM_Mahjong exp45)"}
ROOM = "四鳳南喰赤"
LOG_RE = re.compile(r"log=([0-9a-z-]+)")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_DIR = os.path.normpath(os.path.join(ROOT, "data", "tenhou", "raw"))
INDEX = os.path.normpath(os.path.join(ROOT, "data", "tenhou", "index.jsonl"))


def _get(url: str, tries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def _read_index(url: str) -> str:
    """Fetch one scc archive; '' if missing (404 page / not gzip)."""
    try:
        raw = _get(url, tries=1)
        return gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "replace")
    except (RuntimeError, OSError):
        return ""


def day_ids(date: str, throttle: float = 0.5) -> list:
    """Houou 4p-hanchan log ids for YYYYMMDD.

    Consolidated daily archives live under dat/{year}/ (a few weeks behind);
    recent days exist only as hourly files dat/scc{date}{HH}.html.gz.
    """
    text = _read_index(f"{BASE}/sc/raw/dat/{date[:4]}/scc{date}.html.gz")
    if not text:
        text = _read_index(f"{BASE}/sc/raw/dat/scc{date}.html.gz")
    if not text:
        parts = []
        for hh in range(24):
            parts.append(_read_index(f"{BASE}/sc/raw/dat/scc{date}{hh:02d}.html.gz"))
            time.sleep(throttle)
        text = "\n".join(parts)
    ids = []
    for line in text.splitlines():
        if ROOM in line:
            m = LOG_RE.search(line)
            if m:
                ids.append(m.group(1))
    return ids


def fetch_log(log_id: str, date: str) -> str:
    """Download one mjlog XML; returns path (cached or fresh)."""
    d = os.path.join(RAW_DIR, date)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{log_id}.mjlog")
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return path
    data = _get(f"{BASE}/0/log/?{log_id}")
    if b"<mjloggm" not in data[:200]:
        raise RuntimeError(f"{log_id}: response is not an mjlog")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def daterange(start: str, end: str):
    import datetime as dt
    a = dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    b = dt.date(int(end[:4]), int(end[4:6]), int(end[6:]))
    while a <= b:
        yield a.strftime("%Y%m%d")
        a += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max_games", type=int, default=20000)
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--index_only", action="store_true",
                    help="only build the id index, download no logs")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    have = set()
    if os.path.exists(INDEX):
        with open(INDEX) as f:
            have = {json.loads(l)["id"] for l in f if l.strip()}

    n_dl, n_idx = 0, 0
    with open(INDEX, "a") as idx:
        for date in daterange(a.start, a.end):
            ids = day_ids(date)
            time.sleep(a.throttle)
            fresh = [i for i in ids if i not in have]
            n_idx += len(fresh)
            for log_id in fresh:
                idx.write(json.dumps({"id": log_id, "date": date, "room": ROOM}) + "\n")
            idx.flush()
            print(f"[{date}] {len(ids)} houou games ({len(fresh)} new)", flush=True)
            if a.index_only:
                continue
            for log_id in ids:
                if n_dl >= a.max_games:
                    print(f"[done] hit --max_games {a.max_games}")
                    return
                p = os.path.join(RAW_DIR, date, f"{log_id}.mjlog")
                if os.path.exists(p) and os.path.getsize(p) > 500:
                    continue
                try:
                    fetch_log(log_id, date)
                    n_dl += 1
                except RuntimeError as e:
                    print(f"[warn] {e}", file=sys.stderr, flush=True)
                time.sleep(a.throttle)
    print(f"[done] indexed +{n_idx}, downloaded {n_dl}")


if __name__ == "__main__":
    main()
