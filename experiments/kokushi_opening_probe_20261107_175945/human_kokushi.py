"""Human reference for the terminal-heavy opening probe: Tenhou houou logs, regex over the raw XML. Per INIT, every
seat whose 13-tile start has K >= 8 kinds of terminals/honors: K after its first draw, whether it declared kyuushu
kyuuhai (RYUUKYOKU type=yao9 revealing that seat's hand), terminal/honor share of its first 6 discards, kokushi
shanten after them, won / won with kokushi.
usage: human_kokushi.py RAW_DIR OUT.json"""
import glob
import json
import os
import re
import sys

TAG = re.compile(r"<([A-Za-z]+)(\d*)\b([^>]*)>")
ATTR = re.compile(r"(\w+)=\"([^\"]*)\"")
YAO = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
DRAW, DISC = "TUVW", "DEFG"


def kinds(ids):
    return len({i // 4 for i in ids if i // 4 in YAO})


def kokushi_shanten(ids):
    ks = [i // 4 for i in ids if i // 4 in YAO]
    return 13 - len(set(ks)) - (1 if len(ks) > len(set(ks)) else 0)


def main():
    raw, out = sys.argv[1], sys.argv[2]
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "*", "*.mjlog"))):
        cur = None
        for m in TAG.finditer(open(f).read()):
            tag, num, attrs = m.group(1), m.group(2), m.group(3)
            if tag == "INIT":
                if cur:
                    rows += cur.values()
                at = dict(ATTR.findall(attrs))
                cur = {}
                for p in range(4):
                    h = [int(x) for x in at[f"hai{p}"].split(",")]
                    if kinds(h) >= 8:
                        cur[p] = {"K": kinds(h), "hand": h, "K_draw": None, "disc": [], "ks6": None,
                                  "kyuushu": False, "won": False, "kokushi_win": False}
                continue
            if not cur:
                continue
            if num and tag in DRAW:
                w = cur.get(DRAW.index(tag))
                if w:
                    w["hand"].append(int(num))
                    if w["K_draw"] is None:
                        w["K_draw"] = kinds(w["hand"])
            elif num and tag in DISC:
                w = cur.get(DISC.index(tag))
                if w:
                    w["hand"].remove(int(num))
                    w["disc"].append(int(num))
                    if len(w["disc"]) == 6:
                        w["ks6"] = kokushi_shanten(w["hand"])
            elif tag == "RYUUKYOKU":
                at = dict(ATTR.findall(attrs))
                if at.get("type") == "yao9":
                    for p, w in cur.items():
                        if f"hai{p}" in at:
                            w["kyuushu"] = True
            elif tag == "AGARI":
                at = dict(ATTR.findall(attrs))
                w = cur.get(int(at["who"]))
                if w:
                    w["won"] = True
                    w["kokushi_win"] = bool({"47", "48"} & set(at.get("yakuman", "").split(",")))
            elif tag == "N":
                pass
        if cur:
            rows += cur.values()
    for r in rows:
        d6 = r["disc"][:6]
        r["yao6"], r["n6"] = sum(1 for i in d6 if i // 4 in YAO), len(d6)
        del r["hand"], r["disc"]
    json.dump(rows, open(out, "w"))
    print(f"{len(rows)} human seat-deals with K >= 8")


if __name__ == "__main__":
    main()
