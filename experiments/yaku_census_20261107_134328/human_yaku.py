"""Human reference for the yaku census: Tenhou houou logs (data/tenhou/raw/*/*.mjlog), regex over the raw XML.
Per deal (INIT) and per win (AGARI): Tenhou yaku ids -> the engine's yaku names (style_stats spelling), dora /
ura / aka counted once per win when they score (ura is listed with 0 han when it misses), yakuman from the
`yakuman` attribute (Tenhou has no double yakuman), kazoe = >= 13 han without one.
usage: human_yaku.py RAW_DIR OUT.json"""
import glob
import json
import os
import re
import sys

TAG = re.compile(r"<(INIT|AGARI)\b([^>]*)>")
ATTR = re.compile(r"(\w+)=\"([^\"]*)\"")
W = ["east", "south", "west", "north"]
NAMES = {0: "Menzen Tsumo", 1: "Riichi", 2: "Ippatsu", 3: "Chankan", 4: "Rinshan Kaihou", 5: "Haitei Raoyue",
         6: "Houtei Raoyui", 7: "Pinfu", 8: "Tanyao", 9: "Iipeiko",
         **{10 + i: f"Yakuhai (seat wind {w})" for i, w in enumerate(W)},
         **{14 + i: f"Yakuhai (round wind {w})" for i, w in enumerate(W)},
         18: "Yakuhai (haku)", 19: "Yakuhai (hatsu)", 20: "Yakuhai (chun)", 21: "Double Riichi", 22: "Chiitoitsu",
         23: "Chantai", 24: "Ittsu", 25: "Sanshoku Doujun", 26: "Sanshoku Doukou", 27: "San Kantsu", 28: "Toitoi",
         29: "San Ankou", 30: "Shou Sangen", 31: "Honroutou", 32: "Ryanpeikou", 33: "Junchan", 34: "Honitsu",
         35: "Chinitsu", 36: "Renhou (yakuman)", 37: "Tenhou", 38: "Chiihou", 39: "Daisangen", 40: "Suu Ankou",
         41: "Suu Ankou Tanki", 42: "Tsuu Iisou", 43: "Ryuuiisou", 44: "Chinroutou", 45: "Chuuren Poutou",
         46: "Daburu Chuuren Poutou", 47: "Kokushi Musou", 48: "Kokushi Musou Juusanmen Matchi", 49: "Dai Suushii",
         50: "Shousuushii", 51: "Suu Kantsu", 52: "Dora", 53: "Ura Dora", 54: "Aka Dora"}


def main():
    raw, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(raw, "*", "*.mjlog")))
    c = {"_deals": 0, "_wins": 0, "_yakuman": 0, "_kazoe": 0, "_hanchan": len(files)}
    for f in files:
        for m in TAG.finditer(open(f).read()):
            if m.group(1) == "INIT":
                c["_deals"] += 1
                continue
            at = dict(ATTR.findall(m.group(2)))
            c["_wins"] += 1
            if at.get("yakuman"):
                c["_yakuman"] += 1
                for i in at["yakuman"].split(","):
                    n = NAMES.get(int(i), f"id{i}")
                    c[n] = c.get(n, 0) + 1
                continue
            v = [int(x) for x in at.get("yaku", "").split(",") if x != ""]
            han = 0
            for i, h in zip(v[0::2], v[1::2]):
                han += h
                if h > 0:
                    n = NAMES.get(i, f"id{i}")
                    c[n] = c.get(n, 0) + 1
            if han >= 13:
                c["_kazoe"] += 1
    json.dump(dict(sorted(c.items())), open(out, "w"), indent=1)
    print(f"{c['_hanchan']} hanchan {c['_deals']} deals {c['_wins']} wins yakuman {c['_yakuman']} kazoe {c['_kazoe']}")


if __name__ == "__main__":
    main()
