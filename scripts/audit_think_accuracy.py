import sys, os, re, glob
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tools/webui")
from server import parse_rollout_file
from src.tasks.mahjong.shanten import TileEfficiency, pad_for_melds
te = TileEfficiency()
CLAIM = re.compile(r'打(\S+?)→(\d)向听/受入(\d+)种')

def audit(run, tag, epoch):
    files = glob.glob(f"experiments/{run}/mahjong_epoch_{epoch}_rollouts.txt")
    if not files: return
    eps = parse_rollout_file(files[0])
    total = correct_sh = correct_uk = 0
    examples = []
    for e in eps:
        for s in e["steps"]:
            hand = s.get("self", {}).get("hand", [])
            if len(hand) % 3 != 2: continue
            n_melds = len(s.get("self", {}).get("melds", []))
            claims = CLAIM.findall(s.get("think", ""))
            if not claims: continue
            try:
                ranked = te.evaluate_discards_ranked(pad_for_melds(hand, n_melds))
            except Exception:
                continue
            for tile, sh_c, uk_c in claims:
                if tile not in ranked or tile not in hand: continue
                sh_t, uk_t = ranked[tile][0], len(ranked[tile][1])
                total += 1
                if int(sh_c) == sh_t: correct_sh += 1
                if int(uk_c) == uk_t: correct_uk += 1
                if len(examples) < 3 and (int(sh_c) != sh_t or int(uk_c) != uk_t):
                    examples.append(f"    手牌[{' '.join(hand)}] 声明:打{tile}→{sh_c}向听/{uk_c}种  真值:{sh_t}向听/{uk_t}种")
    if total:
        print(f"{tag} ep{epoch}: 声明数={total}  向听正确率={correct_sh/total:.0%}  受入正确率={correct_uk/total:.0%}")
        for x in examples: print(x)

audit("v2_engine_ppo_value_run_20260802_054921", "ARM-B", 1)
audit("v2_engine_pbrs_run_20260802_054918", "BASE", 1)
audit("v2_engine_pbrs_run_20260802_054918", "BASE", 20)
audit("v2_engine_ppo_value_run_20260802_054921", "ARM-B", 24)
# 对照:SFT 语料本身的忠实度(教师生成,应接近100%)
import json, random
lines = open("data/sft_mahjong.jsonl").readlines()
random.seed(0); total = ok_sh = ok_uk = 0
for ln in random.sample(lines, 300):
    d = json.loads(ln)
    txt = d["text"]
    hm = re.search(r'手牌: ((?:[1-9][mpsz] )*[1-9][mpsz])', txt)
    fm = re.search(r'副露: ([^\n]*)', txt)
    if not hm: continue
    hand = hm.group(1).split()
    if len(hand) % 3 != 2: continue
    n_melds = fm.group(1).count('(') if fm else 0
    for tile, sh_c, uk_c in CLAIM.findall(txt):
        if tile not in hand: continue
        try:
            ranked = te.evaluate_discards_ranked(pad_for_melds(hand, n_melds))
        except Exception: continue
        if tile not in ranked: continue
        sh_t, uk_t = ranked[tile][0], len(ranked[tile][1])
        total += 1; ok_sh += int(sh_c)==sh_t; ok_uk += int(uk_c)==uk_t
print(f"SFT语料(教师) 抽样: 声明数={total}  向听正确率={ok_sh/max(total,1):.0%}  受入正确率={ok_uk/max(total,1):.0%}")
