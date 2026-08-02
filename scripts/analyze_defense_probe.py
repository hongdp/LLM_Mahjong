"""Defense probe: fold-rate and deal-in-rate after an opponent declares riichi.

Parses per-epoch rollout transcripts (``mahjong_epoch_N_rollouts.txt``) written
by ``save_trajectory_log`` and reports, per epoch:

  - riichi-exposed steps: decision points where >=1 opponent shows a riichi
    declaration tile (marked ``*``) in their river and we ourselves have not
    declared riichi;
  - genbutsu rate (弃和 proxy): fraction of exposed discards that are 100% safe
    against every riichi opponent (tile already present in that opponent's
    river — the engine has no one-tile-per-row suji model, genbutsu is the
    honest lower bound of deliberate folding);
  - push rate: fraction of exposed discards that are NOT genbutsu;
  - deal-in rate (放铳率): among games with >=1 riichi exposure, the fraction
    where an exposed player's last discard fed a ron (reconstructed by pairing
    the ron winner's terminal step with the payer trajectory whose final
    discard matches the winning tile).

Usage:
    python scripts/analyze_defense_probe.py experiments/<run>/mahjong_epoch_*_rollouts.txt
"""

import argparse
import glob
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

STEP_RE = re.compile(r'^\[Step (\d+)\] Reward: (-?[\d.]+) \| Terminal: (True|False)')
EPISODE_RE = re.compile(r'^--- Episode (\d+) \(Total Steps: (\d+)\) ---')
HAND_RE = re.compile(r'手牌: ((?:[0-9][mpsz] )*[0-9][mpsz])')
WALL_RE = re.compile(r'剩余牌数: (\d+)')
SELF_RIICHI_RE = re.compile(r'私有[^\n]*立直')
RIVER_RE = re.compile(r'玩家\d \([东南西北]\): 点数: -?\d+, 牌河: ([^,]*), 副露')
ACTION_RE = re.compile(r'<action type="(\w+)"(?: tile="(\w+)")?')


@dataclass
class Step:
    reward: float
    terminal: bool
    wall: int
    exposed: bool           # >=1 opponent riichi visible, self not riichi
    genbutsu_rivers: list   # rivers of riichi opponents (tile lists, * stripped)
    action_type: str = ""
    action_tile: str = ""


@dataclass
class Episode:
    steps: list = field(default_factory=list)


def parse_file(path):
    """Returns list of Episodes in file order (4 consecutive episodes = 1 game)."""
    episodes = []
    cur_ep = None
    cur_step = None
    prompt_lines = []
    in_prompt = False

    def close_step():
        nonlocal cur_step
        if cur_step is None:
            return
        prompt = "\n".join(prompt_lines)
        cur_step.wall = int(m.group(1)) if (m := WALL_RE.search(prompt)) else -1
        self_riichi = bool(SELF_RIICHI_RE.search(prompt))
        riichi_rivers = []
        for river in RIVER_RE.findall(prompt):
            if '*' in river:
                tiles = [t.rstrip('*') for t in river.split() if t != '无']
                riichi_rivers.append(tiles)
        cur_step.exposed = bool(riichi_rivers) and not self_riichi
        cur_step.genbutsu_rivers = riichi_rivers
        cur_ep.steps.append(cur_step)
        cur_step = None

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if m := EPISODE_RE.match(line):
            close_step()
            cur_ep = Episode()
            episodes.append(cur_ep)
        elif m := STEP_RE.match(line):
            close_step()
            cur_step = Step(reward=float(m.group(2)), terminal=m.group(3) == "True",
                            wall=-1, exposed=False, genbutsu_rivers=[])
            prompt_lines = []
            in_prompt = False
        elif cur_step is not None:
            if line.startswith("PROMPT:"):
                in_prompt = True
            elif line.startswith("ACTION:"):
                in_prompt = False
                if m := ACTION_RE.search(line):
                    cur_step.action_type, cur_step.action_tile = m.group(1), m.group(2) or ""
            elif in_prompt:
                prompt_lines.append(line)
            elif not cur_step.action_type and (m := ACTION_RE.search(line)):
                # action tag on a continuation line of the generated text
                cur_step.action_type, cur_step.action_tile = m.group(1), m.group(2) or ""
    close_step()
    return episodes


def analyze(paths):
    header = (f"{'epoch':>5} | {'exposed':>7} | {'genbutsu%':>9} | {'push%':>6} | "
              f"{'exp.games':>9} | {'deal-in%':>8} | {'wins':>4} | {'draws':>5}")
    print(header)
    print("-" * len(header))

    for path in sorted(paths, key=lambda p: int(re.search(r'epoch_(\d+)_', p).group(1))):
        epoch = int(re.search(r'epoch_(\d+)_', path).group(1))
        episodes = parse_file(path)
        games = [episodes[i:i + 4] for i in range(0, len(episodes) - len(episodes) % 4, 4)]

        exposed_discards = genbutsu_discards = 0
        exposed_games = dealin_games = wins = draws = 0

        for game in games:
            game_exposed = False
            # winner trajectory: last step is ron/tsumo
            ron_tile, ron_wall, winner = None, None, None
            for ep in game:
                if not ep.steps:
                    continue
                last = ep.steps[-1]
                if last.action_type == "ron":
                    ron_tile, ron_wall, winner = last.action_tile, last.wall, ep
                elif last.action_type == "tsumo":
                    winner = ep
            if winner is not None:
                wins += 1
            else:
                draws += 1

            for ep in game:
                for st in ep.steps:
                    if st.exposed and st.action_type in ("discard", "riichi") and st.action_tile:
                        exposed_discards += 1
                        if all(st.action_tile in river for river in st.genbutsu_rivers):
                            genbutsu_discards += 1
                if any(st.exposed for st in ep.steps):
                    game_exposed = True

            if game_exposed:
                exposed_games += 1
                # deal-in: some exposed trajectory's final discard == ron tile,
                # at the same wall count as the ron (same table moment).
                if ron_tile:
                    for ep in game:
                        if ep is winner or not ep.steps:
                            continue
                        last = ep.steps[-1]
                        if (last.exposed and last.action_type in ("discard", "riichi")
                                and last.action_tile == ron_tile
                                and (ron_wall is None or abs(last.wall - ron_wall) <= 1)):
                            dealin_games += 1
                            break

        gr = 100 * genbutsu_discards / exposed_discards if exposed_discards else float("nan")
        dr = 100 * dealin_games / exposed_games if exposed_games else float("nan")
        print(f"{epoch:>5} | {exposed_discards:>7} | {gr:>8.1f}% | {100 - gr:>5.1f}% | "
              f"{exposed_games:>9} | {dr:>7.1f}% | {wins:>4} | {draws:>5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="mahjong_epoch_N_rollouts.txt files (globs ok)")
    args = ap.parse_args()
    paths = [p for pat in args.logs for p in glob.glob(pat)]
    if not paths:
        sys.exit("no rollout logs matched")
    analyze(paths)
