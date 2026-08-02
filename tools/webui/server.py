"""Local web UI for inspecting training runs: metrics curves + rollout game viewer.

Stdlib HTTP server + tensorboard's EventAccumulator (already in the training env).

Run:
    conda run -n rlhf_mahjong python tools/webui/server.py [--port 8642]
Then open http://localhost:8642

Endpoints:
    GET  /api/experiments                 -> experiment dirs w/ available data
    GET  /api/metrics?exp=NAME            -> tensorboard scalars
    GET  /api/rollouts?exp=NAME&epoch=N   -> parsed games for one epoch
    POST /api/sync?exp=NAME               -> rsync live logs/tensorboard from the VM
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_ROOT = os.path.join(REPO, "experiments")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
VM_HOST = "mahjong-a100.us-central1-b.workstation-185016"
VM_EXP_ROOT = "LLM_Mahjong/experiments"
# Experiment-name prefix -> ssh host alias (longest prefix wins; VM_HOST is
# the fallback). Three concurrent A100 runs live on three VMs.
VM_HOSTS = {
    "v2_engine_ppo_value_run": "mahjong-a100-b.europe-west4-a.workstation-185016",
    "v2_engine_ppo_run": "mahjong-a100-e.us-east1-b.workstation-185016",
    "tune_": "mahjong-a100-e.us-east1-b.workstation-185016",
}


def host_for_exp(exp_name: str) -> str:
    for prefix in sorted(VM_HOSTS, key=len, reverse=True):
        if exp_name.startswith(prefix):
            return VM_HOSTS[prefix]
    return VM_HOST

# ---------------------------------------------------------------- rollout parsing

STEP_RE = re.compile(r'^\[Step (\d+)\] Reward: (-?[\d.]+) \| Terminal: (True|False)')
EPISODE_RE = re.compile(r'^--- Episode (\d+) \(Total Steps: (\d+)\) ---')
SEP_RE = re.compile(r'^-{40}$')
GLOBAL_RE = re.compile(
    r'场风: (\S+), 局数: (\S+), 宝牌指示牌: (\S+), 供托: (\d+), 剩余牌数: (\d+)')
PRIVATE_RE = re.compile(
    r'私有 \(Private\)： 自风: (\S+), 点数: (-?\d+), 手牌: ([^,]*), 副露: ([^\n]*)')
OPP_RE = re.compile(
    r'玩家(\d) \((\S)\): 点数: (-?\d+), 牌河: ([^,]*), 副露: ([^\n]*)')
LEGAL_RE = re.compile(r'^\s+- (<action [^\n]*/>)', re.M)
ACTION_TAG_RE = re.compile(r'<action type="(\w+)"(?: tile="(\w+)")?(?: with="([^"]*)")?\s*/?>')
THINK_RE = re.compile(r'<think>(.*?)(?:</think>|$)', re.S)
MELD_RE = re.compile(r'(\w+)\(([^)]*)\)')


def _parse_melds(text):
    return [{"type": m.group(1), "tiles": m.group(2).split()}
            for m in MELD_RE.finditer(text)]


def _parse_river(text):
    tiles = []
    for tok in text.split():
        if tok == "无":
            continue
        tiles.append({"t": tok.rstrip("*"), "riichi": tok.endswith("*")})
    return tiles


def _parse_prompt(prompt):
    state = {}
    if m := GLOBAL_RE.search(prompt):
        state["global"] = {"round_wind": m.group(1), "kyoku": m.group(2),
                           "dora_ind": m.group(3), "kyotaku": int(m.group(4)),
                           "wall": int(m.group(5))}
    if m := PRIVATE_RE.search(prompt):
        hand_txt, meld_txt = m.group(3).strip(), m.group(4)
        state["self"] = {
            "seat": m.group(1), "points": int(m.group(2)),
            "hand": [] if hand_txt in ("", "无") else hand_txt.split(),
            "melds": _parse_melds(meld_txt),
            # the state line appends ", 已立直" to the meld field when set
            # (value-facts variants may append more after it — substring check)
            "riichi": "已立直" in meld_txt,
        }
    opps = []
    for m in OPP_RE.finditer(prompt):
        meld_txt = m.group(5)
        opps.append({"idx": int(m.group(1)), "seat": m.group(2),
                     "points": int(m.group(3)),
                     "river": _parse_river(m.group(4)),
                     "melds": _parse_melds(meld_txt),
                     "riichi": "已立直" in meld_txt})
    state["opponents"] = opps
    state["legal"] = LEGAL_RE.findall(prompt)
    return state


def _parse_response(text):
    # The chat template pre-opens <think>, so the generation usually starts
    # with bare think content and only carries the closing tag.
    if "</think>" in text:
        think, _, visible = text.partition("</think>")
        think = think.replace("<think>", "").strip()
    elif m := THINK_RE.search(text):
        think, visible = m.group(1).strip(), THINK_RE.sub("", text)
    else:
        think, visible = "", text
    m = ACTION_TAG_RE.search(visible)
    a_type, a_tile, a_with = ("", "", "")
    if m:
        a_type, a_tile, a_with = m.group(1), m.group(2) or "", m.group(3) or ""
    return think, a_type, a_tile, a_with, not bool(m)


def parse_rollout_file(path):
    """Returns [{episode, steps:[...]}] — 4 consecutive episodes = 1 game."""
    episodes = []
    cur_ep = None
    cur = None          # dict for step under construction
    mode = None         # None | 'prompt' | 'action'
    prompt_lines, action_lines = [], []

    def close_step():
        nonlocal cur
        if cur is None:
            return
        prompt = "\n".join(prompt_lines)
        state = _parse_prompt(prompt)
        raw_resp = "\n".join(action_lines)
        think, a_type, a_tile, a_with, bad_format = _parse_response(raw_resp)
        cur.update(state)
        cur["think"] = think
        cur["action"] = {"type": a_type, "tile": a_tile, "with": a_with,
                         "bad_format": bad_format}
        cur_ep["steps"].append(cur)
        cur = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if m := EPISODE_RE.match(line):
                close_step()
                cur_ep = {"episode": int(m.group(1)), "steps": []}
                episodes.append(cur_ep)
            elif m := STEP_RE.match(line):
                close_step()
                cur = {"step": int(m.group(1)), "reward": float(m.group(2)),
                       "terminal": m.group(3) == "True"}
                prompt_lines, action_lines = [], []
                mode = None
            elif cur is not None:
                if line.startswith("PROMPT:"):
                    mode = "prompt"
                elif line.startswith("ACTION:"):
                    mode = "action"
                    action_lines.append(line[len("ACTION:"):].lstrip())
                elif SEP_RE.match(line):
                    mode = None
                elif mode == "prompt":
                    prompt_lines.append(line)
                elif mode == "action":
                    action_lines.append(line)
    close_step()
    return episodes


# ---------------------------------------------------------------- live rollout

LIVE_HDR_RE = re.compile(r'^=== \[(?:game (\d+) \| )?Player (\d)(?: \(Interrupt\))?\] ===$')
LIVE_END_RE = re.compile(r'^=== 对局结束(?: \(game (\d+)\))?: (.*) ===$')


def parse_live_file(path):
    """Parse the append-only live_rollout.txt written during an epoch's rollout.

    Block format (orchestrator._log_live):
        === [Player N] ===
        [INPUT PROMPT]:
        ...
        [MODEL OUTPUT]:
        ...
        [PARSED ACTION]: <action .../>
        ------------------------------------------------------------
    Games are separated by "=== 对局结束: ... ===" lines. No rewards here —
    those are computed by the trainer at epoch end.

    Returns [{"result": str|None, "episodes": [{episode, steps}]}]; the last
    game has result=None while still in progress.
    """
    finished = []           # completed games, in completion order
    open_games = {}         # gid -> {pid: [steps]} still in progress
    order = []              # gid first-seen order (for stable in-progress output)
    seq_gid = 0             # implicit id for legacy (sequential) logs
    cur_gid = None          # game of the block under construction
    pid = None              # player of the block under construction
    mode = None             # None | 'prompt' | 'raw'
    prompt_lines, raw_lines = [], []

    def close_block():
        nonlocal pid, cur_gid, mode, prompt_lines, raw_lines
        if pid is not None and prompt_lines and raw_lines:
            state = _parse_prompt("\n".join(prompt_lines))
            think, a_type, a_tile, a_with, bad = _parse_response(
                "\n".join(raw_lines))
            if cur_gid not in open_games:
                open_games[cur_gid] = {}
                order.append(cur_gid)
            steps = open_games[cur_gid].setdefault(pid, [])
            step = {"step": len(steps), "reward": 0.0, "terminal": False}
            step.update(state)
            step["think"] = think
            step["action"] = {"type": a_type, "tile": a_tile, "with": a_with,
                              "bad_format": bad}
            steps.append(step)
        pid, cur_gid, mode = None, None, None
        prompt_lines, raw_lines = [], []

    def finish_game(gid, result):
        players = open_games.pop(gid, None)
        if gid in order:
            order.remove(gid)
        if players:
            for steps in players.values():
                steps[-1]["terminal"] = result is not None
            finished.append({"result": result, "players": players})

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if m := LIVE_HDR_RE.match(line):
                close_block()
                # batched logs tag every block with its game; legacy logs
                # have a single implicit game at a time
                cur_gid = int(m.group(1)) if m.group(1) is not None else ("seq", seq_gid)
                pid = int(m.group(2))
            elif m := LIVE_END_RE.match(line):
                close_block()
                if m.group(1) is not None:
                    finish_game(int(m.group(1)), m.group(2))
                else:
                    finish_game(("seq", seq_gid), m.group(2))
                    seq_gid += 1
            elif pid is not None:
                if line.startswith("[INPUT PROMPT]:"):
                    mode = "prompt"
                elif line.startswith("[MODEL OUTPUT]:"):
                    mode = "raw"
                elif line.startswith("[PARSED ACTION]:") or re.fullmatch(r'-{40,}', line):
                    pass  # parsed tag is re-derived from the raw output
                elif mode == "prompt":
                    prompt_lines.append(line)
                elif mode == "raw":
                    raw_lines.append(line)
    close_block()

    games = []
    for entry in finished + [{"result": None, "players": open_games[g]}
                             for g in order if open_games.get(g)]:
        base = 4 * len(games)
        games.append({"result": entry["result"], "episodes": [
            {"episode": base + p, "steps": entry["players"][p]}
            for p in sorted(entry["players"])]})
    return games


# ---------------------------------------------------------------- metrics

def read_metrics(exp_name):
    tb_dir = os.path.join(EXP_ROOT, exp_name, "tensorboard")
    if not os.path.isdir(tb_dir):
        return {"error": "no tensorboard dir"}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
    except ImportError:
        return {"error": "tensorboard not importable — run inside rlhf_mahjong env"}
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [{"step": e.step, "value": e.value, "wall": e.wall_time}
                    for e in ea.Scalars(tag)]
    return out


# ---------------------------------------------------------------- experiments

def list_experiments():
    exps = []
    if not os.path.isdir(EXP_ROOT):
        return exps
    for name in sorted(os.listdir(EXP_ROOT), reverse=True):
        d = os.path.join(EXP_ROOT, name)
        if not os.path.isdir(d):
            continue
        epochs = sorted(
            int(m.group(1)) for f in os.listdir(d)
            if (m := re.match(r'mahjong_epoch_(\d+)_rollouts\.txt$', f)))
        exps.append({
            "name": name,
            "has_tensorboard": os.path.isdir(os.path.join(d, "tensorboard")),
            "has_experiment_md": os.path.isfile(os.path.join(d, "EXPERIMENT.md")),
            "has_live": os.path.isfile(os.path.join(d, "live_rollout.txt")),
            "rollout_epochs": epochs,
            "mtime": os.path.getmtime(d),
        })
    return exps


def sync_from_vm(exp_name):
    """Pull rollout logs + tensorboard for a (possibly running) remote experiment."""
    remote_dir = f"{VM_EXP_ROOT}/{exp_name}"
    host = host_for_exp(exp_name)
    try:
        chk = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             host, f"test -d {shlex.quote(remote_dir)}"],
            capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stderr": "连接 VM 超时"}
    if chk.returncode != 0:
        if chk.stderr.strip():
            return {"ok": False, "stderr": "无法连接 VM：\n" + chk.stderr[-2000:]}
        return {"ok": False,
                "stderr": f"VM 上没有该实验目录（{remote_dir}）。"
                          "这是一个本地实验，无需从 VM 同步。"}
    src = f"{host}:{remote_dir}/"
    dst = os.path.join(EXP_ROOT, exp_name) + "/"
    os.makedirs(dst, exist_ok=True)
    cmd = ["rsync", "-az", "--timeout=30",
           "--include=*.txt", "--include=*.json", "--include=*.log",
           "--include=tensorboard/", "--include=tensorboard/**",
           "--include=*.png", "--exclude=*", src, dst]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {"ok": r.returncode == 0, "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stderr": "rsync timed out"}


# ---------------------------------------------------------------- http plumbing

_rollout_cache = {}
_live_cache = {}
_cache_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):  # quieter console
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self._file(os.path.join(STATIC, "index.html"),
                       "text/html; charset=utf-8")
        elif u.path == "/api/experiments":
            self._json(list_experiments())
        elif u.path == "/api/metrics":
            exp = os.path.basename(q.get("exp", ""))
            self._json(read_metrics(exp))
        elif u.path == "/api/live":
            exp = os.path.basename(q.get("exp", ""))
            path = os.path.join(EXP_ROOT, exp, "live_rollout.txt")
            if not os.path.isfile(path):
                self._json({"error": "not found"}, 404)
                return
            key = (path, os.path.getmtime(path))
            with _cache_lock:
                if key not in _live_cache:
                    _live_cache.clear()
                    _live_cache[key] = parse_live_file(path)
                data = _live_cache[key]
            self._json({"live": True, "games": data})
        elif u.path == "/api/rollouts":
            exp = os.path.basename(q.get("exp", ""))
            epoch = int(q.get("epoch", "1"))
            path = os.path.join(EXP_ROOT, exp,
                                f"mahjong_epoch_{epoch}_rollouts.txt")
            if not os.path.isfile(path):
                self._json({"error": "not found"}, 404)
                return
            key = (path, os.path.getmtime(path))
            with _cache_lock:
                if key not in _rollout_cache:
                    _rollout_cache.clear()
                    _rollout_cache[key] = parse_rollout_file(path)
                data = _rollout_cache[key]
            self._json({"episodes": data})
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/sync":
            exp = os.path.basename(q.get("exp", ""))
            if not exp:
                self._json({"ok": False, "stderr": "missing exp"}, 400)
                return
            self._json(sync_from_vm(exp))
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address; use 127.0.0.1 to restrict to this machine")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"webui: http://{args.host}:{args.port}  (experiments root: {EXP_ROOT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
