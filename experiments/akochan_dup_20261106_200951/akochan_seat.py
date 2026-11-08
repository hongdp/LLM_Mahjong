"""akochan (github.com/critter-mj/akochan, (c) critter-mj; used unmodified, non-commercial research) as a seat in our
engine, modelled on scripts/arena_mortal_mjai.MortalSeat. One `system.exe pipe <tactics> <seat>` subprocess per seat:
we stream that seat's masked mjai view; akochan answers (one JSON action list per line) after its own tsumo and after
another seat's dahai / kakan. The list is consumed by successive engine decisions (e.g. [pon, dahai], [reach, dahai])."""
import json
import os
import subprocess

from scripts.arena_mortal_mjai import mask_for_seat
from src.agents.dnn.mjai_bridge import mjai_to_engine, mjai_to_engine_spelled
from tests.test_mjai_bridge import _norm

AKO = "/home/hongdp/Workspace/akochan_ext/akochan"
LIB = "/home/hongdp/miniconda3/envs/akochan/lib"


class AkochanSeat:
    def __init__(self, seat, stats, tactics, threads=None):
        threads = threads or int(os.environ.get("AKO_THREADS", "1"))
        self.seat, self.stats, self.pending, self.hist = seat, stats, [], []
        self.dump = os.environ.get("AKO_FALLBACK_LOG")   # 11-07: on fallback, dump this seat's event history
        env = dict(os.environ, LD_LIBRARY_PATH=f"{AKO}:{LIB}", OMP_NUM_THREADS=str(threads))
        self.p = subprocess.Popen([f"{AKO}/system.exe", "pipe", tactics, str(seat)], cwd=AKO, env=env, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1)
        self._send({"type": "start_game", "id": seat, "names": ["p0", "p1", "p2", "p3"], "kyoku_first": 4,
                    "aka_flag": True})

    def _send(self, ev):
        self.p.stdin.write(json.dumps(ev) + "\n")
        self.p.stdin.flush()

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:                       # noqa: BLE001
            self.p.kill()

    def feed(self, ev):
        m = mask_for_seat(ev, self.seat)
        self._send(m)
        if self.dump:
            self.hist.append(m)
            if m.get("type") == "start_kyoku":
                self.hist = [m]
        a, t = m.get("actor"), m.get("type")
        if (a == self.seat and t == "tsumo") or (a is not None and a != self.seat and t in ("dahai", "kakan")):
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"akochan seat {self.seat} died")
            self.pending = list(json.loads(line))
            self.stats["replies"] += 1

    def _to_xml(self, r):
        t = r.get("type")
        if t in (None, "none"):
            return '<action type="skip" />'
        if t == "hora":
            return '<action type="tsumo" />' if r.get("target") == self.seat else '<action type="ron" />'
        if t == "dahai":
            return f'<action type="discard" tile="{mjai_to_engine_spelled(r["pai"])}" />'
        if t == "reach":
            nxt = self.pending.pop(0) if self.pending and self.pending[0].get("type") == "dahai" else None
            return (f'<action type="riichi" tile="{mjai_to_engine_spelled(nxt["pai"])}" />' if nxt
                    else '<action type="skip" />')
        if t == "chi":
            a, b = [mjai_to_engine(c) for c in r["consumed"]]
            return f'<action type="chi" tile="{mjai_to_engine(r["pai"])}" with="{a} {b}" />'
        if t == "pon":
            return f'<action type="pon" tile="{mjai_to_engine(r["pai"])}" />'
        if t in ("daiminkan", "kakan"):
            return f'<action type="kan" tile="{mjai_to_engine(r["pai"])}" />'
        if t == "ankan":
            return f'<action type="kan" tile="{mjai_to_engine(r["consumed"][0])}" />'
        if t == "ryukyoku":
            return '<action type="kyuushu" />'
        return '<action type="skip" />'

    def policy(self, table, pid, actions):
        r = self.pending.pop(0) if self.pending else None
        xml = self._to_xml(r) if r is not None else '<action type="skip" />'
        if _norm([xml]) <= _norm(actions):
            self.stats["ok"] += 1
            return xml
        self.stats["fallback"] += 1
        if self.stats["fallback"] <= 10:
            print(f"[fallback] seat{pid} akochan {r} -> {xml} legal={actions[:6]}", flush=True)
        if self.dump:
            with open(self.dump, "a") as f:
                f.write(json.dumps({"seat": pid, "akochan": r, "xml": xml, "legal": actions,
                                    "engine_hand": table.display_hand(pid), "events": self.hist}) + "\n")
        drawn = table.last_drawn[pid]
        cand = f'<action type="discard" tile="{drawn}" />' if drawn else None
        if cand and _norm([cand]) <= _norm(actions):
            return cand
        skip = '<action type="skip" />'
        return skip if _norm([skip]) <= _norm(actions) else actions[0]
