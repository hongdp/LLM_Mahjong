"""Local HTTP server exposing the DNN agent as an MJAI bot.

MahjongCopilot runs in its own Python environment (tkinter, mitmproxy,
playwright); this server keeps our torch stack on our side and talks
JSON over localhost. The MahjongCopilot-side client is
tools/majsoul_bridge/bot_llmmahjong.py.

API (all JSON):
  GET  /health                       -> {"ok": true, "ckpt": ..., "decisions": n}
  POST /start  {"seat": 0..3}        -> {"ok": true}        (new game; resets state)
  POST /react  {"msg": <mjai event>} -> {"reaction": <mjai reaction> | null}
  POST /react_batch {"msgs": [...]}  -> {"reaction": ...}   (only the last may act)
  GET  /last                         -> last decision (phase / legal / probs / value)
  GET  /panel                        -> assist-mode web panel (auto-refreshing)
  GET  /state                        -> panel payload (hand, dora, rivers, decision history)

Two ways to use it with MahjongCopilot:
  * auto-play : MC "enable automation" on  -> MC clicks the bot's reaction.
  * assist    : MC automation off; open http://127.0.0.1:8765/panel and
                play by hand. The panel shows the legal actions with the
                policy's probabilities, the sampled/greedy pick and V(s).
                --temperature > 0 samples (the pick is highlighted, the
                full distribution is always shown); 0 = greedy.

Record keeping (both modes):
  * --log  (JSONL): every raw event in/out, plus one "decision" line per
    decision once its outcome is known: state snapshot before acting,
    legal actions, policy probabilities, V(s), the policy's pick, our
    reaction, and `executed` = what actually happened at the table
    (`override`=true when a human played something else in assist mode).
  * --games-dir: one self-contained JSON per game (all kyokus, decisions,
    liqi round results, final result) written at end_game / next start.

Usage:
  PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt <path.pt> --port 8765
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dnn.mjai_bridge import MjaiDnnBot, load_policy   # noqa: E402


class State:
    def __init__(self, ckpt, device, temperature, log_path, name=None):
        self.ckpt = ckpt
        self.temperature = temperature
        # human-readable model name shown in MahjongCopilot's model bar:
        # "<experiment dir>/<file>" unless --name is given
        self.name = name or "/".join(os.path.normpath(ckpt).split(os.sep)[-2:])
        self.policy = load_policy(ckpt, device, temperature)
        self.bot = MjaiDnnBot(self.policy)
        self.lock = threading.Lock()
        self.log = open(log_path, "a") if log_path else None
        self.started = time.time()
        self.history = []          # decisions of the current kyoku (panel)
        self.games_dir = None
        self.bot.on_game_end = self._write_game

    def _write_game(self, rec):
        if not self.games_dir:
            return
        os.makedirs(self.games_dir, exist_ok=True)
        path = os.path.join(self.games_dir, f"game_{time.strftime('%Y%m%d_%H%M%S')}_seat{rec['seat']}.json")
        clean = {**rec, "kyokus": [{**ky, "decisions": [{k: v for k, v in d.items() if k != "_logged"}
                                                          for d in ky["decisions"]]} for ky in rec["kyokus"]]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, default=str)
        self.record("game_file", path)

    def flush_decisions(self):
        """Log decisions whose outcome is now known (durable even if the
        process dies before the game file is written)."""
        for ky in (self.bot.game_record or {}).get("kyokus", []):
            for d in ky["decisions"]:
                if d["executed"] is not None and not d.get("_logged"):
                    d["_logged"] = True
                    self.record("decision", {k: v for k, v in d.items() if k != "_logged"})

    def snapshot(self):
        bot, tb = self.bot, self.bot.table
        if tb is None or not bot.in_kyoku:
            return {"in_kyoku": False, "history": self.history[-12:], "decisions": bot.n_decisions}
        me = bot.seat
        from src.tasks.mahjong.shanten import dora_from_indicator
        return {
            "in_kyoku": True, "seat": me, "dealer": tb.dealer, "turn": tb.turn,
            "hand": list(tb.my_tiles_mjai), "drawn": tb.my_drawn_mjai,
            "dora": [dora_from_indicator(i) for i in tb.dora_indicators],
            "melds": [[m["type"], m["tiles"]] for m in tb.melds[me]],
            "rivers": [tb.discards[p] for p in range(4)],
            "riichi": list(tb.riichi), "points": list(tb.points), "wall": len(tb.wall),
            "last_discard": tb.last_discard, "last_discarder": tb.last_discarder,
            "history": self.history[-12:], "decisions": bot.n_decisions,
        }

    def note_decision(self):
        d = self.bot.last_decision
        if d and (not self.history or self.history[-1]["n"] != self.bot.n_decisions):
            probs = sorted(d["probs"].items(), key=lambda kv: -kv[1])
            self.history.append({"n": self.bot.n_decisions, "phase": d["phase"], "chosen": d["chosen"],
                                 "value": round(d["value"], 3),
                                 "probs": [(a, round(p, 4)) for a, p in probs]})

    def record(self, kind, payload):
        if self.log:
            self.log.write(json.dumps({"t": time.time(), "kind": kind, "data": payload},
                                      ensure_ascii=False) + "\n")
            self.log.flush()


PANEL_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>LLM_Mahjong assist</title>
<style>
body{font-family:system-ui,sans-serif;background:#151a1e;color:#e6e6e6;margin:0;padding:14px}
h1{font-size:16px;margin:0 0 8px;color:#9fd}
.row{display:flex;gap:18px;flex-wrap:wrap}
.card{background:#1f262c;border-radius:8px;padding:10px 14px;min-width:260px}
.tile{display:inline-block;font:bold 15px/1 system-ui;margin:1px;padding:6px 4px;background:#fafafa;color:#111;border-radius:4px;min-width:22px;text-align:center}
.tile.red{color:#c22}.tile.drawn{outline:3px solid #ffb300}.tile.p{color:#1a5fb4}.tile.s{color:#26a269}.tile.z{color:#333}
.bar{height:14px;background:#3a7;border-radius:3px}
.act{display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;margin:2px 0}
.act.pick{background:#2e4a3a;border-radius:4px}
.small{color:#9aa;font-size:12px}
table{border-collapse:collapse}td{padding:1px 8px}
</style></head><body>
<h1>LLM_Mahjong assist panel <span id=st class=small></span></h1>
<div class=row>
 <div class=card><div class=small>hand (drawn tile outlined)</div><div id=hand></div>
   <div class=small>melds: <span id=melds></span> &nbsp; dora: <span id=dora></span></div>
   <div class=small id=info></div></div>
 <div class=card style="flex:1"><div class=small>latest decision — policy distribution (highlighted = sampled/greedy pick)</div><div id=dec></div></div>
</div>
<div class=card style="margin-top:12px"><div class=small>this kyoku</div><table id=hist></table></div>
<script>
const HON={'1z':'\u6771','2z':'\u5357','3z':'\u897f','4z':'\u5317','5z':'\u767d','6z':'\u767c','7z':'\u4e2d',E:'\u6771',S:'\u5357',W:'\u897f',N:'\u5317',P:'\u767d',F:'\u767c',C:'\u4e2d'};
const ORD={m:0,p:1,s:2,z:3,E:30,S:31,W:32,N:33,P:34,F:35,C:36};
function key(t){if(HON[t]&&t.length===1)return ORD[t];const n=parseInt(t[0]);return ORD[t[1]]*10+n+(t.endsWith('r')?-0.5:0)}
function uni(t){return HON[t]||(t[0]+{m:'\u842c',p:'\u7b52',s:'\u7d22'}[t[1]])}
function tile(t,cls){const suit=HON[t]?'z':t[1];return '<span class="tile '+suit+' '+(cls||'')+(t.endsWith('r')?' red':'')+'" title="'+t+'">'+uni(t)+'</span>'}
function actLabel(x){const m=/type="(\\w+)"(?:[^>]*tile="([^"]+)")?(?:[^>]*with="([^"]+)")?/.exec(x);if(!m)return x;
 let s=m[1];if(m[2])s+=' '+tile(m[2]);if(m[3])s+=' ('+m[3].split(' ').map(t=>tile(t)).join('')+')';return s}
async function tick(){try{const r=await fetch('/state');const s=await r.json();
 document.getElementById('st').textContent=(s.in_kyoku?'in kyoku · seat '+s.seat+' · wall '+s.wall:'idle')+' · decisions '+s.decisions;
 if(s.in_kyoku){const h=s.hand.slice().sort((a,b)=>key(a)-key(b));let drawnShown=false;
  document.getElementById('hand').innerHTML=h.map(t=>{const c=(!drawnShown&&t===s.drawn)?'drawn':'';if(c)drawnShown=true;return tile(t,c)}).join('');
  document.getElementById('melds').innerHTML=s.melds.map(m=>m[0]+':'+m[1].map(t=>tile(t)).join('')).join(' ')||'–';
  document.getElementById('dora').innerHTML=s.dora.map(t=>tile(t)).join('');
  document.getElementById('info').textContent='points '+s.points.join(' / ')+' · riichi '+s.riichi.map(x=>x?1:0).join('')+' · last discard '+(s.last_discard||'–')+' by '+(s.last_discarder??'–')+' · turn '+s.turn;}
 const hist=s.history;const d=hist[hist.length-1];
 if(d){document.getElementById('dec').innerHTML='<div class=small>#'+d.n+' '+d.phase+' · V(s)='+d.value+'</div>'+
   d.probs.map(([a,p])=>'<div class="act'+(a===d.chosen?' pick':'')+'"><span>'+actLabel(a)+'</span><div class=bar style="width:'+(p*100).toFixed(1)+'%"></div><span>'+(p*100).toFixed(1)+'%</span></div>').join('');}
 document.getElementById('hist').innerHTML=hist.slice().reverse().map(x=>'<tr><td>#'+x.n+'</td><td>'+x.phase+'</td><td>'+actLabel(x.chosen)+'</td><td>'+(x.probs[0]?(x.probs.find(q=>q[0]===x.chosen)||[0,0])[1]*100:0).toFixed(0)+'%</td><td>V '+x.value+'</td></tr>').join('');
 }catch(e){document.getElementById('st').textContent='server unreachable';}}
setInterval(tick,400);tick();
</script></body></html>"""


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, fmt, *args):      # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": state.ckpt, "model_name": state.name,
                                 "mode": "greedy" if state.temperature <= 0 else f"sample T={state.temperature:g}",
                                 "decisions": state.bot.n_decisions,
                                 "uptime_s": round(time.time() - state.started)})
            elif self.path == "/last":
                self._send(200, state.bot.last_decision or {})
            elif self.path == "/state":
                with state.lock:
                    self._send(200, state.snapshot())
            elif self.path in ("/", "/panel"):
                body = PANEL_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
                with state.lock:
                    if self.path == "/start":
                        seat = int(body["seat"])
                        state.flush_decisions()
                        state.bot.start_game(seat)
                        state.record("start", body)
                        self._send(200, {"ok": True, "seat": seat})
                    elif self.path == "/react":
                        msg = body["msg"]
                        state.record("in", msg)
                        if msg.get("type") == "start_kyoku":
                            state.history.clear()
                        r = state.bot.react(msg)
                        state.note_decision()
                        state.flush_decisions()
                        state.record("out", r)
                        self._send(200, {"reaction": r})
                    elif self.path == "/react_batch":
                        msgs = body["msgs"]
                        state.record("in_batch", msgs)
                        if any(m.get("type") == "start_kyoku" for m in msgs):
                            state.history.clear()
                        r = state.bot.react_batch(msgs)
                        state.note_decision()
                        state.flush_decisions()
                        state.record("out", r)
                        self._send(200, {"reaction": r})
                    else:
                        self._send(404, {"error": "not found"})
            except Exception as e:             # never leave the client hanging
                state.record("error", repr(e))
                self._send(500, {"error": repr(e)})
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (argmax); >0 samples")
    ap.add_argument("--log", default="experiments/majsoul_sessions/mjai_session.jsonl")
    ap.add_argument("--name", default=None,
                    help="model label shown in MahjongCopilot (default: <exp dir>/<file>)")
    ap.add_argument("--games-dir", default=None,
                    help="per-game JSON records (default: <log dir>/games)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.log:
        os.makedirs(os.path.dirname(a.log), exist_ok=True)
    state = State(a.ckpt, a.device, a.temperature, a.log, a.name)
    state.games_dir = a.games_dir or (os.path.join(os.path.dirname(a.log), "games") if a.log else None)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    print(f"[serve_mjai_bot] {state.name} ({a.ckpt}) on http://{a.host}:{a.port}  log={a.log}  games={state.games_dir}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
