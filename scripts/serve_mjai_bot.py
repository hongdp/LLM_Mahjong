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

Every event and reaction is appended to --log (JSONL) so a Majsoul
session can be replayed offline and audited.

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
    def __init__(self, ckpt, device, temperature, log_path):
        self.ckpt = ckpt
        self.policy = load_policy(ckpt, device, temperature)
        self.bot = MjaiDnnBot(self.policy)
        self.lock = threading.Lock()
        self.log = open(log_path, "a") if log_path else None
        self.started = time.time()

    def record(self, kind, payload):
        if self.log:
            self.log.write(json.dumps({"t": time.time(), "kind": kind, "data": payload},
                                      ensure_ascii=False) + "\n")
            self.log.flush()


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
                self._send(200, {"ok": True, "ckpt": state.ckpt,
                                 "decisions": state.bot.n_decisions,
                                 "uptime_s": round(time.time() - state.started)})
            elif self.path == "/last":
                self._send(200, state.bot.last_decision or {})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
                with state.lock:
                    if self.path == "/start":
                        seat = int(body["seat"])
                        state.bot.start_game(seat)
                        state.record("start", body)
                        self._send(200, {"ok": True, "seat": seat})
                    elif self.path == "/react":
                        msg = body["msg"]
                        state.record("in", msg)
                        r = state.bot.react(msg)
                        state.record("out", r)
                        self._send(200, {"reaction": r})
                    elif self.path == "/react_batch":
                        msgs = body["msgs"]
                        state.record("in_batch", msgs)
                        r = state.bot.react_batch(msgs)
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
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.log:
        os.makedirs(os.path.dirname(a.log), exist_ok=True)
    state = State(a.ckpt, a.device, a.temperature, a.log)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    print(f"[serve_mjai_bot] {a.ckpt} on http://{a.host}:{a.port}  log={a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
