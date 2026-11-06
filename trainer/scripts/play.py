#!/usr/bin/env python3
"""Play duck chess against the engine in your browser.

    python scripts\\play.py                                  # embedded network, 800 nodes per move
    python scripts\\play.py --net ..\\runs\\first\\latest.pdn --movetime 2000
    python scripts\\play.py --engine ..\\target\\release\\peaduckzero.exe --nodes 3000

Opens http://localhost:8800. Click a piece, then its target square, then the
duck's square. The rules come from the engine's own move generator
(peaduck_rs, `pip install ..\\crates\\py`); the engine plays over UCI.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import peaduck_rs as pz

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Engine:
    def __init__(self, path, options):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.send("uci")
        self.wait("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, prefix):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return line, lines

    def think(self, moves, go):
        with self.lock:
            self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
            self.send(go)
            best, lines = self.wait("bestmove")
        info = next((l for l in reversed(lines) if l.startswith("info depth")), "")
        return best.split()[1], info

    def new_game(self):
        with self.lock:
            self.send("ucinewgame")
            self.send("isready")
            self.wait("readyok")


class Game:
    def __init__(self):
        self.moves = []
        self.human_white = True
        self.last_info = ""

    def state(self):
        info = pz.position_info(None, self.moves)
        return {"moves": self.moves, "human_white": self.human_white, "info": self.last_info,
                "fen": info["fen"], "legal": info["legal"], "white_to_move": info["white_to_move"],
                "result": info["result"], "reason": info["reason"]}

    def engine_to_move(self):
        s = pz.position_info(None, self.moves)
        return s["result"] == "*" and s["white_to_move"] != self.human_white


def make_handler(game, engine, go):
    def engine_move():
        if game.engine_to_move():
            mv, info = engine.think(game.moves, go)
            game.moves.append(mv)
            game.last_info = info

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, obj, ctype="application/json"):
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state"):
                self.reply(game.state())
            elif self.path.startswith("/ducks"):
                # duck squares after the piece move ?m=e2e4
                m = self.path.split("m=")[-1]
                self.reply(pz.position_info(None, game.moves + [m]))
            else:
                self.reply(PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/move":
                try:
                    pz.position_info(None, game.moves + [req["move"]])
                except ValueError as e:
                    return self.reply({"error": str(e)})
                game.moves.append(req["move"])
                game.last_info = ""
            elif self.path == "/new":
                game.moves, game.last_info = [], ""
                game.human_white = req.get("white", True)
                engine.new_game()
            elif self.path == "/undo":
                # back to the human's previous turn
                game.moves = game.moves[:-2] if len(game.moves) >= 2 else []
                if game.engine_to_move():
                    game.moves = game.moves[:-1]
                game.last_info = ""
            elif self.path == "/think":
                engine_move()
            self.reply(game.state())

    return H


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>PeaDuckZero</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--light:#ecd9b9;--dark:#ae8a68;--sel:#f6f669;--last:#cdd26a;--bg:#1e1f22;--fg:#e6e6e6}
body{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:0;display:flex;gap:24px;
 padding:20px;flex-wrap:wrap;justify-content:center}
#board{display:grid;grid-template-columns:repeat(8,min(11vw,72px));grid-template-rows:repeat(8,min(11vw,72px));
 border:3px solid #444}
.sq{display:flex;align-items:center;justify-content:center;font-size:min(8vw,52px);cursor:pointer;position:relative;
 user-select:none;line-height:1}
.l{background:var(--light)}.d{background:var(--dark)}
.last{box-shadow:inset 0 0 0 100px rgba(155,199,0,.35)}
.sel{box-shadow:inset 0 0 0 100px rgba(246,246,105,.6)}
.dot::after{content:"";position:absolute;width:28%;height:28%;border-radius:50%;background:rgba(0,0,0,.25)}
.cap::after{content:"";position:absolute;inset:6%;border-radius:50%;border:5px solid rgba(0,0,0,.25)}
.w{color:#fff;text-shadow:0 0 2px #000,0 0 2px #000}.b{color:#111}
.coord{position:absolute;font-size:11px;left:3px;bottom:2px;color:rgba(0,0,0,.5)}
#side{max-width:380px;min-width:260px}
button,select{background:#333;color:var(--fg);border:1px solid #555;border-radius:6px;padding:7px 12px;margin:2px;
 cursor:pointer;font-size:14px}
#status{font-size:18px;margin:12px 0;font-weight:600}
#info{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;color:#aaa;margin:8px 0}
#moves{font-family:ui-monospace,monospace;font-size:13px;max-height:45vh;overflow:auto;background:#26272b;
 padding:8px;border-radius:6px}
</style></head><body>
<div id="board"></div>
<div id="side">
 <h2 style="margin:0 0 8px">PeaDuckZero</h2>
 <button onclick="newGame(true)">New game (White)</button><button onclick="newGame(false)">New game (Black)</button>
 <button onclick="post('/undo')">Take back</button><button onclick="flip=!flip;draw()">Flip</button>
 <div>Promote to <select id="promo"><option value="q">queen</option><option value="r">rook</option>
 <option value="b">bishop</option><option value="n">knight</option></select></div>
 <div id="status"></div><div id="info"></div><div id="moves"></div>
</div>
<script>
const G = {K:"♚",Q:"♛",R:"♜",B:"♝",N:"♞",P:"♟"};
let st = null, flip = false, sel = null, pending = null, ducks = [], thinking = false;
const sqName = (f, r) => "abcdefgh"[f] + (r + 1);
function boardOf(fen) {
  const b = {}; const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, i) => { let f = 0; for (const c of row) {
    if (/\d/.test(c)) f += +c; else { b[sqName(f, 7 - i)] = c; f++; } } });
  return b;
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", body: JSON.stringify(body)});
  return r.json();
}
async function post(path, body) { st = await api(path, body || {}); sel = pending = null; ducks = []; draw(); maybeThink(); }
async function newGame(white) { flip = !white; await post("/new", {white}); }
async function maybeThink() {
  if (!st || st.result !== "*" || st.white_to_move === st.human_white || thinking) return;
  thinking = true; draw(); st = await api("/think", {}); thinking = false; draw();
}
function lastSquares() {
  const m = st.moves[st.moves.length - 1]; if (!m) return [];
  const [a, b] = m.split(","); return [a.slice(0, 2), a.slice(2, 4), b ? b.slice(2, 4) : ""];
}
async function click(sq) {
  if (!st || st.result !== "*" || thinking || st.white_to_move !== st.human_white) return;
  if (pending) {                                    // placing the duck
    if (ducks.includes(sq)) { const m = pending; pending = null;
      await post("/move", {move: `${m},${m.slice(2, 4)}${sq}`}); }
    else { pending = null; ducks = []; draw(); }
    return;
  }
  const b = boardOf(st.fen);
  const mine = (p) => p && p !== "*" && (p === p.toUpperCase()) === st.white_to_move;
  if (sel) {
    let m = st.legal.find((x) => x.slice(0, 4) === sel + sq && (x.length === 4 || x[4] === document.getElementById("promo").value));
    if (m) {
      const info = await api("/ducks?m=" + m);
      if (info.result !== "*") { await post("/move", {move: m}); return; }   // king captured: no duck
      pending = m; ducks = info.legal; sel = null; draw(); return;
    }
    if (mine(b[sq])) { sel = sq; draw(); return; }
    sel = null; draw(); return;
  }
  if (mine(b[sq]) && st.legal.some((x) => x.startsWith(sq))) { sel = sq; draw(); }
}
function draw() {
  if (!st) return;
  const b = boardOf(st.fen); const el = document.getElementById("board"); el.innerHTML = "";
  if (pending) {  // show the piece move made, before the duck
    const from = pending.slice(0, 2), to = pending.slice(2, 4); b[to] = b[from]; delete b[from];
  }
  const last = lastSquares();
  const targets = sel ? st.legal.filter((x) => x.startsWith(sel)).map((x) => x.slice(2, 4)) : [];
  for (let i = 0; i < 64; i++) {
    const r = flip ? i >> 3 : 7 - (i >> 3), f = flip ? 7 - (i & 7) : i & 7, n = sqName(f, r);
    const d = document.createElement("div");
    d.className = "sq " + ((f + r) % 2 ? "l" : "d");
    if (last.includes(n)) d.classList.add("last");
    if (n === sel || (pending && n === pending.slice(2, 4))) d.classList.add("sel");
    const p = b[n];
    if (p === "*") d.textContent = "🦆";
    else if (p) { d.textContent = G[p.toUpperCase()]; d.classList.add(p === p.toUpperCase() ? "w" : "b"); }
    if (targets.includes(n) || ducks.includes(n)) d.classList.add(p && p !== "*" ? "cap" : "dot");
    if ((flip ? r === 7 : r === 0)) { const c = document.createElement("span"); c.className = "coord"; c.textContent = n; d.appendChild(c); }
    d.onclick = () => click(n); el.appendChild(d);
  }
  const s = document.getElementById("status");
  if (st.result !== "*") s.textContent = `Game over: ${st.result} (${st.reason})`;
  else if (thinking) s.textContent = "Engine is thinking…";
  else if (pending) s.textContent = "Place the duck";
  else s.textContent = st.white_to_move === st.human_white ? "Your move" : "Engine to move";
  document.getElementById("info").textContent = st.info;
  let t = ""; st.moves.forEach((m, k) => { if (k % 2 === 0) t += `${k / 2 + 1}. `; t += m + (k % 2 ? "\n" : "  "); });
  const mv = document.getElementById("moves"); mv.textContent = t; mv.scrollTop = 1e9;
}
api("/state").then((x) => { st = x; flip = !st.human_white; draw(); maybeThink(); });
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    exe = os.path.join(REPO, "target", "release", "peaduckzero" + (".exe" if os.name == "nt" else ""))
    ap.add_argument("--engine", default=exe, help="engine binary (default target/release/peaduckzero)")
    ap.add_argument("--net", help="network file (.pdn) for the engine (default: its embedded one)")
    ap.add_argument("--nodes", type=int, default=800, help="playouts per engine move (default 800)")
    ap.add_argument("--movetime", type=int, help="milliseconds per engine move instead of --nodes")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(args.engine):
        sys.exit(f"no engine at {args.engine}: build it with `cargo build --release`")
    opts = {"Threads": args.threads, "Hash": 64}
    if args.net:
        opts["EvalFile"] = os.path.abspath(args.net)
    engine = Engine(args.engine, opts)
    go = f"go movetime {args.movetime}" if args.movetime else f"go nodes {args.nodes}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Game(), engine, go))
    url = f"http://localhost:{args.port}"
    print(f"playing at {url} ({go}); Ctrl+C to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    engine.send("quit")


if __name__ == "__main__":
    main()
