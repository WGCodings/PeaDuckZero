#!/usr/bin/env python3
"""Background strength evaluation of a training run, on the CPU.

Every --every generations (a "milestone"): gen N plays gen N - every, --games
games with the engine binary at --tc (default 8+0.08) through the match
runner (tools/duckmatch: random openings, each played twice with colours
swapped). Results are appended to <run>/evals.jsonl and shown on the
dashboard. Milestones already in evals.jsonl are skipped, so it catches up
after a restart and can backfill an old run:

    python -m peaduck.evaluator --run runs/first --once     # evaluate what exists, then exit
    python -m peaduck.evaluator --run runs/first            # keep watching for new networks

train.py starts it automatically (see --eval-every there). It needs the
engine and the match runner built: `cargo build --release`.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback

TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINER_DIR)
EXE = ".exe" if os.name == "nt" else ""


def find_binary(name, given=None):
    """`given`, else target/release/<name> in the repository, else None."""
    if given:
        return given if os.path.isfile(given) else None
    path = os.path.join(REPO_ROOT, "target", "release", name + EXE)
    return path if os.path.isfile(path) else None


def net_files(run):
    """{generation: path} of <run>/nets/gen_XXXX.pdn."""
    out = {}
    d = os.path.join(run, "nets")
    if os.path.isdir(d):
        for name in os.listdir(d):
            m = re.fullmatch(r"gen_(\d+)\.pdn", name)
            if m:
                out[int(m.group(1))] = os.path.join(d, name)
    return out


def read_evals(run):
    out = []
    path = os.path.join(run, "evals.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def append_eval(run, record):
    with open(os.path.join(run, "evals.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


class Status:
    """<run>/eval_status.json: what the evaluator is doing (dashboard)."""

    def __init__(self, run, settings):
        self.path = os.path.join(run, "eval_status.json")
        self.settings = settings

    def set(self, state, task=None):
        data = {"state": state, "task": task, "updated": time.time(), **self.settings}
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


# Set when train.py has finished: evaluate what is left, then exit.
FINISHING = threading.Event()


def stable(path, age=10.0):
    """A network file the trainer has finished writing."""
    if FINISHING.is_set():
        return True
    try:
        return time.time() - os.path.getmtime(path) > age
    except OSError:
        return False


def pending(args, nets, evals):
    mode = args.tc
    done = {(e["generation"], e["opponent"]) for e in evals if e.get("type") == "match" and e.get("mode") == mode}
    todo = []
    for g in sorted(nets):
        if g % args.every or not stable(nets[g]):
            continue
        opp = g - args.every
        if args.games > 0 and opp in nets and (g, opp) not in done:
            todo.append(g)
    return todo


def run_match(args, nets, g, status):
    opp = g - args.every
    t = time.time()
    cmd = [args.duckmatch,
           "--engine", f"cmd={args.engine}", f"name=gen{g}", f"option.EvalFile={nets[g]}",
           "--engine", f"cmd={args.engine}", f"name=gen{opp}", f"option.EvalFile={nets[opp]}",
           "--each", f"tc={args.tc}", "option.Hash=16", "option.Threads=1",
           "--games", str(args.games), "--concurrency", str(args.concurrency),
           "--random-turns", str(args.random_turns), "--seed", str(args.seed), "--json"]
    status.set("running", f"gen {g} vs {opp}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    result = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("final", True):
                result = r
        elif line.startswith("Finished game"):
            status.set("running", f"gen {g} vs {opp}: {line.split(' (')[0][9:]}")
    proc.wait()
    if result is None:
        raise RuntimeError(f"duckmatch failed (exit {proc.returncode})")
    rec = {"type": "match", "generation": g, "opponent": opp, "mode": args.tc,
           "games": result["games"], "wins": result["wins"], "draws": result["draws"],
           "losses": result["losses"], "score": round((result["wins"] + 0.5 * result["draws"]) / result["games"], 4),
           "elo": round(result["elo"], 1), "elo_lo": round(result["elo"] - result["elo_error"], 1),
           "elo_hi": round(result["elo"] + result["elo_error"], 1),
           "pentanomial": result["pentanomial"], "terminations": result["terminations"],
           "avg_plies": result.get("avg_turns"),
           "seconds": round(time.time() - t, 1), "time": round(time.time(), 1)}
    bad = {k: v for k, v in rec["terminations"].items()
           if k in ("TimeForfeit", "EngineError") or k.startswith("IllegalMove")}
    print(f"[eval] gen {g} vs {opp} ({args.tc}): {rec['elo']:+.0f} Elo [{rec['elo_lo']:+.0f}, {rec['elo_hi']:+.0f}] "
          f"(+{rec['wins']} ={rec['draws']} -{rec['losses']}, {rec['seconds']:.0f}s)"
          + (f" WARNING {bad}" if bad else ""), flush=True)
    return rec


def watch_parent():
    """train.py holds our stdin open: a line "finish" = training is done,
    evaluate the rest and exit; end of file = exit now."""
    def wait():
        try:
            for line in sys.stdin:
                if line.strip() == "finish":
                    FINISHING.set()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=wait, daemon=True).start()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=25, help="milestone interval in generations")
    ap.add_argument("--games", type=int, default=100, help="match games per milestone (0 = off)")
    ap.add_argument("--tc", default="8+0.08", help="time control, base+increment seconds")
    ap.add_argument("--engine", help="engine binary (default target/release/peaduckzero)")
    ap.add_argument("--duckmatch", help="match runner (default target/release/duckmatch)")
    ap.add_argument("--concurrency", type=int, default=4, help="games at the same time")
    ap.add_argument("--random-turns", type=int, default=4, help="random turns of every opening")
    ap.add_argument("--seed", type=int, default=0, help="opening seed (same openings at every milestone)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between looks for new networks")
    ap.add_argument("--once", action="store_true", help="evaluate what exists now, then exit")
    ap.add_argument("--parent-stdin", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.parent_stdin:
        watch_parent()
    status = Status(args.run, {"every": args.every, "games": args.games, "mode": args.tc,
                               "threads": args.concurrency, "puzzles": 0})
    failures = {}
    warned = False
    while True:
        args.engine = find_binary("peaduckzero", args.engine) or args.engine
        args.duckmatch = find_binary("duckmatch", args.duckmatch) or args.duckmatch
        if not (args.engine and os.path.isfile(args.engine) and args.duckmatch and os.path.isfile(args.duckmatch)):
            if not warned:
                print("[eval] engine or duckmatch not built (cargo build --release); waiting", flush=True)
                warned = True
            if args.once:
                return
            status.set("waiting", "engine not built")
            FINISHING.wait(args.poll)
            continue
        nets = net_files(args.run)
        todo = [g for g in pending(args, nets, read_evals(args.run)) if failures.get(g, 0) < 3]
        if not todo:
            if args.once or FINISHING.is_set():
                break
            status.set("waiting")
            FINISHING.wait(args.poll)
            continue
        g = todo[0]
        try:
            append_eval(args.run, run_match(args, nets, g, status))
        except Exception:
            traceback.print_exc()
            failures[g] = failures.get(g, 0) + 1
            print(f"[eval] match for gen {g} failed; {'retrying' if failures[g] < 3 else 'giving up on it'}",
                  flush=True)
            status.set("error", f"gen {g}")
            FINISHING.wait(args.poll)
    status.set("idle")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
