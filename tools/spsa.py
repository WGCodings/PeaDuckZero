#!/usr/bin/env python3
"""SPSA tuning of the engine's search parameters (OpenBench's method),
with games played by duckmatch.

    python tools\\spsa.py --name cpuct --params CPuct,CPuctExponent,FpuReduction,FpuReductionRoot `
        --net runs\\big\\nets\\gen_0200.pdn --tc 8+0.08 --games 20000 --concurrency 7
    python tools\\spsa.py --name cpuct                 # continue a stopped tune (same settings)
    python tools\\spsa.py --name cpuct --show          # current values, in OpenBench format

Parameters: the engine's `spsa` command lists every tunable one in
OpenBench's format (name, int, value, min, max, c_end, r_end). --params picks
some of them by name; --config FILE takes lines in that format instead (copy
them from `peaduckzero.exe` → `spsa` and edit values, ranges or c_end).

Each iteration plays `--batch` game pairs between θ+cδ and θ−cδ (δ = random
±1 per parameter, both engines the same binary and network, every opening
twice with colours swapped), then moves θ by r·c·(wins − losses)·δ. The step
sizes follow OpenBench: c_k = c / (k+1)^0.101, a_k = a / (A+k+1)^0.602 with
A = 0.1·pairs, c = c_end·pairs^0.101, a = r_end·c_end²·(A+pairs)^0.602, and
r_k = a_k / c_k² (k counts game pairs).

State is saved after every iteration in .spsa/<name>.json (Ctrl+C is safe;
run the same --name again to continue); the parameter history goes to
.spsa/<name>.csv. When it ends, copy the printed values into the defaults
in engine/src/search/params.rs and confirm the result with an SPRT.
"""
import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPSA_DIR = os.path.join(REPO, ".spsa")
EXE = ".exe" if os.name == "nt" else ""
ALPHA, GAMMA, A_RATIO = 0.602, 0.101, 0.1


def engine_params(engine):
    """The engine's tunable parameters: [(name, value, min, max, c_end, r_end)]."""
    out = subprocess.run([engine], input="uci\nspsa\nquit\n", capture_output=True, text=True, timeout=60).stdout
    return parse_lines(out.splitlines())


def parse_lines(lines):
    params = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7 and parts[1] == "int":
            name, _, value, lo, hi, c_end, r_end = parts
            params.append({"name": name, "value": float(value), "min": float(lo), "max": float(hi),
                           "c_end": float(c_end), "r_end": float(r_end)})
    return params


def build(target):
    r = subprocess.run(["cargo", "build", "--release", "-p", target], cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"building {target} failed")


def new_state(args):
    engine = os.path.abspath(args.engine)
    if args.config:
        params = parse_lines(open(args.config).read().splitlines())
    else:
        params = engine_params(engine)
        if args.params:
            want = [p.strip() for p in args.params.split(",") if p.strip()]
            known = {p["name"] for p in params}
            unknown = [w for w in want if w not in known]
            if unknown:
                sys.exit(f"unknown parameters: {', '.join(unknown)} (the engine's `spsa` command lists them)")
            params = [p for p in params if p["name"] in want]
    if not params:
        sys.exit("no parameters to tune")
    pairs = args.games // 2
    A = A_RATIO * pairs
    for p in params:
        p["start"] = p["value"]
        p["c"] = p["c_end"] * pairs ** GAMMA
        p["a"] = p["r_end"] * p["c_end"] ** 2 * (A + pairs) ** ALPHA
    return {
        "name": args.name, "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "engine": engine, "net": os.path.abspath(args.net) if args.net else None,
        "limit": f"nodes={args.nodes}" if args.nodes else f"tc={args.tc}",
        "hash": args.hash, "threads": args.threads, "concurrency": args.concurrency,
        "batch": args.batch or args.concurrency, "pairs_total": pairs, "A": A,
        "pairs_done": 0, "iteration": 0, "wins": 0, "losses": 0, "draws": 0,
        "params": params,
    }


def save(state):
    path = os.path.join(SPSA_DIR, f"{state['name']}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)


def show(state):
    print(f"{state['name']}: {state['pairs_done']:,} / {state['pairs_total']:,} pairs, iteration {state['iteration']}"
          f" ({state['limit']}, W {state['wins']} L {state['losses']} D {state['draws']} for θ+)")
    print(f"{'parameter':22} {'start':>9} {'now':>9} {'change':>8}   {'min':>7} {'max':>7}")
    for p in state["params"]:
        print(f"{p['name']:22} {p['start']:9.1f} {p['value']:9.1f} {p['value'] - p['start']:+8.1f}   "
              f"{p['min']:7.0f} {p['max']:7.0f}")
    print("OpenBench format (paste as new defaults after confirming with an SPRT):")
    for p in state["params"]:
        print(f"{p['name']}, int, {round(p['value'])}, {p['min']:g}, {p['max']:g}, {p['c_end']:g}, {p['r_end']:g}")


def run_batch(state, duckmatch, plus, minus, seed):
    """Play `batch` pairs θ+ vs θ−; returns (wins, losses, draws) for θ+."""
    def engine(name, values):
        e = ["--engine", f"cmd={state['engine']}", f"name={name}"]
        e += [f"option.{k}={v}" for k, v in values.items()]
        return e

    cmd = [duckmatch, *engine("plus", plus), *engine("minus", minus),
           "--each", state["limit"], f"option.Hash={state['hash']}", f"option.Threads={state['threads']}"]
    if state["net"]:
        cmd.append(f"option.EvalFile={state['net']}")
    cmd += ["--games", str(2 * state["batch"]), "--concurrency", str(state["concurrency"]),
            "--seed", str(seed), "--report", "1000000", "--json"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    result = None
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("final"):
                result = r
    if result is None:
        sys.exit(f"duckmatch failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    bad = {k: v for k, v in result["terminations"].items()
           if k in ("TimeForfeit", "EngineError") or k.startswith("IllegalMove")}
    if bad:
        print(f"  warning: {bad} (CPU overloaded? lower --concurrency)")
    return result["wins"], result["losses"], result["draws"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="name of the tune (state in .spsa/<name>.json)")
    ap.add_argument("--params", help="comma-separated parameter names (default: all of them)")
    ap.add_argument("--config", help="file with parameter lines in OpenBench format instead of --params")
    ap.add_argument("--engine", default=os.path.join(REPO, "target", "release", f"peaduckzero{EXE}"),
                    help="engine binary (default target/release/peaduckzero, built first)")
    ap.add_argument("--net", help="network (.pdn) for both sides (default: the embedded one)")
    ap.add_argument("--tc", default="8+0.08", help="time control base+inc seconds (default 8+0.08)")
    ap.add_argument("--nodes", type=int, help="fixed playouts per move instead of --tc")
    ap.add_argument("--games", type=int, default=20000, help="total games of the tune (default 20000)")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch", type=int, help="game pairs per iteration (default: --concurrency)")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--show", action="store_true", help="print the tune's current values and exit")
    args = ap.parse_args()

    os.makedirs(SPSA_DIR, exist_ok=True)
    path = os.path.join(SPSA_DIR, f"{args.name}.json")
    if os.path.exists(path):
        state = json.load(open(path))
        if args.show:
            show(state)
            return
        print(f"continuing {args.name} (settings from {path}; delete it to start over)")
    else:
        if args.show:
            sys.exit(f"no tune named {args.name}")
        if os.path.abspath(args.engine) == os.path.join(REPO, "target", "release", f"peaduckzero{EXE}"):
            build("peaduckzero")
        state = new_state(args)
        save(state)
        print(f"new tune {args.name}: {len(state['params'])} parameters, {state['pairs_total']:,} pairs, "
              f"{state['batch']} pairs per iteration, {state['limit']}")
    build("duckmatch")
    duckmatch = os.path.join(REPO, "target", "release", f"duckmatch{EXE}")
    csv_path = os.path.join(SPSA_DIR, f"{args.name}.csv")
    new_csv = not os.path.exists(csv_path)
    rng = random.Random()

    try:
        while state["pairs_done"] < state["pairs_total"]:
            k, A = state["pairs_done"], state["A"]
            plus, minus, flips, cks = {}, {}, {}, {}
            for p in state["params"]:
                ck = p["c"] / (k + 1) ** GAMMA
                f = rng.choice((-1, 1))
                plus[p["name"]] = round(min(p["max"], max(p["min"], p["value"] + ck * f)))
                minus[p["name"]] = round(min(p["max"], max(p["min"], p["value"] - ck * f)))
                flips[p["name"]], cks[p["name"]] = f, ck
            w, l, d = run_batch(state, duckmatch, plus, minus, seed=rng.randrange(1 << 30))
            for p in state["params"]:
                ck = cks[p["name"]]
                ak = p["a"] / (A + k + 1) ** ALPHA
                rk = ak / ck ** 2
                p["value"] = min(p["max"], max(p["min"], p["value"] + rk * ck * (w - l) * flips[p["name"]]))
            state["pairs_done"] += state["batch"]
            state["iteration"] += 1
            state["wins"] += w
            state["losses"] += l
            state["draws"] += d
            save(state)
            with open(csv_path, "a", newline="") as f:
                wr = csv.writer(f)
                if new_csv:
                    wr.writerow(["iteration", "pairs"] + [p["name"] for p in state["params"]])
                    new_csv = False
                wr.writerow([state["iteration"], state["pairs_done"]] + [round(p["value"], 2) for p in state["params"]])
            print(f"iter {state['iteration']:5d} | pairs {state['pairs_done']:6,}/{state['pairs_total']:,} | "
                  f"θ+ {w}-{l}-{d} | " + "  ".join(f"{p['name']} {p['value']:.1f}" for p in state["params"][:6])
                  + (" ..." if len(state["params"]) > 6 else ""), flush=True)
            if state["iteration"] % 25 == 0:
                show(state)
    except KeyboardInterrupt:
        print(f"\nstopped; continue with: python tools/spsa.py --name {args.name}")
    show(state)


if __name__ == "__main__":
    main()
