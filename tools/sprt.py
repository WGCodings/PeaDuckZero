#!/usr/bin/env python3
"""SPRT between two git branches (or any commits/tags): build both, play
them with duckmatch, keep a history.

    python tools\\sprt.py my-feature                     # my-feature vs main, 8+0.08, [0, 5]
    python tools\\sprt.py my-feature master --elo1 10 --concurrency 7
    python tools\\sprt.py my-feature --nodes 800         # fixed nodes instead of a clock
    python tools\\sprt.py --list                         # earlier tests

* Each ref is checked out in a temporary git worktree and built with
  `cargo build --release -p peaduckzero` (shared target folder, so the
  second build only recompiles what changed). The binary is kept in
  .sprt/bins/<commit>.exe, so a commit is built only once.
* Only committed work is used: commit your changes on the branch first.
* Each binary embeds that commit's networks/default.pdn, unless --evalfile
  (both builds) or --dev-net / --base-net (UCI EvalFile per side) are given.
* duckmatch is built from the current checkout.
* Every test appends one line to .sprt/results.jsonl (date, refs, commits,
  settings, Elo, LLR, verdict); the games go to .sprt/logs/.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRT_DIR = os.path.join(REPO, ".sprt")
EXE = ".exe" if os.name == "nt" else ""


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def resolve(ref):
    try:
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        sys.exit(f"unknown branch or commit: {ref}")
    return sha, git("log", "-1", "--format=%s", sha)


def warn_if_dirty(ref):
    """Uncommitted changes on the checked-out branch are not part of its build."""
    try:
        current = git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return
    if ref == current and git("status", "--porcelain", "--untracked-files=no"):
        print(f"note: {ref} has uncommitted changes; they are NOT in this test (commit them first)")


def build(sha, evalfile):
    """Build the engine at commit `sha`; returns the binary's path (cached)."""
    tag = sha[:12] + ("-" + os.path.splitext(os.path.basename(evalfile))[0] if evalfile else "")
    out = os.path.join(SPRT_DIR, "bins", f"{tag}{EXE}")
    if os.path.isfile(out):
        print(f"[build] {sha[:12]}: cached")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    src = os.path.join(SPRT_DIR, "src", sha[:12])
    if os.path.isdir(src):
        subprocess.run(["git", "worktree", "remove", "--force", src], cwd=REPO, capture_output=True)
        shutil.rmtree(src, ignore_errors=True)
    git("worktree", "add", "--detach", src, sha)
    try:
        env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(SPRT_DIR, "target"))
        if evalfile:
            env["EVALFILE"] = os.path.abspath(evalfile)
        else:
            env.pop("EVALFILE", None)
        print(f"[build] {sha[:12]} ...", flush=True)
        r = subprocess.run(["cargo", "build", "--release", "-p", "peaduckzero"], cwd=src, env=env)
        if r.returncode != 0:
            sys.exit(f"build of {sha[:12]} failed")
        shutil.copyfile(os.path.join(SPRT_DIR, "target", "release", f"peaduckzero{EXE}"), out)
        if os.name != "nt":
            os.chmod(out, 0o755)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", src], cwd=REPO, capture_output=True)
        shutil.rmtree(src, ignore_errors=True)
    return out


def build_duckmatch():
    r = subprocess.run(["cargo", "build", "--release", "-p", "duckmatch"], cwd=REPO)
    if r.returncode != 0:
        sys.exit("building duckmatch failed")
    return os.path.join(REPO, "target", "release", f"duckmatch{EXE}")


def list_results():
    path = os.path.join(SPRT_DIR, "results.jsonl")
    if not os.path.exists(path):
        print("no tests yet")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{'date':16}  {'dev':22} {'base':22} {'limit':10} {'bounds':8} {'games':>6} "
          f"{'Elo':>13} {'LLR':>6}  verdict")
    for r in rows:
        dev = f"{r['dev']} ({r['dev_sha'][:7]})"
        base = f"{r['base']} ({r['base_sha'][:7]})"
        elo = f"{r['elo']:+.1f}±{r['elo_error']:.1f}"
        print(f"{r['date']:16}  {dev[:22]:22} {base[:22]:22} {r['limit'][:10]:10} "
              f"{'[%g,%g]' % (r['elo0'], r['elo1']):8} {r['games']:>6} {elo:>13} {r['llr']:>6.2f}  "
              f"{r['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dev", nargs="?", help="branch/commit with the change")
    ap.add_argument("base", nargs="?", default="main", help="branch/commit to compare against (default main)")
    ap.add_argument("--list", action="store_true", help="show earlier tests and exit")
    ap.add_argument("--tc", default="8+0.08", help="time control base+inc seconds (default 8+0.08)")
    ap.add_argument("--nodes", type=int, help="fixed playouts per move instead of --tc")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--games", type=int, default=40000, help="maximum games (default 40000)")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1, help="engine threads per side")
    ap.add_argument("--openings", help="opening file for duckmatch (default: random turns)")
    ap.add_argument("--evalfile", help="build both with this network embedded")
    ap.add_argument("--dev-net", help="EvalFile option for dev (UCI)")
    ap.add_argument("--base-net", help="EvalFile option for base (UCI)")
    ap.add_argument("--seed", type=int, default=None, help="opening seed (default: new each test)")
    args = ap.parse_args()

    if args.list:
        list_results()
        return
    if not args.dev:
        ap.error("give the branch to test (and optionally the base, default main)")
    for ref in (args.dev, args.base):
        warn_if_dirty(ref)
    (dev_sha, dev_msg), (base_sha, base_msg) = resolve(args.dev), resolve(args.base)
    print(f"dev  {args.dev} {dev_sha[:12]}  {dev_msg}")
    print(f"base {args.base} {base_sha[:12]}  {base_msg}")
    if dev_sha == base_sha and not (args.dev_net or args.base_net):
        print("note: dev and base are the same commit")

    dev_bin = build(dev_sha, args.evalfile)
    base_bin = build(base_sha, args.evalfile)
    duckmatch = build_duckmatch()

    limit = f"nodes={args.nodes}" if args.nodes else f"tc={args.tc}"
    seed = args.seed if args.seed is not None else int(datetime.datetime.now().timestamp())
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    log = os.path.join(SPRT_DIR, "logs", f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{args.dev}-vs-{args.base}.txt"
                       .replace("/", "_"))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    dev_engine = ["--engine", f"cmd={dev_bin}", f"name={args.dev}"]
    base_engine = ["--engine", f"cmd={base_bin}", f"name={args.base}" + ("-base" if args.base == args.dev else "")]
    if args.dev_net:
        dev_engine.append(f"option.EvalFile={os.path.abspath(args.dev_net)}")
    if args.base_net:
        base_engine.append(f"option.EvalFile={os.path.abspath(args.base_net)}")
    cmd = [duckmatch, *dev_engine, *base_engine,
           "--each", limit, f"option.Hash={args.hash}", f"option.Threads={args.threads}",
           "--games", str(args.games), "--concurrency", str(args.concurrency), "--seed", str(seed),
           "--sprt", f"elo0={args.elo0}", f"elo1={args.elo1}", f"alpha={args.alpha}", f"beta={args.beta}",
           "--log", log, "--json"]
    if args.openings:
        cmd += ["--openings", args.openings]
    print(f"SPRT [{args.elo0:g}, {args.elo1:g}] {limit}, {args.concurrency} games at a time; games -> {log}",
          flush=True)

    result, stopped = None, False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            if line.startswith("{"):
                result = json.loads(line)  # progress every 10 games, then the final one
            else:
                print(line, end="", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        stopped = True  # Ctrl+C reaches duckmatch too: keep the last progress line
        proc.kill()
    if result is None:
        print("no result recorded (stopped before 10 games)")
        return
    if stopped or not result.get("final"):
        result["sprt"] = "stopped"
    elif result["sprt"] is None:
        result["sprt"] = "max games"
    record = {"date": stamp, "dev": args.dev, "dev_sha": dev_sha, "dev_msg": dev_msg,
              "base": args.base, "base_sha": base_sha, "base_msg": base_msg,
              "limit": limit, "elo0": args.elo0, "elo1": args.elo1, "hash": args.hash, "threads": args.threads,
              "dev_net": args.dev_net, "base_net": args.base_net, "evalfile": args.evalfile,
              "games": result["games"], "wins": result["wins"], "draws": result["draws"],
              "losses": result["losses"], "pentanomial": result["pentanomial"], "elo": result["elo"],
              "elo_error": result["elo_error"], "llr": result["llr"], "verdict": result["sprt"],
              "terminations": result["terminations"], "log": os.path.relpath(log, REPO)}
    with open(os.path.join(SPRT_DIR, "results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved to .sprt/results.jsonl (python tools/sprt.py --list)")


if __name__ == "__main__":
    main()
