#!/usr/bin/env python3
"""PeaDuckZero training loop: alternate tree-search self-play and training.

    python train.py --run ..\\runs\\first                 # new run, default settings
    python train.py --run ..\\runs\\first --resume        # continue with the run's settings
    python train.py --run ..\\runs\\first --resume --lr 3e-4   # ... and change one

Each generation:
  1. self-play until --positions-per-gen new training positions are recorded
     (every step of a turn, piece move and duck move, is a position)
  2. train ~positions × --reuse / --batch steps on the replay buffer
  3. save the checkpoint and export the network

Run directory layout:
  <run>/config.json            the run's settings (reused by --resume)
  <run>/checkpoint.pt          latest model + optimizer (used by --resume)
  <run>/metrics.jsonl          one line per generation
  <run>/status.json            live progress (read by the dashboard)
  <run>/latest.pdn             copy of the newest network
  <run>/nets/gen_XXXX.pdn      network of every --keep-nets-th generation (+ the newest)
  <run>/evals.jsonl            strength evaluation (matches; see below)
  <run>/data/gen_XXXX/*.bin    self-play games (crates/selfplay/src/record.rs)

--qat (quantisation-aware training): the engine runs the residual blocks in
int8 (crates/nn/src/int8.rs); with --qat the trainer rounds weights and
activations the same way in every training step. Best switched on for a
run's last stretch, together with a low learning rate.

Settings: every run stores its settings in config.json. --resume reuses
them; a flag given on the command line overrides the stored value and is
saved. --init starts a new run from another run's checkpoint.pt (its size
and weights, fresh optimizer).

Warm start of a bigger network from a smaller one (the teacher):

    python train.py --run ..\\runs\\big --net 96x8 --teacher ..\\runs\\first\\checkpoint.pt `
        --seed-data ..\\runs\\first\\data

* --seed-data: the teacher's recent self-play games fill the replay buffer
  and the new network trains on them for --seed-epochs passes before the
  first generation.
* --teacher: for the first --teacher-gens generations the self-play games
  are played by the teacher (with the run's search settings); the new
  network trains on them. After that it plays its own games.

A live dashboard opens at http://localhost:8765 while training; see
--dashboard-port / --no-browser.

Strength: every --eval-every generations a background CPU process
(peaduck/evaluator.py) plays gen N against gen N - eval-every
(--match-games games with the engine binary at --match-tc 8+0.08, through
tools/duckmatch). --eval-every 0 = off.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peaduck_rs as pz  # noqa: E402  (pip install ./crates/py)
from peaduck import dashboard  # noqa: E402
from peaduck import encoding as enc  # noqa: E402
from peaduck.model import DuckNet, NetConfig, export_pdn  # noqa: E402
from peaduck.selfplay_gpu import SearchRunner  # noqa: E402
from peaduck.train_lib import az_loss, batch_to_device, eval_metrics, eval_net, search_step  # noqa: E402


def net_size(s):
    """'96x8' -> (96, 8): channels x residual blocks."""
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--net: want CHANNELSxBLOCKS like 96x8, got {s!r}")
    return int(m.group(1)), int(m.group(2))


# Settings of a new run. All of these are stored in <run>/config.json.
# Step counts (max_plies, noise_plies, temp_decay_plies) count both steps of
# a turn: 2 per turn.
DEFAULTS = {
    "net": "64x6", "generations": 1000,
    # warm start: self-play by another run's network for the first generations
    "teacher": None, "teacher_gens": 30,
    # self-play
    "games": 1024, "simulations": 64, "considered": 16, "max_plies": 1024, "threads": 0,
    "fast_simulations": 0, "full_prob": 0.25, "fast_considered": 8,
    # game variety: opening book + random turns, step temperature
    "book": None, "book_prob": 0.75, "random_prob": 0.0, "random_plies": 2,
    "temperature": 0.0, "temp_decay_plies": 60, "temp_endgame": 0.0, "temp_value_cutoff": 0.1,
    "root_explore": 0.25,
    "positions_per_gen": 100_000, "noise_plies": 60, "late_noise": 0.0,
    # training
    "batch": 512, "reuse": 2.0, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 500,
    "replay": 1_000_000, "keep_data_gens": -1, "keep_nets": 25,
    "q_ratio": 0.5, "qat": False, "seed": 0,
    # background strength evaluation (CPU)
    "eval_every": 25, "match_games": 100, "match_tc": "8+0.08", "eval_threads": 4,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--resume", action="store_true", help="continue the run with its stored settings")
    ap.add_argument("--init", help="new run only: start from this checkpoint.pt (its size and weights)")
    ap.add_argument("--net", help="network size CHANNELSxBLOCKS, e.g. 64x6, 96x8, 128x10")
    ap.add_argument("--generations", type=int, help="total generations to reach")
    ap.add_argument("--teacher", help="checkpoint.pt of another run whose network plays the self-play games "
                                      "for the first --teacher-gens generations ('none' = off)")
    ap.add_argument("--teacher-gens", type=int, help="generations of teacher self-play (default 30)")
    ap.add_argument("--seed-data", help="new run only: a data folder of another run (e.g. ..\\runs\\first\\data); "
                                        "its newest games fill the replay buffer and are trained on first")
    ap.add_argument("--seed-epochs", type=float, default=2.0,
                    help="passes over the --seed-data positions before the first generation (default 2)")
    # self-play
    ap.add_argument("--games", type=int, help="concurrent self-play games")
    ap.add_argument("--simulations", type=int, help="tree-search simulations per step (Gumbel search)")
    ap.add_argument("--considered", type=int, help="Gumbel m: root candidates (Sequential Halving width)")
    ap.add_argument("--fast-simulations", type=int,
                    help="playout cap randomization: each step is a full search with probability --full-prob, "
                         "otherwise a fast search with this many simulations that is not a training sample; 0 = off")
    ap.add_argument("--full-prob", type=float, help="share of steps with the full search")
    ap.add_argument("--fast-considered", type=int, help="Sequential Halving width of a fast search")
    ap.add_argument("--book", help="opening book: one line per opening, 'startpos moves e2e4,e4d5 ...', "
                                   "'fen <FEN> [moves ...]' or a bare FEN; with probability --book-prob a new game "
                                   "starts after 1..all turns of a random line ('none' = no book)")
    ap.add_argument("--book-prob", type=float, help="share of games that start from the book")
    ap.add_argument("--random-prob", type=float,
                    help="share of games that then play 1..--random-plies uniformly random turns (not recorded)")
    ap.add_argument("--random-plies", type=int, help="at most this many random turns after the opening")
    ap.add_argument("--temperature", type=float,
                    help="sample the step played from the search policy with this temperature "
                         "(decays over --temp-decay-plies steps to --temp-endgame); 0 = always the best step")
    ap.add_argument("--temp-decay-plies", type=int, help="steps over which the temperature decays (0 = constant)")
    ap.add_argument("--temp-endgame", type=float, help="lowest temperature for the rest of the game")
    ap.add_argument("--temp-value-cutoff", type=float,
                    help="only sample steps whose search value (W-L) is within this of the best step")
    ap.add_argument("--root-explore", type=float,
                    help="root exploration floor: candidates drawn from (1-e)*prior + e/legal steps (0 = off)")
    ap.add_argument("--max-plies", type=int, help="steps per game before it is adjudicated a draw")
    ap.add_argument("--threads", type=int, help="CPU threads for the Rust self-play work (0 = all)")
    ap.add_argument("--positions-per-gen", type=int)
    ap.add_argument("--noise-plies", type=int, help="steps per game with full Gumbel noise, then --late-noise")
    ap.add_argument("--late-noise", type=float, help="Gumbel noise scale after --noise-plies (0 = none)")
    # training
    ap.add_argument("--batch", type=int)
    ap.add_argument("--reuse", type=float, help="avg times each new position is sampled")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--replay", type=int, help="replay buffer size in positions")
    ap.add_argument("--keep-nets", type=int,
                    help="keep the network of every N-th generation (and every evaluation milestone, and the "
                         "newest); others are deleted. 0 = keep all")
    ap.add_argument("--keep-data-gens", type=int,
                    help="generations of self-play data kept on disk; -1 = just enough to refill the replay "
                         "buffer on --resume, 0 = keep everything")
    ap.add_argument("--q-ratio", type=float, help="value target = (1-q)·game result + q·search value")
    ap.add_argument("--qat", action=argparse.BooleanOptionalAction,
                    help="quantisation-aware training (simulate the engine's int8 residual blocks)")
    ap.add_argument("--seed", type=int)
    # strength evaluation
    ap.add_argument("--eval-every", type=int,
                    help="every N generations: match vs the net N generations older, in a background "
                         "CPU process (0 = off)")
    ap.add_argument("--match-games", type=int, help="games per evaluation match")
    ap.add_argument("--match-tc", help="evaluation matches: time control (base+inc seconds)")
    ap.add_argument("--eval-threads", type=int, help="games at the same time in the evaluation")
    # this session only (not stored)
    ap.add_argument("--engine", help="engine binary for the matches (default target/release/peaduckzero)")
    ap.add_argument("--pools", type=int, default=2,
                    help="self-play speed (same results): split the games over N pools so the GPU evaluates "
                         "one while the CPU searches the other (2, default); 1 = one pool; 0 = plain loop")
    ap.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True,
                    help="self-play inference as recorded CUDA graphs")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="self-play inference in NHWC memory layout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    ap.add_argument("--dashboard-port", type=int, default=8765, help="live dashboard port; 0 = no dashboard")
    ap.add_argument("--no-browser", action="store_true", help="don't open the dashboard in a browser")
    return resolve_settings(ap.parse_args())


def resolve_settings(args):
    """Fill unset settings from the run's config.json (--resume) or the
    defaults, store the result in config.json and print it."""
    config_path = os.path.join(args.run, "config.json")
    ckpt = os.path.join(args.run, "checkpoint.pt")
    if not args.resume and os.path.exists(ckpt):
        try:
            gen = torch.load(ckpt, map_location="cpu", weights_only=False).get("generation", "?")
        except Exception:
            gen = "?"
        sys.exit(f"{args.run} already holds a run (checkpoint at generation {gen}).\n"
                 f"  continue it:     python train.py --run {args.run} --resume\n"
                 f"  or start fresh:  use another --run folder (or delete this one)")
    saved = {}
    if args.resume and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
    overridden = []
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        if value is not None:
            if args.resume and saved.get(key, value) != value:
                overridden.append(key)
            continue
        setattr(args, key, saved.get(key, default) if args.resume else default)
    os.makedirs(args.run, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({k: getattr(args, k) for k in DEFAULTS}, f, indent=2)
    print("settings: " + ", ".join(f"{k}={getattr(args, k)}" + (" (changed)" if k in overridden else "")
                                   for k in DEFAULTS))
    return args


def log(run, record):
    with open(os.path.join(run, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def data_gen_dirs(run):
    """(generation, path) of every data/gen_XXXX folder, oldest first."""
    out = []
    for path in glob.glob(os.path.join(run, "data", "gen_*")):
        m = re.fullmatch(r"gen_(\d+)", os.path.basename(path))
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def replay_gens(args):
    """Generations of data needed to fill the replay buffer, plus one spare."""
    return math.ceil(args.replay / args.positions_per_gen) + 1


def prune_nets(args, newest):
    """Delete nets/gen_XXXX.pdn except every --keep-nets-th generation, the
    evaluation milestones and `newest`. Returns the number deleted."""
    if args.keep_nets <= 0:
        return 0
    removed = 0
    for path in glob.glob(os.path.join(args.run, "nets", "gen_*.pdn")):
        m = re.fullmatch(r"gen_(\d+)\.pdn", os.path.basename(path))
        if not m:
            continue  # hand-made copies stay
        g = int(m.group(1))
        if g >= newest or g % args.keep_nets == 0 or (args.eval_every > 0 and g % args.eval_every == 0):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def prune_data(args):
    keep = args.keep_data_gens if args.keep_data_gens >= 0 else replay_gens(args)
    if keep <= 0:
        return
    for _, path in data_gen_dirs(args.run)[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def start_evaluator(args):
    """Background strength evaluation at lower CPU priority. It exits when we
    do: it watches its stdin, which we hold."""
    if args.eval_every <= 0 or args.match_games <= 0:
        return None
    cmd = [sys.executable, "-m", "peaduck.evaluator", "--run", os.path.abspath(args.run),
           "--every", str(args.eval_every), "--games", str(args.match_games), "--tc", str(args.match_tc),
           "--concurrency", str(args.eval_threads), "--seed", str(args.seed), "--parent-stdin"]
    if args.engine:
        cmd += ["--engine", os.path.abspath(args.engine)]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        kw["preexec_fn"] = lambda: os.nice(10)
    try:
        return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdin=subprocess.PIPE, **kw)
    except OSError as e:
        print(f"could not start the evaluator: {e}")
        return None


def stop_evaluator(proc, finish=False):
    if proc is None or proc.poll() is not None:
        return
    try:
        if finish:
            print("waiting for the evaluator to finish the last milestones (Ctrl+C to skip)")
            proc.stdin.write(b"finish\n")
            proc.stdin.flush()
            proc.wait()
        else:
            proc.stdin.close()
            proc.wait(timeout=5)
    except (Exception, KeyboardInterrupt):
        proc.kill()


def load_book(path):
    """Opening lines [(FEN or None, [UCI turns])], one per line:
    'startpos moves e2e4,e4d5 ...', 'fen <FEN> [moves ...]' or a bare FEN."""
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            setup, _, moves = raw.partition(" moves ")
            setup = setup.strip().removesuffix(" moves")
            fen = None if setup == "startpos" else setup.removeprefix("fen ").strip()
            lines.append((fen, moves.split()))
    if not lines:
        sys.exit(f"no openings found in {path}")
    print(f"book: {len(lines):,} openings from {path}")
    return lines


def variety_settings(args):
    kw = {}
    if args.book and args.book.lower() != "none":
        kw.update(openings=load_book(args.book), book_prob=args.book_prob)
    if args.random_prob > 0 and args.random_plies > 0:
        kw.update(random_prob=args.random_prob, random_plies=args.random_plies)
    if args.root_explore > 0:
        kw["root_explore"] = args.root_explore
    if args.temperature > 0 or args.temp_endgame > 0:
        kw.update(temperature=args.temperature, temp_decay_plies=args.temp_decay_plies,
                  temp_endgame=args.temp_endgame, temp_value_cutoff=args.temp_value_cutoff)
    if args.fast_simulations > 0:
        kw.update(fast_simulations=args.fast_simulations, full_prob=args.full_prob,
                  fast_considered=args.fast_considered)
    return kw


def seed_from_data(args, net, opt, scaler, replay, dev, amp, status):
    """Fill the replay buffer with the newest games of another run's data
    folder and train on them for --seed-epochs passes (new runs only)."""
    p = os.path.abspath(args.seed_data)  # the run folder or its data folder
    gens = data_gen_dirs(p if os.path.isdir(os.path.join(p, "data")) else os.path.dirname(p))
    if not gens:
        sys.exit(f"--seed-data: no gen_XXXX folders in {args.seed_data}")
    loaded = 0
    for _, d in gens[-replay_gens(args):]:
        for path in sorted(glob.glob(os.path.join(d, "*.bin"))):
            with open(path, "rb") as f:
                replay.add_chunk(f.read())
            loaded += 1
    n_steps = max(1, int(replay.positions * args.seed_epochs / args.batch))
    print(f"seed data: {replay.positions:,} positions from {loaded} chunks of {args.seed_data}; "
          f"training {n_steps:,} steps ({args.seed_epochs:g} passes) before generation 1")
    net.train()
    t0 = time.time()
    sums, count = {}, 0
    status.update(force=True, phase="training", phase_start=t0, steps_done=0, steps_total=n_steps,
                  replay_positions=replay.positions)
    batch = batch_to_device(replay.sample(args.batch), dev)
    for step in range(1, n_steps + 1):
        lr = args.lr * min(1.0, step / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss, m = az_loss(net, batch, dev, amp, q_ratio=args.q_ratio)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step < n_steps:
            batch = batch_to_device(replay.sample(args.batch), dev)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        for k, v in m.items():
            sums[k] = sums[k] + v if k in sums else v
        count += 1
        if step % 500 == 0 or step == n_steps:
            avg = {k: float(v) / count for k, v in sums.items()}
            print(f"  seed step {step:6d}/{n_steps} | loss {avg['loss']:.3f} (p {avg['policy_loss']:.3f} "
                  f"v {avg['value_loss']:.3f} agree {avg['policy_agreement']:.2f}) | {time.time() - t0:.0f}s",
                  flush=True)
            sums, count = {}, 0
            status.update(steps_done=step, step=step)
    args.seed_steps_done = n_steps
    net.eval()


class PlainPool:
    """One pool driven by the plain loop (train_lib.search_step)."""

    def __init__(self, pool):
        self.pool = pool

    def stats(self):
        return self.pool.stats()

    def take_chunks(self):
        chunk = self.pool.take_chunk()
        return [chunk] if chunk[1] else []

    @property
    def num_threads(self):
        return self.pool.num_threads


def main():
    args = parse_args()
    for d in ("data", "nets"):
        os.makedirs(os.path.join(args.run, d), exist_ok=True)
    status = dashboard.StatusWriter(
        args.run, run=os.path.abspath(args.run), target_generations=args.generations, batch=args.batch,
        reuse=args.reuse, games=args.games, considered=args.considered,
        positions_target=args.positions_per_gen, replay_capacity=args.replay)
    status.update(force=True, phase="loading", generation=0)
    if args.dashboard_port:
        url = dashboard.start(args.run, args.dashboard_port, not args.no_browser)
        if url:
            print(f"dashboard: {url}")
    evaluator = start_evaluator(args)
    finished = False
    try:
        train(args, status)
        finished = True
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C); continue with --resume")
    finally:
        status.update(force=True, phase="stopped")
        stop_evaluator(evaluator, finish=finished)


def train(args, status):
    dev = torch.device(args.device)
    amp = not args.no_amp and dev.type == "cuda"
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if pz.ENCODING_VERSION != enc.ENCODING_VERSION:
        sys.exit("peaduck_rs is out of date (encoding version); run `pip install ./crates/py`")

    ckpt_path = os.path.join(args.run, "checkpoint.pt")
    c, blocks = net_size(args.net)
    cfg = NetConfig(channels=c, blocks=blocks)
    gen, step = 0, 0
    ckpt = None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = NetConfig(**ckpt["config"])
        gen, step = ckpt["generation"], ckpt["step"]
    elif args.init:
        ick = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = NetConfig(**ick["config"])
    net = DuckNet(cfg).to(dev)
    net.set_qat(args.qat)
    if args.qat and not net.qat:
        print(f"--qat: {cfg.channels} channels is not a multiple of 32, the engine uses float; QAT off")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if ckpt is not None:
        net.load_state_dict(ckpt["model"])
        opt_state = ckpt["optimizer"]
        for g in opt_state["param_groups"]:
            g["fused"] = dev.type == "cuda" or None
            g["foreach"] = None
        opt.load_state_dict(opt_state)
        print(f"resumed {args.run} at generation {gen}, step {step}")
    elif args.init:
        net.load_state_dict(ick["model"])
        print(f"started from {args.init} ({cfg.channels}x{cfg.blocks}, generation {ick.get('generation')})")

    replay = pz.Replay(capacity=args.replay, seed=args.seed + gen)
    teacher = None
    if args.teacher and args.teacher.lower() != "none" and gen < args.teacher_gens:
        tck = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher = DuckNet(NetConfig(**tck["config"])).to(dev)
        teacher.load_state_dict(tck["model"])
        teacher = eval_net(teacher)
        print(f"teacher: {args.teacher} ({teacher.cfg.channels}x{teacher.cfg.blocks}, generation "
              f"{tck.get('generation')}) plays the self-play games up to generation {args.teacher_gens}")
    if args.resume:  # refill the replay buffer, oldest data first
        refill = data_gen_dirs(args.run)[-replay_gens(args):]
        for _, d in refill:
            for path in sorted(glob.glob(os.path.join(d, "*.bin"))):
                with open(path, "rb") as f:
                    replay.add_chunk(f.read())
        print(f"replay buffer: {replay.positions:,} positions from {len(refill)} generations on disk")
    if args.seed_data and gen == 0:
        seed_from_data(args, net, opt, scaler, replay, dev, amp, status)
        step = args.seed_steps_done
    prune_nets(args, gen)

    n_pools = max(1, args.pools)
    kw = variety_settings(args)
    sizes = [args.games // n_pools + (i < args.games % n_pools) for i in range(n_pools)]
    pools = [pz.SearchSelfPlay(games=n, simulations=args.simulations, considered=args.considered,
                               max_plies=args.max_plies, seed=args.seed + gen + 1_000_003 * i,
                               threads=args.threads, noise_plies=args.noise_plies,
                               late_gumbel_scale=args.late_noise, **kw)
             for i, n in enumerate(sizes)]
    runner = None
    if args.pools > 0:
        runner = SearchRunner(pools, dev, amp, cuda_graphs=args.cuda_graphs, channels_last=args.channels_last)
        sp = runner
    else:
        sp = PlainPool(pools[0])
    mode = f"{args.simulations} simulations per step, {args.considered} considered"
    if kw.get("openings"):
        mode += f"; book {len(kw['openings']):,} lines ({args.book_prob:.0%} of games)"
    if kw.get("random_prob"):
        mode += f"; {args.random_prob:.0%} 1-{args.random_plies} random turns"
    if kw.get("root_explore"):
        mode += f"; root exploration floor {args.root_explore:g}"
    if kw.get("temperature"):
        mode += (f"; temperature {args.temperature:g} -> {args.temp_endgame:g} over {args.temp_decay_plies} "
                 f"steps, cutoff {args.temp_value_cutoff:g}")
    if args.fast_simulations > 0:
        mode += (f" on {args.full_prob:.0%} of steps, else {args.fast_simulations} simulations, "
                 f"{args.fast_considered} considered (playout cap randomization)")
    if runner:
        mode += (f", {n_pools} pipelined pools" if n_pools > 1 else ", 1 pool")
        mode += ", CUDA graphs" if runner.graphs_wanted else ""
        mode += ", channels_last" if runner.channels_last else ""
    else:
        mode += ", plain loop"
    params = sum(p.numel() for p in net.parameters()) / 1e6
    net_name = f"{cfg.channels}x{cfg.blocks}"
    print(f"net {net_name} ({params:.2f}M params) on {dev}{' fp16' if amp else ''}; "
          f"{args.games} games, {mode}, {sp.num_threads} self-play threads, batch {args.batch}")
    status.update(force=True, net=net_name, params=params, device=dev.type, amp=amp, step=step,
                  generation=gen, session_start_generation=gen, replay_positions=replay.positions)

    sp_net = teacher if teacher is not None else eval_net(net)
    if runner:
        runner.set_net(sp_net)
        if runner.graphs_wanted and not runner.graphs:
            print("CUDA graphs unavailable here; eager inference")
    recent_keys = collections.deque(maxlen=5)
    lr = args.lr

    while gen < args.generations:
        gen += 1
        # ---------------------------------------------------------------- self-play
        by_teacher = teacher is not None
        t0 = time.time()
        new_positions, new_games, new_plies, part = 0, 0, 0, 0
        gen_keys = []
        gen_dir = os.path.join(args.run, "data", f"gen_{gen:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        stats_before = sp.stats()
        status.update(force=True, phase="selfplay", generation=gen, phase_start=t0, positions=0,
                      games_finished=0, moves_per_second=None)
        chunk_every = 8 * (args.simulations + 1)
        steps = 0
        if runner:
            runner.reset_timing()
        while new_positions < args.positions_per_gen:
            if runner:
                runner.step()
            else:
                search_step(sp_net, sp.pool, dev, amp)
            steps += 1
            if time.time() - status.last_write >= 1.0:
                moves = sp.stats()["plies"] - stats_before["plies"]
                status.update(positions=new_positions, games_finished=new_games,
                              moves_per_second=moves / max(time.time() - t0, 1e-6),
                              replay_positions=replay.positions)
            if steps % chunk_every == 0:
                for chunk, games, positions, plies in sp.take_chunks():
                    with open(os.path.join(gen_dir, f"part_{part:03d}.bin"), "wb") as f:
                        f.write(chunk)
                    replay.add_chunk(chunk)
                    gen_keys.append(pz.chunk_position_keys(chunk))
                    part += 1
                    new_positions += positions
                    new_plies += plies
                    new_games += games
        sp_timing = None
        if runner:
            runner.drain()
            sp_timing = runner.reset_timing()
        sp_time = time.time() - t0
        st = sp.stats()
        uniq = {}
        if gen_keys:
            keys = np.unique(np.concatenate(gen_keys))
            total = sum(len(k) for k in gen_keys)
            uniq["unique_positions"] = round(len(keys) / max(total, 1), 4)
            if recent_keys:
                seen = np.isin(keys, np.concatenate(list(recent_keys)))
                uniq["new_vs_recent"] = round(float((~seen).sum()) / max(total, 1), 4)
            recent_keys.append(keys)
        moves = st["plies"] - stats_before["plies"]
        d = {k: st[k] - stats_before[k] for k in ("games", "white_wins", "black_wins", "draws")}
        terms = {k: st["terminations"][k] - stats_before["terminations"][k] for k in st["terminations"]}

        # ---------------------------------------------------------------- training
        net.train()
        t1 = time.time()
        n_steps = max(1, int(new_positions * args.reuse / args.batch))
        sums = {}
        status.update(force=True, phase="training", phase_start=t1, steps_done=0, steps_total=n_steps,
                      running=None, positions=new_positions, games_finished=new_games,
                      replay_positions=replay.positions)
        batch = batch_to_device(replay.sample(args.batch), dev)
        for i in range(n_steps):
            step += 1
            lr = args.lr * min(1.0, step / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, m = az_loss(net, batch, dev, amp, q_ratio=args.q_ratio)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if i + 1 < n_steps:
                batch = batch_to_device(replay.sample(args.batch), dev)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in m.items():
                sums[k] = sums[k] + v if k in sums else v
            if time.time() - status.last_write >= 1.0:
                status.update(steps_done=i + 1, step=step,
                              running={k: float(v) / (i + 1) for k, v in sums.items()},
                              steps_per_second=(i + 1) / max(time.time() - t1, 1e-6))
        avg = {k: float(v) / n_steps for k, v in sums.items()}
        train_time = time.time() - t1

        # ---------------------------------------------------------------- save
        status.update(force=True, phase="saving", steps_done=n_steps, step=step)
        net.eval()
        student = eval_net(net)
        ev = eval_metrics(student, replay, dev, amp)
        if teacher is not None and gen >= args.teacher_gens:
            print(f"generation {gen}: teacher done, the network plays its own games from now on")
            teacher = None
        sp_net = teacher if teacher is not None else student
        if runner:
            runner.set_net(sp_net)
        net_path = os.path.join(args.run, "nets", f"gen_{gen:04d}.pdn")
        export_pdn(student, net_path)
        shutil.copyfile(net_path, os.path.join(args.run, "latest.pdn"))
        prune_data(args)
        prune_nets(args, gen)
        tmp = ckpt_path + ".tmp"
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg.__dict__,
                    "generation": gen, "step": step}, tmp)
        os.replace(tmp, ckpt_path)

        record = {
            "generation": gen, "step": step, "lr": lr, "batch": args.batch, "time": round(time.time(), 1),
            "selfplay_seconds": round(sp_time, 1), "train_seconds": round(train_time, 1),
            "moves_per_second": round(moves / sp_time), "new_games": new_games, "new_positions": new_positions,
            "avg_game_length": round(new_plies / max(new_games, 1) / 2, 1),  # in turns
            "full_search_share": round((st["full_plies"] - stats_before["full_plies"]) / max(moves, 1), 3),
            **({"sp_gpu_wait_seconds": round(sp_timing["gpu_wait"], 1),
                "sp_rust_seconds": round(sp_timing["rust"], 1),
                "sp_launch_seconds": round(sp_timing["launch"], 1)} if sp_timing else {}),
            **uniq,
            "results": d, "terminations": terms, "replay_positions": replay.positions, "qat": net.qat,
            "teacher_selfplay": by_teacher,
            **avg, **ev,
        }
        log(args.run, record)
        print(f"gen {gen:4d}{' T' if record['teacher_selfplay'] else ''} | selfplay {sp_time:6.1f}s {moves / sp_time:7.0f} steps/s "
              + (f"(gpu wait {sp_timing['gpu_wait']:.0f}s rust {sp_timing['rust']:.0f}s "
                 f"launch {sp_timing['launch']:.0f}s) " if sp_timing else "")
              + f"{new_games:5d} games (W{d['white_wins']} D{d['draws']} B{d['black_wins']}, "
              + (f"unique {uniq['unique_positions']:.0%}, " if uniq else "")
              + f"{record['avg_game_length']} turns) | train {n_steps} steps {train_time:6.1f}s "
              f"loss {avg['loss']:.3f} (p {avg['policy_loss']:.3f} [move {avg['policy_loss_move']:.3f} "
              f"duck {avg['policy_loss_duck']:.3f}] v {avg['value_loss']:.3f} agree {avg['policy_agreement']:.2f}) "
              f"| exported p {ev.get('eval_policy_loss', 0):.3f} v {ev.get('eval_value_loss', 0):.3f}"
              + (f" | int8 {ev['int8_agreement'] * 100:.1f}%" if "int8_agreement" in ev else "")
              + f" | {net_path}")


if __name__ == "__main__":
    main()
