# PeaDuckZero

A duck chess engine trained from zero with Gumbel AlphaZero self-play. It is
built from PeaZero's parts (bitboards, network inference, MCTS, self-play
and trainer), adapted to duck chess only.

## Duck chess

* A turn is a normal chess move followed by moving the duck to a different
  empty square (White's first turn places it anywhere empty).
* The duck can't be captured and blocks every piece: nothing moves onto it
  or slides through it.
* There is no check: kings may move into attacks and castle out of, through
  or into them. **Capturing the enemy king wins.**
* A player with no legal move at the start of their turn **wins**.
* Draws: fifty-move rule, threefold repetition.

These are the chess.com / duckchess.com rules as Fairy-Stockfish implements
them. The move generator was checked against Fairy-Stockfish (pyffish) on
33,000 positions from random games, with no differences in legal turns, FENs
(duck, en passant, castling) or game ends (`trainer/scripts/compare_fsf.py`).

Notation (Fairy-Stockfish): a turn is written `e2e4,e4d5` (the piece move, a
comma, the piece's square and the duck's square); in FENs the duck is `*`,
e.g. `rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1`.

## Layout

| path | what |
|---|---|
| `crates/core` | board, duck, move generation, FEN/UCI notation, game results, perft, network encoding |
| `crates/nn` | network inference (AVX-512 / AVX2 / SSE2, int8 residual blocks), `.pdn` network files |
| `crates/selfplay` | Gumbel self-play search, training data format, replay buffer |
| `crates/py` | Python bindings for the trainer (`peaduck_rs`) |
| `engine` | the UCI engine `peaduckzero` (MCTS, evaluation table, tree reuse, threads, time management) |
| `tools/duckmatch` | match runner with SPRT (OpenBench and fastchess don't do duck chess) |
| `trainer` | training loop, model, evaluator, dashboard |
| `networks/default.pdn` | the embedded network (random until a trained one replaces it) |

## How the two-step turn works

Internally a turn is two steps on the same position type: the **move step**
(a piece move) and the **duck step** (a duck move). The side to move only
changes after the duck step. Everything is built on that:

* **Search tree:** each step is one tree level. Every node stores its value
  for the side that made the step into it, so a backed-up result changes
  sign only across a duck move.
* **Network:** one policy head with 74 planes × 64 squares: the 73 chess
  move planes (AlphaZero layout) for piece moves, plus one plane for the
  duck's square. The input (23 planes) has the duck's square and a plane
  that says the duck step is pending. The same network evaluates both steps.
* **Training:** every step is a training sample with its own search target,
  so the network learns duck placement as well as piece moves.
* **King capture shortcut:** a position below the root where the side to
  move can capture the king is scored as a win straight away (that capture
  is its next step). It works like mate detection one step early, and the
  search finds duck blocks and king safety much faster with it.
* **Duck evasions:** one step earlier, in the duck step, the search checks
  which duck squares stop the opponent from capturing the king next
  (`Position::duck_defence`). If none does (a knight, pawn or king attack,
  or two lines with no shared square), the position is scored as lost at
  once; if only some do, only those are searched. In self-play the other
  squares get a policy target of 0.
* **Solver (engine):** these exact results are passed up the tree. A node
  is proven won when one of its steps wins, and proven lost when all of
  them lose. The engine plays a proven win at once, stops searching, and
  avoids proven losses while another move is left.
* **Duck-step parameters:** `CPuctDuck`, `FpuReductionDuck` and
  `PolicyTempDuck` replace `CPuct`, `FpuReduction` and `PolicyTemp` at duck
  steps (same defaults, separate for SPSA).

## Build and use

```powershell
cargo build --release        # target\release\peaduckzero.exe and duckmatch.exe
cargo test --release
make                         # .\peaduckzero(.exe), built for this CPU; EVALFILE=<.pdn> embeds a network
```

UCI as usual, with duck turns:

```
position startpos moves e2e4,e4d5 e7e5,e5e6
go wtime 8000 btime 8000 winc 80 binc 80
bestmove g1f3,f3h3
```

* Options: `Hash`, `Threads`, `MoveOverhead`, `EvalFile` (a `.pdn`),
  `UCI_ShowWDL`, `UCI_Variant` (only `duck`), `SearchMode` (`Mcts`,
  `Searchless`) and the tunable search parameters (`spsa` lists them).
* `depth`/`seldepth` in info lines count steps (two per turn); the PV is
  written in turns.
* `peaduckzero bench` prints a node count and speed; `peaduckzero perft N`
  counts whole turns, the way Fairy-Stockfish does.

The engine speaks Fairy-Stockfish's duck chess UCI notation, so it should
work with tools that drive Fairy-Stockfish for duck chess; that hasn't been
tested here.

## Playing against it

```powershell
cd trainer
python scripts\play.py                                   # embedded network, 800 playouts per move
python scripts\play.py --net ..\runs\first\latest.pdn --movetime 2000
```

A board opens in the browser (http://localhost:8800). Click a piece, then
its target, then the duck's square. It needs `peaduck_rs` (`pip install
..\crates\py`) and the built engine.

## Testing changes: branch against branch

```powershell
python tools\sprt.py my-feature                 # my-feature vs main, 8+0.08, SPRT [0, 5]
python tools\sprt.py my-feature master --elo1 10 --concurrency 7
python tools\sprt.py my-feature --nodes 800     # fixed playouts instead of a clock
python tools\sprt.py --list                     # history of all tests
```

* Each branch (or commit) is built in its own git worktree; the binaries
  are kept in `.sprt/bins`, so a commit is built only once. Only committed
  work counts.
* Then the two play with duckmatch until the SPRT decides (or `--games`
  runs out, or Ctrl+C).
* Every test is appended to `.sprt/results.jsonl` (refs, commits, settings,
  Elo, LLR, verdict); games go to `.sprt/logs`.
* `python tools\sprt.py --dashboard` shows all tests in the browser
  (http://localhost:8766), running ones live, like OpenBench's test list.
* `--dev-net` / `--base-net` give each side another network; `--evalfile`
  embeds one network in both builds.

## Tuning search parameters: SPSA

```powershell
python tools\spsa.py --name cpuct --params CPuct,CPuctExponent,FpuReduction,FpuReductionRoot `
    --net runs\big\nets\gen_0200.pdn --tc 8+0.08 --games 20000 --concurrency 7
python tools\spsa.py --name cpuct          # continue after Ctrl+C
python tools\spsa.py --name cpuct --show   # current values
```

OpenBench's SPSA method (same step-size formulas, `c_end`/`r_end` from the
engine's `spsa` command), with the games played by duckmatch. `--config
FILE` takes parameter lines in OpenBench format instead of `--params`.
State and history are kept in `.spsa/`.

## Testing changes: duckmatch

```powershell
target\release\duckmatch.exe --engine cmd=dev.exe name=dev --engine cmd=base.exe name=base `
    --each tc=8+0.08 option.Hash=16 option.Threads=1 --games 20000 --concurrency 8 `
    --sprt elo0=0 elo1=5
```

* Each opening (random turns from the start position, or `--openings FILE`)
  is played twice with colours swapped.
* The SPRT uses a pentanomial model and logistic Elo, like OpenBench.
* It prints Elo ± 95%, LLR, the pentanomial counts and how the games ended.
  `--log games.txt` writes every game.
* Different networks with one binary:
  `--engine cmd=peaduckzero.exe name=new option.EvalFile=new.pdn --engine cmd=peaduckzero.exe name=old option.EvalFile=old.pdn`.
* Fixed nodes instead of a clock: `--each nodes=800`.

## Training

See [docs/TRAINING.md](docs/TRAINING.md).

## Status

* Done:
  * Rules, verified against Fairy-Stockfish.
  * Encoding and inference: the Rust inference matches PyTorch on test networks.
  * UCI engine with MCTS (tree reuse, evaluation table, threads, time management).
  * Self-play trainer.
  * Match runner with SPRT.
* Not done yet: a trained network. The search parameters are PeaZero's chess
  values, a starting point to tune once there is a decent network.
