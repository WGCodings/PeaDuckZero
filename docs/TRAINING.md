# Training PeaDuckZero

AlphaZero-style: the network plays itself with Gumbel tree search (Rust,
batched on the GPU from Python), trains on those games, and repeats. There
is no outside data, so a run starts from a random network.

## One-time setup (Windows, PowerShell)

```powershell
cd trainer
pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA build of torch
pip install numpy onnx maturin
pip install ..\crates\py                                   # peaduck_rs (Rust self-play + replay buffer)
cd ..; cargo build --release; cd trainer                   # engine + duckmatch, for the evaluation matches
python -c "import peaduck_rs, torch; print(peaduck_rs.ENCODING_VERSION, torch.cuda.is_available())"
```

After any change in `crates/`, run `pip install ..\crates\py` again.

## Run

```powershell
python train.py --run ..\runs\first
```

* Stop it any time with Ctrl+C.
* Continue with `python train.py --run ..\runs\first --resume`.
* Change a setting on resume, e.g. `--resume --lr 3e-4`. The new value is
  stored in `config.json`.

**Generations.** Each generation:
1. **Self-play** until `--positions-per-gen` new training positions exist.
   Every step of a turn is a position: the piece move and the duck move each
   get their own search and their own policy target.
2. **Training** for about positions × `--reuse` / `--batch` steps.
3. **Export** to `nets/gen_XXXX.pdn` (also copied to `latest.pdn`), plus a
   checkpoint.

**Evaluation.** Every `--eval-every` generations a background CPU process
plays gen N against gen N − eval-every. It uses `--match-games` games with
the engine binary at `--match-tc` through duckmatch. The Elo ladder shows up
on the dashboard at http://localhost:8765.

## Settings

Defaults for a new run (`python train.py --help` shows all of them). Step
counts count both steps of a turn, so 2 per turn.

| setting | default | meaning |
|---|---|---|
| `--net` | 64x6 | channels × residual blocks (channels a multiple of 32 keeps the engine's int8 path) |
| `--simulations` / `--considered` | 64 / 16 | search simulations per step, Gumbel root candidates |
| `--fast-simulations` / `--full-prob` / `--fast-considered` | 0 (off) / 0.25 / 8 | playout cap randomization: a share of the steps gets the full search and becomes a training sample, the rest a cheap search that only keeps the game going |
| `--games` | 1024 | concurrent self-play games |
| `--positions-per-gen` | 100k | new training positions per generation |
| `--noise-plies` / `--late-noise` | 60 / 0 | full Gumbel noise for the first 60 steps (30 turns), then none |
| `--root-explore` | 0.25 | exploration floor at the root: no step's prior can collapse beyond rediscovery |
| `--temperature` / `--temp-decay-plies` / `--temp-endgame` / `--temp-value-cutoff` | 0 (off) / 60 / 0 / 0.1 | sample the step played from the search policy |
| `--book` / `--book-prob` | none / 0.75 | opening book: one line per opening, `startpos moves e2e4,e4d5 ...`, `fen <FEN> [moves ...]` or a bare FEN |
| `--random-prob` / `--random-plies` | 0 / 2 | a share of games starts with 1..N random turns (not recorded) |
| `--max-plies` | 1024 | steps (512 turns) before a game is adjudicated a draw |
| `--batch` / `--reuse` / `--lr` / `--warmup` | 512 / 2.0 / 1e-3 / 500 | training; the learning rate stays constant after the warm-up |
| `--replay` | 1M | replay buffer (positions) |
| `--q-ratio` | 0.5 | value target = 50 % game result + 50 % search value |
| `--qat` | off | quantisation-aware training (the engine's int8 residual blocks); for the last stretch, with a low learning rate |
| `--eval-every` / `--match-games` / `--match-tc` / `--eval-threads` | 25 / 100 / 8+0.08 / 4 | background evaluation |
| `--init <checkpoint.pt>` | | a new run starting from another run's network (size and weights) |
| `--pools` / `--cuda-graphs` / `--channels-last` | 2 / on / on | self-play speed only (same results) |

## A first run

Start small and fast. The first goal is a network that clearly beats the
random one:

```powershell
python train.py --run ..\runs\first --net 64x6 --simulations 64 --fast-simulations 16 --full-prob 0.25
```

* **What to watch on the dashboard:**
  * Policy loss, split into piece moves and duck moves.
  * How games end: at first almost every game ends with a king capture,
    because the random network leaves its king hanging. That share drops as
    the network learns to guard the king with the duck.
  * Game length.
  * The Elo ladder.
* **Learning rate:** lower it in factors of about 3 (`--resume --lr 3e-4`,
  then `1e-4`) once the ladder stops gaining. Turn on `--qat` with the last
  drop.
* **Bigger networks:** `--init ..\runs\first\checkpoint.pt` starts a new run
  from an existing network, size included. A larger size needs a run of its
  own, or distillation from the smaller network's games, which hasn't been
  ported from PeaZero yet.

## A bigger network, warm-started by a smaller one

A new, larger network doesn't have to start from random play. A trained
smaller run can serve as its teacher:

```powershell
python train.py --run ..\runs\big --net 96x8 --teacher ..\runs\first\checkpoint.pt `
    --seed-data ..\runs\first --simulations 128 --fast-simulations 32
```

1. **Seed data:** the teacher run's newest games (as many generations as
   the replay buffer holds) are loaded, and the new network trains on them
   for `--seed-epochs` passes (default 2) before generation 1.
2. **Teacher self-play:** for the first `--teacher-gens` generations
   (default 30) the teacher plays the self-play games with this run's
   search settings, and the new network learns from those games. The
   console marks these generations with `T`.
3. **Own games:** after that the new network plays itself, and the run
   carries on like any other (`--resume`, learning-rate drops, `--qat`).

* **The teacher must be finished first:** use its final checkpoint, e.g.
  after the 64×6's last learning-rate drop with QAT. A copy of that
  `checkpoint.pt` is safest, since the teacher run's file changes while it
  trains.
* **Checking the warm start:** compare the new network with the teacher
  around the switch (fixed nodes first, then your time control, since the
  bigger network is slower). The ladder in the dashboard only compares the
  new run's own generations.

## Testing a network

* **In the engine:** `setoption name EvalFile value C:\...\runs\first\nets\gen_0100.pdn`,
  or embed it with `make EVALFILE=...`.
* **Against another network:** use duckmatch (see the README), for example:
  ```powershell
  ..\target\release\duckmatch.exe --engine cmd=..\target\release\peaduckzero.exe name=new option.EvalFile=new.pdn `
      --engine cmd=..\target\release\peaduckzero.exe name=old option.EvalFile=old.pdn `
      --each nodes=800 option.Threads=1 --games 2000 --concurrency 8 --sprt elo0=0 elo1=10
  ```

## Files

| file | what |
|---|---|
| `config.json` | the run's settings |
| `checkpoint.pt` | model + optimizer (`--resume`) |
| `metrics.jsonl` | one line per generation (losses, game results, speed) |
| `evals.jsonl` | evaluation matches |
| `nets/gen_XXXX.pdn` | networks (every `--keep-nets`-th generation, the milestones and the newest) |
| `data/gen_XXXX/*.bin` | self-play games (format: `crates/selfplay/src/record.rs`); old generations are deleted once the replay buffer can't use them |

## Test networks for the inference tests

`crates/nn/testdata` holds two small random networks with their PyTorch
outputs; `cargo test` checks that every inference path (AVX-512, AVX2,
SSE2, Winograd, int8) matches them. They are regenerated with
`python scripts\make_test_net.py` (after any change to the model or the
encoding). `python scripts\make_default_net.py` writes a fresh random
`networks/default.pdn`.
