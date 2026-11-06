#!/usr/bin/env python3
"""Check PeaDuckZero's rules against Fairy-Stockfish (pyffish) along random
games: legal turns, FENs (duck, en passant, castling) and game ends.

    pip install pyffish
    cargo build --release --example turns
    python trainer/scripts/compare_fsf.py target/release/examples/turns 120

The random games mostly avoid king captures, so they get long enough for
endgames, promotions and the fifty-move rule.
"""
import random, subprocess, sys
import pyffish as sf

V = "duck"
exe = sys.argv[1]
games = int(sys.argv[2]) if len(sys.argv) > 2 else 200
rng = random.Random(1)

def expand(fen):
    rows = fen.split()[0].split("/")
    b = [""] * 64
    for i, r in enumerate(rows):
        f = 0
        for c in r:
            if c.isdigit():
                for _ in range(int(c)):
                    b[(7 - i) * 8 + f] = "."; f += 1
            else:
                b[(7 - i) * 8 + f] = c; f += 1
    return b

def sq(s):
    return (ord(s[0]) - 97) + 8 * (int(s[1]) - 1)

fens, info = [], []
for g in range(games):
    fen0 = sf.start_fen(V)
    fen = fen0
    moves = []
    for ply in range(400):
        legal = sf.legal_moves(V, fen, [])
        end, val = sf.is_immediate_game_end(V, fen0, moves)
        opt, oval = sf.is_optional_game_end(V, fen0, moves)
        fens.append(fen)
        info.append((sorted(legal), end, val, opt, oval, fen0, None))
        if end or opt or not legal:
            break
        board = expand(fen)
        # mostly avoid king captures so games get long (endgames, promotions,
        # 50-move rule, repetitions, no-move wins)
        noking = [m for m in legal if board[sq(m[2:4])] not in "kK"] or legal
        if rng.random() < 0.02:
            noking = legal
        caps = [m for m in noking if board[sq(m[2:4])] not in ".*"]
        pick = rng.choice(caps) if caps and rng.random() < 0.3 else rng.choice(noking)
        moves.append(pick)
        fen = sf.get_fen(V, fen, [pick])

out = subprocess.run([exe], input="\n".join(fens) + "\n", capture_output=True, text=True, check=True).stdout.splitlines()
bad = 0
results = {}
for fen, line, (legal, end, val, opt, oval, f0, mv) in zip(fens, out, info):
    myfen, myres, myturns = line.split("\t")
    myturns = myturns.split() if myturns else []
    results[myres] = results.get(myres, 0) + 1
    if myfen != fen:
        bad += 1; print("FEN", fen, "->", myfen); continue
    kingless = "Win" in myres and "KingCaptured" in myres
    if kingless:
        if not end:
            bad += 1; print("FSF says not over:", fen)
        continue
    if myturns != legal and not end:
        bad += 1
        a, b = set(myturns), set(legal)
        print("MOVES", fen, "only mine:", sorted(a - b)[:8], "only FSF:", sorted(b - a)[:8])
    if (myres != "Ongoing") != bool(end or opt):
        bad += 1; print("RESULT", fen, myres, end, val, opt, oval)
print(f"{len(fens)} positions, {bad} mismatches; results: {results}")
