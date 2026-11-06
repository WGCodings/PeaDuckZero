//! Perft in whole turns (a piece move plus a duck move), counted the way
//! Fairy-Stockfish counts duck chess moves (`e2e4,e4d5` is one move).

use crate::moves::{Move, MoveList};
use crate::position::Position;

/// Number of duck squares after a piece move (Fairy-Stockfish also counts
/// a king capture once per duck square).
fn duck_options(pos: &Position) -> u64 {
    pos.duck_targets().count() as u64
}

/// Count whole turns to `depth` from a position in the move step. Bulk
/// counting at the last turn: piece moves × duck squares.
pub fn perft(pos: &Position, depth: u32) -> u64 {
    debug_assert!(!pos.is_duck_step());
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let mut ducks = MoveList::new();
    let mut nodes = 0;
    for &m in list.iter() {
        let mut child = *pos;
        child.make_move(m);
        if depth == 1 {
            nodes += duck_options(&child);
            continue;
        }
        child.generate_legal(&mut ducks);
        for &d in ducks.iter() {
            let mut grandchild = child;
            grandchild.make_move(d);
            nodes += perft(&grandchild, depth - 1);
        }
    }
    nodes
}

/// Per-piece-move breakdown, sorted by UCI string.
pub fn divide(pos: &Position, depth: u32) -> Vec<(Move, u64)> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let mut out: Vec<(Move, u64)> = list
        .iter()
        .map(|&m| {
            let mut child = *pos;
            child.make_move(m);
            let n = if depth <= 1 {
                duck_options(&child)
            } else {
                child
                    .legal_moves()
                    .iter()
                    .map(|&d| {
                        let mut g = child;
                        g.make_move(d);
                        perft(&g, depth - 1)
                    })
                    .sum()
            };
            (m, n)
        })
        .collect();
    out.sort_by_key(|(m, _)| m.to_uci());
    out
}

/// Perft that checks the incremental hash at every step. Slow; tests only.
pub fn perft_checked(pos: &Position, depth: u32) -> u64 {
    assert_eq!(
        pos.hash(),
        pos.compute_hash(),
        "hash mismatch at {}",
        pos.to_fen()
    );
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0;
    for &m in pos.legal_moves().iter() {
        let mut child = *pos;
        child.make_move(m);
        assert_eq!(
            child.hash(),
            child.compute_hash(),
            "hash mismatch at {}",
            child.to_fen()
        );
        if depth == 1 {
            nodes += duck_options(&child);
            continue;
        }
        for &d in child.legal_moves().iter() {
            let mut g = child;
            g.make_move(d);
            nodes += perft_checked(&g, depth - 1);
        }
    }
    nodes
}
