//! `peaduckzero bench` — deterministic node count (a build signature) plus
//! an NPS figure.
//!
//! Runs exactly what `go depth 3` does with the MCTS search (the default
//! mode) on every bench position, silently, and prints the total node count
//! (playouts) and the overall speed (total nodes / total time).
//!
//! The node count is deterministic (no randomness in the search,
//! single-threaded inference) but depends on the network: MCTS depth
//! is the average playout depth, and how fast that grows depends on the
//! policy and values. A new embedded network gives a new bench number.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use peaduck_core::Game;

use crate::engine::{CommonOptions, SearchLimits};
use crate::policy::PolicyNet;
use crate::search::{mcts, SearchContext};

pub const BENCH_DEPTH: u32 = 6;
/// Per-position playout cap: a flat (e.g. untrained) policy deepens the
/// tree very slowly, so without it the bench could run for minutes.
pub const BENCH_MAX_NODES: u64 = 20_000;

pub const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2*2/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "4k3/8/8/q7/8/8/8/4K3 w - - 0 1 duck",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/7* w - - 0 1",
    "8/8/4k3/3p4/3P1K2/8/8/8 w - - 0 1",
];

pub fn run(net: &mut PolicyNet) -> u64 {
    let limits = SearchLimits {
        depth: Some(BENCH_DEPTH),
        nodes: Some(BENCH_MAX_NODES),
        ..Default::default()
    };
    let common = CommonOptions::default();
    let stop = AtomicBool::new(false);
    let ctx = SearchContext {
        limits: &limits,
        common: &common,
        stop: &stop,
        show_wdl: false,
        verbose_move_stats: false,
        params: &crate::search::SearchParams::default(),
    };

    let start = Instant::now();
    let mut total = 0u64;
    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let game = Game::from_fen(fen).expect("bench FEN");
        let t = Instant::now();
        let (_, nodes) = mcts::run(net, &game, &ctx, false);
        let ms = t.elapsed().as_millis();
        total += nodes;
        println!(
            "position {:>2} nodes {nodes:>6} nps {:>6} time {ms:>6} ms  {fen}",
            i + 1,
            nodes as u128 * 1000 / ms.max(1)
        );
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (total as f64 / elapsed) as u64;
    println!("{total} nodes {nps} nps");
    total
}
