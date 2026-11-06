//! Testing aid: for every FEN on stdin (move step), print the FEN as this
//! crate writes it, the game result and all legal turns in
//! Fairy-Stockfish notation, tab separated. Used to compare move
//! generation with Fairy-Stockfish (pyffish).

use std::io::BufRead;

use peaduck_core::game::result_of;
use peaduck_core::moves::turn_uci;
use peaduck_core::Position;

fn main() {
    peaduck_core::init();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let pos = Position::from_fen(line.trim()).expect("bad FEN");
        let mut turns: Vec<String> = Vec::new();
        for &m in pos.legal_moves().iter() {
            let mut child = pos;
            child.make_move(m);
            for sq in child.duck_targets() {
                turns.push(turn_uci(m, Some(sq)));
            }
        }
        turns.sort();
        println!(
            "{}\t{:?}\t{}",
            pos.to_fen(),
            result_of(&pos, 0),
            turns.join(" ")
        );
    }
}
