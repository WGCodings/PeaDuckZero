//! PeaDuckZero core: duck chess board representation, move generation,
//! perft and game-state bookkeeping (repetitions, results).
//!
//! # Duck chess rules (as played on chess.com / duckchess.com and by Fairy-Stockfish)
//! * A turn is a normal chess move followed by moving the duck to a
//!   different empty square (White's first turn places it anywhere empty).
//! * The duck can't be captured and blocks every piece: nothing moves onto
//!   it or slides through it.
//! * There is no check: kings may move into attacks and castle out of,
//!   through or into them. Capturing the enemy king wins.
//! * A player with no legal move at the start of their turn wins.
//! * Draws: fifty-move rule and threefold repetition.
//!
//! Internally a turn is two steps on [`Position`]: the move step (a piece
//! move) and the duck step (a duck move, [`Move::duck`]). The side to move
//! only changes after the duck step, so search trees have two levels per
//! turn and the value only changes sign below a duck step.
//!
//! No external dependencies.

pub mod attacks;
pub mod bitboard;
pub mod encoding;
pub mod game;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod position;
pub mod rng;
pub mod types;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use game::{DrawReason, Game, GameResult, WinReason};
pub use movegen::DuckDefence;
pub use moves::{Move, MoveList};
pub use position::{FenError, Position, START_FEN};
pub use types::{Color, Piece, PieceType, Square};

/// Force initialisation of lazily-built lookup tables (slider attacks).
/// Call once at startup so the first search/perft isn't charged for it.
pub fn init() {
    attacks::init();
}
