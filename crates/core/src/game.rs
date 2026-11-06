//! A game = current position + hash history, so repetitions and the final
//! result can be detected. Self-play and UCI both work on a `Game`.

use crate::moves::{Move, MoveList};
use crate::position::{FenError, Position};
use crate::types::{Color, Square};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinReason {
    /// The loser's king was captured.
    KingCaptured,
    /// The winner had no legal move at the start of their turn.
    NoMoves,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    FiftyMoves,
    Repetition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    /// The given colour won.
    Win(Color, WinReason),
    Draw(DrawReason),
}

impl GameResult {
    /// PGN-style result string.
    pub fn pgn(self) -> &'static str {
        match self {
            GameResult::Ongoing => "*",
            GameResult::Win(Color::White, _) => "1-0",
            GameResult::Win(Color::Black, _) => "0-1",
            GameResult::Draw(_) => "1/2-1/2",
        }
    }

    pub fn is_over(self) -> bool {
        self != GameResult::Ongoing
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pos: Position,
    /// Hashes of the earlier positions at the start of a turn (move step),
    /// one per turn.
    history: Vec<u64>,
    /// Every step played (piece moves and duck moves).
    moves: Vec<Move>,
    start: Position,
}

impl Default for Game {
    fn default() -> Self {
        Game::new(Position::startpos())
    }
}

impl Game {
    pub fn new(pos: Position) -> Game {
        Game {
            pos,
            history: Vec::with_capacity(256),
            moves: Vec::with_capacity(512),
            start: pos,
        }
    }

    pub fn from_fen(fen: &str) -> Result<Game, FenError> {
        Ok(Game::new(Position::from_fen(fen)?))
    }

    #[inline]
    pub fn position(&self) -> &Position {
        &self.pos
    }

    pub fn start_position(&self) -> &Position {
        &self.start
    }

    /// Every step played: piece moves and duck moves alternate.
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Steps played (two per turn).
    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    /// Play a step assumed to be legal.
    pub fn push(&mut self, m: Move) {
        if !self.pos.is_duck_step() {
            self.history.push(self.pos.hash());
        }
        self.moves.push(m);
        self.pos.make_move(m);
    }

    /// Play one step from text: a piece move or a duck square (see
    /// [`Position::parse_move`]). Returns false (and does nothing) if illegal.
    pub fn push_step(&mut self, s: &str) -> bool {
        match self.pos.parse_move(s) {
            Some(m) => {
                self.push(m);
                true
            }
            None => false,
        }
    }

    /// Play a whole turn in UCI notation: `e2e4,e4d5` (Fairy-Stockfish:
    /// piece move, comma, the piece's square and the duck's square). The
    /// duck part may be missing when the move ends the game. In the duck
    /// step a duck square on its own (`d5`) is accepted too. Returns false
    /// (and changes nothing) if illegal.
    pub fn push_uci(&mut self, s: &str) -> bool {
        let saved = (self.pos, self.history.len(), self.moves.len());
        let ok = (|| {
            let (mv, duck) = match s.split_once(',') {
                Some((m, d)) => (m, Some(d)),
                None => (s, None),
            };
            if !self.push_step(mv) {
                return false;
            }
            match duck {
                _ if !self.pos.is_duck_step() => true, // a duck square on its own
                _ if self.result().is_over() => true,  // king captured: duck irrelevant
                Some(d) if d.len() >= 2 => self.push_step(&d[d.len() - 2..]),
                _ => false,
            }
        })();
        if !ok {
            self.pos = saved.0;
            self.history.truncate(saved.1);
            self.moves.truncate(saved.2);
        }
        ok
    }

    /// How many earlier positions are identical to the current one (same
    /// pieces, duck, side to move, castling and ep rights). Counted at the
    /// start of a turn; always 0 in the duck step.
    pub fn repetitions(&self) -> usize {
        if self.pos.is_duck_step() {
            return 0;
        }
        let key = self.pos.hash();
        let n = self.history.len();
        let lookback = (self.pos.halfmove_clock() as usize).min(n);
        // Same side to move: step back 2 turns at a time, starting 2 back.
        (2..=lookback)
            .step_by(2)
            .filter(|&back| self.history[n - back] == key)
            .count()
    }

    pub fn result(&self) -> GameResult {
        result_of(&self.pos, self.repetitions())
    }
}

/// Result of a position given its repetition count (see
/// [`Game::repetitions`]); search uses this on its own copies.
pub fn result_of(pos: &Position, repetitions: usize) -> GameResult {
    for c in Color::BOTH {
        if !pos.has_king(c) {
            return GameResult::Win(!c, WinReason::KingCaptured);
        }
    }
    if pos.is_duck_step() {
        return GameResult::Ongoing; // there is always an empty square
    }
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    if list.is_empty() {
        return GameResult::Win(pos.side_to_move(), WinReason::NoMoves);
    }
    if pos.halfmove_clock() >= 100 {
        return GameResult::Draw(DrawReason::FiftyMoves);
    }
    if repetitions >= 2 {
        return GameResult::Draw(DrawReason::Repetition);
    }
    GameResult::Ongoing
}

/// A legal duck square in `pos` (duck step), used to complete a UCI turn
/// when the engine has no preference (e.g. after a king capture, where
/// Fairy-Stockfish still expects one).
pub fn any_duck_square(pos: &Position) -> Option<Square> {
    let t = pos.duck_targets();
    t.any().then(|| t.lsb())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefold_by_knight_shuffle() {
        crate::init();
        let mut g = Game::default();
        // the ducks shuffle between a3/h6 too, so positions repeat exactly
        let turns = [
            "g1f3,f3a3",
            "g8f6,f6h6",
            "f3g1,g1a4",
            "f6g8,g8h5",
            "g1f3,f3a3",
            "g8f6,f6h6",
            "f3g1,g1a4",
            "f6g8,g8h5",
        ];
        for t in turns {
            assert_eq!(g.result(), GameResult::Ongoing);
            assert!(g.push_uci(t), "{t}");
        }
        assert_eq!(g.repetitions(), 1);
        for t in &turns[..4] {
            assert!(g.push_uci(t), "{t}");
        }
        assert_eq!(g.repetitions(), 2);
        assert_eq!(g.result(), GameResult::Draw(DrawReason::Repetition));
    }

    #[test]
    fn king_capture_wins() {
        crate::init();
        let mut g = Game::from_fen("4k3/8/8/8/8/8/8/4K2r b - - 0 1").unwrap();
        assert!(g.push_uci("h1e1,e1h1"));
        assert_eq!(
            g.result(),
            GameResult::Win(Color::Black, WinReason::KingCaptured)
        );
        assert_eq!(g.ply(), 1); // the duck part is ignored after the capture
    }

    #[test]
    fn no_moves_wins() {
        crate::init();
        // every white piece is blocked (by own pieces, black pawns, the duck)
        let g = Game::from_fen("4k3/8/8/8/5p1p/4pP*P/4P1PP/6NK w - - 0 1").unwrap();
        assert_eq!(
            g.result(),
            GameResult::Win(Color::White, WinReason::NoMoves)
        );
    }

    #[test]
    fn fifty_moves() {
        crate::init();
        let g = Game::from_fen("7k/8/6K1/8/8/8/8/R7 w - - 100 80").unwrap();
        assert_eq!(g.result(), GameResult::Draw(DrawReason::FiftyMoves));
    }

    #[test]
    fn illegal_uci_rejected_atomically() {
        crate::init();
        let mut g = Game::default();
        assert!(!g.push_uci("e2e5,e5d5"));
        assert!(!g.push_uci("e2e4")); // duck part missing
        assert!(!g.push_uci("e2e4,e4e4")); // duck onto the pawn
        assert_eq!(g.ply(), 0);
        assert!(g.push_uci("e2e4,e4d5"));
        assert!(!g.push_uci("d7d5,d5e6")); // d5 is the duck
        assert_eq!(g.ply(), 2);
    }
}
