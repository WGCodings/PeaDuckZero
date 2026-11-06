use std::fmt;

use crate::attacks::pawn_attacks;
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square};
use crate::zobrist::KEYS;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const WHITE_OO: u8 = 1;
pub const WHITE_OOO: u8 = 2;
pub const BLACK_OO: u8 = 4;
pub const BLACK_OOO: u8 = 8;

/// Castling rights that survive a move touching each square.
const CASTLE_KEEP: [u8; 64] = {
    let mut t = [0xFu8; 64];
    t[0] = !WHITE_OOO & 0xF; // a1
    t[4] = !(WHITE_OO | WHITE_OOO) & 0xF; // e1
    t[7] = !WHITE_OO & 0xF; // h1
    t[56] = !BLACK_OOO & 0xF; // a8
    t[60] = !(BLACK_OO | BLACK_OOO) & 0xF; // e8
    t[63] = !BLACK_OO & 0xF; // h8
    t
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenError(pub String);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl std::error::Error for FenError {}

/// Full board state, including the duck and which step of the turn is
/// next. `Copy` on purpose: search uses copy-make.
///
/// A turn is two steps: in the move step the side to move plays a piece
/// move ([`Position::make_move`] with a normal move), which leaves the same
/// side to move in the duck step; the duck move ([`Move::duck`]) then hands
/// the turn to the opponent.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    board: [Piece; 64],
    stm: Color,
    castling: u8,
    /// Only set when an enemy pawn of the side that will move next can
    /// capture onto it and the duck isn't on it. Keeps repetitions exact.
    ep: Option<Square>,
    /// None until White's first duck step.
    duck: Option<Square>,
    duck_step: bool,
    halfmove: u16,
    fullmove: u16,
    hash: u64,
}

impl Default for Position {
    fn default() -> Self {
        Position::startpos()
    }
}

impl Position {
    fn empty() -> Position {
        Position {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            board: [Piece::NONE; 64],
            stm: Color::White,
            castling: 0,
            ep: None,
            duck: None,
            duck_step: false,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("start FEN is valid")
    }

    // ----------------------------------------------------------- accessors

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.stm
    }
    /// Is the duck step of the turn next (the piece move was just played)?
    #[inline(always)]
    pub fn is_duck_step(&self) -> bool {
        self.duck_step
    }
    #[inline(always)]
    pub fn duck(&self) -> Option<Square> {
        self.duck
    }
    #[inline(always)]
    pub fn duck_bb(&self) -> Bitboard {
        match self.duck {
            Some(d) => d.bb(),
            None => Bitboard::EMPTY,
        }
    }
    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }
    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep
    }
    /// Piece moves since the last capture or pawn move (duck steps don't count).
    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove
    }
    #[inline(always)]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove
    }
    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.idx()]
    }
    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()]
    }
    #[inline(always)]
    pub fn color_bb(&self, c: Color) -> Bitboard {
        self.colors[c.idx()]
    }
    #[inline(always)]
    pub fn pieces_of(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()] & self.colors[c.idx()]
    }
    /// Squares with a piece (the duck not included).
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }
    /// Squares nothing can move onto or through: pieces and the duck.
    #[inline(always)]
    pub fn blockers(&self) -> Bitboard {
        self.occupied() | self.duck_bb()
    }
    #[inline(always)]
    pub fn has_king(&self, c: Color) -> bool {
        self.pieces_of(c, PieceType::King).any()
    }

    // ------------------------------------------------------ piece updates

    #[inline(always)]
    fn add_piece(&mut self, p: Piece, sq: Square) {
        let b = sq.bb();
        self.pieces[p.kind().idx()] |= b;
        self.colors[p.color().idx()] |= b;
        self.board[sq.idx()] = p;
        self.hash ^= KEYS.piece_sq[p.idx()][sq.idx()];
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let p = self.board[sq.idx()];
        debug_assert!(!p.is_none());
        let b = sq.bb();
        self.pieces[p.kind().idx()] ^= b;
        self.colors[p.color().idx()] ^= b;
        self.board[sq.idx()] = Piece::NONE;
        self.hash ^= KEYS.piece_sq[p.idx()][sq.idx()];
        p
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        let p = self.remove_piece(from);
        self.add_piece(p, to);
    }

    /// Could a pawn of `side` capture en passant onto `sq`?
    #[inline(always)]
    fn ep_capturable(&self, sq: Square, side: Color) -> bool {
        (pawn_attacks(!side, sq) & self.pieces_of(side, PieceType::Pawn)).any()
    }

    #[inline(always)]
    fn clear_ep(&mut self) {
        if let Some(ep) = self.ep.take() {
            self.hash ^= KEYS.ep_file[ep.file() as usize];
        }
    }

    // ------------------------------------------------------------- moves

    /// Play one step: a piece move in the move step, a duck move
    /// ([`Move::duck`]) in the duck step. No legality check is done.
    pub fn make_move(&mut self, m: Move) {
        if m.is_duck() {
            self.place_duck(m.to());
        } else {
            self.make_piece_move(m);
        }
    }

    fn make_piece_move(&mut self, m: Move) {
        debug_assert!(!self.duck_step, "piece move {m} in the duck step");
        let us = self.stm;
        let from = m.from();
        let to = m.to();
        let moving = self.board[from.idx()];
        debug_assert!(
            !moving.is_none() && moving.color() == us,
            "bad move {m} in {}",
            self.to_fen()
        );

        self.hash ^= KEYS.castling[self.castling as usize];
        self.clear_ep();
        self.halfmove += 1;

        if m.is_capture() {
            let cap_sq = if m.is_ep() { Square(to.0 ^ 8) } else { to };
            self.remove_piece(cap_sq);
            self.halfmove = 0;
        }

        self.move_piece(from, to);

        match moving.kind() {
            PieceType::Pawn => {
                self.halfmove = 0;
                if let Some(promo) = m.promo_piece() {
                    self.remove_piece(to);
                    self.add_piece(Piece::new(us, promo), to);
                } else if m.is_double_push() {
                    let ep = Square((from.0 + to.0) / 2);
                    if self.ep_capturable(ep, !us) {
                        self.ep = Some(ep);
                        self.hash ^= KEYS.ep_file[ep.file() as usize];
                    }
                }
            }
            PieceType::King if m.is_castle() => {
                let (rf, rt) = if m.flag() == Move::KING_CASTLE {
                    (Square(to.0 + 1), Square(to.0 - 1))
                } else {
                    (Square(to.0 - 2), Square(to.0 + 1))
                };
                self.move_piece(rf, rt);
            }
            _ => {}
        }

        self.castling &= CASTLE_KEEP[from.idx()] & CASTLE_KEEP[to.idx()];
        self.hash ^= KEYS.castling[self.castling as usize];

        self.duck_step = true;
        self.hash ^= KEYS.duck_step;
    }

    fn place_duck(&mut self, sq: Square) {
        debug_assert!(self.duck_step, "duck move in the move step");
        debug_assert!(Some(sq) != self.duck && !self.occupied().contains(sq));
        if let Some(old) = self.duck {
            self.hash ^= KEYS.duck[old.idx()];
        }
        self.duck = Some(sq);
        self.hash ^= KEYS.duck[sq.idx()];
        if self.ep == Some(sq) {
            self.clear_ep(); // the duck blocks the en passant capture
        }
        self.duck_step = false;
        self.hash ^= KEYS.duck_step;
        if self.stm == Color::Black {
            self.fullmove += 1;
        }
        self.stm = !self.stm;
        self.hash ^= KEYS.side;
    }

    /// The 12 piece bitboards in the order white P N B R Q K, black P N B R Q K
    /// (the self-play data format's compact board).
    pub fn piece_bitboards(&self) -> [u64; 12] {
        let mut out = [0u64; 12];
        for (ci, c) in Color::BOTH.iter().enumerate() {
            for pt in PieceType::ALL {
                out[ci * 6 + pt.idx()] = self.pieces_of(*c, pt).0;
            }
        }
        out
    }

    /// Rebuild a position from the compact form stored in self-play data.
    /// No legality validation: the data was produced by this engine.
    #[allow(clippy::too_many_arguments)]
    pub fn from_compact(
        pieces: &[u64; 12],
        stm: Color,
        castling: u8,
        ep: Option<Square>,
        duck: Option<Square>,
        duck_step: bool,
        halfmove: u16,
    ) -> Position {
        let mut pos = Position::empty();
        for (i, &bb) in pieces.iter().enumerate() {
            let color = if i < 6 { Color::White } else { Color::Black };
            let p = Piece::new(color, PieceType::from_index(i % 6));
            for sq in Bitboard(bb) {
                pos.add_piece(p, sq);
            }
        }
        pos.stm = stm;
        pos.castling = castling & 0xF;
        pos.ep = ep;
        pos.duck = duck;
        pos.duck_step = duck_step;
        pos.halfmove = halfmove;
        pos.hash = pos.compute_hash();
        pos
    }

    /// Hash recomputed from scratch (used to verify incremental updates).
    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in self.occupied() {
            h ^= KEYS.piece_sq[self.board[sq.idx()].idx()][sq.idx()];
        }
        h ^= KEYS.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            h ^= KEYS.ep_file[ep.file() as usize];
        }
        if let Some(d) = self.duck {
            h ^= KEYS.duck[d.idx()];
        }
        if self.stm == Color::Black {
            h ^= KEYS.side;
        }
        if self.duck_step {
            h ^= KEYS.duck_step;
        }
        h
    }

    // --------------------------------------------------------------- FEN

    /// FEN with the duck as `*` (Fairy-Stockfish's notation). An optional
    /// 7th field `duck` marks a position in the duck step (internal use;
    /// Fairy-Stockfish FENs are always in the move step).
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().ok_or_else(|| FenError("empty".into()))?;
        let stm = parts.next().unwrap_or("w");
        let castling = parts.next().unwrap_or("-");
        let ep = parts.next().unwrap_or("-");
        let halfmove = parts.next().unwrap_or("0");
        let fullmove = parts.next().unwrap_or("1");
        let step = parts.next();

        let mut pos = Position::empty();

        let ranks: Vec<&str> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError(format!("expected 8 ranks, got {}", ranks.len())));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for c in rank_str.chars() {
                if let Some(d) = c.to_digit(10) {
                    if !(1..=8).contains(&d) {
                        return Err(FenError(format!("bad digit '{c}'")));
                    }
                    file += d as u8;
                } else {
                    if file >= 8 {
                        return Err(FenError(format!("rank {} too long", rank + 1)));
                    }
                    let sq = Square::new(file, rank);
                    if c == '*' {
                        if pos.duck.is_some() {
                            return Err(FenError("more than one duck".into()));
                        }
                        pos.duck = Some(sq);
                    } else {
                        let p = Piece::from_char(c)
                            .ok_or_else(|| FenError(format!("bad piece '{c}'")))?;
                        pos.add_piece(p, sq);
                    }
                    file += 1;
                }
                if file > 8 {
                    return Err(FenError(format!("rank {} too long", rank + 1)));
                }
            }
            if file != 8 {
                return Err(FenError(format!("rank {} has {} files", rank + 1, file)));
            }
        }

        for c in Color::BOTH {
            // no king = it was captured (the game is over)
            if pos.pieces_of(c, PieceType::King).count() > 1 {
                return Err(FenError(format!("{c:?} has more than one king")));
            }
        }
        if (pos.pieces(PieceType::Pawn) & (Bitboard::RANK_1 | Bitboard::RANK_8)).any() {
            return Err(FenError("pawn on back rank".into()));
        }

        pos.stm = match stm {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError(format!("bad side to move '{stm}'"))),
        };
        pos.duck_step = match step {
            None => false,
            Some("duck") => true,
            Some(s) => return Err(FenError(format!("bad step '{s}' (only 'duck')"))),
        };

        if castling != "-" {
            for c in castling.chars() {
                pos.castling |= match c {
                    'K' => WHITE_OO,
                    'Q' => WHITE_OOO,
                    'k' => BLACK_OO,
                    'q' => BLACK_OOO,
                    _ => return Err(FenError(format!("bad castling char '{c}'"))),
                };
            }
        }
        // Drop rights that don't match king/rook placement.
        let wk = pos.piece_on(Square::E1) == Piece::new(Color::White, PieceType::King);
        let bk = pos.piece_on(Square::E8) == Piece::new(Color::Black, PieceType::King);
        let wr = Piece::new(Color::White, PieceType::Rook);
        let br = Piece::new(Color::Black, PieceType::Rook);
        if !(wk && pos.piece_on(Square::H1) == wr) {
            pos.castling &= !WHITE_OO;
        }
        if !(wk && pos.piece_on(Square::A1) == wr) {
            pos.castling &= !WHITE_OOO;
        }
        if !(bk && pos.piece_on(Square::H8) == br) {
            pos.castling &= !BLACK_OO;
        }
        if !(bk && pos.piece_on(Square::A8) == br) {
            pos.castling &= !BLACK_OOO;
        }

        if ep != "-" {
            let sq = Square::parse(ep).ok_or_else(|| FenError(format!("bad ep square '{ep}'")))?;
            // In the move step the side to move captures; in the duck step
            // the opponent will.
            let capturer = if pos.duck_step { !pos.stm } else { pos.stm };
            let expected_rank = if capturer == Color::White { 5 } else { 2 };
            if sq.rank() == expected_rank
                && pos.ep_capturable(sq, capturer)
                && pos.duck != Some(sq)
                && !pos.occupied().contains(sq)
            {
                pos.ep = Some(sq);
            }
        }

        pos.halfmove = halfmove
            .parse()
            .map_err(|_| FenError(format!("bad halfmove '{halfmove}'")))?;
        pos.fullmove = fullmove
            .parse::<u16>()
            .map_err(|_| FenError(format!("bad fullmove '{fullmove}'")))?
            .max(1);

        pos.hash = pos.compute_hash();
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::with_capacity(96);
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                let p = self.board[sq.idx()];
                let c = if self.duck == Some(sq) {
                    Some('*')
                } else if p.is_none() {
                    None
                } else {
                    Some(p.to_char())
                };
                match c {
                    None => empty += 1,
                    Some(c) => {
                        if empty > 0 {
                            s.push(char::from(b'0' + empty));
                            empty = 0;
                        }
                        s.push(c);
                    }
                }
            }
            if empty > 0 {
                s.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, c) in [
                (WHITE_OO, 'K'),
                (WHITE_OOO, 'Q'),
                (BLACK_OO, 'k'),
                (BLACK_OOO, 'q'),
            ] {
                if self.castling & bit != 0 {
                    s.push(c);
                }
            }
        }
        s.push(' ');
        match self.ep {
            Some(sq) => s.push_str(&sq.to_string()),
            None => s.push('-'),
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        if self.duck_step {
            s.push_str(" duck");
        }
        s
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, " +---+---+---+---+---+---+---+---+")?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::new(file, rank);
                let c = if self.duck == Some(sq) {
                    '*'
                } else {
                    self.board[sq.idx()].to_char()
                };
                write!(f, " | {c}")?;
            }
            writeln!(f, " | {}", rank + 1)?;
            writeln!(f, " +---+---+---+---+---+---+---+---+")?;
        }
        writeln!(f, "   a   b   c   d   e   f   g   h")?;
        writeln!(f)?;
        writeln!(f, "Fen: {}", self.to_fen())?;
        write!(f, "Key: {:016X}", self.hash)
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Position({})", self.to_fen())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fen_roundtrip() {
        for fen in [
            START_FEN,
            "rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/7* w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 duck",
        ] {
            assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
        }
    }

    #[test]
    fn compact_roundtrip() {
        let p =
            Position::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/2*5/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let q = Position::from_compact(
            &p.piece_bitboards(),
            p.side_to_move(),
            p.castling_rights(),
            p.ep_square(),
            p.duck(),
            p.is_duck_step(),
            p.halfmove_clock(),
        );
        assert_eq!(p.to_fen(), q.to_fen().replace(" 1", " 3"));
        assert_eq!(p.hash(), q.hash());
    }

    #[test]
    fn duck_on_ep_square_drops_it() {
        let p =
            Position::from_fen("rnbqkbnr/ppp1p1pp/5*2/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        assert_eq!(p.ep_square(), None);
    }

    #[test]
    fn a_turn_is_two_steps() {
        crate::init();
        let mut p = Position::startpos();
        let m = p.parse_move("e2e4").unwrap();
        p.make_move(m);
        assert!(p.is_duck_step());
        assert_eq!(p.side_to_move(), Color::White);
        assert_eq!(p.hash(), p.compute_hash());
        p.make_move(Move::duck(Square::parse("d5").unwrap()));
        assert!(!p.is_duck_step());
        assert_eq!(p.side_to_move(), Color::Black);
        assert_eq!(p.hash(), p.compute_hash());
        assert_eq!(
            p.to_fen(),
            "rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        );
    }

    #[test]
    fn rejects_bad_fens() {
        for fen in [
            "",
            "4kk2/8/8/8/8/8/8/4K3 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",
            "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/**6/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ] {
            assert!(Position::from_fen(fen).is_err(), "{fen}");
        }
        // No check in duck chess: the side not to move may be attacked.
        assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K2r b - - 0 1").is_ok());
    }
}
