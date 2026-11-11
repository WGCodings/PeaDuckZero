//! Move generation.
//!
//! Duck chess has no check, so every pseudo-legal move is legal: the only
//! extra rule is that the duck blocks (nothing moves onto it or through it).
//! * Move step: all piece moves, castling with an empty path (the king may
//!   be attacked anywhere on it), en passant unless the duck is on the
//!   target square.
//! * Duck step: every empty square except the duck's current one.
//!
//! A position where a king is missing (it was just captured) has no moves.

use crate::attacks::{
    between, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveList};
use crate::position::{Position, BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
use crate::types::{Color, PieceType, Square};

/// What the duck step can do about an attack on our king; see
/// [`Position::duck_defence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuckDefence {
    /// The king isn't attacked: every duck square is safe.
    Free,
    /// Only these duck squares (never empty) block every attack.
    Block(Bitboard),
    /// No duck square saves the king.
    Lost,
}

const PROMOS: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

#[inline(always)]
fn push_pawn_moves(list: &mut MoveList, from: Square, to: Square, capture: bool, promo_rank: bool) {
    if promo_rank {
        for p in PROMOS {
            list.push(Move::new_promo(from, to, p, capture));
        }
    } else {
        list.push(Move::new(
            from,
            to,
            if capture { Move::CAPTURE } else { Move::QUIET },
        ));
    }
}

#[inline(always)]
fn push_targets(list: &mut MoveList, from: Square, targets: Bitboard, enemy: Bitboard) {
    for to in targets {
        let flag = if enemy.contains(to) {
            Move::CAPTURE
        } else {
            Move::QUIET
        };
        list.push(Move::new(from, to, flag));
    }
}

impl Position {
    /// Fill `list` with all legal moves of the next step (cleared first):
    /// piece moves in the move step, duck moves in the duck step.
    pub fn generate_legal(&self, list: &mut MoveList) {
        list.clear();
        if !self.has_king(Color::White) || !self.has_king(Color::Black) {
            return;
        }
        if self.is_duck_step() {
            self.gen_duck(list);
        } else {
            self.gen_pieces(list);
        }
    }

    /// Can the side to move capture the enemy king with its next piece
    /// move (move step only)? Then it has won: search treats such a
    /// position as a win without looking further.
    pub fn king_capture_available(&self) -> bool {
        if self.is_duck_step() {
            return false;
        }
        let us = self.side_to_move();
        let kings = self.pieces_of(!us, PieceType::King);
        if kings.is_empty() {
            return false;
        }
        let k = kings.lsb();
        let blk = self.blockers();
        let rq = self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen);
        let bq = self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen);
        let attackers = (pawn_attacks(!us, k) & self.pieces(PieceType::Pawn))
            | (knight_attacks(k) & self.pieces(PieceType::Knight))
            | (king_attacks(k) & self.pieces(PieceType::King))
            | (bishop_attacks(k, blk) & bq)
            | (rook_attacks(k, blk) & rq);
        (attackers & self.color_bb(us)).any()
    }

    /// Duck step only: which duck squares stop the opponent from capturing
    /// our king with its next move.
    ///
    /// The duck has to leave its square, so the attacks on our king are
    /// computed without it. Then:
    /// * no attack: every duck square is safe ([`DuckDefence::Free`]);
    /// * an attack by a knight, pawn or king, or by sliders whose lines
    ///   share no empty square: every duck square loses ([`DuckDefence::Lost`]);
    /// * otherwise only the squares on every attack line are safe
    ///   ([`DuckDefence::Block`]); every other duck square loses.
    ///
    /// "Loses" means the opponent can capture the king next, unless the
    /// game is drawn first (fifty-move rule, repetition): callers check that.
    /// In the move step, or without both kings, it returns `Free`.
    pub fn duck_defence(&self) -> DuckDefence {
        if !self.is_duck_step() {
            return DuckDefence::Free;
        }
        let us = self.side_to_move();
        let (own_king, their_king) = (
            self.pieces_of(us, PieceType::King),
            self.pieces_of(!us, PieceType::King),
        );
        if own_king.is_empty() || their_king.is_empty() {
            return DuckDefence::Free;
        }
        let k = own_king.lsb();
        let them = self.color_bb(!us);
        let occ = self.occupied();
        let jumpers = (pawn_attacks(us, k) & self.pieces(PieceType::Pawn))
            | (knight_attacks(k) & self.pieces(PieceType::Knight))
            | (king_attacks(k) & self.pieces(PieceType::King));
        if (jumpers & them).any() {
            return DuckDefence::Lost;
        }
        let rq = self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen);
        let bq = self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen);
        let sliders = ((bishop_attacks(k, occ) & bq) | (rook_attacks(k, occ) & rq)) & them;
        if sliders.is_empty() {
            return DuckDefence::Free;
        }
        let mut safe = !(occ | self.duck_bb());
        for s in sliders {
            safe &= between(s, k);
        }
        if safe.is_empty() {
            DuckDefence::Lost
        } else {
            DuckDefence::Block(safe)
        }
    }

    /// Squares the duck can move to in the duck step.
    #[inline]
    pub fn duck_targets(&self) -> Bitboard {
        !self.blockers()
    }

    fn gen_duck(&self, list: &mut MoveList) {
        for sq in self.duck_targets() {
            list.push(Move::duck(sq));
        }
    }

    fn gen_pieces(&self, list: &mut MoveList) {
        let us = self.side_to_move();
        let own = self.color_bb(us);
        let enemy = self.color_bb(!us);
        let blk = self.blockers();
        let targets = !(own | self.duck_bb());

        for from in self.pieces_of(us, PieceType::King) {
            push_targets(list, from, king_attacks(from) & targets, enemy);
        }
        self.gen_castling(list, us, blk);
        for from in self.pieces_of(us, PieceType::Knight) {
            push_targets(list, from, knight_attacks(from) & targets, enemy);
        }
        let bq = (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)) & own;
        let rq = (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)) & own;
        for from in bq {
            push_targets(list, from, bishop_attacks(from, blk) & targets, enemy);
        }
        for from in rq {
            push_targets(list, from, rook_attacks(from, blk) & targets, enemy);
        }

        let up: i8 = if us == Color::White { 8 } else { -8 };
        let start_rank = if us == Color::White {
            Bitboard::RANK_2
        } else {
            Bitboard::RANK_7
        };
        for from in self.pieces_of(us, PieceType::Pawn) {
            let promo_rank = from.relative_rank(us) == 6;
            let one = Square((from.0 as i8 + up) as u8);
            if !blk.contains(one) {
                push_pawn_moves(list, from, one, false, promo_rank);
                let two = Square((one.0 as i8 + up) as u8);
                if start_rank.contains(from) && !blk.contains(two) {
                    list.push(Move::new(from, two, Move::DOUBLE_PUSH));
                }
            }
            let att = pawn_attacks(us, from);
            for to in att & enemy {
                push_pawn_moves(list, from, to, true, promo_rank);
            }
            if let Some(ep) = self.ep_square() {
                // the duck can't be on `ep` (placing it there clears ep)
                if att.contains(ep) {
                    list.push(Move::new(from, ep, Move::EP_CAPTURE));
                }
            }
        }
    }

    /// Castling needs the rights and an empty path between king and rook
    /// (no duck either). Attacks don't matter: there is no check.
    fn gen_castling(&self, list: &mut MoveList, us: Color, blk: Bitboard) {
        let rights = self.castling_rights();
        let (oo, ooo, k) = match us {
            Color::White => (WHITE_OO, WHITE_OOO, Square::E1),
            Color::Black => (BLACK_OO, BLACK_OOO, Square::E8),
        };
        if rights & oo != 0 && (blk & Bitboard(0b11 << (k.0 + 1))).is_empty() {
            list.push(Move::new(k, Square(k.0 + 2), Move::KING_CASTLE));
        }
        if rights & ooo != 0 && (blk & Bitboard(0b111 << (k.0 - 3))).is_empty() {
            list.push(Move::new(k, Square(k.0 - 2), Move::QUEEN_CASTLE));
        }
    }

    /// Parse a step in this position: a piece move (`e2e4`, `e7e8q`) in the
    /// move step, a square (`d5`, also `@d5`) in the duck step.
    pub fn parse_move(&self, s: &str) -> Option<Move> {
        if self.is_duck_step() {
            let sq = Square::parse(s.trim_start_matches('@'))?;
            return self.duck_targets().contains(sq).then(|| Move::duck(sq));
        }
        let mut list = MoveList::new();
        self.generate_legal(&mut list);
        list.iter().copied().find(|m| m.to_uci() == s)
    }

    /// Convenience: legal moves of the next step as a fresh list.
    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.generate_legal(&mut list);
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moves(fen: &str) -> Vec<String> {
        crate::init();
        let mut v: Vec<String> = Position::from_fen(fen)
            .unwrap()
            .legal_moves()
            .iter()
            .map(|m| m.to_uci())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn start_has_20_moves_and_32_duck_squares() {
        assert_eq!(moves(crate::START_FEN).len(), 20);
        let mut p = Position::startpos();
        p.make_move(p.parse_move("e2e4").unwrap());
        assert_eq!(p.legal_moves().len(), 32); // e2 is free again, e4 is not
    }

    #[test]
    fn duck_blocks_sliders_pawns_and_castling() {
        // duck on f1: no O-O, bishop can't pass; duck on e3 blocks e-pawn
        let m = moves("4k3/8/8/8/8/4*3/4P3/R3K2R w KQ - 0 1");
        assert!(m.contains(&"e1c1".to_string()));
        assert!(m.contains(&"e1g1".to_string()));
        assert!(!m.iter().any(|s| s.starts_with("e2")));
        let m = moves("4k3/8/8/8/8/8/8/R3K*1R w KQ - 0 1");
        assert!(!m.contains(&"e1g1".to_string()));
        assert!(m.contains(&"e1c1".to_string()));
        assert!(!m.contains(&"h1e1".to_string()));
        assert!(m.contains(&"h1g1".to_string()));
    }

    #[test]
    fn no_check_rules() {
        // king may move next to the enemy king and castle through attacks
        let m = moves("4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1");
        assert!(m.contains(&"e1f2".to_string()));
        assert!(m.contains(&"e1g1".to_string()));
        assert!(m.contains(&"e1d2".to_string()));
    }

    #[test]
    fn duck_must_move() {
        crate::init();
        let p = Position::from_fen("4k3/8/8/3*4/8/8/8/4K3 w - - 0 1 duck").unwrap();
        let m = p.legal_moves();
        assert_eq!(m.len(), 64 - 3);
        assert!(m
            .iter()
            .all(|m| m.is_duck() && m.to() != Square::parse("d5").unwrap()));
    }

    /// `king_capture_available` agrees with the move list in random games.
    #[test]
    fn king_capture_detection_matches_movegen() {
        crate::init();
        let mut rng = crate::rng::Rng::new(5);
        let (mut yes, mut no) = (0, 0);
        for _ in 0..300 {
            let mut p = Position::startpos();
            for _ in 0..200 {
                let moves = p.legal_moves();
                if moves.is_empty() {
                    break;
                }
                if !p.is_duck_step() {
                    let capture = moves
                        .iter()
                        .any(|m| m.is_capture() && p.piece_on(m.to()).kind() == PieceType::King);
                    assert_eq!(p.king_capture_available(), capture, "{}", p.to_fen());
                    if capture {
                        yes += 1;
                    } else {
                        no += 1;
                    }
                } else {
                    assert!(!p.king_capture_available());
                }
                p.make_move(moves[rng.below(moves.len() as u64) as usize]);
            }
        }
        assert!(yes > 100 && no > 1000, "{yes} {no}");
    }

    #[test]
    fn king_capture_ends_movegen() {
        crate::init();
        let mut p = Position::from_fen("4k3/8/8/8/8/8/8/4K2r b - - 0 1").unwrap();
        p.make_move(p.parse_move("h1e1").unwrap());
        assert!(!p.has_king(Color::White));
        assert!(p.legal_moves().is_empty());
    }

    /// `duck_defence` agrees with trying every duck square in random games.
    #[test]
    fn duck_defence_matches_brute_force() {
        crate::init();
        let mut rng = crate::rng::Rng::new(11);
        let (mut free, mut block, mut lost) = (0, 0, 0);
        for _ in 0..400 {
            let mut p = Position::startpos();
            for _ in 0..240 {
                let moves = p.legal_moves();
                if moves.is_empty() {
                    break;
                }
                if p.is_duck_step() {
                    let mut safe = Bitboard(0);
                    for m in moves.iter() {
                        let mut q = p;
                        q.make_move(*m);
                        if !q.king_capture_available() {
                            safe |= Bitboard(1u64 << m.to().0);
                        }
                    }
                    let expect = if safe == p.duck_targets() {
                        free += 1;
                        DuckDefence::Free
                    } else if safe.is_empty() {
                        lost += 1;
                        DuckDefence::Lost
                    } else {
                        block += 1;
                        DuckDefence::Block(safe)
                    };
                    assert_eq!(p.duck_defence(), expect, "{}", p.to_fen());
                } else {
                    assert_eq!(p.duck_defence(), DuckDefence::Free);
                }
                let m = moves[rng.below(moves.len() as u64) as usize];
                p.make_move(m);
            }
        }
        assert!(
            free > 1000 && block > 100 && lost > 100,
            "{free} {block} {lost}"
        );
    }
}
