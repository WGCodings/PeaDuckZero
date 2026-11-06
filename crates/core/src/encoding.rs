//! Network encodings shared by the engine, self-play and (mirrored in
//! Python) the trainer. **Changing anything here invalidates every trained
//! network**: bump `ENCODING_VERSION`.
//!
//! Everything is from the side to move's point of view: for Black the board
//! is mirrored vertically (`sq ^ 56`) and colours are swapped, so the network
//! always "plays upwards". In the duck step the side to move is still the
//! player who just moved a piece (they place the duck).
//!
//! # Actions (74 planes × 64 = 4736)
//! `action = plane * 64 + square_relative`
//!
//! | planes | meaning (move step: square = from square) |
//! |---|---|
//! | 0..56  | queen-like moves: `dir * 7 + (distance - 1)`, dir = N, NE, E, SE, S, SW, W, NW |
//! | 56..64 | knight moves in order (1,2) (2,1) (2,-1) (1,-2) (-1,-2) (-2,-1) (-2,1) (-1,2) as (file, rank) deltas |
//! | 64..73 | under-promotions: `64 + piece * 3 + (file_delta + 1)`, piece = N, B, R |
//! | 73     | duck step: square = the duck's new square |
//!
//! Queen promotions use the queen-like plane of the pawn move; castling is
//! the king's two-square move. Only planes 0..73 are legal in the move step
//! and only plane 73 in the duck step. This matches a convolutional policy
//! head with 74 output channels flattened as NCHW.
//!
//! # Observation (23 planes × 64 squares, f32)
//! `index = plane * 64 + square_relative`
//!
//! | plane | content |
//! |---|---|
//! | 0..6   | own P, N, B, R, Q, K |
//! | 6..12  | opponent P, N, B, R, Q, K |
//! | 12     | the duck (one-hot; empty before White's first duck step) |
//! | 13     | duck step: all ones while the duck move of this turn is pending |
//! | 14     | position occurred before (≥1 repetition) — all ones |
//! | 15     | position occurred twice before (≥2) — all ones |
//! | 16..20 | castling rights: own O-O, own O-O-O, opp O-O, opp O-O-O — all ones |
//! | 20     | en passant target square (one-hot) |
//! | 21     | halfmove clock / 100 (clamped to 1) — constant plane |
//! | 22     | all ones (lets convolutions see the board edge) |

use crate::game::Game;
use crate::moves::{Move, MoveList};
use crate::position::{Position, BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
use crate::types::{Color, PieceType, Square};

pub const ENCODING_VERSION: u32 = 1;

pub const POLICY_PLANES: usize = 74;
/// The policy plane of duck moves.
pub const DUCK_PLANE: usize = 73;
pub const NUM_ACTIONS: usize = POLICY_PLANES * 64; // 4736

pub const OBS_PLANES: usize = 23;
pub const OBS_SIZE: usize = OBS_PLANES * 64; // 1472

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const DIRECTIONS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

/// Square seen from `stm`'s side of the board.
#[inline(always)]
pub fn relative_square(sq: Square, stm: Color) -> Square {
    match stm {
        Color::White => sq,
        Color::Black => sq.flip_rank(),
    }
}

/// Policy index (0..NUM_ACTIONS) of a step played by `stm`.
pub fn move_to_action(m: Move, stm: Color) -> usize {
    if m.is_duck() {
        return DUCK_PLANE * 64 + relative_square(m.to(), stm).idx();
    }
    let from = relative_square(m.from(), stm);
    let to = relative_square(m.to(), stm);
    let df = to.file() as i8 - from.file() as i8;
    let dr = to.rank() as i8 - from.rank() as i8;

    let plane = match m.promo_piece() {
        Some(p) if p != PieceType::Queen => {
            let piece = match p {
                PieceType::Knight => 0,
                PieceType::Bishop => 1,
                PieceType::Rook => 2,
                _ => unreachable!(),
            };
            64 + piece * 3 + (df + 1) as usize
        }
        _ => {
            if let Some(k) = KNIGHT_DELTAS.iter().position(|&d| d == (df, dr)) {
                56 + k
            } else {
                let dir = DIRECTIONS
                    .iter()
                    .position(|&d| d == (df.signum(), dr.signum()))
                    .expect("move is neither a line nor a knight move");
                let dist = df.abs().max(dr.abs()) as usize;
                debug_assert!(df == 0 || dr == 0 || df.abs() == dr.abs());
                dir * 7 + dist - 1
            }
        }
    };
    plane * 64 + from.idx()
}

/// All legal steps with their policy indices.
pub fn legal_actions(pos: &Position) -> Vec<(usize, Move)> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let stm = pos.side_to_move();
    list.iter().map(|&m| (move_to_action(m, stm), m)).collect()
}

/// Inverse of `move_to_action` for a concrete position (None if the index
/// doesn't correspond to a legal step there).
pub fn action_to_move(pos: &Position, action: usize) -> Option<Move> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let stm = pos.side_to_move();
    list.iter()
        .copied()
        .find(|&m| move_to_action(m, stm) == action)
}

/// Encode a position into `out` (length `OBS_SIZE`, overwritten).
/// `repetitions` = how many times this exact position occurred before
/// (see `Game::repetitions`).
pub fn encode_observation(pos: &Position, repetitions: usize, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let us = pos.side_to_move();
    let them = !us;

    for (side_offset, color) in [(0, us), (6, them)] {
        for pt in PieceType::ALL {
            let plane = side_offset + pt.idx();
            for sq in pos.pieces_of(color, pt) {
                out[plane * 64 + relative_square(sq, us).idx()] = 1.0;
            }
        }
    }

    let fill = |out: &mut [f32], plane: usize, v: f32| out[plane * 64..(plane + 1) * 64].fill(v);

    if let Some(d) = pos.duck() {
        out[12 * 64 + relative_square(d, us).idx()] = 1.0;
    }
    if pos.is_duck_step() {
        fill(out, 13, 1.0);
    }
    if repetitions >= 1 {
        fill(out, 14, 1.0);
    }
    if repetitions >= 2 {
        fill(out, 15, 1.0);
    }

    let cr = pos.castling_rights();
    let (own_oo, own_ooo, opp_oo, opp_ooo) = match us {
        Color::White => (WHITE_OO, WHITE_OOO, BLACK_OO, BLACK_OOO),
        Color::Black => (BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO),
    };
    for (plane, bit) in [(16, own_oo), (17, own_ooo), (18, opp_oo), (19, opp_ooo)] {
        if cr & bit != 0 {
            fill(out, plane, 1.0);
        }
    }

    if let Some(ep) = pos.ep_square() {
        out[20 * 64 + relative_square(ep, us).idx()] = 1.0;
    }

    fill(out, 21, (pos.halfmove_clock() as f32 / 100.0).min(1.0));
    fill(out, 22, 1.0);
}

/// Encode the current position of a game (repetitions taken from history).
pub fn encode_game(game: &Game, out: &mut [f32]) {
    encode_observation(game.position(), game.repetitions(), out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn known_actions() {
        crate::init();
        let pos = Position::startpos();
        let m = pos.parse_move("e2e4").unwrap();
        // N, distance 2 → plane 1; from e2 = 12
        assert_eq!(move_to_action(m, Color::White), 64 + 12);
        let m = pos.parse_move("g1f3").unwrap();
        // knight (-1, 2) → plane 56 + 7; from g1 = 6
        assert_eq!(move_to_action(m, Color::White), 63 * 64 + 6);

        // Black's e7e5 mirrors White's e2e4; Black's duck on d4 = White's d5.
        let mut g = Game::default();
        assert!(g.push_uci("e2e4,e4h3"));
        let m = g.position().parse_move("e7e5").unwrap();
        assert_eq!(move_to_action(m, Color::Black), 64 + 12);
        let d4 = Square::parse("d4").unwrap();
        let d5 = Square::parse("d5").unwrap();
        assert_eq!(
            move_to_action(Move::duck(d4), Color::Black),
            DUCK_PLANE * 64 + d5.idx()
        );

        // Under-promotion with capture to the left: b7xa8=N → plane 64 + 0*3 + 0
        let pos = Position::from_fen("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = pos.parse_move("b7a8n").unwrap();
        assert_eq!(move_to_action(m, Color::White), 64 * 64 + 49);
        // Queen promotion uses the queen-like plane (NW, distance 1 → plane 7*7)
        let m = pos.parse_move("b7a8q").unwrap();
        assert_eq!(move_to_action(m, Color::White), 49 * 64 + 49);
    }

    /// Every legal step in many random games maps to a unique in-range index
    /// on the right planes and back.
    #[test]
    fn roundtrip_random_games() {
        crate::init();
        let mut rng = Rng::new(2024);
        let mut checked = 0;
        for _ in 0..200 {
            let mut g = Game::default();
            for _ in 0..300 {
                let pos = *g.position();
                let acts = legal_actions(&pos);
                if acts.is_empty() {
                    break;
                }
                let mut seen = std::collections::HashSet::new();
                for &(a, m) in &acts {
                    assert!(a < NUM_ACTIONS);
                    assert_eq!(a / 64 == DUCK_PLANE, pos.is_duck_step());
                    assert!(seen.insert(a), "duplicate action {a} in {}", pos.to_fen());
                    assert_eq!(action_to_move(&pos, a), Some(m));
                    checked += 1;
                }
                // avoid king captures most of the time so games get long
                let pick = (0..4)
                    .map(|_| acts[rng.below(acts.len() as u64) as usize].1)
                    .find(|m| {
                        let p = pos.piece_on(m.to());
                        p.is_none() || p.kind() != PieceType::King
                    })
                    .unwrap_or(acts[0].1);
                g.push(pick);
                if g.result().is_over() {
                    break;
                }
            }
        }
        assert!(checked > 100_000, "{checked}");
    }

    #[test]
    fn observation_is_side_relative() {
        crate::init();
        let mut a = vec![0.0; OBS_SIZE];
        let mut b = vec![0.0; OBS_SIZE];
        let mut g = Game::default();
        for t in ["g1f3,f3a3", "g8f6,f6h6", "f3g1,g1a4", "f6g8,g8h5"] {
            assert!(g.push_uci(t));
        }
        encode_game(&g, &mut a);
        for t in ["g1f3,f3a3", "g8f6,f6h6", "f3g1,g1a4", "f6g8,g8h5"] {
            assert!(g.push_uci(t));
        }
        encode_game(&g, &mut b);
        // Same position again → only the repetition plane differs.
        let diff: Vec<usize> = (0..OBS_SIZE).filter(|&i| a[i] != b[i]).collect();
        assert!(
            diff.iter().all(|&i| i / 64 == 14 || i / 64 == 21),
            "{diff:?}"
        );
        assert!(diff.iter().any(|&i| i / 64 == 14));

        // Black to move after 1. e4 (duck d5): own pawns on rank 2
        // (relative), the duck seen on relative d4.
        let mut g = Game::default();
        assert!(g.push_uci("e2e4,e4d5"));
        encode_game(&g, &mut a);
        let own_pawns: Vec<usize> = (0..64).filter(|&s| a[s] == 1.0).collect();
        assert_eq!(own_pawns, (8..16).collect::<Vec<_>>());
        assert_eq!(a[12 * 64 + Square::parse("d4").unwrap().idx()], 1.0);
        assert_eq!(a[13 * 64], 0.0);
        assert_eq!(a[16 * 64], 1.0); // own O-O still allowed
        assert_eq!(a[22 * 64], 1.0);

        // In the duck step the player who just moved is still "own".
        assert!(g.push_step("e7e5"));
        encode_game(&g, &mut a);
        assert_eq!(a[13 * 64 + 5], 1.0);
        assert_eq!(a[Square::parse("e4").unwrap().idx()], 1.0); // own e-pawn, black's e5 mirrored
    }
}
