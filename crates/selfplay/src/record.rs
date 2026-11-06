//! Self-play data format. One chunk = one file (little-endian):
//!
//! ```text
//! header   : "PDSP" | u32 format version (1) | u32 ENCODING_VERSION | u32 game count
//! per game : u16 step count | i8 result (+1 white wins, 0 draw, -1 black wins) | u8 termination
//! per step : 12 × u64 piece bitboards (white P N B R Q K, black P N B R Q K)
//!            u8 side to move (0 white, 1 black) | u8 castling | u8 ep square (64 = none)
//!            u8 duck square (64 = none) | u8 halfmove clock | u8 repetitions
//!            u8 flags: 1 = duck step, 2 = has search value, 4 = fast search
//!            u16 action played
//!            u8 legal count L (0 = 256) | L × (u16 action, u16 target probability × 65535)
//!            if flags & 2: u16 W × 65535, u16 D × 65535 (search value for the
//!            side to move; L = 1 − W − D)
//! ```
//!
//! A step is a piece move (move step) or a duck move (duck step). Fast-search
//! steps (playout cap randomization) are not training samples.
//!
//! The position after the last step (the terminal position) is not stored.

use peaduck_core::encoding::ENCODING_VERSION;
use peaduck_core::{Color, DrawReason, GameResult, Position, Square, WinReason};

pub const MAGIC: &[u8; 4] = b"PDSP";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Termination {
    KingCaptured = 0,
    /// The winner had no legal move.
    NoMoves = 1,
    FiftyMoves = 2,
    Repetition = 3,
    /// Reached the self-play step limit; scored as a draw.
    MaxPlies = 4,
}

pub const TERMINATIONS: usize = 5;

impl Termination {
    pub fn from_u8(v: u8) -> Option<Termination> {
        use Termination::*;
        Some(match v {
            0 => KingCaptured,
            1 => NoMoves,
            2 => FiftyMoves,
            3 => Repetition,
            4 => MaxPlies,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use Termination::*;
        match self {
            KingCaptured => "king_captured",
            NoMoves => "no_moves",
            FiftyMoves => "fifty_moves",
            Repetition => "repetition",
            MaxPlies => "max_plies",
        }
    }

    /// (result from White's view, termination) for a finished game result.
    pub fn from_result(r: GameResult) -> Option<(i8, Termination)> {
        let side = |c: Color| if c == Color::White { 1 } else { -1 };
        Some(match r {
            GameResult::Ongoing => return None,
            GameResult::Win(c, WinReason::KingCaptured) => (side(c), Termination::KingCaptured),
            GameResult::Win(c, WinReason::NoMoves) => (side(c), Termination::NoMoves),
            GameResult::Draw(DrawReason::FiftyMoves) => (0, Termination::FiftyMoves),
            GameResult::Draw(DrawReason::Repetition) => (0, Termination::Repetition),
        })
    }
}

const FLAG_DUCK_STEP: u8 = 1;
const FLAG_VALUE: u8 = 2;
const FLAG_FAST: u8 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct PlyRecord {
    pub pieces: [u64; 12],
    pub stm: Color,
    pub castling: u8,
    pub ep: Option<Square>,
    pub duck: Option<Square>,
    pub duck_step: bool,
    pub halfmove: u8,
    pub repetitions: u8,
    pub action: u16,
    /// (action, probability × 65535)
    pub targets: Vec<(u16, u16)>,
    /// Search value (W, D) × 65535 for the side to move, if recorded.
    pub value: Option<(u16, u16)>,
    /// false: played after a fast search (playout cap randomization); not a
    /// training sample.
    pub full_search: bool,
}

impl PlyRecord {
    pub fn new(
        pos: &Position,
        repetitions: usize,
        action: usize,
        targets: &[(usize, f32)],
    ) -> PlyRecord {
        PlyRecord {
            pieces: pos.piece_bitboards(),
            stm: pos.side_to_move(),
            castling: pos.castling_rights(),
            ep: pos.ep_square(),
            duck: pos.duck(),
            duck_step: pos.is_duck_step(),
            halfmove: pos.halfmove_clock().min(255) as u8,
            repetitions: repetitions.min(255) as u8,
            action: action as u16,
            targets: targets
                .iter()
                .map(|&(a, p)| (a as u16, (p.clamp(0.0, 1.0) * 65535.0).round() as u16))
                .collect(),
            value: None,
            full_search: true,
        }
    }

    /// Mark the step as played after a fast search (or not).
    pub fn with_full_search(mut self, full: bool) -> PlyRecord {
        self.full_search = full;
        self
    }

    /// Attach the search value as W/D/L probabilities for the side to move.
    pub fn with_value(mut self, [w, d, _]: [f32; 3]) -> PlyRecord {
        let q = |x: f32| (x.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let (w, d) = (q(w), q(d));
        self.value = Some((w, d.min(65535 - w)));
        self
    }

    /// The stored search value as W/D/L, if any.
    pub fn value_wdl(&self) -> Option<[f32; 3]> {
        self.value.map(|(w, d)| {
            let (w, d) = (w as f32 / 65535.0, d as f32 / 65535.0);
            [w, d, (1.0 - w - d).max(0.0)]
        })
    }

    pub fn position(&self) -> Position {
        Position::from_compact(
            &self.pieces,
            self.stm,
            self.castling,
            self.ep,
            self.duck,
            self.duck_step,
            self.halfmove as u16,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    pub plies: Vec<PlyRecord>,
    /// +1 white won, 0 draw, -1 black won.
    pub result: i8,
    pub termination: Termination,
}

pub fn write_chunk(games: &[GameRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + games.iter().map(|g| g.plies.len() * 250).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    out.extend_from_slice(&(games.len() as u32).to_le_bytes());
    for g in games {
        out.extend_from_slice(&(g.plies.len() as u16).to_le_bytes());
        out.push(g.result as u8);
        out.push(g.termination as u8);
        for p in &g.plies {
            for bb in p.pieces {
                out.extend_from_slice(&bb.to_le_bytes());
            }
            out.push(p.stm as u8);
            out.push(p.castling);
            out.push(p.ep.map(|s| s.0).unwrap_or(64));
            out.push(p.duck.map(|s| s.0).unwrap_or(64));
            out.push(p.halfmove);
            out.push(p.repetitions);
            let flags = ((p.duck_step as u8) * FLAG_DUCK_STEP)
                | ((p.value.is_some() as u8) * FLAG_VALUE)
                | ((!p.full_search as u8) * FLAG_FAST);
            out.push(flags);
            out.extend_from_slice(&p.action.to_le_bytes());
            assert!(p.targets.len() <= 256);
            out.push(p.targets.len() as u8); // 256 wraps to 0
            for &(a, prob) in &p.targets {
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&prob.to_le_bytes());
            }
            if let Some((w, d)) = p.value {
                out.extend_from_slice(&w.to_le_bytes());
                out.extend_from_slice(&d.to_le_bytes());
            }
        }
    }
    out
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!("truncated chunk at byte {}", self.pos));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn square(&mut self) -> Result<Option<Square>, String> {
        match self.u8()? {
            64 => Ok(None),
            s if s < 64 => Ok(Some(Square(s))),
            s => Err(format!("bad square {s}")),
        }
    }
}

pub fn read_chunk(bytes: &[u8]) -> Result<Vec<GameRecord>, String> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err("not a PeaDuckZero self-play chunk (bad magic)".into());
    }
    let version = r.u32()?;
    let encoding = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported data format v{version}"));
    }
    if encoding != ENCODING_VERSION {
        return Err(format!(
            "data uses encoding v{encoding}, expected v{ENCODING_VERSION}"
        ));
    }
    let count = r.u32()? as usize;
    let mut games = Vec::with_capacity(count);
    for _ in 0..count {
        let n = r.u16()? as usize;
        let result = r.u8()? as i8;
        let termination = Termination::from_u8(r.u8()?).ok_or("bad termination code")?;
        let mut plies = Vec::with_capacity(n);
        for _ in 0..n {
            let mut pieces = [0u64; 12];
            for bb in &mut pieces {
                *bb = r.u64()?;
            }
            let stm = if r.u8()? == 0 {
                Color::White
            } else {
                Color::Black
            };
            let castling = r.u8()?;
            let ep = r.square()?;
            let duck = r.square()?;
            let halfmove = r.u8()?;
            let repetitions = r.u8()?;
            let flags = r.u8()?;
            if flags > 7 {
                return Err(format!("bad step flags {flags}"));
            }
            let action = r.u16()?;
            let l = match r.u8()? {
                0 => 256,
                l => l as usize,
            };
            let mut targets = Vec::with_capacity(l);
            for _ in 0..l {
                targets.push((r.u16()?, r.u16()?));
            }
            let value = if flags & FLAG_VALUE != 0 {
                Some((r.u16()?, r.u16()?))
            } else {
                None
            };
            plies.push(PlyRecord {
                pieces,
                stm,
                castling,
                ep,
                duck,
                duck_step: flags & FLAG_DUCK_STEP != 0,
                halfmove,
                repetitions,
                action,
                targets,
                value,
                full_search: flags & FLAG_FAST == 0,
            });
        }
        games.push(GameRecord {
            plies,
            result,
            termination,
        });
    }
    if r.pos != bytes.len() {
        return Err(format!(
            "{} trailing bytes after chunk",
            bytes.len() - r.pos
        ));
    }
    Ok(games)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        peaduck_core::init();
        let mut g = peaduck_core::Game::default();
        let a = PlyRecord::new(g.position(), 0, 76, &[(76, 0.75), (9, 0.25)]);
        assert!(g.push_step("e2e4"));
        let b = PlyRecord::new(g.position(), 0, 4700, &[(4700, 1.0)]).with_full_search(false);
        let games = vec![GameRecord {
            plies: vec![a.clone().with_value([0.5, 0.3, 0.2]), b, a],
            result: 1,
            termination: Termination::KingCaptured,
        }];
        let back = read_chunk(&write_chunk(&games)).unwrap();
        assert_eq!(back, games);
        let wdl = back[0].plies[0].value_wdl().unwrap();
        assert!((wdl[0] - 0.5).abs() < 1e-4 && (wdl[1] - 0.3).abs() < 1e-4);
        assert!(back[0].plies[1].duck_step && !back[0].plies[1].full_search);
        assert_eq!(back[0].plies[1].position(), *g.position());
    }
}
