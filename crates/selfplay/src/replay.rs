//! Replay buffer + training batch sampler.
//!
//! For a sampled step `t` of a game:
//! * observation: encoded position `t` (same encoder as the engine)
//! * policy target: recorded improved policy at `t`
//! * value target: final result from the side to move's view at `t` as a
//!   W/D/L class (0 = win, 1 = draw, 2 = loss)
//! * search value at `t` (W/D/L) where recorded
//!
//! Playout cap randomization: only steps played after a full search are
//! training samples. `positions()` and `capacity` count training samples.

use std::collections::VecDeque;

use peaduck_core::encoding::{encode_observation, OBS_SIZE};
use peaduck_core::rng::Rng;
use peaduck_core::Color;

use crate::record::{read_chunk, GameRecord};

/// Upper bound on legal steps; policy targets are padded to this width.
pub const MAX_LEGAL: usize = peaduck_core::moves::MAX_MOVES;

pub struct Batch {
    pub batch: usize,
    /// `B × OBS_SIZE`
    pub obs: Vec<f32>,
    /// `B × MAX_LEGAL` action indices (padding: 0 with prob 0)
    pub policy_index: Vec<i64>,
    /// `B × MAX_LEGAL` target probabilities
    pub policy_prob: Vec<f32>,
    /// `B` W/D/L class of the game result
    pub value_target: Vec<i64>,
    /// `B × 3` search value (W/D/L, side to move) where recorded
    pub value_search: Vec<f32>,
    /// `B`: 1 where `value_search` is valid
    pub value_search_mask: Vec<f32>,
    /// `B`: 1 for duck steps
    pub duck_step: Vec<f32>,
}

pub struct Replay {
    /// Games with their training samples only (fast-search steps dropped).
    games: VecDeque<GameRecord>,
    /// cumulative sample counts, `cum[i]` = samples in games[..=i]
    cum: Vec<usize>,
    capacity: usize,
    rng: Rng,
    total_added: u64,
}

pub(crate) fn value_class(result: i8, stm: Color) -> i64 {
    match result {
        0 => 1,
        1 if stm == Color::White => 0,
        -1 if stm == Color::Black => 0,
        _ => 2,
    }
}

impl Replay {
    /// `capacity`: maximum number of training samples kept (oldest games
    /// evicted).
    pub fn new(capacity: usize, seed: u64) -> Replay {
        Replay {
            games: VecDeque::new(),
            cum: Vec::new(),
            capacity,
            rng: Rng::new(seed ^ 0x2E91_A7B3),
            total_added: 0,
        }
    }

    pub fn positions(&self) -> usize {
        self.cum.last().copied().unwrap_or(0)
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    /// Training samples ever added (including evicted ones).
    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    pub fn add_games(&mut self, games: Vec<GameRecord>) {
        for mut g in games {
            // only samples are ever read: drop the rest to save memory
            g.plies.retain(|p| p.full_search);
            if g.plies.is_empty() {
                continue;
            }
            self.total_added += g.plies.len() as u64;
            self.games.push_back(g);
        }
        let mut total: usize = self.games.iter().map(|g| g.plies.len()).sum();
        while total > self.capacity && self.games.len() > 1 {
            total -= self.games.pop_front().unwrap().plies.len();
        }
        self.cum.clear();
        let mut acc = 0;
        for g in &self.games {
            acc += g.plies.len();
            self.cum.push(acc);
        }
    }

    pub fn add_chunk(&mut self, bytes: &[u8]) -> Result<usize, String> {
        let games = read_chunk(bytes)?;
        let n = games.len();
        self.add_games(games);
        Ok(n)
    }

    fn pick(&mut self) -> (usize, usize) {
        let total = self.positions();
        let idx = self.rng.below(total as u64) as usize;
        let g = self.cum.partition_point(|&c| c <= idx);
        let start = if g == 0 { 0 } else { self.cum[g - 1] };
        (g, idx - start)
    }

    pub fn sample(&mut self, batch: usize) -> Batch {
        assert!(self.positions() > 0, "replay buffer is empty");
        let mut b = Batch {
            batch,
            obs: vec![0.0; batch * OBS_SIZE],
            policy_index: vec![0; batch * MAX_LEGAL],
            policy_prob: vec![0.0; batch * MAX_LEGAL],
            value_target: vec![1; batch],
            value_search: vec![0.0; batch * 3],
            value_search_mask: vec![0.0; batch],
            duck_step: vec![0.0; batch],
        };
        for i in 0..batch {
            let (gi, t) = self.pick();
            let game = &self.games[gi];
            let p = &game.plies[t];
            encode_observation(
                &p.position(),
                p.repetitions as usize,
                &mut b.obs[i * OBS_SIZE..(i + 1) * OBS_SIZE],
            );
            b.value_target[i] = value_class(game.result, p.stm);
            if let Some(wdl) = p.value_wdl() {
                b.value_search[i * 3..i * 3 + 3].copy_from_slice(&wdl);
                b.value_search_mask[i] = 1.0;
            }
            b.duck_step[i] = p.duck_step as u8 as f32;
            let sum: f32 = p
                .targets
                .iter()
                .map(|&(_, q)| q as f32)
                .sum::<f32>()
                .max(1.0);
            for (n, &(a, q)) in p.targets.iter().enumerate().take(MAX_LEGAL) {
                b.policy_index[i * MAX_LEGAL + n] = a as i64;
                b.policy_prob[i * MAX_LEGAL + n] = q as f32 / sum;
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_targets_follow_side_to_move() {
        assert_eq!(value_class(1, Color::White), 0);
        assert_eq!(value_class(1, Color::Black), 2);
        assert_eq!(value_class(-1, Color::Black), 0);
        assert_eq!(value_class(0, Color::Black), 1);
    }
}
