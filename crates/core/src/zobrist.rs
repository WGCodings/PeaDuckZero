//! Zobrist keys, generated at compile time from a fixed seed.

use crate::rng::splitmix64;

pub struct Keys {
    pub piece_sq: [[u64; 64]; 12],
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub duck: [u64; 64],
    pub side: u64,
    /// Set while the duck step of a turn is pending.
    pub duck_step: u64,
}

const fn fill<const N: usize>(mut state: u64, out: &mut [u64; N], from: usize) -> u64 {
    let mut i = from;
    while i < N {
        let (ns, v) = splitmix64(state);
        state = ns;
        out[i] = v;
        i += 1;
    }
    state
}

pub const KEYS: Keys = {
    let mut state = 0x5045_4144_5543_4B21u64; // "PEADUCK!"
    let mut piece_sq = [[0u64; 64]; 12];
    let mut p = 0;
    while p < 12 {
        state = fill(state, &mut piece_sq[p], 0);
        p += 1;
    }
    let mut castling = [0u64; 16];
    state = fill(state, &mut castling, 1); // [0] stays 0: "no rights" doesn't perturb the hash
    let mut ep_file = [0u64; 8];
    state = fill(state, &mut ep_file, 0);
    let mut duck = [0u64; 64];
    state = fill(state, &mut duck, 0);
    let (state, side) = splitmix64(state);
    let (_, duck_step) = splitmix64(state);
    Keys {
        piece_sq,
        castling,
        ep_file,
        duck,
        side,
        duck_step,
    }
};
