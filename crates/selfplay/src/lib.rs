//! Self-play for PeaDuckZero.
//!
//! * [`search`] — Gumbel AlphaZero tree search on the real board, a pool of
//!   concurrent games driven in lock-step rounds so the GPU always sees
//!   large batches.
//! * [`record`] — self-play data format (writer + reader).
//! * [`replay`] — replay buffer that samples training batches and encodes
//!   observations with the engine's own encoder.
//!
//! Every step of a turn (piece move, duck move) is a separate search and a
//! separate training sample.
//!
//! No Python or GPU code here; `crates/py` wraps this for the trainer.

pub mod record;
pub mod replay;
pub mod search;

/// W/D/L probabilities from W/D/L logits.
pub fn wdl_logits_to_probs(logits: &[f32]) -> [f32; 3] {
    let m = logits[0].max(logits[1]).max(logits[2]);
    let e = [
        (logits[0] - m).exp(),
        (logits[1] - m).exp(),
        (logits[2] - m).exp(),
    ];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

/// Scalar value `W - L` from W/D/L logits (side-to-move perspective).
pub fn wdl_logits_to_q(logits: &[f32]) -> f32 {
    let [w, _, l] = wdl_logits_to_probs(logits);
    w - l
}
