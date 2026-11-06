//! Policy/value network inference on CPU, batch 1 ([`fast`]: hand-written
//! AVX-512 / AVX2 / SSE2 kernels picked at runtime, int8 residual blocks).
//! x86-64 only.
//!
//! A network is one ONNX graph exported by `trainer/peaduck/model.py`:
//!   obs (N,23,8,8) → policy (N,4736), value (N,3)
//!
//! It ships as a `.pdn` file (what `EVALFILE` embeds):
//!
//! ```text
//! "PDZN" | u32 format version (1) | u32 ENCODING_VERSION
//! u64 net.onnx length | bytes
//! ```
//! (all integers little-endian)

#[cfg(not(target_arch = "x86_64"))]
compile_error!("PeaDuckZero's network inference needs an x86-64 CPU");

pub mod fast;
pub mod int8;
pub mod onnx;
mod winograd;

use peaduck_core::encoding::ENCODING_VERSION;

pub const PDN_MAGIC: &[u8; 4] = b"PDZN";
pub const PDN_VERSION: u32 = 1;

/// The ONNX graph inside a `.pdn` file.
pub fn parse_pdn(bytes: &[u8]) -> Result<&[u8], String> {
    let mut rest = bytes;
    let mut take = |n: usize| -> Result<&[u8], String> {
        if rest.len() < n {
            return Err("truncated .pdn file".into());
        }
        let (a, b) = rest.split_at(n);
        rest = b;
        Ok(a)
    };
    if take(4)? != PDN_MAGIC {
        return Err("not a PeaDuckZero network file (bad magic)".into());
    }
    let u32le = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let version = u32le(take(4)?);
    let encoding = u32le(take(4)?);
    if version != PDN_VERSION {
        return Err(format!(
            "unsupported .pdn version {version} (expected {PDN_VERSION})"
        ));
    }
    if encoding != ENCODING_VERSION {
        return Err(format!(
            "network uses encoding v{encoding}, engine uses v{ENCODING_VERSION}"
        ));
    }
    let n = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
    take(n)
}

/// Softmax of 3 W/D/L logits.
pub fn wdl_probs(value_logits: &[f32]) -> [f32; 3] {
    let m = value_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = value_logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    [e[0] / s, e[1] / s, e[2] / s]
}
