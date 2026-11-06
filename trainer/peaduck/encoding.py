"""Constants of crates/core/src/encoding.rs (observations and legal-move
indices are always produced by Rust; see that file for the layout)."""
ENCODING_VERSION = 1

POLICY_PLANES = 74  # 73 move planes + 1 duck plane
DUCK_PLANE = 73
NUM_ACTIONS = POLICY_PLANES * 64  # 4736
OBS_PLANES = 23
OBS_SIZE = OBS_PLANES * 64
MAX_LEGAL = 256  # crates/selfplay/src/replay.rs
