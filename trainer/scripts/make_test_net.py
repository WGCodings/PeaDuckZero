#!/usr/bin/env python3
"""Test networks for the engine's inference (crates/nn/src/fast.rs tests).

Writes crates/nn/testdata/test-32x2.pdn and test-64x2.pdn (random weights
and BatchNorm statistics, so the folding is exercised) and next to each a
.expected file: the PyTorch outputs for POSITIONS (f32 little-endian, per
position 4736 policy logits then 3 value logits). Needs peaduck_rs
(`pip install ./crates/py`) for the observations.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import peaduck_rs as pz  # noqa: E402
from peaduck.model import DuckNet, NetConfig, export_pdn  # noqa: E402

# Must match `positions()` in crates/nn/src/fast.rs.
POSITIONS = [
    [],
    ["e2e4,e4d5"],
    ["e2e4,e4d5", "e7e5"],
    ["g1f3,f3a3", "b8c6,c6h6", "e2e4,e4d4", "d7d5,d5e6"],
    ["d2d4,d4h5", "e7e5,e5a6", "d4e5,e5b4"],
]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "crates", "nn", "testdata")


def randomize(net, gen):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0.0, 0.3, generator=gen)
                m.running_var.uniform_(0.5, 2.0, generator=gen)
                m.weight.uniform_(0.5, 1.5, generator=gen)
                m.bias.normal_(0.0, 0.2, generator=gen)
            elif isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                m.weight.mul_(1.5)
                if m.bias is not None:
                    m.bias.normal_(0.0, 0.1, generator=gen)


def main():
    os.makedirs(OUT, exist_ok=True)
    obs = np.stack([pz.encode_position(None, moves)[0] for moves in POSITIONS])
    for c, b in [(32, 2), (64, 2)]:
        torch.manual_seed(c)
        gen = torch.Generator().manual_seed(c + b)
        net = DuckNet(NetConfig(c, b))
        randomize(net, gen)
        net.eval()
        with torch.no_grad():
            p, v = net(torch.from_numpy(obs))
        out = np.concatenate([p.numpy(), v.numpy()], axis=1).astype("<f4")
        base = os.path.join(OUT, f"test-{c}x{b}")
        export_pdn(net, base + ".pdn")
        out.tofile(base + ".expected")
        print(f"wrote {base}.pdn / .expected")


if __name__ == "__main__":
    main()
