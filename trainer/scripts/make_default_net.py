#!/usr/bin/env python3
"""Write networks/default.pdn: a randomly initialised 64x6 network (the
engine's embedded network until a trained one replaces it)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peaduck.model import DuckNet, NetConfig, export_pdn  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "networks", "default.pdn")
    torch.manual_seed(0)
    net = DuckNet(NetConfig(64, 6)).eval()
    export_pdn(net, out)
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")
