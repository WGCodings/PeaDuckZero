"""Policy/value network for duck chess (AlphaZero style).

    obs (23×8×8) → stem + residual tower → latent (C×8×8, min-max normalised)
                 → policy logits (4736 = 74 planes × 64), value logits (W/D/L)

Exported to ONNX as one graph (obs → policy, value) and bundled into a
.pdn file (format: crates/nn/src/lib.rs). The engine's hand-written
inference (crates/nn/src/fast.rs) reads the weights straight from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoding as enc


@dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 6
    policy_channels: int = 32
    value_channels: int = 8
    value_hidden: int = 128


# ---------------------------------------------------------------------------
# Quantisation-aware training (QAT): simulate the engine's int8 residual
# blocks (crates/nn/src/int8.rs) in the forward pass, so the network learns
# weights whose rounded version plays well. Straight-through estimator: the
# forward pass sees the rounded values, the gradient passes through as if
# rounding were the identity.
#
# Engine (per residual-block convolution):
#   weights     w_q = round(w / s_w) in -127..127, s_w = max|w[co]| / 127 per
#               output channel. BatchNorm folding multiplies each output
#               channel by one constant, so quantising the unfolded weight
#               gives exactly the same integers.
#   activations x_q = round(x · 127 / max x) in 0..127, one scale per
#               convolution input per position (dynamic, no calibration).
#   bias, residual add, ReLU, stem and heads stay float.
QMAX = 127.0


def fake_quant_weight(w: torch.Tensor) -> torch.Tensor:
    w32 = w.float()
    s = (w32.abs().amax(dim=(1, 2, 3), keepdim=True) / QMAX).clamp(min=1e-12).detach()
    q = torch.round(w32 / s).clamp(-QMAX, QMAX) * s
    return (w32 + (q - w32).detach()).to(w.dtype)


def fake_quant_act(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    s = (x32.flatten(1).amax(dim=1).view(-1, 1, 1, 1) / QMAX).clamp(min=1e-12).detach()
    q = torch.round(x32 / s).clamp(0.0, QMAX) * s
    return (x32 + (q - x32).detach()).to(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.qat = False  # simulate the engine's int8 convolutions (DuckNet.set_qat)

    def forward(self, x):
        if self.qat:
            y = F.relu(self.bn1(F.conv2d(fake_quant_act(x), fake_quant_weight(self.conv1.weight), padding=1)))
            y = self.bn2(F.conv2d(fake_quant_act(y), fake_quant_weight(self.conv2.weight), padding=1))
            return F.relu(x + y)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


def conv_bn_relu(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


def normalize_latent(s: torch.Tensor) -> torch.Tensor:
    """Min-max scale each sample's latent to [0, 1] (MuZero appendix G)."""
    flat = s.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-5)
    return flat.view_as(s)


class DuckNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        self.stem = conv_bn_relu(enc.OBS_PLANES, cfg.channels, 3)
        self.blocks = nn.Sequential(*[ResBlock(cfg.channels) for _ in range(cfg.blocks)])
        self.policy = nn.Sequential(
            conv_bn_relu(cfg.channels, cfg.policy_channels, 1),
            nn.Conv2d(cfg.policy_channels, enc.POLICY_PLANES, 1),
        )
        self.value_conv = conv_bn_relu(cfg.channels, cfg.value_channels, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(cfg.value_channels * 64, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 3),
        )

    def set_qat(self, on: bool) -> None:
        """Simulate the engine's int8 residual blocks. The engine skips int8
        when the channel count isn't a multiple of 32, so does this."""
        on = on and self.cfg.channels % 32 == 0
        for b in self.blocks:
            b.qat = on

    @property
    def qat(self) -> bool:
        return any(b.qat for b in self.blocks)

    def forward(self, obs):
        latent = normalize_latent(self.blocks(self.stem(obs)))
        # (N, 74, 8, 8) flattened NCHW → index = plane * 64 + square, as in encoding.rs
        policy = self.policy(latent).flatten(1)
        value = self.value_fc(self.value_conv(latent).flatten(1))
        return policy, value


PDN_MAGIC = b"PDZN"
PDN_VERSION = 1


def export_onnx(net: DuckNet, path: str, opset: int = 17) -> None:
    """Write the network as ONNX (dynamic batch axis), float weights: the
    engine does the int8 rounding itself (same integers as QAT)."""
    was_training, was_qat = net.training, net.qat
    net.set_qat(False)
    net.eval()
    obs = torch.zeros(1, enc.OBS_PLANES, 8, 8)
    with torch.no_grad():
        torch.onnx.export(
            net, (obs,), path, input_names=["obs"], output_names=["policy", "value"],
            dynamic_axes={"obs": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    net.train(was_training)
    net.set_qat(was_qat)


def export_pdn(net: DuckNet, out_path: str) -> None:
    """Export straight to a .pdn file (works for a net on any device)."""
    import copy
    import os
    import struct
    import tempfile

    cpu_net = copy.deepcopy(net).float().cpu()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "net.onnx")
        export_onnx(cpu_net, onnx_path)
        with open(onnx_path, "rb") as f:
            blob = f.read()
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "wb") as f:
        f.write(PDN_MAGIC)
        f.write(struct.pack("<II", PDN_VERSION, enc.ENCODING_VERSION))
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    os.replace(tmp_out, out_path)
