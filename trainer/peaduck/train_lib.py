"""Training loss and evaluation helpers (AlphaZero style).

Rust (`peaduck_rs`) owns the games, the search, the data format and batch
sampling; this file only runs the network.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .model import DuckNet


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """numpy batch → tensors on `device`. On CUDA the copy goes through pinned
    memory and is asynchronous, so it overlaps with work already queued."""
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v)
        if device.type == "cuda":
            t = t.pin_memory()
        out[k] = t.to(device, non_blocking=True)
    return out


def _policy_terms(policy: torch.Tensor, t: dict):
    """Per-sample policy cross-entropy against the search target, and
    whether the network's best step is the target's best step."""
    logp = F.log_softmax(policy.float(), dim=-1)
    target = torch.zeros_like(logp).scatter_add_(1, t["policy_index"], t["policy_prob"])
    ce = -(target * logp).sum(-1)
    agree = (policy.argmax(-1) == target.argmax(-1)).float()
    return ce, agree


def az_loss(net: DuckNet, batch: dict, device: torch.device, amp: bool, q_ratio: float = 0.0):
    """Policy + value loss. Returns (loss tensor, metrics dict of detached
    0-dim tensors; call float() on them only when needed).

    Value target: the game result, or with `q_ratio` > 0 the mix
    `(1 - q_ratio)·result + q_ratio·search value`. The logged value loss is
    always against the game result, so it stays comparable between runs.
    The policy loss is also logged separately for piece moves and duck
    moves."""
    t = batch if isinstance(next(iter(batch.values())), torch.Tensor) else batch_to_device(batch, device)
    with autocast(device, amp):
        policy, value = net(t["obs"])
    ce, agree = _policy_terms(policy, t)
    pl = ce.mean()
    vlogp = F.log_softmax(value.float(), dim=-1)
    result = F.one_hot(t["value_target"], 3).float()
    vl = -(result * vlogp).sum(-1).mean()
    if q_ratio > 0:
        w = q_ratio * t["value_search_mask"].unsqueeze(-1)
        mixed = (1 - w) * result + w * t["value_search"]
        vl_train = -(mixed * vlogp).sum(-1).mean()
    else:
        vl_train = vl
    loss = pl + vl_train
    duck = t["duck_step"]
    n_duck = duck.sum().clamp(min=1)
    n_move = (1 - duck).sum().clamp(min=1)
    metrics = {
        "loss": loss.detach().float(),
        "policy_loss": pl.detach().float(),
        "value_loss": vl.detach().float(),
        "policy_loss_move": ((ce * (1 - duck)).sum() / n_move).detach().float(),
        "policy_loss_duck": ((ce * duck).sum() / n_duck).detach().float(),
        "policy_agreement": agree.mean().detach().float(),
    }
    return loss, metrics


def eval_net(net: DuckNet) -> DuckNet:
    """Eval-mode copy for self-play and export (float, no QAT: the engine
    rounds the weights itself)."""
    e = copy.deepcopy(net).eval()
    e.set_qat(False)
    return e


@torch.no_grad()
def eval_metrics(net: DuckNet, replay, device: torch.device, amp: bool,
                 batches: int = 4, batch_size: int = 1024) -> dict:
    """Losses of the network as exported (eval mode, float).

    int8_agreement / int8_value_diff: same best step and mean |W-L|
    difference between the engine's int8 simulation and the float network."""
    if replay.positions == 0:
        return {}
    out = {"eval_policy_loss": 0.0, "eval_value_loss": 0.0}
    for _ in range(batches):
        t = {k: torch.from_numpy(v).to(device) for k, v in replay.sample(batch_size).items()}
        with autocast(device, amp):
            p, v = net(t["obs"])
        ce, _ = _policy_terms(p, t)
        out["eval_policy_loss"] += float(ce.mean()) / batches
        out["eval_value_loss"] += float(F.cross_entropy(v.float(), t["value_target"])) / batches
        if net.cfg.channels % 32 == 0:
            net.set_qat(True)
            with autocast(device, amp):
                pq, vq = net(t["obs"])
            net.set_qat(False)
            out["int8_agreement"] = out.get("int8_agreement", 0.0) + float(
                (pq.float().argmax(-1) == p.float().argmax(-1)).float().mean()) / batches
            wl = lambda v: (lambda x: x[:, 0] - x[:, 2])(F.softmax(v.float(), dim=-1))  # noqa: E731
            out["int8_value_diff"] = out.get("int8_value_diff", 0.0) + float(
                (wl(vq) - wl(v)).abs().mean()) / batches
    return out


@torch.inference_mode()
def search_step(net: DuckNet, sp, device: torch.device, amp: bool) -> int:
    """One tree-search simulation for every game of a SearchSelfPlay pool
    (the plain loop, see selfplay_gpu.SearchRunner for the fast one).
    Returns games finished."""
    obs, legal = sp.requests()
    obs_t = torch.from_numpy(obs).to(device, non_blocking=True)
    idx = torch.from_numpy(legal).to(device, non_blocking=True).long()
    with autocast(device, amp):
        policy, value = net(obs_t)
    legal_logits = policy.float().gather(1, idx)
    return sp.respond(
        np.ascontiguousarray(legal_logits.cpu().numpy()),
        np.ascontiguousarray(value.float().cpu().numpy()),
    )
