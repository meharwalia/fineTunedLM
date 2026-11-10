"""Train the reward model: python -m ste_rlhf.reward.train [--set reward_training.epochs=1 ...]"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel

from ste_rlhf.common import RunLogger, add_config_args, get_device, load_config, read_jsonl, set_seed

from .model import RewardModel, bt_loss, collate, encode_pair, load_reward_tokenizer, now_iso


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def mem_gb(device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def encode_pairs(tok, pairs, max_length):
    return [(encode_pair(tok, p["prompt"], p["chosen"], max_length),
             encode_pair(tok, p["prompt"], p["rejected"], max_length)) for p in pairs]


def make_batches(enc, batch_size, shuffle, rng):
    """Length-bucketed batches: chunk a shuffled list into groups of 8 batches, sort each group by length."""
    idx = list(range(len(enc)))
    if shuffle:
        rng.shuffle(idx)
    batches = []
    group = batch_size * 8
    for s in range(0, len(idx), group):
        g = sorted(idx[s:s + group], key=lambda i: max(len(enc[i][0]), len(enc[i][1])))
        batches += [g[b:b + batch_size] for b in range(0, len(g), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def pair_forward(model, tok, enc, ids, device):
    """(r_chosen, r_rejected) for the pairs in ids, one padded batch of 2B sequences."""
    seqs = [enc[i][0] for i in ids] + [enc[i][1] for i in ids]
    batch = {k: v.to(device) for k, v in collate(tok, seqs).items()}
    r = model(**batch)
    return r[:len(ids)], r[len(ids):]


def summarize(margins, types):
    acc = sum(m > 0 for m in margins) / max(len(margins), 1)
    by = defaultdict(list)
    for m, t in zip(margins, types):
        by[t].append(m > 0)
    return acc, {t: sum(v) / len(v) for t, v in sorted(by.items())}


@torch.no_grad()
def evaluate_pairs(model, tok, enc, pairs, device, batch_size=8):
    """Returns (accuracy, accuracy_by_type, margins)."""
    was = model.training
    model.eval()
    margins = [0.0] * len(enc)
    for ids in make_batches(enc, batch_size, False, None):
        rc, rr = pair_forward(model, tok, enc, ids, device)
        for i, m in zip(ids, (rc - rr).float().cpu().tolist()):
            margins[i] = m
    if was:
        model.train()
    acc, by = summarize(margins, [p.get("pair_type", "all") for p in pairs])
    return acc, by, margins


@torch.no_grad()
def raw_rewards(model, tok, enc, device, batch_size=8) -> torch.Tensor:
    """Raw reward of every chosen and rejected response in enc."""
    model.eval()
    out = []
    for ids in make_batches(enc, batch_size, False, None):
        rc, rr = pair_forward(model, tok, enc, ids, device)
        out += rc.float().cpu().tolist() + rr.float().cpu().tolist()
    return torch.tensor(out)


def train(cfg: dict, device=None, train_pairs=None, val_pairs=None, out_dir=None, log_dir="logs/reward_model"):
    device = device or get_device(cfg)
    set_seed(cfg["seed"])
    rt, rc = cfg["reward_training"], cfg["reward"]
    max_length = int(rc["max_length"])
    data_dir = Path(cfg["data"]["dir"]) / "pairs"
    train_pairs = train_pairs if train_pairs is not None else read_jsonl(data_dir / "train.jsonl")
    val_pairs = val_pairs if val_pairs is not None else read_jsonl(data_dir / "val.jsonl")
    out_dir = Path(out_dir or rc["checkpoint"])

    tok = load_reward_tokenizer(rc["base_model"])
    model = RewardModel(AutoModel.from_pretrained(rc["base_model"], dtype=torch.float32)).to(device)
    if rt.get("gradient_checkpointing", False):
        model.backbone.gradient_checkpointing_enable()
    model.train()
    tr_enc = encode_pairs(tok, train_pairs, max_length)
    va_enc = encode_pairs(tok, val_pairs, max_length)

    bs, accum, epochs = int(rt["batch_size"]), int(rt.get("grad_accum", 1)), int(rt["epochs"])
    micro = max(bs // accum, 1)  # pairs per forward pass; the optimiser step still uses bs pairs
    opt = torch.optim.AdamW(model.parameters(), lr=float(rt["lr"]), weight_decay=float(rt["weight_decay"]))
    steps_per_epoch = math.ceil(len(tr_enc) / bs)
    total = steps_per_epoch * epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    logger = RunLogger(log_dir, cfg)
    rng = random.Random(cfg["seed"])

    best_acc, best_by, best_state, step = -1.0, {}, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for ids in make_batches(tr_enc, bs, True, rng):
            t0 = time.time()
            loss_sum, correct = 0.0, 0
            for s in range(0, len(ids), micro):
                sub = ids[s:s + micro]
                r_c, r_r = pair_forward(model, tok, tr_enc, sub, device)
                loss = bt_loss(r_c, r_r) * len(sub) / len(ids)
                loss.backward()
                loss_sum += loss.item()
                correct += int((r_c > r_r).sum())
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            _sync(device)
            logger.log(step, {"epoch": ep, "loss": loss_sum, "train_acc": correct / len(ids), "grad_norm": float(gn),
                              "lr": sched.get_last_lr()[0], "seq_len": max(len(tr_enc[i][0]) for i in ids),
                              "mem_gb": mem_gb(device), "sec": time.time() - t0})
            if step % 5 == 0:
                _empty_cache(device)
        _empty_cache(device)
        acc, by, margins = evaluate_pairs(model, tok, va_enc, val_pairs, device, micro)
        logger.log(step, {"epoch": ep, "val_accuracy": acc, **{f"val_acc_{t}": a for t, a in by.items()},
                          "val_margin_mean": sum(margins) / max(len(margins), 1)})
        print(f"epoch {ep}: val_acc={acc:.3f} {by}", flush=True)
        if acc >= best_acc:  # ties: prefer the later (lower-loss) epoch
            best_acc, best_by = acc, by
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    rewards = raw_rewards(model, tok, tr_enc, device, micro)
    mean, std = float(rewards.mean()), float(rewards.std().clamp(min=1e-6))
    meta = {"base_model": rc["base_model"], "max_length": max_length, "train_reward_mean": mean,
            "train_reward_std": std, "val_accuracy": best_acc, "val_accuracy_by_type": best_by,
            "epochs": epochs, "created": now_iso()}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save(out_dir, tok, meta)
    print(f"saved {out_dir}: {meta}", flush=True)
    return meta


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
