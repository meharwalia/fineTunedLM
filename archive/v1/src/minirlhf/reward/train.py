"""Train the reward model (Bradley-Terry on preference pairs).

    python -m minirlhf.reward.train --source files            # data/prefs/{train,val}.jsonl
    python -m minirlhf.reward.train --source hh --n 3000      # Anthropic/hh-rlhf subset
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from minirlhf.common.config import load_config
from minirlhf.common.device import get_device
from minirlhf.common.io import read_jsonl
from minirlhf.common.seed import set_seed
from minirlhf.reward.model import bt_loss, build_model, raw_rewards, save_checkpoint

H, A = "\n\nHuman:", "\n\nAssistant:"


# ----------------------------------------------------------------------------- data
def parse_hh_single_turn(text: str):
    """Return (prompt, response) for a single-turn hh-rlhf transcript, else None.

    Prompt = last Human turn, response = final Assistant turn.
    """
    if text.count(H) != 1 or text.count(A) != 1:
        return None
    h_i, a_i = text.index(H), text.index(A)
    if h_i > a_i:
        return None
    prompt = text[h_i + len(H):a_i].strip()
    response = text[a_i + len(A):].strip()
    if not prompt or not response:
        return None
    return prompt, response


def hh_pair(chosen: str, rejected: str, max_words: int = 120):
    c, r = parse_hh_single_turn(chosen), parse_hh_single_turn(rejected)
    if c is None or r is None or c[0] != r[0] or c[1] == r[1]:
        return None
    if len(c[1].split()) > max_words or len(r[1].split()) > max_words:
        return None
    return {"prompt": c[0], "chosen": c[1], "rejected": r[1], "source": "hh-rlhf"}


def load_hh(n: int, seed: int = 0, max_words: int = 120) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    rows = []
    for i in order:
        p = hh_pair(ds[i]["chosen"], ds[i]["rejected"], max_words)
        if p is None:
            continue
        p["id"] = f"hh{i}"
        rows.append(p)
        if len(rows) >= n:
            break
    return rows


def split_rows(rows, val_fraction, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_fraction)))
    return rows[n_val:], rows[:n_val]


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def pair_margins(model, tok, rows, max_length, device, batch_size=32) -> np.ndarray:
    """r_chosen - r_rejected for every row (raw logits)."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        b = rows[i:i + batch_size]
        ps = [r["prompt"] for r in b]
        rc = raw_rewards(model, tok, ps, [r["chosen"] for r in b], max_length, device)
        rr = raw_rewards(model, tok, ps, [r["rejected"] for r in b], max_length, device)
        out.append((rc - rr).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def accuracy_report(margins: np.ndarray, rows) -> tuple[float, dict]:
    acc = float((margins > 0).mean()) if len(margins) else 0.0
    by = defaultdict(list)
    for m, r in zip(margins, rows):
        by[r.get("source", "unknown")].append(m > 0)
    return acc, {s: float(np.mean(v)) for s, v in sorted(by.items())}


@torch.no_grad()
def reward_stats(model, tok, rows, max_length, device, batch_size=32):
    """Mean/std of raw reward over all chosen+rejected responses (for normalisation)."""
    model.eval()
    vals = []
    for i in range(0, len(rows), batch_size):
        b = rows[i:i + batch_size]
        ps = [r["prompt"] for r in b] * 2
        rs = [r["chosen"] for r in b] + [r["rejected"] for r in b]
        vals.append(raw_rewards(model, tok, ps, rs, max_length, device).float().cpu())
    v = torch.cat(vals)
    return float(v.mean()), float(v.std().clamp_min(1e-6))


# ----------------------------------------------------------------------------- train
def train_reward_model(train_rows, val_rows, *, base_model, device, max_length, lr, epochs,
                       batch_size, weight_decay, out_dir=None, warmup_frac=0.1, seed=0, log=print):
    set_seed(seed)
    model, tok = build_model(base_model)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    total = steps_per_epoch * epochs
    warm = max(1, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    rng = random.Random(seed)
    best_acc, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        run_loss, run_acc, nb = 0.0, 0.0, 0
        for i in range(0, len(order), batch_size):
            b = [train_rows[j] for j in order[i:i + batch_size]]
            ps = [r["prompt"] for r in b] * 2
            rs = [r["chosen"] for r in b] + [r["rejected"] for r in b]
            r_all = raw_rewards(model, tok, ps, rs, max_length, device)
            rc, rr = r_all[:len(b)], r_all[len(b):]
            loss = bt_loss(rc, rr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            run_acc += (rc > rr).float().mean().item()
            nb += 1
            if nb % 20 == 0:
                log(f"  ep {ep + 1} step {nb}/{steps_per_epoch} loss {run_loss / nb:.4f} "
                    f"acc {run_acc / nb:.3f} ({time.time() - t0:.0f}s)")
        val_m = pair_margins(model, tok, val_rows, max_length, device)
        val_acc, per_src = accuracy_report(val_m, val_rows)
        history.append({"epoch": ep + 1, "train_loss": run_loss / nb, "train_acc": run_acc / nb,
                        "val_acc": val_acc})
        log(f"epoch {ep + 1}/{epochs}: train loss {run_loss / nb:.4f} acc {run_acc / nb:.3f} | "
            f"val acc {val_acc:.4f} {per_src} ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_per_src = per_src

    model.load_state_dict(best_state)
    mean, std = reward_stats(model, tok, train_rows, max_length, device)
    meta = {
        "base_model": base_model, "max_length": max_length, "val_accuracy": best_acc,
        "per_source_accuracy": best_per_src, "train_reward_mean": mean, "train_reward_std": std,
        "data_counts": {"train": len(train_rows), "val": len(val_rows),
                        "train_by_source": dict(Counter(r.get("source", "unknown") for r in train_rows)),
                        "val_by_source": dict(Counter(r.get("source", "unknown") for r in val_rows))},
    }
    if out_dir is not None:
        save_checkpoint(model, tok, out_dir, meta)
        log(f"saved best (epoch {best_epoch}, val acc {best_acc:.4f}) to {out_dir}")
    return model, tok, meta, history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--source", choices=["files", "hh"], default="files")
    ap.add_argument("--n", type=int, default=None, help="hh pairs to load (default data.hh_rlhf_n)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default reward.checkpoint)")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--prefs-dir", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    rt, rw, d = cfg["reward_training"], cfg["reward"], cfg["data"]
    device = get_device(cfg)
    if a.source == "hh":
        rows = load_hh(a.n or d["hh_rlhf_n"], cfg["seed"])
        train_rows, val_rows = split_rows(rows, d["val_fraction"], cfg["seed"])
    else:
        pdir = Path(a.prefs_dir or d["prefs_dir"])
        train_rows, val_rows = read_jsonl(pdir / "train.jsonl"), read_jsonl(pdir / "val.jsonl")
    print(f"device {device}; train {len(train_rows)} val {len(val_rows)}")
    t0 = time.time()
    _, _, meta, _ = train_reward_model(
        train_rows, val_rows, base_model=rw["base_model"], device=device,
        max_length=a.max_length or rw["max_length"], lr=a.lr or rt["lr"],
        epochs=a.epochs or rt["epochs"], batch_size=a.batch_size or rt["batch_size"],
        weight_decay=rt["weight_decay"], out_dir=a.out or rw["checkpoint"], seed=cfg["seed"])
    print(f"done in {time.time() - t0:.0f}s: val_accuracy={meta['val_accuracy']:.4f} "
          f"per_source={meta['per_source_accuracy']}")


if __name__ == "__main__":
    main()
