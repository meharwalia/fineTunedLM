"""Evaluate the reward model on val pairs: python -m ste_rlhf.reward.evaluate"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ste_rlhf.common import add_config_args, get_device, load_config, read_jsonl

from .model import RewardModel, load_meta, load_reward_tokenizer
from .train import encode_pairs, evaluate_pairs


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg)
    ckpt = Path(cfg["reward"]["checkpoint"])
    meta = load_meta(ckpt)
    model = RewardModel.load(ckpt).to(device).eval()
    tok = load_reward_tokenizer(ckpt)
    pairs = read_jsonl(Path(cfg["data"]["dir"]) / "pairs" / "val.jsonl")
    enc = encode_pairs(tok, pairs, int(meta["max_length"]))
    acc, by, margins = evaluate_pairs(model, tok, enc, pairs, device)
    t = torch.tensor(margins)
    types = [p["pair_type"] for p in pairs]
    out = {"n": len(pairs), "accuracy": acc, "accuracy_by_type": by,
           "margin_mean": float(t.mean()), "margin_std": float(t.std()) if len(t) > 1 else 0.0,
           "margin_min": float(t.min()), "margin_max": float(t.max()),
           "margin_mean_by_type": {k: float(torch.tensor([m for m, y in zip(margins, types) if y == k]).mean())
                                   for k in sorted(set(types))},
           "train_reward_mean": meta["train_reward_mean"], "train_reward_std": meta["train_reward_std"]}
    out_dir = Path(cfg["eval"]["out_dir"]) / "reward_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(out, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in sorted(set(types)):
        ax.hist([m for m, y in zip(margins, types) if y == k], bins=20, alpha=0.6, label=k)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("margin  r(chosen) - r(rejected)")
    ax.set_ylabel("pairs")
    ax.set_title(f"Val margins, accuracy {acc:.1%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "margins.png", dpi=150)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
