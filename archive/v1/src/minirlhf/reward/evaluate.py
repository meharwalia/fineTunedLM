"""Evaluate a reward-model checkpoint on preference pairs.

    python -m minirlhf.reward.evaluate --checkpoint checkpoints/reward_model --data data/prefs/val.jsonl
    python -m minirlhf.reward.evaluate --source hh --n 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from minirlhf.common.config import load_config
from minirlhf.common.device import get_device
from minirlhf.common.io import read_jsonl
from minirlhf.reward.model import load_meta
from minirlhf.reward.train import accuracy_report, load_hh, pair_margins


def plot_margins(margins, per_source_margins, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(-1, margins.min()), max(1, margins.max()), 40)
    if len(per_source_margins) > 1:
        for s, m in per_source_margins.items():
            ax.hist(m, bins=bins, alpha=0.5, label=f"{s} (n={len(m)})")
        ax.legend()
    else:
        ax.hist(margins, bins=bins, alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("margin = r(chosen) - r(rejected)")
    ax.set_ylabel("pairs")
    ax.set_title(f"Reward margins (accuracy {(margins > 0).mean():.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def evaluate(checkpoint, rows, device, max_length=None, plot_path="results/plots/rm_margins.png"):
    meta = load_meta(checkpoint)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    margins = pair_margins(model, tok, rows, max_length or meta["max_length"], device)
    acc, per_src = accuracy_report(margins, rows)
    by = {}
    for m, r in zip(margins, rows):
        by.setdefault(r.get("source", "unknown"), []).append(m)
    by = {k: np.array(v) for k, v in by.items()}
    if plot_path:
        plot_margins(margins, by, plot_path)
    counts = {s: len(v) for s, v in by.items()}
    return {"accuracy": acc, "per_source": per_src, "counts": counts, "n": len(rows),
            "margin_mean": float(margins.mean()), "margin_std": float(margins.std())}


def format_table(res) -> str:
    lines = ["| source | n | accuracy |", "|---|---|---|"]
    for s, a in res["per_source"].items():
        lines.append(f"| {s} | {res['counts'][s]} | {a:.4f} |")
    lines.append(f"| **overall** | {res['n']} | {res['accuracy']:.4f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default=None, help="pairs jsonl (default data/prefs/val.jsonl)")
    ap.add_argument("--source", choices=["files", "hh"], default="files")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345, help="hh sampling seed (differs from training)")
    ap.add_argument("--plot", default="results/plots/rm_margins.png")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    device = get_device(cfg)
    rows = load_hh(a.n, a.seed) if a.source == "hh" else read_jsonl(
        a.data or Path(cfg["data"]["prefs_dir"]) / "val.jsonl")
    res = evaluate(a.checkpoint or cfg["reward"]["checkpoint"], rows, device, plot_path=a.plot)
    print(format_table(res))
    print(f"margin mean {res['margin_mean']:.3f} std {res['margin_std']:.3f}; plot -> {a.plot}")


if __name__ == "__main__":
    main()
