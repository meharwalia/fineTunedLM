"""Plots from logs/*/metrics.jsonl and results/metrics/*.json (matplotlib, Agg).

python -m ste_rlhf.eval.plots [--logs logs] [--results results] [--smooth N] [--base base --tuned tuned]
Runs named pilot_* are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._util import metrics_dir, moving_average, parse_beta, seed_group  # noqa: E402

KEYS = {"reward": ["reward", "reward_mean", "rm_score"],
        "kl": ["kl", "kl_mean"],
        "ste": ["ste_score_mean", "ste_score", "ste"]}


def _pick(row, names):
    for n in names:
        if row.get(n) is not None:
            return row[n]
    return None


def load_runs(logs_dir) -> dict:
    """{run_name: {"step": [...], "reward": [...], "kl": [...], "ste": [...]}} (pilot_* skipped)."""
    runs = {}
    for p in sorted(Path(logs_dir).glob("*/metrics.jsonl")):
        name = p.parent.name
        if name.startswith("pilot_"):
            continue
        d = {"step": [], "reward": [], "kl": [], "ste": []}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            d["step"].append(row.get("step", len(d["step"])))
            for k, names in KEYS.items():
                d[k].append(_pick(row, names))
        if d["step"]:
            runs[name] = d
    return runs


def _clean(steps, vals):
    pts = [(s, v) for s, v in zip(steps, vals) if v is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def curve_plot(runs, key, ylabel, out, smooth=1):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n = 0
    for name, d in runs.items():
        x, y = _clean(d["step"], d[key])
        if not x:
            continue
        ax.plot(x, moving_average(y, smooth), label=name, linewidth=1.4)
        n += 1
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if n:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return n > 0


def seed_plot(runs, out, smooth=1):
    """Runs named *_seed<k> grouped; mean +/- std over seeds for reward and KL."""
    groups = {}
    for name, d in runs.items():
        g = seed_group(name)
        if g:
            groups.setdefault(g, []).append(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, [("reward", "reward"), ("kl", "KL")]):
        for g, ds in groups.items():
            series = []
            for d in ds:
                x, y = _clean(d["step"], d[key])
                series.append(dict(zip(x, moving_average(y, smooth))))
            steps = sorted(set.intersection(*[set(s) for s in series])) if series else []
            if not steps:
                continue
            import numpy as np
            arr = np.array([[s[t] for t in steps] for s in series])
            mu, sd = arr.mean(0), arr.std(0)
            ax.plot(steps, mu, label=f"{g} (n={len(ds)})")
            ax.fill_between(steps, mu - sd, mu + sd, alpha=0.25)
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if groups:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return bool(groups)


def final_value(vals, frac=0.1):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    k = max(1, int(round(len(v) * frac)))
    return sum(v[-k:]) / k


def beta_plot(runs, out):
    """Scatter of final KL vs final reward, one point per run with a beta in its name; no connecting line."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    n = 0
    for name, d in runs.items():
        b = parse_beta(name)
        fr, fk = final_value(d["reward"]), final_value(d["kl"])
        if b is None or fr is None or fk is None:
            continue
        ax.scatter([fk], [fr], s=40)
        ax.annotate(f"β={b:g}", (fk, fr), textcoords="offset points", xytext=(5, 4), fontsize=8)
        n += 1
    ax.set_xlabel("final KL (mean of last 10% steps)")
    ax.set_ylabel("final reward (mean of last 10% steps)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return n > 0


BEFORE_AFTER = [("ste_score_mean", "STE score"), ("avg_sentence_words_mean", "Avg sentence words"),
                ("pct_passive_mean", "% passive"), ("list_items_mean", "List items"),
                ("filler_phrases_mean", "Filler phrases"), ("complex_words_mean", "Complex words")]


def before_after_plot(results_dir, out, base="base", tuned="tuned"):
    """Small multiples (own y-axis each) so differently scaled metrics stay readable."""
    md = metrics_dir(results_dir)
    pb, pt = md / f"{base}.json", md / f"{tuned}.json"
    if not (pb.exists() and pt.exists()):
        return False
    mb, mt = json.loads(pb.read_text()), json.loads(pt.read_text())
    fig, axes = plt.subplots(1, len(BEFORE_AFTER), figsize=(13, 3.4))
    for ax, (k, label) in zip(axes, BEFORE_AFTER):
        vals = [mb.get(k) or 0.0, mt.get(k) or 0.0]
        bars = ax.bar([base, tuned], vals, color=["#8a8a8a", "#2a7fbf"])
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, max(vals + [1e-9]) * 1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def make_all(logs_dir, results_dir, smooth=1, base="base", tuned="tuned", log=print):
    out = Path(results_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(logs_dir)
    made = {}
    made["reward_curves.png"] = curve_plot(runs, "reward", "reward (RM score - β·KL)", out / "reward_curves.png", smooth)
    made["kl_curves.png"] = curve_plot(runs, "kl", "KL vs reference", out / "kl_curves.png", smooth)
    made["ste_curves.png"] = curve_plot(runs, "ste", "STE score (mean)", out / "ste_curves.png", smooth)
    made["seed_mean_std.png"] = seed_plot(runs, out / "seed_mean_std.png", smooth)
    made["beta_sweep.png"] = beta_plot(runs, out / "beta_sweep.png")
    made["ste_before_after.png"] = before_after_plot(results_dir, out / "ste_before_after.png", base, tuned)
    for k, v in made.items():
        log(f"{'wrote' if v else 'empty/skipped'}: {out / k}")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window (steps)")
    ap.add_argument("--base", default="base")
    ap.add_argument("--tuned", default="tuned")
    a = ap.parse_args(argv)
    make_all(a.logs, a.results, a.smooth, a.base, a.tuned)


if __name__ == "__main__":
    main()
