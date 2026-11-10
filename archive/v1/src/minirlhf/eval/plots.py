"""Training-curve plots from logs/*/metrics.jsonl.

python -m minirlhf.eval.plots --config configs/default.yaml [--logs-dir logs] [--smooth 5]
Plots whose required runs are missing are skipped.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from minirlhf.common.io import read_jsonl  # noqa: E402
from minirlhf.eval._util import base_parser, cfg_from_args, out_dir  # noqa: E402

SEED_RE = re.compile(r"_seed\d+")
SWEEP_RE = re.compile(r"^reinforce_beta(.+)_seed0$")


def load_runs(logs_dir) -> dict[str, dict[str, np.ndarray]]:
    """{run: {'step': arr, 'reward_mean': arr, 'kl_mean': arr, ...}} for runs with a metrics.jsonl."""
    runs = {}
    for f in sorted(Path(logs_dir).glob("*/metrics.jsonl")):
        rows = read_jsonl(f)
        if not rows:
            continue
        rows.sort(key=lambda r: r["step"])
        runs[f.parent.name] = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                               for k in ("step", "reward_mean", "kl_mean")}
    return runs


def moving_average(y, window: int = 1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) == 0:
        return y
    w = min(window, len(y))
    pad = np.concatenate([np.full(w - 1, y[0]), y])  # causal, no edge shrinkage
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def group_seeds(runs) -> dict[str, list[str]]:
    """Group runs differing only by _seed<N>; returns groups with >= 2 members."""
    groups = defaultdict(list)
    for name in runs:
        groups[SEED_RE.sub("", name)].append(name)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def final_value(y, frac: float = 0.1) -> float:
    k = max(1, int(len(y) * frac))
    return float(np.mean(y[-k:]))


def sweep_points(runs) -> list[tuple[float, float, float]]:
    """[(beta, final_reward, final_kl)] sorted by beta, from reinforce_beta<β>_seed0 runs."""
    pts = []
    for name, r in runs.items():
        m = SWEEP_RE.match(name)
        if m:
            try:
                pts.append((float(m.group(1)), final_value(r["reward_mean"]), final_value(r["kl_mean"])))
            except ValueError:
                pass
    return sorted(pts)


def ablation_pairs(runs) -> list[tuple[str, str]]:
    return [(n[:-len("_nobaseline")], n) for n in sorted(runs)
            if n.endswith("_nobaseline") and n[:-len("_nobaseline")] in runs]


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_overlay(runs, key, ylabel, path, smooth=1):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in runs.items():
        if name.startswith("pilot"):  # short smoke/pilot runs clutter the comparison
            continue
        ax.plot(r["step"], moving_average(r[key], smooth), label=name)
    ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.legend(fontsize=7)
    return _save(fig, path)


def plot_seed_groups(runs, groups, path, smooth=1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, [("reward_mean", "reward"), ("kl_mean", "KL")]):
        for gname, members in groups.items():
            n = min(len(runs[m]["step"]) for m in members)
            steps = runs[members[0]]["step"][:n]
            ys = np.stack([moving_average(runs[m][key][:n], smooth) for m in members])
            mu, sd = ys.mean(0), ys.std(0)
            ax.plot(steps, mu, label=f"{gname} (n={len(members)})")
            ax.fill_between(steps, mu - sd, mu + sd, alpha=.2)
        ax.set_xlabel("step"); ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend(fontsize=7)
    return _save(fig, path)


def plot_beta_sweep(points, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    kl = [p[2] for p in points]; rew = [p[1] for p in points]
    ax.scatter(kl, rew)  # no connecting line: points are separate runs, ordering by β is not a path
    for b, r, k in points:
        ax.annotate(f"β={b:g}", (k, r), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("final KL"); ax.set_ylabel("final reward"); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_ablation(runs, pairs, path, smooth=1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, [("reward_mean", "reward"), ("kl_mean", "KL")]):
        for with_b, without in pairs:
            ax.plot(runs[with_b]["step"], moving_average(runs[with_b][key], smooth), label=f"{with_b} (baseline)")
            ax.plot(runs[without]["step"], moving_average(runs[without][key], smooth), "--", label=without)
        ax.set_xlabel("step"); ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend(fontsize=7)
    return _save(fig, path)


def make_plots(logs_dir, out, smooth=1) -> list[str]:
    runs = load_runs(logs_dir)
    out = Path(out)
    written = []
    if not runs:
        print(f"[plots] no runs found under {logs_dir}; nothing to plot")
        return written
    written.append(plot_overlay(runs, "reward_mean", "reward (mean)", out / "reward_curves.png", smooth))
    written.append(plot_overlay(runs, "kl_mean", "KL (mean)", out / "kl_curves.png", smooth))
    groups = group_seeds(runs)
    if groups:
        written.append(plot_seed_groups(runs, groups, out / "seed_mean_std.png", smooth))
    else:
        print("[plots] skip seed_mean_std: no runs differing only by _seed<N>")
    pts = sweep_points(runs)
    if len(pts) >= 2:
        written.append(plot_beta_sweep(pts, out / "beta_sweep.png"))
    else:
        print("[plots] skip beta_sweep: need >= 2 runs named reinforce_beta<β>_seed0")
    pairs = ablation_pairs(runs)
    if pairs:
        written.append(plot_ablation(runs, pairs, out / "baseline_ablation.png", smooth))
    else:
        print("[plots] skip baseline_ablation: no run/<run>_nobaseline pair")
    for w in written:
        print(f"[plots] wrote {w}")
    return written


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window (1 = off)")
    p.add_argument("--out", default=None, help="output dir (default <eval.out_dir>/plots)")
    a = p.parse_args(argv)
    cfg = cfg_from_args(a)
    make_plots(a.logs_dir, a.out or out_dir(cfg) / "plots", a.smooth)


if __name__ == "__main__":
    main()
