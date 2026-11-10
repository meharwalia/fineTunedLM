"""Robustness probe: how does the reward model score degenerate answers?

    python -m ste_rlhf.reward.probe [--n 40] [--set reward.checkpoint=checkpoints/reward_model_nodegen]

Scores RM-validation prompts with the real originals and STE rewrites, and with answers an RL
policy could fall into (shuffled words, word salad, symbol gibberish, one line repeated,
off-topic steps, a bare "."). A robust reward model ranks every degenerate answer below the
fluent original. Writes results/reward_model/probe.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ste_rlhf.common import add_config_args, get_device, load_config, read_jsonl

GIBBERISH = "![x](https://a.b/{[*:_{4}}) {{Set}]i}{|A} |~=~i{[*:_{4}}}f(1,1) <==! essentialScalar " * 6


def probe_cases(rows, seed=0):
    rng = random.Random(seed)

    def shuffled(t):
        w = t.split()
        rng.shuffle(w)
        return " ".join(w)

    def salad(t):
        w = t.split()
        return " ".join(rng.choice(w) for _ in w)

    n = len(rows)
    return {
        "original (fluent, free-form)": [r["original"] for r in rows],
        "STE rewrite": [r["ste"] for r in rows],
        "STE words shuffled": [shuffled(r["ste"]) for r in rows],
        "word salad": [salad(r["original"]) for r in rows],
        "symbol gibberish": [GIBBERISH] * n,
        "one line repeated 40x": ["1. Use the tool.\n" * 40] * n,
        "off-topic short steps": ["1. Open the door.\n2. Close the door.\n3. Open the door."] * n,
        "STE answer to another question": [rows[(k + 1) % n]["ste"] for k in range(n)],
        "bare '.'": ["."] * n,
    }


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    from ste_rlhf.reward.scorer import load_scorer

    d = Path(cfg["data"]["dir"])
    val_ids = set(json.loads((d / "prompts" / "rm_val_ids.json").read_text()))
    rows = [r for r in read_jsonl(d / "ste_answers.jsonl") if r["id"] in val_ids][: a.n]
    scorer = load_scorer(cfg, get_device(cfg))
    prompts = [r["prompt"] for r in rows]
    res = {}
    for name, answers in probe_cases(rows).items():
        s = scorer.score(prompts, answers).float()
        res[name] = {"mean": round(s.mean().item(), 3), "min": round(s.min().item(), 3),
                     "max": round(s.max().item(), 3)}
    orig = res["original (fluent, free-form)"]["mean"]
    print(f"reward model: {cfg['reward']['checkpoint']}   (n={len(rows)} validation prompts)")
    print(f"{'answer type':34s} {'mean':>7s} {'min':>7s} {'max':>7s}  below original?")
    for name, v in res.items():
        flag = "" if name in ("original (fluent, free-form)", "STE rewrite") else ("yes" if v["mean"] < orig else "NO")
        print(f"{name:34s} {v['mean']:+7.2f} {v['min']:+7.2f} {v['max']:+7.2f}  {flag}")
    out = Path(a.out) if a.out else Path(cfg["eval"]["out_dir"]) / "reward_model" / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": cfg["reward"]["checkpoint"], "n": len(rows), "scores": res}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
