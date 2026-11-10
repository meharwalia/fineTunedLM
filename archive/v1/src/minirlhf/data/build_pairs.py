"""Merge preference files by ``data.mix``, dedupe, and make a stratified train/val split."""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from minirlhf.common.config import load_config
from minirlhf.common.io import read_jsonl, write_jsonl
from minirlhf.common.schemas import validate_record

FILES = {"human": "human.jsonl", "rule": "rule.jsonl", "hh-rlhf": "hh.jsonl"}


def _n(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _key(p: dict):
    return (_n(p["prompt"]), _n(p["chosen"]), _n(p["rejected"]))


def _pair_key(p: dict):
    """Order-free key: the same two responses for a prompt, whichever one was preferred."""
    return (_n(p["prompt"]), frozenset((_n(p["chosen"]), _n(p["rejected"]))))


def dedupe(prefs):
    """Drop exact duplicates (case/whitespace-insensitive) and pairs with chosen == rejected."""
    seen, out = set(), []
    for p in prefs:
        k = _key(p)
        if k in seen or k[1] == k[2]:
            continue
        seen.add(k)
        out.append(p)
    return out


def apply_mix(by_source: dict, mix: dict, seed: int = 0) -> dict:
    """mix[source] is the fraction (0..1, capped) of that source's pairs to keep."""
    rng = random.Random(seed)
    out = {}
    for s in sorted(by_source):
        rows = list(by_source[s])
        rng.shuffle(rows)
        frac = min(max(float(mix.get(s, 1.0)), 0.0), 1.0)
        out[s] = rows[:round(len(rows) * frac)]
    return out


def stratified_split(by_source: dict, val_fraction: float, seed: int = 0):
    """Per source, split by prompt so no prompt appears in both sets (across all sources)."""
    rng = random.Random(seed)
    train, val = [], []
    assigned = {}  # prompt -> "train" | "val", shared across sources
    for s in sorted(by_source):
        groups = defaultdict(list)
        for p in by_source[s]:
            groups[_n(p["prompt"])].append(p)
        prompts = sorted(groups)
        rng.shuffle(prompts)
        n_val = round(len(by_source[s]) * val_fraction)
        if val_fraction > 0 and len(prompts) > 1:
            n_val = max(n_val, 1)
        got = 0
        # prompts already placed by an earlier source keep their split
        for pr in [p for p in prompts if p in assigned]:
            if assigned[pr] == "val":
                val += groups[pr]
                got += len(groups[pr])
            else:
                train += groups[pr]
        for pr in [p for p in prompts if p not in assigned]:
            if got < n_val and len(prompts) > 1:
                val += groups[pr]
                got += len(groups[pr])
                assigned[pr] = "val"
            else:
                train += groups[pr]
                assigned[pr] = "train"
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def merge_and_split(by_source: dict, mix: dict, val_fraction: float, seed: int = 0):
    by_source = {s: dedupe(r) for s, r in by_source.items()}
    mixed = apply_mix(by_source, mix, seed)
    seen, final = set(), {}
    for s in FILES:  # cross-source dedupe, human > rule > hh; order-free so a
        rows = []    # lower-priority label can't contradict a higher one
        for p in mixed.get(s, []):
            if _pair_key(p) not in seen:
                seen.add(_pair_key(p))
                rows.append(p)
        if rows:
            final[s] = rows
    return stratified_split(final, val_fraction, seed)


def build(cfg, prefs_dir=None):
    prefs_dir = Path(prefs_dir or cfg["data"]["prefs_dir"])
    by_source = {}
    for s, fn in FILES.items():
        path = prefs_dir / fn
        if not path.exists():
            print(f"  (missing {path}, skipping {s})")
            continue
        by_source[s] = [validate_record("pref", r) for r in read_jsonl(path)]
    train, val = merge_and_split(by_source, cfg["data"].get("mix", {}), cfg["data"]["val_fraction"],
                                 cfg.get("seed", 0))
    write_jsonl(prefs_dir / "train.jsonl", train)
    write_jsonl(prefs_dir / "val.jsonl", val)
    return train, val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    train, val = build(cfg)
    for name, rows in (("train", train), ("val", val)):
        c = defaultdict(int)
        for r in rows:
            c[r["source"]] += 1
        print(f"{name}: {len(rows)} {dict(c)}")


if __name__ == "__main__":
    main()
