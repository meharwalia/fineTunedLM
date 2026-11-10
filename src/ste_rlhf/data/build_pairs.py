"""Build reward-model preference pairs (contract C2).

Run: ``python -m ste_rlhf.data.build_pairs [--config ...] [--set data.dir=...]``

Reads  data/ste_answers.jsonl, data/hard_negatives.jsonl, data/prompts/{rm_pool.jsonl,rm_val_ids.json,hardneg_ids.json}
Writes data/pairs/{train,val}.jsonl  (id, prompt, chosen, rejected, pair_type)
Val = every pair whose id is in rm_val_ids.json; train = the rest (shuffled, seeded).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, read_jsonl, write_jsonl


DEGENERATE_TYPES = ("ste_vs_shuffled", "ste_vs_offtopic", "ste_vs_repeated", "ste_vs_truncated")


def _lines(text):
    return [l for l in text.splitlines() if l.strip()]


def degenerate_pairs(ste, ids, seed, n_per_type, n_offtopic=None):
    """Pairs that teach the reward model what STE-looking text must still be: fluent, on topic,
    non-repetitive and complete. Each rejected answer is made by code from the STE rewrites
    (no new data source). The probe before this change showed the RM scoring shuffled words
    (+0.48), one line repeated 40x (+0.56) and off-topic steps (+0.42) far above the fluent
    originals (-0.88), which RL would exploit.
    `ids` must all come from one split (train or val) so off-topic answers never leak across."""
    rng = random.Random(seed)
    ids = list(ids)
    out = []
    if len(ids) < 2 or n_per_type <= 0:
        return out

    def pick(cond=lambda i: True):
        cands = [i for i in ids if cond(i)]
        return rng.sample(cands, min(n_per_type, len(cands)))

    for i in pick():  # same words, random order: not fluent
        w = ste[i]["ste"].split()
        rng.shuffle(w)
        out.append((i, " ".join(w), "ste_vs_shuffled"))
    # a good STE answer to a different question: not relevant. Relevance is the hardest
    # thing for the RM to learn, so it gets more pairs.
    n_off = len(ids) if n_offtopic is None else n_offtopic
    for i in rng.sample(ids, min(n_off, len(ids))):
        j = rng.choice([k for k in ids if k != i])
        out.append((i, ste[j]["ste"], "ste_vs_offtopic"))
    for i in pick(lambda k: any(len(l) >= 15 for l in _lines(ste[k]["ste"]))):  # looping
        line = rng.choice([l for l in _lines(ste[i]["ste"]) if len(l) >= 15])
        out.append((i, "\n".join([line] * rng.randint(8, 25)), "ste_vs_repeated"))
    for i in pick(lambda k: len(_lines(ste[k]["ste"])) >= 4):  # stops after the first line
        out.append((i, _lines(ste[i]["ste"])[0], "ste_vs_truncated"))
    return [dict(id=i, prompt=ste[i]["prompt"], chosen=ste[i]["ste"], rejected=rej, pair_type=t)
            for i, rej, t in out]


def build_pairs(ste_rows, hard_rows, pool_ids, hardneg_ids, val_ids, seed=0,
                n_per_type=150, n_val_per_type=20):
    """Return (train, val) lists of pair dicts."""
    ste = {r["id"]: r for r in ste_rows}
    hard = {r["id"]: r for r in hard_rows}
    pairs = []
    for i in pool_ids:
        r = ste[i]
        pairs.append(dict(id=i, prompt=r["prompt"], chosen=r["ste"], rejected=r["original"],
                          pair_type="ste_vs_original"))
    for i in hardneg_ids:
        pairs.append(dict(id=i, prompt=ste[i]["prompt"], chosen=ste[i]["ste"],
                          rejected=hard[i]["hard_negative"], pair_type="ste_vs_hardneg"))
    val_set = set(val_ids)
    pairs += degenerate_pairs(ste, [i for i in pool_ids if i not in val_set], seed, n_per_type)
    pairs += degenerate_pairs(ste, [i for i in pool_ids if i in val_set], seed + 1, n_val_per_type)
    val = [p for p in pairs if p["id"] in val_set]
    train = [p for p in pairs if p["id"] not in val_set]
    assert not {p["prompt"] for p in train} & {p["prompt"] for p in val}, "prompt overlap train/val"
    assert not {p["id"] for p in train} & val_set
    random.Random(seed).shuffle(train)
    return train, val


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    d = Path(cfg["data"]["dir"])
    pool_ids = [r["id"] for r in read_jsonl(d / "prompts" / "rm_pool.jsonl")]
    val_ids = json.loads((d / "prompts" / "rm_val_ids.json").read_text())
    hardneg_ids = json.loads((d / "prompts" / "hardneg_ids.json").read_text())
    train, val = build_pairs(read_jsonl(d / "ste_answers.jsonl"), read_jsonl(d / "hard_negatives.jsonl"),
                             pool_ids, hardneg_ids, val_ids, cfg["seed"])
    write_jsonl(d / "pairs" / "train.jsonl", train)
    write_jsonl(d / "pairs" / "val.jsonl", val)
    from collections import Counter
    for name, rows in (("train", train), ("val", val)):
        c = Counter(p["pair_type"] for p in rows)
        print(f"{name}: {len(rows)} pairs " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
