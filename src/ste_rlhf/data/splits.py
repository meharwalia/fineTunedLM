"""Disjoint prompt splits (project_v2.md 4.2, contract C2).

Run: ``python -m ste_rlhf.data.splits [--config ...] [--set ...]``  (after select_prompts)

Reads  data/prompts/candidates.jsonl
Writes data/prompts/{rm_pool,rm_spares,eval,rl_train}.jsonl   (id, prompt, hs2_row)
       data/prompts/rm_val_ids.json, data/prompts/hardneg_ids.json
       data/originals.jsonl                                   (rm_pool + spares only)
and fills the "rm_pool sample" section of data/selection_report.md.

Soft preference for rm_pool: candidates whose original has >= 3 sentences or a procedure
(list / "first, then" structure) are placed first, so rm_pool + spares come from them while
they last (there is then something to restructure). Within each group the order is a seeded
shuffle. eval and rl_train are drawn at random (seeded) from what is left.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, read_jsonl, write_jsonl
from ste_rlhf.data.select_prompts import SAMPLE_MARKER

ORIGINAL_FIELDS = ["id", "prompt", "original", "helpfulness", "verbosity", "original_tokens"]
PROMPT_FIELDS = ["id", "prompt", "hs2_row"]


def is_structured(c: dict) -> bool:
    return c["original_sentences"] >= 3 or c["has_procedure"]


def make_splits(cands: list[dict], dcfg: dict, seed: int) -> dict:
    """Return dict with rm_pool, spares, eval, rl_train (lists of candidate dicts),
    val_ids, hardneg_ids (lists of ids)."""
    rng = random.Random(seed)
    cands = sorted(cands, key=lambda c: c["hs2_row"])  # deterministic base order
    rng.shuffle(cands)
    cands = sorted(cands, key=lambda c: not is_structured(c))  # stable: structured first
    n_pool, n_spare = dcfg["n_rm_pool"], dcfg["n_spare"]
    if len(cands) < n_pool + n_spare + dcfg["n_eval"]:
        raise ValueError(f"only {len(cands)} candidates, need {n_pool + n_spare + dcfg['n_eval']}+")
    rm_pool = cands[:n_pool]
    spares = cands[n_pool:n_pool + n_spare]
    rest = cands[n_pool + n_spare:]
    rng.shuffle(rest)
    eval_ = rest[:dcfg["n_eval"]]
    rl_train = rest[dcfg["n_eval"]:dcfg["n_eval"] + dcfg["n_rl_train"]]

    pool_ids = [c["id"] for c in rm_pool]
    val_ids = sorted(rng.sample(pool_ids, dcfg["n_val_prompts"]))
    val_set = set(val_ids)
    train_ids = [i for i in pool_ids if i not in val_set]
    n_hard = dcfg["n_hardneg"]
    n_hard_val = round(n_hard * dcfg["n_val_prompts"] / n_pool)
    hardneg_ids = sorted(rng.sample(val_ids, n_hard_val) + rng.sample(train_ids, n_hard - n_hard_val))
    return dict(rm_pool=rm_pool, spares=spares, eval=eval_, rl_train=rl_train,
                val_ids=val_ids, hardneg_ids=hardneg_ids)


def _trunc(s: str, n: int = 400) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def sample_section(rm_pool: list[dict], seed: int, n: int = 50) -> str:
    rng = random.Random(seed + 1)
    pick = rng.sample(rm_pool, min(n, len(rm_pool)))
    out = [f"## {len(pick)} random rm_pool prompts (full text, truncated at 400 chars)", ""]
    for i, c in enumerate(pick, 1):
        out += [f"**{i}. {c['id']}** ({c['prompt_tokens']} prompt tok, {c['original_tokens']} original tok)", "",
                "```text", _trunc(c["prompt"]), "```", ""]
    return "\n".join(out)


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description="Make disjoint prompt splits"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    dcfg = cfg["data"]
    out = Path(dcfg["dir"])
    cands = read_jsonl(out / "prompts" / "candidates.jsonl")
    s = make_splits(cands, dcfg, cfg["seed"])
    proj = lambda rows, fields: [{k: r[k] for k in fields} for r in rows]
    write_jsonl(out / "prompts" / "rm_pool.jsonl", proj(s["rm_pool"], PROMPT_FIELDS))
    write_jsonl(out / "prompts" / "rm_spares.jsonl", proj(s["spares"], PROMPT_FIELDS))
    write_jsonl(out / "prompts" / "eval.jsonl", proj(s["eval"], PROMPT_FIELDS))
    write_jsonl(out / "prompts" / "rl_train.jsonl", proj(s["rl_train"], PROMPT_FIELDS))
    (out / "prompts" / "rm_val_ids.json").write_text(json.dumps(s["val_ids"], indent=1))
    (out / "prompts" / "hardneg_ids.json").write_text(json.dumps(s["hardneg_ids"], indent=1))
    write_jsonl(out / "originals.jsonl", proj(s["rm_pool"] + s["spares"], ORIGINAL_FIELDS))

    n_struct = sum(is_structured(c) for c in s["rm_pool"])
    summary = (f"## Splits (seed {cfg['seed']})\n\n| split | size |\n|---|---|\n"
               f"| rm_pool | {len(s['rm_pool'])} ({n_struct} with >=3 sentences or a procedure) |\n"
               f"| rm_spares | {len(s['spares'])} |\n| rm_val_ids | {len(s['val_ids'])} |\n"
               f"| hardneg_ids | {len(s['hardneg_ids'])} ({len(set(s['hardneg_ids']) & set(s['val_ids']))} val, "
               f"{len(set(s['hardneg_ids']) - set(s['val_ids']))} train) |\n"
               f"| eval | {len(s['eval'])} |\n| rl_train | {len(s['rl_train'])} |\n\n")
    report_path = out / "selection_report.md"
    report = report_path.read_text(encoding="utf-8")
    head = report.split(SAMPLE_MARKER)[0]
    report_path.write_text(head + SAMPLE_MARKER + "\n\n" + summary + sample_section(s["rm_pool"], cfg["seed"]),
                           encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
