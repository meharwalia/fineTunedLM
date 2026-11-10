"""Single-turn, short pairs from Anthropic/hh-rlhf."""
from __future__ import annotations

import argparse
import random
import re

from minirlhf.common.config import load_config
from minirlhf.common.io import write_jsonl
from minirlhf.common.schemas import validate_record

MAX_WORDS = 120
MAX_PROMPT_WORDS = 80
_TURN = re.compile(r"\n\n(Human|Assistant):")


def parse_dialogue(text: str):
    """Split an hh-rlhf transcript into [(role, text), ...]."""
    parts = _TURN.split(text)  # [pre, role, content, role, content, ...]
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def parse_single_turn(chosen: str, rejected: str):
    """Return (prompt, chosen_answer, rejected_answer) for short single-turn pairs, else None."""
    c, r = parse_dialogue(chosen), parse_dialogue(rejected)
    for d in (c, r):
        if [role for role, _ in d] != ["Human", "Assistant"]:
            return None
    if c[0][1] != r[0][1]:
        return None
    prompt, ca, ra = c[0][1], c[1][1], r[1][1]
    if not prompt or not ca or not ra or ca == ra:
        return None
    if max(len(ca.split()), len(ra.split())) > MAX_WORDS or len(prompt.split()) > MAX_PROMPT_WORDS:
        return None
    return prompt, ca, ra


def build_hh(rows, n: int, seed: int = 0) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        p = parse_single_turn(row["chosen"], row["rejected"])
        if p:
            out.append({"id": f"hh{i}", "prompt": p[0], "chosen": p[1], "rejected": p[2], "source": "hh-rlhf"})
    random.Random(seed).shuffle(out)
    out = out[:n]
    for r in out:
        validate_record("pref", r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--limit", type=int, default=None, help="override data.hh_rlhf_n")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    from datasets import load_dataset
    rows = load_dataset("Anthropic/hh-rlhf", split="train")
    n = args.limit or cfg["data"]["hh_rlhf_n"]
    prefs = build_hh(rows, n, cfg.get("seed", 0))
    out = f"{cfg['data']['prefs_dir']}/hh.jsonl"
    write_jsonl(out, prefs)
    print(f"wrote {len(prefs)} single-turn pairs (from {len(rows)} rows) to {out}")


if __name__ == "__main__":
    main()
