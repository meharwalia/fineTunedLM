"""Build train/eval prompt sets from Dolly-15k (+ optional handwritten prompts)."""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from minirlhf.common.config import load_config
from minirlhf.common.io import write_jsonl
from minirlhf.common.schemas import validate_record

CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")
MAX_WORDS = 40


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def filter_dolly(rows, max_words: int = MAX_WORDS, categories=CATEGORIES) -> list[dict]:
    """Keep context-free rows from the chosen categories with at most ``max_words`` words."""
    out = []
    for r in rows:
        if (r.get("context") or "").strip():
            continue
        if r.get("category") not in categories:
            continue
        text = re.sub(r"\s+", " ", (r.get("instruction") or "").strip())
        if not text or len(text.split()) > max_words:
            continue
        out.append({"prompt": text, "source": f"dolly:{r['category']}"})
    return out


def read_handwritten(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [{"prompt": ln, "source": "handwritten"} for ln in lines if ln and not ln.startswith("#")]


def dedupe(items) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = _norm(it["prompt"])
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def split_prompts(items, n_train: int, n_eval: int, seed: int = 0):
    """Seeded shuffle, then eval = first n_eval, train = next n_train. Disjoint by construction."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items[n_eval:n_eval + n_train], items[:n_eval]


def to_records(items, prefix: str) -> list[dict]:
    recs = [{"id": f"{prefix}{i}", "prompt": it["prompt"], "source": it["source"]} for i, it in enumerate(items)]
    for r in recs:
        validate_record("prompt", r)
    return recs


def load_dolly():
    from datasets import load_dataset
    return load_dataset("databricks/databricks-dolly-15k", split="train")


def build(cfg, rows=None, handwritten_path=None, out_dir=None):
    d = cfg["data"]
    out_dir = Path(out_dir or d["prompts_dir"])
    rows = load_dolly() if rows is None else rows
    items = filter_dolly(rows)
    items += read_handwritten(handwritten_path or out_dir / "handwritten.txt")
    items = dedupe(items)
    tr, ev = split_prompts(items, d["n_train_prompts"], d["n_eval_prompts"], cfg.get("seed", 0))
    train, evalr = to_records(tr, "train"), to_records(ev, "eval")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", evalr)
    return train, evalr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--handwritten", default=None, help="text file, one prompt per line")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    train, ev = build(cfg, handwritten_path=args.handwritten)
    print(f"wrote {len(train)} train and {len(ev)} eval prompts to {cfg['data']['prompts_dir']}")


if __name__ == "__main__":
    main()
