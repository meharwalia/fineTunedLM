"""Split work for the writer subagents (plan W1-W10, H1-H2).

Run: ``python -m ste_rlhf.data.writer_batches``  (after splits)

Writes data/writes/todo_00..09.jsonl (50 rm_pool items each: id, prompt, original) and
data/writes/hardneg_todo_00..01.jsonl (50 items each from hardneg_ids).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, read_jsonl, write_jsonl

BATCH = 50


def chunk(rows: list[dict], size: int = BATCH) -> list[list[dict]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def make_batches(originals: list[dict], pool_ids: list[str], hardneg_ids: list[str]):
    by_id = {o["id"]: {"id": o["id"], "prompt": o["prompt"], "original": o["original"]} for o in originals}
    return chunk([by_id[i] for i in pool_ids]), chunk([by_id[i] for i in hardneg_ids])


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description="Make writer todo batches"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out = Path(cfg["data"]["dir"])
    originals = read_jsonl(out / "originals.jsonl")
    pool_ids = [r["id"] for r in read_jsonl(out / "prompts" / "rm_pool.jsonl")]
    hardneg_ids = json.loads((out / "prompts" / "hardneg_ids.json").read_text())
    todo, hard = make_batches(originals, pool_ids, hardneg_ids)
    for i, b in enumerate(todo):
        write_jsonl(out / "writes" / f"todo_{i:02d}.jsonl", b)
    for i, b in enumerate(hard):
        write_jsonl(out / "writes" / f"hardneg_todo_{i:02d}.jsonl", b)
    print(f"wrote {len(todo)} todo batches and {len(hard)} hardneg batches to {out / 'writes'}")


if __name__ == "__main__":
    main()
