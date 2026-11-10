"""Shared helpers for the eval tools."""
from __future__ import annotations

import argparse
from pathlib import Path

from minirlhf.common.config import load_config
from minirlhf.common.io import read_jsonl
from minirlhf.common.schemas import validate_record


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides", help="config overrides, e.g. reward.type=heuristic")
    return p


def cfg_from_args(args) -> dict:
    return load_config(args.config, args.overrides)


def out_dir(cfg) -> Path:
    return Path(cfg.get("eval", {}).get("out_dir", "results"))


def gen_path(cfg, tag: str) -> Path:
    return out_dir(cfg) / "generations" / f"{tag}.jsonl"


def read_generations(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run generate_eval first")
    return [validate_record("generation", r) for r in read_jsonl(path)]


def common_prompts(gens_by_tag: dict[str, list[dict]]) -> list[str]:
    """Ids present in every tag, in the order of the first tag."""
    tags = list(gens_by_tag)
    ids = [r["id"] for r in gens_by_tag[tags[0]]]
    for t in tags[1:]:
        have = {r["id"] for r in gens_by_tag[t]}
        ids = [i for i in ids if i in have]
    return ids
