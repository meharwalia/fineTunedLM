"""Greedy generations on the eval prompts -> results/generations/<tag>.jsonl (schema C2).

python -m ste_rlhf.eval.generate_eval --model PATH_OR_HF_NAME --tag TAG [--limit N] [--set K=V ...]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from ste_rlhf.common import add_config_args, append_jsonl, get_device, load_config, read_jsonl

from ._util import gen_path

BATCH = 8


def done_ids(path) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # a half-written last line is regenerated
    return ids


def free_memory(device):
    gc.collect()
    try:
        import torch
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_for_tag(model_name, tag, cfg, prompts_path=None, limit=None, out_dir=None, device=None,
                     batch_size=BATCH, log=print) -> Path:
    from ste_rlhf.common import generate
    from ste_rlhf.common.models import load_policy_from

    ev = cfg["eval"]
    out_dir = out_dir or ev.get("out_dir", "results")
    prompts_path = prompts_path or Path(cfg["data"]["dir"]) / "prompts" / "eval.jsonl"
    rows = read_jsonl(prompts_path)
    if limit:
        rows = rows[:int(limit)]
    path = gen_path(out_dir, tag)
    done = done_ids(path)
    todo = [r for r in rows if r["id"] not in done]
    log(f"{tag}: {len(done)} done, {len(todo)} to generate -> {path}")
    if not todo:
        return path
    device = device or get_device(cfg)
    model, tok = load_policy_from(model_name, device)
    gen_cfg = {"max_new_tokens": ev["max_new_tokens"], "repetition_penalty": ev.get("repetition_penalty", 1.0),
               "max_prompt_tokens": cfg["policy"].get("max_prompt_tokens", 320)}
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        ro = generate(model, tok, [r["prompt"] for r in batch], gen_cfg, greedy=bool(ev.get("greedy", True)))
        for r, resp, fin, m in zip(batch, ro.responses, ro.finished, ro.response_mask):
            append_jsonl(path, {"id": r["id"], "prompt": r["prompt"], "response": resp,
                                "finished": bool(fin), "num_tokens": int(m.sum().item()),
                                "model": str(model_name)})
        del ro
        free_memory(device)
        log(f"  {min(i + batch_size, len(todo))}/{len(todo)}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint dir or HF model name")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompts", default=None, help="default: <data.dir>/prompts/eval.jsonl")
    add_config_args(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    generate_for_tag(a.model, a.tag, cfg, prompts_path=a.prompts, limit=a.limit)


if __name__ == "__main__":
    main()
