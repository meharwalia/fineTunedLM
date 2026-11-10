"""Sample two responses per prompt from the base policy."""
from __future__ import annotations

import argparse

from minirlhf.common.config import load_config
from minirlhf.common.io import read_jsonl, write_jsonl
from minirlhf.common.schemas import validate_record


def make_candidate(pid: str, prompt: str, r1: str, r2: str, gen: dict):
    """Return a candidate record, or None if either response is empty or both are identical."""
    r1, r2 = r1.strip(), r2.strip()
    if not r1 or not r2 or r1 == r2:
        return None
    return validate_record("candidate", {"id": pid, "prompt": prompt, "responses": [r1, r2], "gen": gen})


def generate_candidates(cfg, prompts, model, tokenizer, batch_size: int = 16):
    from minirlhf.common.generation import generate
    pol = cfg["policy"]
    gen = {"model": pol["name"], "temperature": pol["temperature"], "top_p": pol["top_p"],
           "max_new_tokens": pol["max_new_tokens"], "seed": cfg.get("seed", 0)}
    out, dropped = [], 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [b["prompt"] for b in batch]
        a = generate(model, tokenizer, texts, pol, greedy=False).responses
        b = generate(model, tokenizer, texts, pol, greedy=False).responses
        for rec, r1, r2 in zip(batch, a, b):
            c = make_candidate(rec["id"], rec["prompt"], r1, r2, gen)
            if c is None:
                dropped += 1
            else:
                out.append(c)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)
    return out, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train prompts")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prompts", default=None, help="prompt jsonl (default: <prompts_dir>/train.jsonl)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    from minirlhf.common.device import get_device
    from minirlhf.common.models import load_policy
    from minirlhf.common.seed import set_seed
    set_seed(cfg.get("seed", 0))
    prompts = read_jsonl(args.prompts or f"{cfg['data']['prompts_dir']}/train.jsonl")
    if args.limit:
        prompts = prompts[:args.limit]
    model, tok = load_policy(cfg, get_device(cfg))
    cands, dropped = generate_candidates(cfg, prompts, model, tok, args.batch_size)
    write_jsonl(cfg["data"]["candidates_path"], cands)
    print(f"wrote {len(cands)} candidates ({dropped} dropped) to {cfg['data']['candidates_path']}")


if __name__ == "__main__":
    main()
