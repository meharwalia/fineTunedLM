"""Generate answers on the eval prompts (greedy, C4 demo settings).

python -m minirlhf.eval.generate_eval --config configs/default.yaml --model base --tag base [--limit 20]
"""
from __future__ import annotations

from pathlib import Path

from minirlhf.common.io import read_jsonl, write_jsonl
from minirlhf.common.schemas import validate_record
from minirlhf.eval._util import base_parser, cfg_from_args, gen_path

FALLBACK_PROMPTS = Path("tests/fixtures/prompts_small.jsonl")


def resolve_model(cfg, model: str) -> str:
    return cfg["policy"]["name"] if model == "base" else model


def load_eval_prompts(cfg, path=None, limit=None) -> list[dict]:
    p = Path(path) if path else Path(cfg["data"]["prompts_dir"]) / "eval.jsonl"
    if not p.exists():
        print(f"[generate_eval] {p} missing, falling back to {FALLBACK_PROMPTS}")
        p = FALLBACK_PROMPTS
    rows = [validate_record("prompt", r) for r in read_jsonl(p)]
    return rows[:limit] if limit else rows


def run(cfg, model, tag, limit=None, batch_size=8, prompts_path=None, device=None) -> Path:
    from minirlhf.common.device import get_device
    from minirlhf.common.generation import generate
    from minirlhf.common.models import load_policy_from

    device = device or get_device(cfg)
    model_name = resolve_model(cfg, model)
    rows = load_eval_prompts(cfg, prompts_path, limit)
    ev = cfg["eval"]
    gen_cfg = {"max_new_tokens": ev["max_new_tokens"], "repetition_penalty": ev["repetition_penalty"],
               "max_prompt_tokens": cfg["policy"].get("max_prompt_tokens", 128)}
    lm, tok = load_policy_from(model_name, device)
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        ro = generate(lm, tok, [r["prompt"] for r in chunk], gen_cfg, greedy=bool(ev.get("greedy", True)))
        for r, resp in zip(chunk, ro.responses):
            out.append(validate_record("generation", {"id": r["id"], "prompt": r["prompt"],
                                                      "response": resp, "model": model_name}))
        print(f"[generate_eval] {min(i + batch_size, len(rows))}/{len(rows)}")
    path = gen_path(cfg, tag)
    write_jsonl(path, out)
    print(f"[generate_eval] wrote {len(out)} records to {path}")
    return path


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, help="'base' or a checkpoint path")
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prompts", default=None, help="override prompts file")
    a = p.parse_args(argv)
    run(cfg_from_args(a), a.model, a.tag, a.limit, a.batch_size, a.prompts)


if __name__ == "__main__":
    main()
