"""Per-tag metrics: reward, length, diversity, KL vs the base model.

python -m minirlhf.eval.metrics --config configs/default.yaml --tags base tuned
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from minirlhf.eval._util import base_parser, cfg_from_args, gen_path, out_dir, read_generations


def _words(text: str) -> list[str]:
    return text.lower().split()


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams / total n-grams over all texts (word level). 0.0 if no n-grams."""
    total, seen = 0, set()
    for t in texts:
        w = _words(t)
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
            total += 1
    return len(seen) / total if total else 0.0


def mean_std(xs) -> tuple[float, float]:
    t = torch.as_tensor(list(xs), dtype=torch.float64)
    if t.numel() == 0:
        return 0.0, 0.0
    return float(t.mean()), float(t.std(unbiased=False))


def build_rollout(tokenizer, prompts, responses, max_prompt_tokens=128, device="cpu"):
    """Rebuild a Rollout from text so logprobs can be recomputed (response tokens + EOS)."""
    from minirlhf.common.generation import Rollout, format_prompts
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    enc = tokenizer(format_prompts(tokenizer, prompts), return_tensors="pt", padding=True, truncation=True,
                    max_length=max_prompt_tokens, add_special_tokens=False)
    toks = [tokenizer(r, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id] for r in responses]
    L = max(len(t) for t in toks)
    ids = torch.full((len(toks), L), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(toks), L), dtype=torch.long)
    for i, t in enumerate(toks):
        ids[i, :len(t)] = torch.tensor(t)
        mask[i, :len(t)] = 1
    return Rollout(prompts=list(prompts), query_ids=enc["input_ids"].to(device),
                   query_mask=enc["attention_mask"].to(device), response_ids=ids.to(device),
                   response_mask=mask.to(device), responses=list(responses))


@torch.no_grad()
def sequence_logprobs(model, tokenizer, prompts, responses, max_prompt_tokens=128, batch_size=8):
    """Sum over response tokens of log p(token), one value per response (masked positions excluded)."""
    from minirlhf.common.logprobs import response_logprobs
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch_size):
        ro = build_rollout(tokenizer, prompts[i:i + batch_size], responses[i:i + batch_size], max_prompt_tokens, device)
        lp = response_logprobs(model, ro).cpu()
        out.append((lp * ro.response_mask.cpu()).sum(1))
    return torch.cat(out)


def kl_vs_base(model_name, base_name, prompts, responses, max_prompt_tokens, device, batch_size=8):
    """Per-response sum_t (logp_model - logp_base); exactly 0 when model_name is the base model."""
    from minirlhf.common.models import load_policy_from
    if model_name == base_name:
        return [0.0] * len(prompts)
    base, tok = load_policy_from(base_name, device)
    lp_base = sequence_logprobs(base, tok, prompts, responses, max_prompt_tokens, batch_size)
    del base
    model, tok = load_policy_from(model_name, device)
    lp_model = sequence_logprobs(model, tok, prompts, responses, max_prompt_tokens, batch_size)
    del model
    return (lp_model - lp_base).tolist()


def compute_text_metrics(responses, rewards, kls) -> dict:
    rm, rs = mean_std(rewards)
    km, ks = mean_std(kls)
    return {"n": len(responses), "reward_mean": rm, "reward_std": rs,
            "resp_len_words": mean_std([len(_words(r)) for r in responses])[0],
            "distinct1": distinct_n(responses, 1), "distinct2": distinct_n(responses, 2),
            "kl_mean": km, "kl_std": ks}


def format_summary(all_metrics: dict[str, dict]) -> str:
    cols = [("Tag", None), ("N", "n"), ("Reward mean", "reward_mean"), ("Reward std", "reward_std"),
            ("Length (words)", "resp_len_words"), ("Distinct-1", "distinct1"), ("Distinct-2", "distinct2"),
            ("KL vs base", "kl_mean")]
    lines = ["| " + " | ".join(c[0] for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for tag, m in all_metrics.items():
        cells = [tag, str(m["n"])] + [f"{m[k]:.3f}" for _, k in cols[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(mdir) -> dict:
    mdir = Path(mdir)
    allm = {}
    for f in sorted(mdir.glob("*.json")):
        m = json.loads(f.read_text())
        allm[m.get("tag", f.stem)] = m
    allm = dict(sorted(allm.items(), key=lambda kv: (kv[0] != "base", kv[0])))
    (mdir / "summary.md").write_text(format_summary(allm))
    return allm


def run(cfg, tags, device=None, batch_size=8) -> dict:
    from minirlhf.common.device import get_device
    from minirlhf.reward.factory import load_scorer
    device = device or get_device(cfg)
    scorer = load_scorer(cfg, device)
    base_name = cfg["policy"]["name"]
    mdir = out_dir(cfg) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    mpt = cfg["policy"].get("max_prompt_tokens", 128)
    for tag in tags:
        gens = read_generations(gen_path(cfg, tag))
        prompts = [g["prompt"] for g in gens]
        responses = [g["response"] for g in gens]
        rewards = scorer.score(prompts, responses).tolist()
        kls = kl_vs_base(gens[0]["model"], base_name, prompts, responses, mpt, device, batch_size)
        m = compute_text_metrics(responses, rewards, kls)
        m.update(tag=tag, model=gens[0]["model"], reward_type=cfg["reward"]["type"])
        (mdir / f"{tag}.json").write_text(json.dumps(m, indent=2))
        print(f"[metrics] {tag}: {m}")
    return write_summary(mdir)


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args(argv)
    run(cfg_from_args(a), a.tags, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
