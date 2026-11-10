"""Evaluation metrics per generation tag -> results/metrics/<tag>.json, summary.json, summary.md.

python -m ste_rlhf.eval.metrics --tags base tuned [--no-kl] [--no-rm] [--set K=V ...]
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ste_rlhf.common import add_config_args, get_device, load_config, read_jsonl

from ._util import PREFERRED_COLS, distinct_n, gen_path, metrics_dir

COL_LABELS = {
    "ste_score_mean": "STE score", "pct_passes_80": "% >= 80", "avg_sentence_words_mean": "Sent. words",
    "pct_passive_mean": "% passive", "list_items_mean": "List items", "filler_phrases_mean": "Filler",
    "complex_words_mean": "Complex", "flesch_mean": "Flesch", "words_mean": "Words",
    "rm_score_mean": "RM score", "kl_vs_base": "KL vs base", "distinct2": "Distinct-2",
    "finish_rate": "Finish rate", "num_tokens_mean": "Tokens",
}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def basic_metrics(rows) -> dict:
    return {
        "n": len(rows),
        "finish_rate": _mean(1.0 if r.get("finished") else 0.0 for r in rows),
        "num_tokens_mean": _mean(r.get("num_tokens", 0) for r in rows),
        "distinct2": distinct_n([r["response"] for r in rows], 2),
    }


def sequence_kl(rows, base_name, model_name, cfg, device, batch_size=4):
    """Mean over prompts of sum_t [log p_model(y_t) - log p_base(y_t)], teacher-forced on the saved responses."""
    import torch
    from ste_rlhf.common import response_logprobs
    from ste_rlhf.common.models import load_policy_from

    base, tok = load_policy_from(base_name, device)
    base.eval()
    tuned, _ = load_policy_from(model_name, device)
    tuned.eval()
    mpt = int(cfg["policy"].get("max_prompt_tokens", 320))
    vals = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            ro = build_rollout(tok, rows[i:i + batch_size], mpt, device)
            lp_m = response_logprobs(tuned, ro)
            lp_b = response_logprobs(base, ro)
            m = ro.response_mask.to(lp_m.device).to(lp_m.dtype)
            vals.extend(((lp_m - lp_b) * m).sum(1).tolist())
            del ro, lp_m, lp_b
            _free(device)
    return _mean(vals)


def build_rollout(tok, rows, max_prompt_tokens, device=None):
    """Rollout from saved (prompt, response, finished) rows for teacher forcing."""
    import torch
    from ste_rlhf.common import Rollout
    from ste_rlhf.common.generation import format_prompts

    tok.padding_side = "left"
    tok.truncation_side = "left"
    enc = tok(format_prompts(tok, [r["prompt"] for r in rows]), return_tensors="pt", padding=True,
              truncation=True, max_length=max_prompt_tokens, add_special_tokens=False)
    ids = []
    for r in rows:
        t = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("finished"):
            t = t + [tok.eos_token_id]
        ids.append(t or [tok.eos_token_id])
    L = max(len(t) for t in ids)
    resp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(ids), L), dtype=torch.long)
    for j, t in enumerate(ids):
        resp[j, :len(t)] = torch.tensor(t)
        mask[j, :len(t)] = 1
    return Rollout(prompts=[r["prompt"] for r in rows], query_ids=enc["input_ids"], query_mask=enc["attention_mask"],
                   response_ids=resp, response_mask=mask, responses=[r["response"] for r in rows],
                   finished=[bool(r.get("finished")) for r in rows])


def _free(device):
    try:
        import gc
        import torch
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def tag_metrics(tag, rows, *, ste_table_fn, scorer=None, kl_fn=None) -> dict:
    out = basic_metrics(rows)
    if ste_table_fn is not None:
        out.update(ste_table_fn({tag: [r["response"] for r in rows]})[tag])
    out.setdefault("words_mean", _mean(len(r["response"].split()) for r in rows))
    if scorer is not None:
        scores = [float(s) for s in scorer.score([r["prompt"] for r in rows], [r["response"] for r in rows])]
        out["rm_score_mean"], out["rm_score_std"] = _mean(scores), _std(scores)
    if tag == "base":
        out["kl_vs_base"] = 0.0
    elif kl_fn is not None:
        out["kl_vs_base"] = float(kl_fn(rows))
    return out


def format_summary_md(tags: dict) -> str:
    """Markdown table of the key columns (only columns present for at least one tag), base first."""
    cols = [c for c in PREFERRED_COLS + ["num_tokens_mean"] if any(c in m for m in tags.values())]
    order = sorted(tags, key=lambda t: (t != "base", t))
    lines = ["| tag | n | " + " | ".join(COL_LABELS.get(c, c) for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for t in order:
        m = tags[t]
        cells = []
        for c in cols:
            v = m.get(c)
            if v is None:
                cells.append("-")
            elif c in ("rm_score_mean",) and m.get("rm_score_std") is not None:
                cells.append(f"{v:.2f} ± {m['rm_score_std']:.2f}")
            elif c == "finish_rate":
                cells.append(f"{100 * v:.0f}%")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {t} | {m.get('n', '-')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, new: dict) -> dict:
    """Write per-tag json (merged over any earlier values), summary.json and summary.md."""
    md = metrics_dir(out_dir)
    md.mkdir(parents=True, exist_ok=True)
    for tag, m in new.items():
        p = md / f"{tag}.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(m)
        p.write_text(json.dumps(old, indent=2))
    tags = {}
    for p in sorted(md.glob("*.json")):
        if p.name != "summary.json":
            tags[p.stem] = json.loads(p.read_text())
    summary = {"tags": tags, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (md / "summary.json").write_text(json.dumps(summary, indent=2))
    (md / "summary.md").write_text(format_summary_md(tags))
    return summary


def run(tags, cfg, out_dir=None, no_kl=False, no_rm=False, scorer=None, ste_table_fn=None, kl_fn_factory=None,
        log=print) -> dict:
    out_dir = out_dir or cfg["eval"].get("out_dir", "results")
    if ste_table_fn is None:
        from ste_rlhf.eval.ste_eval import ste_table as ste_table_fn
    device = None
    if scorer is None and not no_rm:
        from ste_rlhf.reward.scorer import load_scorer
        device = get_device(cfg)
        scorer = load_scorer(cfg, device)
    new = {}
    for tag in tags:
        rows = read_jsonl(gen_path(out_dir, tag))
        kl_fn = None
        if not no_kl and tag != "base":
            if kl_fn_factory is not None:
                kl_fn = kl_fn_factory(tag, rows)
            else:
                device = device or get_device(cfg)
                model_name = rows[0]["model"]
                kl_fn = lambda r, mn=model_name: sequence_kl(r, cfg["policy"]["name"], mn, cfg, device)
        log(f"metrics: {tag} ({len(rows)} rows)")
        new[tag] = tag_metrics(tag, rows, ste_table_fn=ste_table_fn, scorer=None if no_rm else scorer, kl_fn=kl_fn)
    return write_outputs(out_dir, new)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--no-kl", action="store_true", help="skip KL vs base (loads two models)")
    ap.add_argument("--no-rm", action="store_true", help="skip reward-model scoring")
    add_config_args(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    summary = run(a.tags, cfg, no_kl=a.no_kl, no_rm=a.no_rm)
    print((metrics_dir(cfg["eval"].get("out_dir", "results")) / "summary.md").read_text())
    return summary


if __name__ == "__main__":
    main()
