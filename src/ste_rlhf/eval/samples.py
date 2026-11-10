"""Side-by-side sample answers with STE and RM scores -> results/samples.md.

python -m ste_rlhf.eval.samples --tags base tuned --n 10 [--no-rm] [--set K=V ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ste_rlhf.common import add_config_args, get_device, load_config, read_jsonl

from ._util import gen_path


def build_markdown(tags, by_tag, ste_scores, rm_scores, n) -> str:
    """by_tag {tag: {id: row}}; ste_scores/rm_scores {tag: {id: float}} (rm may be empty)."""
    ids = [i for i in by_tag[tags[0]] if all(i in by_tag[t] for t in tags)][:n]
    out = [f"# Sample answers ({', '.join(tags)})\n"]
    for k, i in enumerate(ids, 1):
        out.append(f"## {k}. {by_tag[tags[0]][i]['prompt'].strip()}\n")
        out.append("| " + " | ".join(tags) + " |")
        out.append("|" + "---|" * len(tags))
        cells = []
        for t in tags:
            r = by_tag[t][i]
            head = f"**STE {ste_scores.get(t, {}).get(i, float('nan')):.0f}"
            if i in rm_scores.get(t, {}):
                head += f" | RM {rm_scores[t][i]:.2f}"
            head += f" | {len(r['response'].split())} words" + ("" if r.get("finished", True) else " | cut off")
            head += "**"
            body = r["response"].strip().replace("|", "\\|").replace("\n", "<br>")
            cells.append(f"{head}<br><br>{body}")
        out.append("| " + " | ".join(cells) + " |\n")
    return "\n".join(out)


def main(argv=None, ste_report_fn=None, scorer=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--no-rm", action="store_true")
    ap.add_argument("--out", default=None, help="default: <eval.out_dir>/samples.md")
    add_config_args(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out_dir = cfg["eval"].get("out_dir", "results")
    if ste_report_fn is None:
        from ste_rlhf.ste.metrics import ste_report as ste_report_fn
    if scorer is None and not a.no_rm:
        from ste_rlhf.reward.scorer import load_scorer
        scorer = load_scorer(cfg, get_device(cfg))
    by_tag, ste, rm = {}, {}, {}
    for t in a.tags:
        rows = read_jsonl(gen_path(out_dir, t))
        by_tag[t] = {r["id"]: r for r in rows}
    ids = [i for i in by_tag[a.tags[0]] if all(i in by_tag[t] for t in a.tags)][:a.n]
    for t in a.tags:
        rs = [by_tag[t][i] for i in ids]
        ste[t] = {r["id"]: float(ste_report_fn(r["response"])["ste_score"]) for r in rs}
        rm[t] = {}
        if scorer is not None:
            sc = scorer.score([r["prompt"] for r in rs], [r["response"] for r in rs])
            rm[t] = {r["id"]: float(s) for r, s in zip(rs, sc)}
    md = build_markdown(a.tags, by_tag, ste, rm, a.n)
    path = Path(a.out) if a.out else Path(out_dir) / "samples.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
