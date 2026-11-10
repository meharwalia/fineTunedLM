"""Merge writer outputs into gated datasets (contract C2).

Run: ``python -m ste_rlhf.data.collect_writes [--config ...] [--set data.dir=...]``

Reads  data/originals.jsonl, data/writes/batch_*.jsonl (id, ste), data/writes/hardneg_*.jsonl (id, hard_negative)
Writes data/ste_answers.jsonl, data/hard_negatives.jsonl, data/pairs_report.md
Gate: STE rewrite score >= 80; hard negative score <= 50 and has_list.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, read_jsonl, write_jsonl
from ste_rlhf.ste.metrics import ste_report

STE_MIN, HARD_MAX = 80.0, 50.0


def collect(originals, ste_rows, hard_rows):
    """Return (ste_answers, hard_negatives, failures)."""
    orig = {r["id"]: r for r in originals}
    failures, ste_out, hard_out = [], [], []
    seen = set()
    for r in ste_rows:
        if r["id"] not in orig:
            failures.append((r["id"], "ste", "unknown id")); continue
        if r["id"] in seen:
            failures.append((r["id"], "ste", "duplicate id")); continue
        seen.add(r["id"])
        rep = ste_report(r["ste"])
        if rep["ste_score"] < STE_MIN:
            failures.append((r["id"], "ste", f"score {rep['ste_score']:.1f} < {STE_MIN}")); continue
        o = orig[r["id"]]
        ste_out.append(dict(id=r["id"], prompt=o["prompt"], original=o["original"], ste=r["ste"],
                            ste_score=rep["ste_score"], attempts=1))
    seen = set()
    for r in hard_rows:
        if r["id"] not in orig:
            failures.append((r["id"], "hardneg", "unknown id")); continue
        if r["id"] in seen:
            failures.append((r["id"], "hardneg", "duplicate id")); continue
        seen.add(r["id"])
        rep = ste_report(r["hard_negative"])
        if rep["ste_score"] > HARD_MAX or not rep["has_list"]:
            failures.append((r["id"], "hardneg", f"score {rep['ste_score']:.1f}, has_list={rep['has_list']}")); continue
        hard_out.append(dict(id=r["id"], prompt=orig[r["id"]]["prompt"], hard_negative=r["hard_negative"],
                             ste_score=rep["ste_score"]))
    return ste_out, hard_out, failures


def summarize(texts: list[str]) -> dict:
    reps = [ste_report(t) for t in texts]
    n = max(len(reps), 1)
    return dict(n=len(reps),
                score=sum(r["ste_score"] for r in reps) / n,
                pass80=100 * sum(r["passes_80"] for r in reps) / n,
                sent_words=sum(r["avg_sentence_words"] for r in reps) / n,
                passive=sum(r["pct_passive"] for r in reps) / n)


def report_table(ste_out, hard_out) -> str:
    rows = [("originals", summarize([r["original"] for r in ste_out])),
            ("STE rewrites", summarize([r["ste"] for r in ste_out])),
            ("hard negatives", summarize([r["hard_negative"] for r in hard_out]))]
    lines = ["| set | n | STE score mean | % passes_80 | avg sentence words | % passive |",
             "|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['n']} | {s['score']:.1f} | {s['pass80']:.1f} | {s['sent_words']:.1f} | {s['passive']:.1f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    d = Path(cfg["data"]["dir"])
    originals = read_jsonl(d / "originals.jsonl")
    ste_rows = [r for p in sorted(glob.glob(str(d / "writes" / "batch_*.jsonl"))) for r in read_jsonl(p)]
    hard_rows = [r for p in sorted(glob.glob(str(d / "writes" / "hardneg_[0-9]*.jsonl"))) for r in read_jsonl(p)]
    ste_out, hard_out, failures = collect(originals, ste_rows, hard_rows)
    write_jsonl(d / "ste_answers.jsonl", ste_out)
    write_jsonl(d / "hard_negatives.jsonl", hard_out)
    table = report_table(ste_out, hard_out)
    md = (f"# Pairs report\n\nSTE rewrites: {len(ste_out)}/{len(ste_rows)} pass the gate; "
          f"hard negatives: {len(hard_out)}/{len(hard_rows)}.\n\n{table}\n")
    if failures:
        md += "\n## Gate failures\n" + "\n".join(f"- {i} ({k}): {why}" for i, k, why in failures) + "\n"
    (d / "pairs_report.md").write_text(md)
    print(table)
    print(f"gate failures: {len(failures)}")
    for f in failures:
        print("  FAIL", *f)


if __name__ == "__main__":
    main()
