"""STE checker CLI.

  python -m ste_rlhf.ste.cli FILE.jsonl --field ste [--show-fails]
  python -m ste_rlhf.ste.cli --text "Open the file. Then save it."
Always exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys

from ste_rlhf.ste.metrics import ste_diagnose


def _print_fails(fails: dict, indent: str = "    ") -> None:
    for rule, msgs in fails.items():
        for m in msgs:
            print(f"{indent}{rule}: {m}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score text against the 8 STE rules.")
    ap.add_argument("file", nargs="?", help="JSONL file (one object per line)")
    ap.add_argument("--field", default="ste", help="JSON field holding the text (default: ste)")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--text", help="score a single text instead of a file")
    ap.add_argument("--show-fails", action="store_true", help="print actionable detail for every failing rule")
    args = ap.parse_args(argv)

    if args.text is not None:
        rep, fails = ste_diagnose(args.text)
        print(f"ste_score={rep['ste_score']} passes_80={rep['passes_80']} "
              f"words={rep['words']} sentences={rep['sentences']} avg_sentence_words={rep['avg_sentence_words']} "
              f"flesch={rep['flesch_reading_ease']}")
        print("rules: " + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in rep["rules"].items()))
        _print_fails(fails, "  ")
        return 0
    if not args.file:
        ap.error("give a JSONL file or --text")

    reports = []
    with open(args.file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get(args.id_field, i)
            text = row.get(args.field)
            if not isinstance(text, str):
                print(f"{rid}\tMISSING field '{args.field}'", file=sys.stderr)
                text = ""
            rep, fails = ste_diagnose(text)
            reports.append(rep)
            failed = ",".join(fails) if fails else "-"
            print(f"{rid}\t{rep['ste_score']:.1f}\tfailed: {failed}")
            if args.show_fails:
                _print_fails(fails)

    n = len(reports)
    if n:
        mean = lambda k: sum(r[k] for r in reports) / n  # noqa: E731
        print("-" * 60)
        print(f"n={n}  mean_ste_score={mean('ste_score'):.1f}  "
              f"pct_passes_80={100 * sum(r['passes_80'] for r in reports) / n:.1f}  "
              f"mean_avg_sentence_words={mean('avg_sentence_words'):.1f}  "
              f"mean_pct_passive={mean('pct_passive'):.1f}")
        rates = {k: 100 * sum(r["rules"][k] for r in reports) / n for k in reports[0]["rules"]}
        print("rule_pass_rates: " + " ".join(f"{k}={v:.0f}%" for k, v in rates.items()))
    else:
        print("n=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
