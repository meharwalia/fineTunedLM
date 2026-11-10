"""Turn a folder of per-item Markdown files into a batch JSONL for the writer agents.

Writers save each answer as data/writes/<batch>/<id>.md (plain text, no escaping needed), then run:

    python -m ste_rlhf.data.assemble_batch --batch batch_03 --field ste
    python -m ste_rlhf.data.assemble_batch --batch hardneg_01 --field hard_negative

which writes data/writes/<batch>.jsonl with fields {id, <field>} in the order of the todo file,
and prints which todo ids are still missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TODO = {"batch": "todo", "hardneg": "hardneg_todo"}


def todo_path(writes: Path, batch: str) -> Path:
    kind, num = batch.rsplit("_", 1)
    return writes / f"{TODO[kind]}_{num}.jsonl"


def assemble(writes: Path, batch: str, field: str) -> tuple[int, list[str]]:
    ids = [json.loads(l)["id"] for l in todo_path(writes, batch).read_text().splitlines() if l.strip()]
    folder = writes / batch
    rows, missing = [], []
    for i in ids:
        f = folder / f"{i}.md"
        if f.is_file() and f.read_text(encoding="utf-8").strip():
            rows.append({"id": i, field: f.read_text(encoding="utf-8").strip()})
        else:
            missing.append(i)
    with open(writes / f"{batch}.jsonl", "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="batch_00..batch_09 or hardneg_00..hardneg_01")
    ap.add_argument("--field", required=True, choices=["ste", "hard_negative"])
    ap.add_argument("--writes", default="data/writes")
    a = ap.parse_args(argv)
    n, missing = assemble(Path(a.writes), a.batch, a.field)
    print(f"wrote {n} items to {a.writes}/{a.batch}.jsonl; missing: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
