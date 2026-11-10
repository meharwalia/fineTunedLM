"""Blind A/B comparison in the terminal.

python -m ste_rlhf.eval.winrate_cli --a tuned --b base [--seed 0] [--limit N] [--set K=V ...]

Keys: 1 = left is better, 2 = right is better, t = tie, s = skip, q = quit. EOF and Ctrl-C quit and
keep progress; rerunning resumes from results/winrate/<a>_vs_<b>.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, read_jsonl

from ._util import gen_path, sign_test, wilson_ci

CRITERION = "Which answer is easier to follow and act on?"


def side_for(item_id, seed) -> str:
    """'a_left' or 'b_left', deterministic per (seed, id)."""
    return "a_left" if random.Random(f"{seed}:{item_id}").random() < 0.5 else "b_left"


def summarize(choices: dict) -> dict:
    wins = sum(1 for v in choices.values() if v == "a")
    losses = sum(1 for v in choices.values() if v == "b")
    ties = sum(1 for v in choices.values() if v == "tie")
    n = wins + losses
    lo, hi = wilson_ci(wins, n)
    p1, p2 = sign_test(wins, losses)
    return {"wins": wins, "ties": ties, "losses": losses, "n_compared": len(choices),
            "win_rate_excl_ties": (wins / n) if n else None, "wilson_95": [lo, hi],
            "p_one_sided": p1, "p_two_sided": p2}


def load_state(path, a, b, seed) -> dict:
    path = Path(path)
    if path.exists():
        st = json.loads(path.read_text())
        st.setdefault("choices", {})
        return st
    return {"a": a, "b": b, "seed": seed, "choices": {}}


def save_state(path, st):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st.update(summarize(st["choices"]))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(path)


def format_summary(st) -> str:
    s = summarize(st["choices"])
    if s["win_rate_excl_ties"] is None:
        return f"{st['a']} vs {st['b']}: no decisive votes yet ({s['ties']} ties)."
    return (f"{st['a']} vs {st['b']}: wins {s['wins']}, ties {s['ties']}, losses {s['losses']}\n"
            f"win rate (excl. ties) {100 * s['win_rate_excl_ties']:.1f}% "
            f"[Wilson 95% CI {100 * s['wilson_95'][0]:.1f}-{100 * s['wilson_95'][1]:.1f}]\n"
            f"sign test: one-sided p = {s['p_one_sided']:.4f}, two-sided p = {s['p_two_sided']:.4f}")


def run_session(rows_a, rows_b, state_path, a="a", b="b", seed=0, limit=None, input_fn=input, out_fn=print) -> dict:
    """Interactive loop. input_fn may raise EOFError / KeyboardInterrupt, which quits keeping progress."""
    A = {r["id"]: r for r in rows_a}
    B = {r["id"]: r for r in rows_b}
    ids = [r["id"] for r in rows_a if r["id"] in B]
    if limit:
        ids = ids[:int(limit)]
    st = load_state(state_path, a, b, seed)
    st["seed"] = st.get("seed", seed)
    todo = [i for i in ids if i not in st["choices"]]
    out_fn(f"{len(ids)} prompts, {len(ids) - len(todo)} already judged, {len(todo)} to go. {CRITERION}")
    save_state(state_path, st)
    try:
        for k, i in enumerate(todo, 1):
            left_is_a = side_for(i, st["seed"]) == "a_left"
            left, right = (A[i], B[i]) if left_is_a else (B[i], A[i])
            out_fn(f"\n=== [{len(ids) - len(todo) + k}/{len(ids)}] {A[i]['prompt'].strip()}\n")
            out_fn("--- 1 (left) ---\n" + left["response"].strip() + "\n")
            out_fn("--- 2 (right) ---\n" + right["response"].strip() + "\n")
            while True:
                key = input_fn("[1] left  [2] right  [t] tie  [s] skip  [q] quit > ").strip().lower()
                if key in ("1", "2", "t", "s", "q"):
                    break
            if key == "q":
                break
            if key == "s":
                continue
            if key == "t":
                st["choices"][i] = "tie"
            else:
                st["choices"][i] = "a" if (key == "1") == left_is_a else "b"
            save_state(state_path, st)
    except (EOFError, KeyboardInterrupt):
        out_fn("\nstopped; progress saved.")
    save_state(state_path, st)
    out_fn(format_summary(st))
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="tag of system A (e.g. tuned)")
    ap.add_argument("--b", required=True, help="tag of system B (e.g. base)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    add_config_args(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out_dir = cfg["eval"].get("out_dir", "results")
    state = Path(out_dir) / "winrate" / f"{a.a}_vs_{a.b}.json"
    run_session(read_jsonl(gen_path(out_dir, a.a)), read_jsonl(gen_path(out_dir, a.b)), state,
                a=a.a, b=a.b, seed=a.seed, limit=a.limit)


if __name__ == "__main__":
    main()
