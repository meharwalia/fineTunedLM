"""Terminal A/B labeller for human preferences.

Keys: 1 = left is better, 2 = right is better, s = skip, q = quit. Resumable.
"""
from __future__ import annotations

import argparse
import random
import sys

from minirlhf.common.config import load_config
from minirlhf.common.io import append_jsonl, read_jsonl
from minirlhf.common.schemas import validate_record


def order_pair(responses, rng: random.Random):
    """Random left/right order. Returns (left, right, swapped); swapped means left is responses[1]."""
    a, b = responses
    return (b, a, True) if rng.random() < 0.5 else (a, b, False)


def interpret_key(key):
    """Map a key press to 'left' | 'right' | 'skip' | 'quit' | None (unknown)."""
    return {"1": "left", "2": "right", "s": "skip", "q": "quit"}.get((key or "").strip().lower()[:1])


def make_pref(cand: dict, left: str, right: str, choice: str) -> dict:
    chosen, rejected = (left, right) if choice == "left" else (right, left)
    return validate_record("pref", {"id": f"human_{cand['id']}", "prompt": cand["prompt"],
                                    "chosen": chosen, "rejected": rejected, "source": "human"})


def pending(cands, out_path, skipped_path=None):
    """Candidates not yet labelled (or skipped, if skipped_path is given)."""
    done = set()
    if out_path:
        try:
            done |= {r["id"] for r in read_jsonl(out_path)}
        except FileNotFoundError:
            pass
    if skipped_path:
        try:
            done |= {"human_" + r["id"] for r in read_jsonl(skipped_path)}
        except FileNotFoundError:
            pass
    return [c for c in cands if f"human_{c['id']}" not in done]


def run_session(cands, out_path, read_key, write=print, rng=None, max_n=None, skipped_path=None) -> int:
    """Label ``cands`` interactively. ``read_key()`` returns one key; ``write(str)`` prints a line.

    Returns the number of labels written. Skips are recorded in ``skipped_path`` if given."""
    rng = rng or random.Random()
    todo = pending(cands, out_path, skipped_path)
    total = len(todo) if max_n is None else min(max_n, len(todo))
    n = 0
    for i, cand in enumerate(todo[:total]):
        left, right, _ = order_pair(cand["responses"], rng)
        write(f"\n[{i + 1}/{total}] Prompt: {cand['prompt']}\n\n  [1] {left}\n\n  [2] {right}\n")
        write("Which is better? 1 / 2 / s=skip / q=quit")
        while True:
            action = interpret_key(read_key())
            if action is not None:
                break
            write("Please press 1, 2, s or q.")
        if action == "quit":
            break
        if action == "skip":
            if skipped_path:
                append_jsonl(skipped_path, {"id": cand["id"]})
            continue
        append_jsonl(out_path, make_pref(cand, left, right, action))
        n += 1
    write(f"\nSaved {n} new labels to {out_path}.")
    return n


def _tty_key():
    """Read one key press from the terminal; falls back to line input without a tty. EOF -> 'q'."""
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:  # no tty (piped stdin), termios.error, or non-POSIX platform
        line = sys.stdin.readline()
        return line.strip()[:1] if line else "q"
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch or "q"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--limit", type=int, default=None, help="label at most N pairs this session")
    ap.add_argument("--seed", type=int, default=None, help="seed for left/right order (default: random)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cands = read_jsonl(cfg["data"]["candidates_path"])
    out = f"{cfg['data']['prefs_dir']}/human.jsonl"
    skipped = f"{cfg['data']['prefs_dir']}/human_skipped.jsonl"
    try:
        run_session(cands, out, _tty_key, print, random.Random(args.seed), args.limit, skipped)
    except KeyboardInterrupt:
        print("\nInterrupted; progress is saved.")


if __name__ == "__main__":
    main()
