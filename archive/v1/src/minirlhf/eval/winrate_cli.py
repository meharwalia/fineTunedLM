"""Blind A/B comparison in the terminal. Keys: 1 = left, 2 = right, t = tie, s = skip, q = quit.

python -m minirlhf.eval.winrate_cli --config configs/default.yaml --a tuned --b base
Resumable: judgments already stored in results/winrate/<a>_vs_<b>.json are skipped.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from minirlhf.eval._util import base_parser, cfg_from_args, gen_path, out_dir, read_generations


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(judgments: list[dict]) -> dict:
    """judgments: [{id, winner: 'a'|'b'|'tie', ...}]; win rate excludes ties."""
    wins = sum(j["winner"] == "a" for j in judgments)
    losses = sum(j["winner"] == "b" for j in judgments)
    ties = sum(j["winner"] == "tie" for j in judgments)
    decisive = wins + losses
    lo, hi = wilson_ci(wins, decisive)
    return {"wins": wins, "ties": ties, "losses": losses, "n": len(judgments),
            "win_rate": (wins / decisive) if decisive else None, "ci95": [lo, hi]}


def build_pairs(gens_a: list[dict], gens_b: list[dict]) -> list[dict]:
    b = {r["id"]: r for r in gens_b}
    return [{"id": r["id"], "prompt": r["prompt"], "a": r["response"], "b": b[r["id"]]["response"]}
            for r in gens_a if r["id"] in b]


def run_session(pairs, judged, get_key, show=print, seed=0, save=None) -> list[dict]:
    """Interactive loop with injectable input. `judged` = existing judgments (a new list is returned).
    Left/right assignment is a deterministic function of (seed, id), so resuming is consistent."""
    judgments = list(judged)
    done = {j["id"] for j in judgments}
    todo = [p for p in pairs if p["id"] not in done]
    for k, p in enumerate(todo, 1):
        a_left = random.Random(f"{seed}:{p['id']}").random() < 0.5
        left, right = (p["a"], p["b"]) if a_left else (p["b"], p["a"])
        show(f"\n[{k}/{len(todo)}] Prompt: {p['prompt']}\n\n(1) {left}\n\n(2) {right}\n\n"
             "1/2 = better, t = tie, s = skip, q = quit")
        while True:
            key = get_key().strip().lower()
            if key in ("1", "2", "t", "s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            continue
        if key == "t":
            winner = "tie"
        else:
            winner = "a" if (key == "1") == a_left else "b"
        judgments.append({"id": p["id"], "winner": winner, "a_left": a_left})
        if save:
            save(judgments)
    return judgments


def result_path(cfg, a, b) -> Path:
    return out_dir(cfg) / "winrate" / f"{a}_vs_{b}.json"


def write_result(path, a, b, judgments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"a": a, "b": b, **summarize(judgments), "judgments": judgments}, indent=2))


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cfg = cfg_from_args(args)
    pairs = build_pairs(read_generations(gen_path(cfg, args.a)), read_generations(gen_path(cfg, args.b)))
    path = result_path(cfg, args.a, args.b)
    judged = json.loads(path.read_text())["judgments"] if path.exists() else []

    def save(js):
        write_result(path, args.a, args.b, js)

    def get_key():
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C / closed stdin = quit, progress is kept
            return "q"

    js = run_session(pairs, judged, get_key, seed=args.seed, save=save)
    write_result(path, args.a, args.b, js)
    s = summarize(js)
    print(f"\n{args.a} vs {args.b}: wins={s['wins']} ties={s['ties']} losses={s['losses']} "
          f"win_rate={s['win_rate']} 95% CI={s['ci95']}\nSaved to {path}")


if __name__ == "__main__":
    main()
