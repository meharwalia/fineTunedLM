"""Shared helpers for the evaluation tools (pure Python, no model imports)."""
from __future__ import annotations

import math
import re
from pathlib import Path

PREFERRED_COLS = ["ste_score_mean", "pct_passes_80", "avg_sentence_words_mean", "pct_passive_mean",
                  "list_items_mean", "filler_phrases_mean", "complex_words_mean", "flesch_mean",
                  "words_mean", "rm_score_mean", "kl_vs_base", "distinct2", "finish_rate"]


def gen_path(out_dir, tag) -> Path:
    return Path(out_dir) / "generations" / f"{tag}.jsonl"


def metrics_dir(out_dir) -> Path:
    return Path(out_dir) / "metrics"


def distinct_n(texts, n: int = 2) -> float:
    """Distinct-n: unique n-grams / total n-grams over all texts (whitespace tokens, lowercased)."""
    total, seen = 0, set()
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i:i + n]))
            total += 1
    return len(seen) / total if total else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n. Returns (lo, hi); (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins: int, losses: int):
    """Exact sign test on non-tied comparisons. Returns (p_one_sided, p_two_sided);
    one-sided tests H1: wins > losses."""
    n = wins + losses
    if n == 0:
        return 1.0, 1.0
    p_one = sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, p_one), min(1.0, 2 * tail)


_BETA = re.compile(r"beta(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
_SEED = re.compile(r"_seed(\d+)")


def parse_beta(run_name: str):
    m = _BETA.search(run_name)
    return float(m.group(1)) if m else None


def parse_seed(run_name: str):
    m = _SEED.search(run_name)
    return int(m.group(1)) if m else None


def seed_group(run_name: str):
    """Run name with the _seed<k> part removed (None if the run has no seed suffix)."""
    return _SEED.sub("", run_name) if _SEED.search(run_name) else None


def moving_average(values, window: int):
    if window is None or window <= 1:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out
