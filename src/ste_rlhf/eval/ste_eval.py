"""Aggregate STE metrics per tag (e.g. base / sft / rl) for tables and plots."""
from __future__ import annotations

from ste_rlhf.ste.metrics import ste_report


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def ste_table(groups: dict[str, list[str]]) -> dict[str, dict]:
    out = {}
    for tag, texts in groups.items():
        reps = [ste_report(t) for t in texts]
        col = lambda k: _mean([r[k] for r in reps])  # noqa: E731
        out[tag] = {
            "n": len(reps),
            "ste_score_mean": col("ste_score"),
            "pct_passes_80": 100.0 * _mean([1.0 if r["passes_80"] else 0.0 for r in reps]),
            "avg_sentence_words_mean": col("avg_sentence_words"),
            "pct_passive_mean": col("pct_passive"),
            "list_items_mean": col("list_items"),
            "pct_has_list": 100.0 * _mean([1.0 if r["has_list"] else 0.0 for r in reps]),
            "filler_phrases_mean": col("filler_phrases"),
            "complex_words_mean": col("complex_words"),
            "vague_terms_mean": col("vague_terms"),
            "flesch_mean": col("flesch_reading_ease"),
            "words_mean": col("words"),
        }
    return out
