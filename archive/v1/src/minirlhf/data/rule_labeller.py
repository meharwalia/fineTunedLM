"""Rule-based preference labels from candidate pairs."""
from __future__ import annotations

import argparse
import re

from minirlhf.common.config import load_config
from minirlhf.common.io import read_jsonl, write_jsonl
from minirlhf.common.schemas import validate_record

POLITE = ("please", "thank", "happy to", "glad", "sure", "certainly", "of course",
          "hope this helps", "great question", "you're welcome", "i'd be happy", "here are", "here is")
STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "with",
        "what", "how", "why", "do", "does", "i", "you", "it", "that", "this", "can", "me", "my", "be",
        "as", "at", "by", "from", "about", "some", "give", "tell", "write"}
MAX_WORDS = 80
MIN_MARGIN = 0.5


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def politeness(response: str) -> float:
    low = response.lower()
    return min(sum(1 for p in POLITE if p in low), 2) * 0.5


def overlap(prompt: str, response: str) -> float:
    """Fraction of the prompt's content words that appear in the response."""
    pw = {w for w in _words(prompt) if w not in STOP}
    if not pw:
        return 0.0
    return len(pw & set(_words(response))) / len(pw)


def length_penalty(response: str) -> float:
    n = len(response.split())
    return min((n - MAX_WORDS) / MAX_WORDS, 1.0) if n > MAX_WORDS else 0.0


def repetition_penalty(response: str) -> float:
    """0 for diverse text, larger for repeated or symbol-heavy (gibberish) text."""
    w = _words(response)
    pen = 0.0
    if len(w) >= 4:
        distinct = len(set(w)) / len(w)
        pen += max(0.0, 0.6 - distinct) / 0.6
        tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        pen += (1 - len(set(tri)) / len(tri)) * 0.5
    if response:
        ok = sum(c.isalpha() or c.isspace() for c in response)
        if ok / len(response) < 0.7:
            pen += 0.5
    return pen


def score(prompt: str, response: str) -> float:
    return politeness(response) + overlap(prompt, response) - length_penalty(response) \
        - 1.5 * repetition_penalty(response)


def label_pair(cand: dict, min_margin: float = MIN_MARGIN):
    """Return a pref record (chosen = higher rule score) or None for near-ties."""
    r1, r2 = cand["responses"]
    s1, s2 = score(cand["prompt"], r1), score(cand["prompt"], r2)
    if abs(s1 - s2) < min_margin:
        return None
    chosen, rejected = (r1, r2) if s1 > s2 else (r2, r1)
    return validate_record("pref", {"id": f"rule_{cand['id']}", "prompt": cand["prompt"],
                                    "chosen": chosen, "rejected": rejected, "source": "rule"})


def label_all(cands, min_margin: float = MIN_MARGIN):
    return [p for p in (label_pair(c, min_margin) for c in cands) if p]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cands = read_jsonl(cfg["data"]["candidates_path"])
    prefs = label_all(cands, args.min_margin)
    out = f"{cfg['data']['prefs_dir']}/rule.jsonl"
    write_jsonl(out, prefs)
    print(f"labelled {len(prefs)}/{len(cands)} pairs (skipped {len(cands) - len(prefs)} near-ties) -> {out}")


if __name__ == "__main__":
    main()
