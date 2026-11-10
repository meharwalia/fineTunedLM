"""Select explanatory / how-to / technical single-turn prompts from HelpSteer2 and pick originals.

Run: ``python -m ste_rlhf.data.select_prompts [--config configs/v2.yaml] [--set data.max_prompt_tokens=300]``

Outputs (under ``cfg.data.dir``):
  prompts/candidates.jsonl   kept prompts + chosen original (intermediate; input to splits.py)
  prompts/dropped.jsonl      every dropped prompt with its drop reason
  selection_report.md        counts, length stats (the rm_pool sample is added by splits.py)

Classifier (keyword/regex lists below). Order of checks for each unique prompt:
  multi_turn -> prompt_too_long -> code_fence -> long_passage (> 150 words of pasted text)
  -> mostly_code -> creative / roleplay / translation / rewrite_text / writing_task / code_writing
  (EXCLUDE lists) -> not_explanatory (no INCLUDE pattern matched) -> original_too_long
  -> original_too_short (< 40 words).
The first failing check is the recorded drop reason.
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

from ste_rlhf.common import add_config_args, load_config, load_tokenizer, write_jsonl

MULTI_TURN_MARK = "<extra_id_1>"
MIN_ORIGINAL_WORDS = 40
MAX_PASSAGE_WORDS = 150  # more words than this in a prompt = embedded reference passage

EXCLUDE = {
    "creative": [
        r"\bpoems?\b", r"\bpoetry\b", r"\bsonnets?\b", r"\bhaikus?\b", r"\blimericks?\b",
        r"\bstory\b", r"\bstories\b", r"\bfiction\b", r"\bnovel\b", r"\bfan ?fic",
        r"\bsongs?\b", r"\blyrics\b", r"\brap\b", r"\bjokes?\b", r"\bpuns?\b", r"\briddles?\b",
        r"\bscreenplay\b", r"\bscript for\b", r"\bdialogue\b", r"\bepisode\b", r"\bmonologue\b",
        r"\bfunny\b", r"\bchapter\b", r"\bcharacters?\b", r"\bplot\b",
        r"\bfantasy\b", r"\brpg\b", r"\bdungeon", r"\bmake up\b", r"\bwhat animal\b",
        r"\b(do|did) you think\b", r"\bwhy you think\b", r"\byour (favou?rite|opinion)\b",
    ],
    "roleplay": [
        r"\bpretend\b", r"\bact as\b", r"\bacting as\b", r"\brole[- ]?play", r"\bpersona\b",
        r"\byou are (a|an|the)\b", r"\bimagine (you|that you)\b", r"\bplay the role\b",
        r"\bas if you (are|were)\b", r"\bI want you to (act|be|play)\b", r"\bpretending\b",
    ],
    "translation": [
        r"\btranslat", r"\bin (french|spanish|german|chinese|japanese|italian|portuguese|russian|"
        r"korean|arabic|hindi|latin)\b", r"\bhow do you say\b", r"\bhow to say\b",
    ],
    "rewrite_text": [
        r"\brewrite\b", r"\brephrase\b", r"\bparaphras", r"\bsummari[sz]e\b", r"\bsummary of the\b",
        r"\bproofread\b", r"\bcorrect (the|my) grammar\b", r"\bfix (the|my) grammar\b",
        r"\bmake (this|it|the following) (sound|shorter|longer|more)\b", r"\bedit (this|the following)\b",
        r"\btl;?dr\b", r"\bshorten (this|the)\b", r"\bexpand (on )?(this|the following)\b",
    ],
    "writing_task": [  # asked to produce a document, not to explain something
        r"\bwrite (me )?(a |an |the |some )?(short |brief |professional |formal |long )?"
        r"(email|e-mail|letter|cover letter|tweet|blog|article|post|caption|slogan|ad|advert|"
        r"product description|bio|resume|essay|review|message|text message|paragraph|description|"
        r"speech|toast|eulogy)",
        r"\bpersonal statement\b", r"\bcraft (a|an|your|the)\b", r"\bnewsletter\b", r"\blinkedin\b",
        r"\bwrite\b[^.?\n]{0,40}\b\d+[- ]?words?\b", r"\bmarketing campaign\b",
        r"\bdraft (a|an|the)\b", r"\bcompose\b", r"\bgenerate (a|an) (name|names|slogan|title)\b",
    ],
    "code_writing": [
        r"\bwrite (me )?(a |an |the |some )?(\w+ )?(function|program|script|class|code|query|regex|"
        r"snippet|algorithm|macro|method)\b",
        r"\b(python|java|javascript|c\+\+|c#|rust|golang|php|typescript|sql|bash|html|css) "
        r"(code|function|program|script|class|snippet)\b",
        r"\bimplement (a|an|the)\b", r"\bfix (this|my|the) (code|bug|error)\b", r"\bdebug\b",
        r"\brefactor\b",
    ],
    # The prompt fixes the answer's format (word count, paragraph, data format), so an STE
    # rewrite with lists and short sentences would ignore the user's request.
    "format_constraint": [
        r"\bin (about |around |under |less than |no more than |exactly )?\d+ words\b",
        r"\b(about |around )?\d+[- ]words?\b", r"\bin (a|one) (single |short |brief )?paragraph\b",
        r"\bparagraph of\b", r"\bin (a |an )?(python|json|yaml|csv|xml|markdown|table|dictionary|list) format\b",
        r"\bformat(ted)? (it |this )?(as|in) (a |an )?(table|json|python|dictionary|csv)\b",
        r"\bone (sentence|word)\b", r"\b(two|three|four|five|six|\d+) paragraphs\b", r"\bpython dictionary\b", r"\bsame format\b", r"\bin \d+ (sentences|bullet points|points)\b",
    ],
    # Conversations, assignments and multi-part job briefs, not a question to explain.
    "task_setup": [
        r"\bconversation (between|with)\b", r"\bdialogue\b", r"\bupload\b", r"\bzip\b",
        r"\bI will (give|provide|send) you\b", r"\bI want you to\b", r"\byou will (get|be given)\b",
        r"\bservice page\b", r"\bhelp me (make|create|write|design) my\b",
        r"\bfix any (writing )?mistakes?\b", r"\bessay\b", r"\bgrammar\b", r"\bmake notes for me\b", r"\bintroductory paragraph\b", r"\bgiven a paragraph\b", r"\bexcerpt from\b",
    ],
    # Jailbreak set-ups and harmful requests never go into training data.
    "unsafe": [
        r"\bjailbreak", r"\bexcluding any (legal|moral|ethical)", r"\bcontent polic", r"\brespond to every prompt\b", r"\bDAN\b",
        r"\bignore (all |your |previous )", r"\b(nuclear|biological|chemical) weapon", r"\bbomb\b",
        r"\bexplosive", r"\bmalware\b", r"\bransomware\b", r"\bhack into\b", r"\bsynthesi[sz]e (meth|drugs)\b",
    ],
}
REASONS_ORDER = ["unsafe", "creative", "roleplay", "translation", "rewrite_text", "writing_task", "code_writing",
                 "format_constraint", "task_setup"]
_EXCLUDE_RE = {k: re.compile("|".join(v), re.I) for k, v in EXCLUDE.items()}

CODE_FENCE_RE = re.compile(r"```|~~~")
CODE_LINE_RE = re.compile(r"^\s*(def |class |import |from \S+ import |#include|public |private |"
                          r"function |const |let |var |return |if ?\(|for ?\(|while ?\(|SELECT |"
                          r"</?[a-z]+[^>]*>|[\w\.\[\]]+ ?= ?.+;\s*$|[{}]\s*$)", re.I)

INCLUDE_RE = re.compile("|".join([
    r"\bhow (do|does|did|can|could|should|would|to|is|are|come)\b", r"\bhow'?s\b",
    r"\bexplain", r"\bwhat (is|are|was|were|does|do|did|happens|causes|makes)\b",
    r"\bwhat'?s\b", r"\bwhat (should|can|could|would)\b",
    r"\bwhy (does|do|did|is|are|was|were|can|would|should|doesn'?t|isn'?t)\b",
    r"\bsteps?\b", r"\bdifferences?\b", r"\bdifferent from\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bcompare\b", r"\bcomparison\b", r"\bguide\b", r"\btutorial\b", r"\binstructions?\b",
    r"\bset ?up\b", r"\binstall", r"\bconfigur", r"\bdeploy", r"\btroubleshoot", r"\bprocedure\b",
    r"\bprocess\b", r"\bdescribe\b", r"\boverview\b", r"\bdefin(e|ition)\b", r"\bmeaning of\b",
    r"\bwhich (is|are|one|should)\b", r"\bbest (way|practice|practices|method|approach)\b",
    r"\bbenefits?\b", r"\badvantages?\b", r"\bdisadvantages?\b", r"\bpros and cons\b",
    r"\btips\b", r"\bways to\b", r"\bcan you (tell|teach|show|help|walk|give|list|recommend)\b",
    r"\bhelp me (understand|learn|figure|set|install|fix)\b",
    r"\bpurpose of\b", r"\bexamples? of\b",
    r"\bcauses?\b", r"\bfix(es)?\b", r"\bsolve\b", r"\bresolve\b",
    r"\bimprove\b", r"\bprevent\b", r"\bavoid\b", r"\bchoose\b", r"\blearn\b", r"\bunderstand\b",
]), re.I)


def normalize_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", p.lower())).strip()


def code_fraction(prompt: str) -> float:
    lines = [l for l in prompt.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(bool(CODE_LINE_RE.match(l)) for l in lines) / len(lines)


def classify_prompt(prompt: str) -> str | None:
    """Return a drop reason, or None if the prompt is an explanatory/how-to/technical keeper."""
    if CODE_FENCE_RE.search(prompt):
        return "code_fence"
    if len(prompt.split()) > MAX_PASSAGE_WORDS or re.search(r"\breference text\b", prompt, re.I):
        return "long_passage"
    if code_fraction(prompt) >= 0.3:
        return "mostly_code"
    for reason in REASONS_ORDER:
        if _EXCLUDE_RE[reason].search(prompt):
            return reason
    if not INCLUDE_RE.search(prompt):
        return "not_explanatory"
    return None


def pick_original(rows: list[dict], tok_len) -> dict:
    """rows: HelpSteer2 rows (each with 'row') for ONE prompt. Higher helpfulness, then lower
    verbosity, then fewer tokens (then lower row index, for determinism)."""
    return min(rows, key=lambda r: (-r["helpfulness"], r["verbosity"], tok_len(r["response"]), r["row"]))


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 3])


def has_procedure(text: str) -> bool:
    return bool(re.search(r"^\s*(\d+[.)]|[-*•])\s+", text, re.M)) or bool(
        re.search(r"\b(step \d|first,|then,? |finally,|next,)", text, re.I))


def select(rows: list[dict], cfg: dict, tok_len) -> tuple[list[dict], list[dict]]:
    """rows: dicts with row, prompt, response, helpfulness, verbosity. Returns (kept, dropped).

    Rows with the same normalised prompt (HelpSteer2 has 2 responses per prompt) form one
    prompt group: this is the dedup step.
    """
    dcfg = cfg["data"]
    groups: dict[str, list[dict]] = {}
    dropped: list[dict] = []
    for r in rows:
        if MULTI_TURN_MARK in r["prompt"]:
            dropped.append({"id": f"hs2_{r['row']}", "hs2_row": r["row"], "prompt": r["prompt"],
                            "reason": "multi_turn"})
            continue
        groups.setdefault(normalize_prompt(r["prompt"]), []).append(r)

    kept: list[dict] = []
    for grp in groups.values():
        first = grp[0]
        pid = f"hs2_{first['row']}"

        def drop(reason):
            dropped.append({"id": pid, "hs2_row": first["row"], "prompt": first["prompt"], "reason": reason})

        ptok = tok_len(first["prompt"])
        if ptok > dcfg["max_prompt_tokens"]:
            drop("prompt_too_long")
            continue
        reason = classify_prompt(first["prompt"])
        if reason:
            drop(reason)
            continue
        best = pick_original(grp, tok_len)
        otok = tok_len(best["response"])
        if otok > dcfg["max_original_tokens"]:
            drop("original_too_long")
            continue
        if len(best["response"].split()) < MIN_ORIGINAL_WORDS:
            drop("original_too_short")
            continue
        kept.append({
            "id": f"hs2_{best['row']}", "prompt": first["prompt"], "hs2_row": best["row"],
            "original": best["response"], "helpfulness": best["helpfulness"],
            "verbosity": best["verbosity"], "original_tokens": otok, "prompt_tokens": ptok,
            "original_sentences": count_sentences(best["response"]),
            "has_procedure": has_procedure(best["response"]),
        })
    return kept, dropped


def _stats(xs: list[int]) -> str:
    if not xs:
        return "n=0"
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return (f"n={len(xs)} min={xs[0]} median={q(.5)} mean={statistics.mean(xs):.0f} "
            f"p90={q(.9)} max={xs[-1]}")


SAMPLE_MARKER = "<!-- RM_POOL_SAMPLE -->"


def build_report(n_rows: int, kept: list[dict], dropped: list[dict]) -> str:
    c = Counter(d["reason"] for d in dropped)
    lines = ["# Prompt selection report", "",
             f"HelpSteer2 train rows: {n_rows}", f"Kept prompts: {len(kept)}",
             f"Dropped: {len(dropped)} (multi_turn counts rows; every other reason counts unique prompts)", "",
             "## Dropped by reason", "", "| reason | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in c.most_common()]
    lines += ["", "## Length stats (SmolLM2 tokens, kept prompts)", "",
              f"- prompt tokens: {_stats([k['prompt_tokens'] for k in kept])}",
              f"- original tokens: {_stats([k['original_tokens'] for k in kept])}",
              f"- original words: {_stats([len(k['original'].split()) for k in kept])}",
              f"- originals with >=3 sentences or a procedure: "
              f"{sum(k['original_sentences'] >= 3 or k['has_procedure'] for k in kept)}",
              "", SAMPLE_MARKER]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description="Select HelpSteer2 prompts and originals"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    from datasets import load_dataset
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    rows = [dict(row=i, **{k: ex[k] for k in ("prompt", "response", "helpfulness", "verbosity")})
            for i, ex in enumerate(ds)]
    tok = load_tokenizer(cfg["policy"]["name"])
    tok_len = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    kept, dropped = select(rows, cfg, tok_len)
    out = Path(cfg["data"]["dir"])
    write_jsonl(out / "prompts" / "candidates.jsonl", kept)
    write_jsonl(out / "prompts" / "dropped.jsonl", dropped)
    report = build_report(len(rows), kept, dropped)
    (out / "selection_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
