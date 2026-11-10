"""STE metrics: ste_report(text) -> dict (contract C1).

Design notes (decisions the spec left open)
-------------------------------------------
* Markdown: fenced code and tables are dropped; headings are not sentences (but their words count);
  inline code becomes the single token CODE, URLs become URL, links keep their text, emphasis markers go.
* List items (-, *, +, 1., 1)) are sentences: each item is split into its own sentences.
* "Procedure" sentence = inside a numbered list item, OR starts with an imperative verb (after dropping
  lead adverbs such as "then", "first,", "please", or a leading "If/When/Once/To ..., " clause).
* R2 passive = be-form (+ optional adverbs/negation/be/been/being) + past participle (-ed/-en suffix or
  irregular list), minus adjective-like words (interested, located, ...) and "used to <verb>-ing".
  "'s" is never treated as "is" (too ambiguous with "has"/possessive).
* R4 needs_list = already has a list OR >=3 sequential steps in prose (First/Then/Finally markers,
  "Step N" x2, or >=3 imperative prose sentences in a row). Plain enumerations ("a, b and c") are NOT
  detected. R5 "lines" are physical lines of a prose paragraph; the sentence limit does most work.
* R6 vague terms: counted occurrences of the VAGUE list in non-code text; "numerous" is only in R7.
"""
from __future__ import annotations

import re

from ste_rlhf.ste import wordlists as W

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE = re.compile(r"^\s*\|")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?")
_INLINE_CODE = re.compile(r"`+[^`\n]*`+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITAL = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_WORD = re.compile(r"\w+(?:['’.\-/]\w+)*")
_DEC = re.compile(r"(?<=\d)\.(?=\d)")
_ABBR = re.compile(
    r"\b(" + "|".join(re.escape(a).replace(r"\.", r"\.") for a in W.ABBREVIATIONS if a != "etc") + r")\.", re.I)
_ETC = re.compile(r"\betc\.(?!\s+[A-Z])", re.I)
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_DOT = "\x00"

_BE = r"(?:am|is|are|was|were|be|been|being|isn't|aren't|wasn't|weren't|i'm|you're|we're|they're)"
_ADV = r"(?:\s+(?:not|then|just|always|never|already|still|only|be|been|being|later|\w+ly))*"
_PASSIVE = re.compile(r"\b" + _BE + r"\b" + _ADV + r"\s+([a-z]+)\b", re.I)
_USED_TO_ING = re.compile(r"\b(?:am|is|are|was|were|be|been|being)\s+used\s+to\s+\w+ing\b", re.I)

_SEQ_WORDS = re.compile(
    r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|after that|afterwards|finally|lastly)\b[,:]?", re.I)
_STEP_N = re.compile(r"\bstep\s+\d+\b", re.I)
_JOIN = re.compile(r"\s*;\s*|,?\s+and\s+then\s+|,?\s+and\s+|,?\s+then\s+", re.I)
_SUBORD = ("if", "when", "once", "after", "before", "to", "while", "unless", "until")


def _alt(items):
    return "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))


_VAGUE_RE = re.compile(r"(?<![\w])(?:" + _alt(W.VAGUE) + r")(?![\w])", re.I)
_FILLER_RE = re.compile(
    r"(?<![\w])(?:" + _alt(W.FILLERS) + r")(?![\w])|" + "|".join(W.FILLER_REGEXES), re.I)
_COMPLEX_MAP: dict[str, tuple[str, str]] = {}
for _c, _s, _k, _x in W.COMPLEX:
    for _f in W.entry_forms(_c, _k, _x):
        _COMPLEX_MAP.setdefault(_f, (_c, _s))
_COMPLEX_RE = re.compile(r"(?<![\w])(?:" + _alt(_COMPLEX_MAP) + r")(?![\w])", re.I)

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def _clean_inline(s: str) -> str:
    s = _IMAGE.sub("", s)
    s = _INLINE_CODE.sub("CODE", s)
    s = _LINK.sub(r"\1", s)
    s = _URL.sub("URL", s)
    s = _BOLD.sub(r"\2", s)
    s = _ITAL.sub(r"\1", s)
    return s.strip()


def _split_sentences(s: str) -> list[str]:
    s = _DEC.sub(_DOT, s)
    s = _ETC.sub("etc" + _DOT, s)
    s = _ABBR.sub(lambda m: m.group(1) + _DOT, s)
    out = []
    for part in _SPLIT.split(s):
        part = part.replace(_DOT, ".").strip()
        if part and _WORD.search(part):
            out.append(part)
    return out


def _words(s: str) -> list[str]:
    return _WORD.findall(s)


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1 if word.isdigit() else 0
    if len(w) <= 3:
        return 1
    n = len(_VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    elif w.endswith("es") or w.endswith("ed"):
        if n > 1 and not w.endswith(("ted", "ded", "ses", "zes", "ces", "ges")):
            n -= 1
    return max(n, 1)


def _imperative_core(sentence: str) -> list[str]:
    """Lowercased tokens of the sentence after dropping lead adverbs / a subordinate lead clause."""
    toks = re.findall(r"[A-Za-z']+|,", sentence.lower())
    # drop a leading "If/When/To ..., " clause
    if toks and toks[0] in _SUBORD and "," in toks[:12]:
        toks = toks[toks.index(",") + 1:]
    while toks and (toks[0] == "," or toks[0] in W.LEAD_ADVERBS):
        toks = toks[1:]
    return toks


def _starts_imperative(sentence: str) -> bool:
    toks = _imperative_core(sentence)
    if not toks:
        return False
    if toks[0] in ("don't", "do") and (toks[0] == "don't" or (len(toks) > 1 and toks[1] == "not")):
        return True
    return toks[0] in W.IMPERATIVES


def _count_imperative_segments(sentence: str) -> int:
    segs = _JOIN.split(sentence)
    n = 0
    for i, seg in enumerate(segs):
        seg = seg.strip()
        if not seg:
            continue
        if i == 0:
            n += _starts_imperative(seg)
        elif len(_words(seg)) >= 2 and _starts_imperative(seg):
            n += 1
    return n


def _passive_hit(sentence: str) -> str | None:
    if _USED_TO_ING.search(sentence):
        sentence = _USED_TO_ING.sub("", sentence)
    for m in _PASSIVE.finditer(sentence):
        p = m.group(1).lower()
        if p in W.PASSIVE_ADJ_EXCLUDE:
            continue
        if p in W.IRREGULAR_PARTICIPLES or (len(p) > 4 and p.endswith("ed")) or (len(p) > 4 and p.endswith("en")):
            return m.group(0)
    return None


def _snip(s: str, n: int = 60) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def _parse(text: str):
    """-> (blocks, lang_text). blocks: list of dicts kind in prose/item; heading words tracked separately."""
    blocks: list[dict] = []
    lang_lines: list[str] = []
    heading_words = 0
    in_code = False
    para: dict | None = None
    item: dict | None = None
    for raw in text.splitlines():
        if _FENCE.match(raw):
            in_code = not in_code
            para = item = None
            continue
        if in_code:
            continue
        line = _QUOTE.sub("", raw)
        if not line.strip():
            para = item = None
            continue
        if _HR.match(line) or _TABLE.match(line):
            para = item = None
            continue
        if _HEADING.match(line):
            heading_words += len(_words(_clean_inline(line.lstrip(" #"))))
            para = item = None
            continue
        m = _LIST.match(line)
        if m:
            para = None
            item = {"kind": "item", "lines": [m.group(3)], "numbered": m.group(2)[0].isdigit()}
            blocks.append(item)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if item is not None and indent >= 2:
            item["lines"].append(line)
            continue
        item = None
        if para is None:
            para = {"kind": "prose", "lines": []}
            blocks.append(para)
        para["lines"].append(line)
    for b in blocks:
        b["text"] = _clean_inline(" ".join(" ".join(l.split()) for l in b["lines"]))
        lang_lines.append(b["text"])
    return blocks, "\n".join(lang_lines), heading_words


def _analyze(text: str) -> tuple[dict, dict]:
    blocks, lang, heading_words = _parse(text or "")
    sents: list[dict] = []
    long_paragraphs: list[str] = []
    list_items = 0
    seq_prose_imp_run = 0
    max_imp_run = 0
    for b in blocks:
        ss = _split_sentences(b["text"])
        if b["kind"] == "item":
            list_items += 1
            for s in ss:
                imp = _starts_imperative(s)
                sents.append({"t": s, "n": len(_words(s)), "proc": b["numbered"] or imp, "imp": imp})
            seq_prose_imp_run = 0
        else:
            if len(ss) > 3 or len(b["lines"]) > 6:
                long_paragraphs.append(b["text"])
            for s in ss:
                imp = _starts_imperative(s)
                sents.append({"t": s, "n": len(_words(s)), "proc": imp, "imp": imp})
                seq_prose_imp_run = seq_prose_imp_run + 1 if imp else 0
                max_imp_run = max(max_imp_run, seq_prose_imp_run)

    n_words = sum(s["n"] for s in sents) + heading_words
    n_sent = len(sents)
    sent_words = sum(s["n"] for s in sents)

    # R1
    long_s = [s for s in sents if s["n"] > (20 if s["proc"] else 25)]
    pct_long = 100.0 * len(long_s) / n_sent if n_sent else 0.0
    # R2
    passive = [(s, h) for s in sents if (h := _passive_hit(s["t"]))]
    pct_passive = 100.0 * len(passive) / n_sent if n_sent else 0.0
    # R3
    instr = [s for s in sents if s["proc"]]
    multi = [s for s in instr if _count_imperative_segments(s["t"]) >= 2]
    # R4
    has_list = list_items > 0
    prose = " ".join(b["text"] for b in blocks if b["kind"] == "prose")
    seq_marks = {m.group(1).lower().split()[0].rstrip("ly") for m in _SEQ_WORDS.finditer(prose)}
    prose_seq = (len(_STEP_N.findall(prose)) >= 2 or max_imp_run >= 3
                 or (len(seq_marks) >= 3 and ("first" in seq_marks or "second" in seq_marks)))
    needs_list = has_list or prose_seq
    # R6-R8
    vague = [m.group(0) for m in _VAGUE_RE.finditer(lang)]
    cplx = [m.group(0) for m in _COMPLEX_RE.finditer(lang)]
    fill = [m.group(0).strip(" .!?") for m in _FILLER_RE.finditer(lang)]
    # Flesch
    syl = sum(_syllables(w) for s in sents for w in _words(s["t"]))
    if sent_words and n_sent:
        flesch = 206.835 - 1.015 * (sent_words / n_sent) - 84.6 * (syl / sent_words)
    else:
        flesch = 0.0

    rules = {
        "R1": pct_long <= 10.0,
        "R2": pct_passive <= 10.0,
        "R3": (len(multi) / len(instr) <= 0.10) if instr else True,
        "R4": (not needs_list) or has_list,
        "R5": not long_paragraphs,
        "R6": len(vague) <= 1,
        "R7": len(cplx) <= 1,
        "R8": len(fill) == 0,
    }
    if n_words == 0:
        rules = {k: False for k in rules}
    score = round(100.0 * sum(rules.values()) / 8, 1)
    report = {
        "words": n_words,
        "sentences": n_sent,
        "avg_sentence_words": round(sent_words / n_sent, 2) if n_sent else 0.0,
        "pct_long_sentences": round(pct_long, 1),
        "pct_passive": round(pct_passive, 1),
        "multi_instruction_sentences": len(multi),
        "list_items": list_items,
        "has_list": has_list,
        "needs_list": needs_list,
        "long_paragraphs": len(long_paragraphs),
        "vague_terms": len(vague),
        "complex_words": len(cplx),
        "filler_phrases": len(fill),
        "flesch_reading_ease": round(flesch, 1),
        "rules": rules,
        "ste_score": score,
        "passes_80": score >= 80.0,
    }
    details = {
        "R1": [f"{len(long_s)} of {n_sent} sentences over limit (20 procedure / 25 other, max 10%): "
               + "; ".join(f"'{_snip(s['t'])}' ({s['n']} words)" for s in long_s[:3]) for _ in [0]] if long_s else [],
        "R2": [f"{len(passive)} of {n_sent} sentences passive (max 10%): "
               + "; ".join(f"'{_snip(s['t'])}' [{h}]" for s, h in passive[:3]) for _ in [0]] if passive else [],
        "R3": [f"{len(multi)} of {len(instr)} instruction sentences have 2+ instructions (max 10%): "
               + "; ".join(f"'{_snip(s['t'])}'" for s in multi[:3]) for _ in [0]] if multi else [],
        "R4": ["steps/items found in prose but no list: use '1.' for steps or '-' for items"]
        if (needs_list and not has_list) else [],
        "R5": [f"{len(long_paragraphs)} paragraph(s) over 3 sentences / 6 lines: '{_snip(long_paragraphs[0])}'"]
        if long_paragraphs else [],
        "R6": [f"{len(vague)} vague terms (max 1): {sorted(set(v.lower() for v in vague))}"] if len(vague) > 1 else [],
        "R7": [f"{len(cplx)} complex words (max 1): "
               + ", ".join(f"'{c.lower()}' -> '{_COMPLEX_MAP[c.lower()][1]}'" for c in cplx[:5])]
        if len(cplx) > 1 else [],
        "R8": [f"{len(fill)} filler phrases (max 0): {sorted(set(f.lower() for f in fill))}"] if fill else [],
    }
    if n_words == 0:
        details["R1"] = ["empty text"]
    return report, details


def ste_report(text: str) -> dict:
    """STE metrics for one answer (contract C1). Pure Python, ~1 ms per 300 words."""
    return _analyze(text)[0]


def ste_diagnose(text: str) -> tuple[dict, dict]:
    """(report, {rule: [actionable messages]}) for failing rules."""
    report, details = _analyze(text)
    return report, {r: m for r, m in details.items() if m and not report["rules"][r]}
