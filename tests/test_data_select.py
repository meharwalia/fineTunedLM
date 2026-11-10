"""Fast tests for ste_rlhf.data (no downloads, tiny in-memory data)."""
import json

import pytest

from ste_rlhf.data.select_prompts import (classify_prompt, normalize_prompt, pick_original, select,
                                          build_report, SAMPLE_MARKER)
from ste_rlhf.data.splits import make_splits, sample_section
from ste_rlhf.data.writer_batches import make_batches

tok_len = lambda s: len(s.split())
CFG = {"data": {"max_prompt_tokens": 300, "max_original_tokens": 768}}
LONG = " ".join(["word"] * 60)


@pytest.mark.parametrize("prompt,reason", [
    ("How do I install Docker on Ubuntu?", None),
    ("Explain the difference between TCP and UDP", None),
    ("what is a webhook", None),
    ("Write a poem about the sea", "creative"),
    ("Tell me a joke about cats", "creative"),
    ("Pretend you are a pirate and explain how sails work", "roleplay"),
    ("Translate this sentence into French", "translation"),
    ("Summarize the following text: abc", "rewrite_text"),
    ("Paraphrase this paragraph", "rewrite_text"),
    ("How do I fix this?\n```python\nprint(1)\n```", "code_fence"),
    ("how to do x\n" + "def f(x):\n    return x\n" * 4, "mostly_code"),
    ("Explain this: " + " ".join(["lorem"] * 200), "long_passage"),
    ("Write a python function to sort a list", "code_writing"),
    ("Write an email to my boss", "writing_task"),
    ("gggg", "not_explanatory"),
])
def test_classify(prompt, reason):
    assert classify_prompt(prompt) == reason


def row(i, prompt, resp=LONG, h=3, v=2):
    return dict(row=i, prompt=prompt, response=resp, helpfulness=h, verbosity=v)


def test_pick_original_rules():
    a = row(0, "p", "x " * 10, h=3, v=2)
    b = row(1, "p", "x " * 20, h=4, v=3)
    assert pick_original([a, b], tok_len)["row"] == 1          # higher helpfulness
    c = row(2, "p", "x " * 20, h=4, v=1)
    assert pick_original([b, c], tok_len)["row"] == 2          # tie -> lower verbosity
    d = row(3, "p", "x " * 5, h=4, v=1)
    assert pick_original([c, d], tok_len)["row"] == 3          # tie -> shorter


def test_select_reasons_and_dedup():
    rows = [
        row(0, "How do I boil an egg?"), row(1, "How do I boil an egg?", h=4),
        row(2, "How  do I boil an EGG??"),                      # normalised duplicate
        row(3, "How to x\n<extra_id_1>Assistant\nhi"),          # multi-turn
        row(4, " ".join(["how"] * 400)),                        # too long prompt
        row(5, "Write a poem"),
        row(6, "How does DNS work?", resp="too short"),
        row(7, "How does DNS work now?", resp=" ".join(["w"] * 800)),
        row(8, "Explain recursion"),
    ]
    kept, dropped = select(rows, CFG, tok_len)
    reasons = {d["hs2_row"]: d["reason"] for d in dropped}
    assert reasons[3] == "multi_turn"
    assert reasons[4] == "prompt_too_long"
    assert reasons[5] == "creative"
    assert reasons[6] == "original_too_short"
    assert reasons[7] == "original_too_long"
    ids = [k["id"] for k in kept]
    assert ids == ["hs2_1", "hs2_8"]                            # egg deduped, best response row
    assert normalize_prompt("How  do I boil an EGG??") == normalize_prompt("How do I boil an egg?")
    rep = build_report(len(rows), kept, dropped)
    assert "original_too_long" in rep and SAMPLE_MARKER in rep


def make_cands(n):
    return [dict(id=f"hs2_{i}", hs2_row=i, prompt=f"how do I do thing {i}?", original=LONG,
                 helpfulness=3, verbosity=2, original_tokens=60, prompt_tokens=8,
                 original_sentences=5 if i % 2 else 1, has_procedure=False) for i in range(n)]


DCFG = dict(n_rm_pool=40, n_spare=4, n_hardneg=8, n_rl_train=30, n_eval=10, n_val_prompts=10)


def test_splits_disjoint_and_sizes():
    s = make_splits(make_cands(100), DCFG, seed=0)
    sets = {k: {c["id"] for c in s[k]} for k in ("rm_pool", "spares", "eval", "rl_train")}
    assert [len(v) for v in sets.values()] == [40, 4, 10, 30]
    names = list(sets)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[names[i]] & sets[names[j]]
    val, hard = set(s["val_ids"]), s["hardneg_ids"]
    assert len(val) == 10 and val <= sets["rm_pool"]
    assert len(hard) == 8 and len(set(hard)) == 8 and set(hard) <= sets["rm_pool"]
    assert len(set(hard) & val) == 2 and len(set(hard) - val) == 6   # 10% val, as 50/500
    # soft preference: all structured (odd) candidates (50) come before any unstructured
    assert all(c["original_sentences"] >= 3 for c in s["rm_pool"] + s["spares"])


def test_splits_deterministic_and_fewer_rl():
    a = make_splits(make_cands(70), DCFG, seed=0)
    b = make_splits(make_cands(70), DCFG, seed=0)
    c = make_splits(make_cands(70), DCFG, seed=1)
    assert [x["id"] for x in a["rm_pool"]] == [x["id"] for x in b["rm_pool"]]
    assert a["hardneg_ids"] == b["hardneg_ids"]
    assert [x["id"] for x in a["eval"]] != [x["id"] for x in c["eval"]]
    assert len(a["rl_train"]) == 70 - 40 - 4 - 10  # as many as remain
    with pytest.raises(ValueError):
        make_splits(make_cands(30), DCFG, seed=0)


def test_writer_batches_and_sample():
    cands = make_cands(120)
    originals = [dict(c, original="orig " + c["id"]) for c in cands]
    pool = [c["id"] for c in cands[:100]]
    hard = pool[:90]
    todo, hn = make_batches(originals, pool, hard)
    assert [len(b) for b in todo] == [50, 50]
    assert [len(b) for b in hn] == [50, 40]
    assert set(todo[0][0]) == {"id", "prompt", "original"}
    sec = sample_section([dict(c, prompt="x" * 1000) for c in cands[:60]], seed=0)
    assert sec.count("```text") == 50 and "[truncated]" in sec
    json.dumps(todo)
