"""Fast tests for the S1 data pipeline (no downloads, no model)."""
import random

from minirlhf.common.io import read_jsonl
from minirlhf.common.schemas import validate_record
from minirlhf.data import build_pairs, generate_candidates, hh_rlhf, label_cli, prompts, rule_labeller

# ---------------- prompts ----------------
DOLLY = [
    {"instruction": "What is a cat?", "context": "", "category": "open_qa"},
    {"instruction": "what is a  cat?", "context": "", "category": "general_qa"},   # duplicate
    {"instruction": "Summarize this", "context": "some text", "category": "open_qa"},  # has context
    {"instruction": "Classify this", "context": "", "category": "classification"},    # wrong category
    {"instruction": " ".join(["word"] * 41), "context": "", "category": "open_qa"},   # too long
    {"instruction": "Give me ideas for a party", "context": "", "category": "brainstorming"},
    {"instruction": "Write a poem about rain", "context": "", "category": "creative_writing"},
]


def test_filter_and_dedupe():
    items = prompts.dedupe(prompts.filter_dolly(DOLLY))
    assert [i["prompt"] for i in items] == ["What is a cat?", "Give me ideas for a party",
                                           "Write a poem about rain"]
    assert items[0]["source"] == "dolly:open_qa"


def test_split_disjoint_and_seeded():
    items = [{"prompt": f"p{i}", "source": "x"} for i in range(50)]
    tr, ev = prompts.split_prompts(items, 30, 10, seed=1)
    assert len(tr) == 30 and len(ev) == 10
    assert not {i["prompt"] for i in tr} & {i["prompt"] for i in ev}
    assert prompts.split_prompts(items, 30, 10, seed=1) == (tr, ev)
    assert prompts.split_prompts(items, 30, 10, seed=2) != (tr, ev)


def test_build_writes_valid_files(cfg, tmp_path):
    cfg["data"]["n_train_prompts"], cfg["data"]["n_eval_prompts"] = 2, 1
    hand = tmp_path / "handwritten.txt"
    hand.write_text("# comment\nHow do I boil an egg?\n\n")
    train, ev = prompts.build(cfg, rows=DOLLY, handwritten_path=hand, out_dir=tmp_path)
    assert len(train) == 2 and len(ev) == 1
    for r in read_jsonl(tmp_path / "train.jsonl") + read_jsonl(tmp_path / "eval.jsonl"):
        validate_record("prompt", r)


# ---------------- candidates ----------------
def test_make_candidate_drops_bad():
    g = {"temperature": 0.9}
    assert generate_candidates.make_candidate("a", "q", "x", "x", g) is None
    assert generate_candidates.make_candidate("a", "q", "", "x", g) is None
    assert generate_candidates.make_candidate("a", "q", "x", "  ", g) is None
    c = generate_candidates.make_candidate("a", "q", "x", "y", g)
    assert c["responses"] == ["x", "y"]


# ---------------- rule labeller ----------------
def test_rule_ordering():
    prompt = "How do I boil an egg?"
    good = "Sure, happy to help! To boil an egg, put the egg in boiling water for ten minutes."
    rep = "I don't know. I don't know. I don't know. I don't know. I don't know."
    long = " ".join(["egg"] * 5 + ["filler"] * 200)
    assert rule_labeller.score(prompt, good) > rule_labeller.score(prompt, rep)
    assert rule_labeller.score(prompt, good) > rule_labeller.score(prompt, long)
    assert rule_labeller.repetition_penalty(rep) > rule_labeller.repetition_penalty(good)
    assert rule_labeller.length_penalty(long) > 0 == rule_labeller.length_penalty(good)
    p = rule_labeller.label_pair({"id": "x", "prompt": prompt, "responses": [rep, good]})
    assert p["chosen"] == good and p["rejected"] == rep and p["source"] == "rule"
    p2 = rule_labeller.label_pair({"id": "x", "prompt": prompt, "responses": [good, rep]})
    assert p2["chosen"] == good


def test_rule_skips_near_ties():
    c = {"id": "t", "prompt": "What is the sky?", "responses": ["Blue things up high.", "Blue stuff up high."]}
    assert rule_labeller.label_pair(c) is None
    assert rule_labeller.label_all([c]) == []


def test_rule_on_fixture(fixtures_dir):
    prefs = rule_labeller.label_all(read_jsonl(fixtures_dir / "candidates_small.jsonl"))
    assert prefs and all(p["chosen"].startswith("Sure") for p in prefs)


# ---------------- hh-rlhf ----------------
H = "\n\nHuman: How do I bake bread?\n\nAssistant: "


def test_hh_parse_single_turn():
    r = hh_rlhf.parse_single_turn(H + "Mix flour and water.", H + "I dunno.")
    assert r == ("How do I bake bread?", "Mix flour and water.", "I dunno.")


def test_hh_rejects_multi_turn_and_long():
    multi = H + "Sure.\n\nHuman: Thanks\n\nAssistant: You're welcome"
    assert hh_rlhf.parse_single_turn(multi, multi) is None
    assert hh_rlhf.parse_single_turn(H + "a", multi) is None
    assert hh_rlhf.parse_single_turn(H + " ".join(["w"] * 121), H + "short") is None
    assert hh_rlhf.parse_single_turn(H + "same", H + "same") is None
    other = "\n\nHuman: Different?\n\nAssistant: ok"
    assert hh_rlhf.parse_single_turn(H + "a", other) is None


def test_hh_build_sampling():
    rows = [{"chosen": f"\n\nHuman: q{i}\n\nAssistant: good{i}", "rejected": f"\n\nHuman: q{i}\n\nAssistant: bad{i}"}
            for i in range(20)]
    rows.append({"chosen": H + "a\n\nHuman: x\n\nAssistant: y", "rejected": H + "b\n\nHuman: x\n\nAssistant: y"})
    out = hh_rlhf.build_hh(rows, 5, seed=0)
    assert len(out) == 5 and all(validate_record("pref", r) for r in out)
    assert out == hh_rlhf.build_hh(rows, 5, seed=0)
    assert len(hh_rlhf.build_hh(rows, 100)) == 20


# ---------------- label cli ----------------
def test_order_pair_mapping_both_orders():
    seen = set()
    for s in range(20):
        left, right, swapped = label_cli.order_pair(["A", "B"], random.Random(s))
        assert (left, right) == (("B", "A") if swapped else ("A", "B"))
        seen.add(swapped)
    assert seen == {True, False}


def test_interpret_key():
    assert [label_cli.interpret_key(k) for k in "12sQx"] == ["left", "right", "skip", "quit", None]
    assert label_cli.interpret_key("") is None and label_cli.interpret_key(None) is None


def test_make_pref_mapping():
    cand = {"id": "c1", "prompt": "q", "responses": ["A", "B"]}
    assert label_cli.make_pref(cand, "B", "A", "left")["chosen"] == "B"
    p = label_cli.make_pref(cand, "B", "A", "right")
    assert (p["chosen"], p["rejected"], p["source"], p["id"]) == ("A", "B", "human", "human_c1")


def _cands(n):
    return [{"id": f"c{i}", "prompt": f"q{i}", "responses": [f"a{i}", f"b{i}"], "gen": {}} for i in range(n)]


def _keys(seq):
    it = iter(seq)
    return lambda: next(it)


def test_session_labels_skip_quit_resume(tmp_path):
    out, skip = tmp_path / "human.jsonl", tmp_path / "skipped.jsonl"
    log = []
    cands = _cands(5)
    # label c0 left, skip c1, label c2 right, quit at c3 (with an invalid key first)
    n = label_cli.run_session(cands, out, _keys(["1", "s", "2", "x", "q"]), log.append,
                              random.Random(0), skipped_path=skip)
    assert n == 2
    rows = read_jsonl(out)
    assert [r["id"] for r in rows] == ["human_c0", "human_c2"]
    assert any("Please press" in m for m in log)
    for r in rows:
        validate_record("pref", r)
        assert r["chosen"][0] != r["rejected"][0] and r["chosen"][1:] == r["rejected"][1:]
    # chosen must be the response shown on the pressed side
    # (rng seed 0 deterministic: reproduce displays)
    rng = random.Random(0)
    left0, right0, _ = label_cli.order_pair(cands[0]["responses"], rng)
    assert rows[0]["chosen"] == left0
    left2, right2, _ = label_cli.order_pair(cands[2]["responses"], rng)  # c1 consumed one draw earlier
    # resume: only c3, c4 remain (c1 skipped is not re-asked)
    assert [c["id"] for c in label_cli.pending(cands, out, skip)] == ["c3", "c4"]
    n = label_cli.run_session(cands, out, _keys(["1", "1"]), log.append, random.Random(1), skipped_path=skip)
    assert n == 2 and len(read_jsonl(out)) == 4


def test_session_max_n_and_displays_prompt(tmp_path):
    log = []
    n = label_cli.run_session(_cands(5), tmp_path / "h.jsonl", _keys(["1"] * 5), log.append,
                              random.Random(0), max_n=2)
    assert n == 2
    assert any("Prompt: q0" in m for m in log)


# ---------------- build_pairs ----------------
def _pref(i, src, prompt=None):
    return {"id": f"{src}{i}", "prompt": prompt or f"prompt {src} {i}", "chosen": f"good {i}",
            "rejected": f"bad {i}", "source": src}


def test_dedupe():
    a = _pref(0, "rule")
    dup = dict(a, id="other", chosen="  GOOD 0 ")
    same = dict(_pref(1, "rule"), rejected="good 1")
    out = build_pairs.dedupe([a, dup, same])
    assert out == [a]


def test_mix_fractions():
    by = {"human": [_pref(i, "human") for i in range(10)], "rule": [_pref(i, "rule") for i in range(10)]}
    out = build_pairs.apply_mix(by, {"human": 1.0, "rule": 0.5, "hh-rlhf": 1.0})
    assert len(out["human"]) == 10 and len(out["rule"]) == 5


def test_split_stratified_and_prompt_disjoint():
    by = {s: [_pref(i, s) for i in range(20)] for s in ("human", "rule", "hh-rlhf")}
    by["rule"].append(dict(_pref(99, "rule"), prompt=by["rule"][0]["prompt"], chosen="alt"))
    train, val = build_pairs.stratified_split(by, 0.1, seed=0)
    assert len(train) + len(val) == 61
    for s in by:
        assert sum(r["source"] == s for r in val) >= 2
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}
    assert (train, val) == build_pairs.stratified_split(by, 0.1, seed=0)


def test_merge_and_split_cross_source_dedupe():
    shared = _pref(0, "human")
    by = {"human": [shared], "rule": [dict(shared, id="r", source="rule"), _pref(1, "rule")]}
    train, val = build_pairs.merge_and_split(by, {}, 0.0)
    assert len(train) == 2 and not val
    assert {r["source"] for r in train} == {"human", "rule"}


def test_build_end_to_end(cfg, tmp_path, fixtures_dir):
    import shutil
    shutil.copy(fixtures_dir / "prefs_small.jsonl", tmp_path / "rule.jsonl")  # mixed sources, validates as pref
    cfg["data"]["val_fraction"] = 0.2
    train, val = build_pairs.build(cfg, prefs_dir=tmp_path)
    assert len(train) + len(val) == len(read_jsonl(tmp_path / "rule.jsonl"))
    assert len(read_jsonl(tmp_path / "train.jsonl")) == len(train)
    assert len(read_jsonl(tmp_path / "val.jsonl")) == len(val)


def test_cross_source_conflict_and_prompt_leak():
    """A rule label reversing a human label is dropped, and no prompt spans train and val."""
    def pref(i, c, r, src):
        return {"id": f"{src}{i}", "prompt": f"p{i}", "chosen": c, "rejected": r, "source": src}
    human = [pref(i, "good", "bad", "human") for i in range(20)]
    rule = [pref(i, "bad", "good", "rule") for i in range(20)]           # reversed: must go
    rule += [pref(i, "x", "y", "rule") for i in range(20)]               # same prompts, new pair: kept
    train, val = build_pairs.merge_and_split({"human": human, "rule": rule}, {}, 0.2, seed=0)
    rows = train + val
    assert not any(r["source"] == "rule" and r["chosen"] == "bad" for r in rows)
    assert sum(r["source"] == "rule" for r in rows) == 20
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}
