import pytest

from ste_rlhf.data.build_pairs import build_pairs
from ste_rlhf.data.collect_writes import collect


def _orig(n):
    return [dict(id=f"p{i}", prompt=f"q{i}", original=f"This is the original answer number {i}.") for i in range(n)]


def test_collect_gate():
    orig = _orig(3)
    ste = [dict(id="p0", ste="Open the file.\nPush the button."),
           dict(id="p1", ste="It should be noted that the button is to be pushed by the operator in order to "
                             "facilitate the utilisation of the system which was described earlier in the manual.")]
    hard = [dict(id="p0", hard_negative="Short answer.")]
    s, h, fails = collect(orig, ste, hard)
    assert [r["id"] for r in s] == ["p0"]
    assert s[0]["ste_score"] >= 80 and s[0]["original"] == orig[0]["original"]
    assert any(f[0] == "p1" and f[1] == "ste" for f in fails)
    assert h == [] and any(f[1] == "hardneg" for f in fails)  # no list -> gate fails


def test_build_pairs_split_and_types():
    n = 10
    ste = [dict(id=f"p{i}", prompt=f"q{i}", original=f"o{i}", ste=f"s{i}") for i in range(n)]
    hard = [dict(id=f"p{i}", hard_negative=f"h{i}") for i in (1, 2, 8)]
    pool = [f"p{i}" for i in range(n)]
    train, val = build_pairs(ste, hard, pool, ["p1", "p2", "p8"], ["p0", "p1"], seed=3,
                             n_per_type=0, n_val_per_type=0)
    assert len(val) == 3 and len(train) == 10
    assert {p["id"] for p in val} == {"p0", "p1"}
    assert sum(p["pair_type"] == "ste_vs_hardneg" for p in val) == 1
    assert all(p["chosen"].startswith("s") for p in train + val)
    assert {p["rejected"] for p in train if p["id"] == "p2"} == {"o2", "h2"}
    train2, _ = build_pairs(ste, hard, pool, ["p1", "p2", "p8"], ["p0", "p1"], seed=3,
                            n_per_type=0, n_val_per_type=0)
    assert train == train2


def test_build_pairs_prompt_overlap_detected():
    ste = [dict(id="a", prompt="same", original="o", ste="s"), dict(id="b", prompt="same", original="o2", ste="s2")]
    with pytest.raises(AssertionError):
        build_pairs(ste, [], ["a", "b"], [], ["a"])


def test_degenerate_pairs_types_and_no_leak():
    from ste_rlhf.data.build_pairs import DEGENERATE_TYPES
    long = "1. Open the valve on the left.\n2. Turn the handle clockwise.\n3. Check the gauge reading.\n4. Close the valve."
    ste = [dict(id=f"p{i}", prompt=f"q{i}", original=f"o{i}", ste=long.replace("valve", f"valve{i}")) for i in range(12)]
    pool = [f"p{i}" for i in range(12)]
    val_ids = ["p0", "p1", "p2"]
    train, val = build_pairs(ste, [], pool, [], val_ids, seed=0, n_per_type=4, n_val_per_type=2)
    types = {p["pair_type"] for p in train}
    assert set(DEGENERATE_TYPES) <= types
    val_ste = {s["ste"] for s in ste if s["id"] in val_ids}
    # an off-topic rejected answer in train never comes from a val prompt (and vice versa)
    assert all(p["rejected"] not in val_ste for p in train if p["pair_type"] == "ste_vs_offtopic")
    assert all(p["rejected"] in val_ste for p in val if p["pair_type"] == "ste_vs_offtopic")
    assert all(p["rejected"] != p["chosen"] for p in train + val)
    rep = [p for p in train if p["pair_type"] == "ste_vs_repeated"][0]["rejected"].splitlines()
    assert len(rep) >= 8 and len(set(rep)) == 1
