import pytest

from minirlhf.common.io import append_jsonl, read_jsonl, write_jsonl
from minirlhf.common.logging import RunLogger
from minirlhf.common.schemas import validate_record


@pytest.mark.parametrize("fname,kind,n", [("prompts_small.jsonl", "prompt", 10),
                                          ("candidates_small.jsonl", "candidate", 5),
                                          ("prefs_small.jsonl", "pref", 20),
                                          ("logs/run_a/metrics.jsonl", "metrics", 5),
                                          ("logs/run_b/metrics.jsonl", "metrics", 5),
                                          ("logs/run_a/samples.jsonl", "sample", 1)])
def test_fixtures_valid(fixtures_dir, fname, kind, n):
    rows = read_jsonl(fixtures_dir / fname)
    assert len(rows) == n
    for r in rows:
        validate_record(kind, r)


def test_prefs_mixed_sources(fixtures_dir):
    assert {r["source"] for r in read_jsonl(fixtures_dir / "prefs_small.jsonl")} == {"human", "rule", "hh-rlhf"}


def test_invalid_records():
    with pytest.raises(ValueError):
        validate_record("pref", {"id": "a", "prompt": "p", "chosen": "c", "rejected": "r", "source": "bogus"})
    with pytest.raises(ValueError):
        validate_record("pref", {"id": "a", "prompt": "p", "chosen": "c"})
    with pytest.raises(ValueError):
        validate_record("candidate", {"id": "a", "prompt": "p", "responses": ["x"], "gen": {}})
    with pytest.raises(ValueError):
        validate_record("nope", {})


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "a" / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    append_jsonl(p, {"a": "é"})
    assert read_jsonl(p) == [{"a": 1}, {"a": "é"}]


def test_run_logger(tmp_path, fixtures_dir):
    lg = RunLogger(tmp_path / "run", {"seed": 0})
    row = read_jsonl(fixtures_dir / "logs/run_a/metrics.jsonl")[0]
    step = row.pop("step")
    lg.log(step, row)
    lg.log_samples(step, ["p"], ["r"], [0.5])
    validate_record("metrics", read_jsonl(lg.metrics_path)[0])
    validate_record("sample", read_jsonl(lg.samples_path)[0])
