import json
import math
from pathlib import Path

import pytest

from minirlhf.eval import metrics, plots, report_tables, winrate_cli


def _write_run(root: Path, name: str, n=20, slope=0.1, kl=0.01):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "metrics.jsonl", "w") as f:
        for i in range(1, n + 1):
            f.write(json.dumps({"step": i, "reward_mean": slope * i, "reward_std": .5, "kl_mean": kl * i,
                                "return_mean": 0, "baseline": 0, "loss": 0, "entropy": 1,
                                "resp_len_mean": 30, "time_s": i}) + "\n")


def test_distinct_n():
    assert metrics.distinct_n(["a b a b"], 1) == 0.5
    assert metrics.distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert metrics.distinct_n(["a b", "a b"], 2) == 0.5
    assert metrics.distinct_n(["hello"], 2) == 0.0
    assert metrics.distinct_n([], 1) == 0.0


def test_wilson_ci():
    lo, hi = winrate_cli.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = winrate_cli.wilson_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert winrate_cli.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = winrate_cli.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)


def _pairs(n=4):
    return [{"id": f"p{i}", "prompt": f"q{i}", "a": f"A{i}", "b": f"B{i}"} for i in range(n)]


def _keys(seq):
    it = iter(seq)
    return lambda: next(it)


def test_winrate_logic_and_resume():
    pairs = _pairs(4)
    shown = []
    # figure out which side A is on so we can always pick A
    keys = []
    for p in pairs[:2]:
        a_left = winrate_cli.random.Random(f"0:{p['id']}").random() < 0.5
        keys.append("1" if a_left else "2")
    saved = []
    js = winrate_cli.run_session(pairs, [], _keys(keys + ["x", "q"]), show=shown.append, save=saved.append)
    assert [j["winner"] for j in js] == ["a", "a"] and len(saved) == 2
    # resume: remaining two; tie then B wins
    p = pairs[3]
    a_left = winrate_cli.random.Random(f"0:{p['id']}").random() < 0.5
    js2 = winrate_cli.run_session(pairs, js, _keys(["t", "2" if a_left else "1"]), show=shown.append)
    assert [j["winner"] for j in js2] == ["a", "a", "tie", "b"]
    s = winrate_cli.summarize(js2)
    assert (s["wins"], s["ties"], s["losses"]) == (2, 1, 1)
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert 0 < s["ci95"][0] < s["win_rate"] < s["ci95"][1] < 1


def test_winrate_skip_and_empty():
    js = winrate_cli.run_session(_pairs(2), [], _keys(["s", "s"]), show=lambda *_: None)
    assert js == []
    assert winrate_cli.summarize(js)["win_rate"] is None


def test_build_pairs_intersection():
    a = [{"id": "1", "prompt": "p", "response": "x", "model": "m"}, {"id": "2", "prompt": "p", "response": "y", "model": "m"}]
    b = [{"id": "2", "prompt": "p", "response": "z", "model": "m"}]
    assert winrate_cli.build_pairs(a, b) == [{"id": "2", "prompt": "p", "a": "y", "b": "z"}]


def test_summary_table_format():
    fake = {"base": {"n": 5, "reward_mean": 0.1, "reward_std": 0.2, "resp_len_words": 30.0, "distinct1": .5,
                     "distinct2": .9, "kl_mean": 0.0},
            "tuned": {"n": 5, "reward_mean": 1.0, "reward_std": 0.3, "resp_len_words": 20.0, "distinct1": .4,
                      "distinct2": .8, "kl_mean": 2.5}}
    md = metrics.format_summary(fake).splitlines()
    assert len(md) == 4
    assert md[0].startswith("| Tag |") and md[1].startswith("|---")
    assert md[3] == "| tuned | 5 | 1.000 | 0.300 | 20.000 | 0.400 | 0.800 | 2.500 |"


def test_write_summary_orders_base_first(tmp_path):
    for t, r in (("zzz", 1.0), ("base", 0.0)):
        (tmp_path / f"{t}.json").write_text(json.dumps({"tag": t, "n": 1, "reward_mean": r, "reward_std": 0,
                                                        "resp_len_words": 1, "distinct1": 1, "distinct2": 1, "kl_mean": 0}))
    metrics.write_summary(tmp_path)
    lines = (tmp_path / "summary.md").read_text().splitlines()
    assert lines[2].startswith("| base") and lines[3].startswith("| zzz")


def test_samples_table():
    g = {t: [{"id": "p0", "prompt": "a|b", "response": f"{t} says\nhi", "model": "m"}] for t in ("base", "tuned")}
    md = report_tables.format_samples(["base", "tuned"], g, {"base": {"p0": 0.5}, "tuned": {"p0": -1.0}}, n=5)
    assert "a\\|b" in md and "base says<br>hi **(+0.50)**" in md and "**(-1.00)**" in md


def test_plots_run(tmp_path):
    logs = tmp_path / "logs"
    _write_run(logs, "run_a")
    _write_run(logs, "run_b", slope=0.2)
    out = tmp_path / "plots"
    written = plots.make_plots(logs, out, smooth=3)
    names = {Path(w).name for w in written}
    assert names == {"reward_curves.png", "kl_curves.png"}
    assert all((out / n).stat().st_size > 0 for n in names)


def test_plots_all_kinds(tmp_path):
    logs = tmp_path / "logs"
    for b, k in ((0.0, 0.5), (0.05, 0.1), (0.5, 0.01)):
        _write_run(logs, f"reinforce_beta{b:g}_seed0", kl=k)
    _write_run(logs, "reinforce_beta0.05_seed1", slope=0.12)
    _write_run(logs, "reinforce_beta0.05_seed0_nobaseline", slope=0.05)
    written = {Path(w).name for w in plots.make_plots(logs, tmp_path / "p")}
    assert {"seed_mean_std.png", "beta_sweep.png", "baseline_ablation.png"} <= written
    runs = plots.load_runs(logs)
    assert plots.group_seeds(runs) == {"reinforce_beta0.05": ["reinforce_beta0.05_seed0", "reinforce_beta0.05_seed1"]}
    assert [p[0] for p in plots.sweep_points(runs)] == [0.0, 0.05, 0.5]
    assert plots.ablation_pairs(runs) == [("reinforce_beta0.05_seed0", "reinforce_beta0.05_seed0_nobaseline")]


def test_plots_empty_dir(tmp_path):
    assert plots.make_plots(tmp_path / "none", tmp_path / "o") == []


def test_moving_average():
    assert list(plots.moving_average([1, 2, 3], 1)) == [1, 2, 3]
    assert list(plots.moving_average([0, 2, 4], 2)) == [0, 1, 3]


@pytest.mark.slow
def test_generate_eval_smoke(cfg, tmp_path):
    from minirlhf.common.io import read_jsonl
    from minirlhf.common.schemas import validate_record
    from minirlhf.eval import generate_eval
    cfg["eval"]["max_new_tokens"] = 8
    cfg["eval"]["out_dir"] = str(tmp_path)
    path = generate_eval.run(cfg, "base", "smoke", limit=2, batch_size=2,
                             prompts_path=Path(__file__).parent / "fixtures" / "prompts_small.jsonl", device="cpu")
    rows = read_jsonl(path)
    assert len(rows) == 2
    for r in rows:
        validate_record("generation", r)
        assert r["model"] == cfg["policy"]["name"]
    # metrics on the base tag: KL exactly 0
    cfg["data"]["prompts_dir"] = str(tmp_path)
    m = metrics.run(cfg, ["smoke"], device="cpu", batch_size=2)
    assert m["smoke"]["kl_mean"] == 0.0 and not math.isnan(m["smoke"]["reward_mean"])
    assert (tmp_path / "metrics" / "summary.md").exists()
