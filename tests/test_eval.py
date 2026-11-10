import json
import math

import pytest
import torch

from ste_rlhf.common import load_config, read_jsonl, write_jsonl
from ste_rlhf.eval import _util, metrics, plots, samples, winrate_cli


def test_distinct2():
    assert _util.distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert _util.distinct_n(["a b", "a b"], 2) == pytest.approx(0.5)
    assert _util.distinct_n([""], 2) == 0.0


def test_wilson_ci():
    lo, hi = _util.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert _util.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = _util.wilson_ci(10, 10)
    assert hi == 1.0 and lo > 0.69


def test_sign_test():
    p1, p2 = _util.sign_test(9, 1)
    assert p1 == pytest.approx(11 / 1024)
    assert p2 == pytest.approx(22 / 1024)
    p1, p2 = _util.sign_test(1, 9)
    assert p1 > 0.99 and p2 == pytest.approx(22 / 1024)
    assert _util.sign_test(5, 5)[1] == 1.0
    assert _util.sign_test(0, 0) == (1.0, 1.0)


def test_parse_beta_seed():
    assert _util.parse_beta("reinforce_beta0.1_seed0") == 0.1
    assert _util.parse_beta("reinforce_beta0_seed2") == 0.0
    assert _util.parse_beta("reinforce_beta0.05_seed1_nobaseline") == 0.05
    assert _util.parse_beta("reinforce_beta1e-2_seed1") == 0.01
    assert _util.parse_beta("foo") is None
    assert _util.parse_seed("reinforce_beta0.1_seed12") == 12
    assert _util.seed_group("reinforce_beta0.1_seed2") == "reinforce_beta0.1"
    assert _util.seed_group("reinforce_beta0.1") is None


def test_moving_average():
    assert _util.moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
    assert _util.moving_average([1, 2], 1) == [1, 2]


def _rows(prefix, n=4):
    return [{"id": f"i{k}", "prompt": f"p{k}", "response": f"{prefix}{k}"} for k in range(n)]


def _scripted(keys):
    it = iter(keys)

    def f(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return f


def test_winrate_session_eof_and_resume(tmp_path):
    state = tmp_path / "w" / "tuned_vs_base.json"
    out = []
    st = winrate_cli.run_session(_rows("A"), _rows("B"), state, "tuned", "base", seed=0,
                                 input_fn=_scripted(["1", "t", "x", "s"]), out_fn=out.append)
    assert len(st["choices"]) == 2  # '1' and 't'; skip not recorded, then EOF
    saved = json.loads(state.read_text())
    assert saved["ties"] == 1 and saved["wins"] + saved["losses"] == 1
    # resume: remaining ids only, answer 'q' immediately
    st2 = winrate_cli.run_session(_rows("A"), _rows("B"), state, "tuned", "base", seed=0,
                                  input_fn=_scripted(["q"]), out_fn=out.append)
    assert st2["choices"] == st["choices"]


def test_winrate_blind_sides_and_counts(tmp_path):
    seen = []
    state = tmp_path / "s.json"

    def inp(_p):
        # always pick the left answer; record which system was on the left
        seen.append(None)
        return "1"
    out = []
    st = winrate_cli.run_session(_rows("A", 20), _rows("B", 20), state, seed=3, input_fn=inp, out_fn=out.append)
    left_a = sum(1 for i in range(20) if winrate_cli.side_for(f"i{i}", 3) == "a_left")
    assert 0 < left_a < 20
    assert st["wins"] == left_a and st["losses"] == 20 - left_a
    assert st["win_rate_excl_ties"] == pytest.approx(left_a / 20)
    assert "Wilson" in out[-1] and "sign test" in out[-1]


def test_winrate_keyboard_interrupt(tmp_path):
    def inp(_p):
        raise KeyboardInterrupt
    st = winrate_cli.run_session(_rows("A"), _rows("B"), tmp_path / "k.json", input_fn=inp, out_fn=lambda *_: None)
    assert st["choices"] == {} and (tmp_path / "k.json").exists()


class FakeScorer:
    def score(self, prompts, responses):
        return torch.tensor([float(len(r.split())) for r in responses])


def fake_ste_table(tag_texts):
    return {t: {"n": len(x), "ste_score_mean": 80.0 if t == "tuned" else 50.0, "pct_passes_80": 100.0,
                "avg_sentence_words_mean": 9.5, "pct_passive_mean": 3.0, "list_items_mean": 2.0,
                "pct_has_list": 50.0, "filler_phrases_mean": 0.1, "complex_words_mean": 1.0,
                "vague_terms_mean": 0.0, "flesch_mean": 70.0, "words_mean": 12.0}
            for t, x in tag_texts.items()}


def _write_gens(out_dir, tag, responses, finished=True):
    write_jsonl(out_dir / "generations" / f"{tag}.jsonl",
                [{"id": f"i{k}", "prompt": f"p{k}", "response": r, "finished": finished if k else True,
                  "num_tokens": len(r.split()), "model": "dummy"} for k, r in enumerate(responses)])


def test_metrics_run_and_summary(tmp_path):
    cfg = load_config()
    _write_gens(tmp_path, "base", ["one two three", "a b c d e f"], finished=False)
    _write_gens(tmp_path, "tuned", ["one two", "x y z"])
    summary = metrics.run(["base", "tuned"], cfg, out_dir=tmp_path, scorer=FakeScorer(), ste_table_fn=fake_ste_table,
                          kl_fn_factory=lambda tag, rows: (lambda r: 1.5), log=lambda *_: None)
    b, t = summary["tags"]["base"], summary["tags"]["tuned"]
    assert b["kl_vs_base"] == 0.0 and t["kl_vs_base"] == 1.5
    assert b["finish_rate"] == 0.5 and t["finish_rate"] == 1.0
    assert b["rm_score_mean"] == pytest.approx(4.5) and b["rm_score_std"] == pytest.approx(1.5)
    assert t["ste_score_mean"] == 80.0 and "created" in summary
    md = (tmp_path / "metrics" / "summary.md").read_text().splitlines()
    assert md[0].startswith("| tag | n |") and "STE score" in md[0] and "KL vs base" in md[0]
    assert md[2].startswith("| base |") and md[3].startswith("| tuned |")
    assert len({len(l.split("|")) for l in md}) == 1  # rectangular table
    assert (tmp_path / "metrics" / "tuned.json").exists()
    assert json.loads((tmp_path / "metrics" / "summary.json").read_text())["tags"].keys() == {"base", "tuned"}


def test_metrics_no_rm_no_kl(tmp_path):
    cfg = load_config()
    _write_gens(tmp_path, "tuned", ["a b", "c d"])
    s = metrics.run(["tuned"], cfg, out_dir=tmp_path, no_kl=True, no_rm=True, ste_table_fn=fake_ste_table,
                    log=lambda *_: None)
    assert "rm_score_mean" not in s["tags"]["tuned"] and "kl_vs_base" not in s["tags"]["tuned"]


def test_summary_md_missing_columns():
    md = metrics.format_summary_md({"base": {"n": 2, "ste_score_mean": 55.0}, "tuned": {"n": 2, "ste_score_mean": 70.0}})
    assert "RM score" not in md and "55.00" in md


def test_samples_markdown():
    by = {"base": {"i0": {"prompt": "Q?", "response": "long | text\nmore", "finished": False}},
          "tuned": {"i0": {"prompt": "Q?", "response": "short", "finished": True}}}
    md = samples.build_markdown(["base", "tuned"], by, {"base": {"i0": 40.0}, "tuned": {"i0": 90.0}},
                                {"base": {"i0": -0.5}, "tuned": {"i0": 1.25}}, 5)
    assert "STE 40" in md and "RM 1.25" in md and "cut off" in md and "\\|" in md


def _write_run(logs, name, n=30, k=1.0):
    d = logs / name
    d.mkdir(parents=True)
    write_jsonl(d / "metrics.jsonl", [{"step": s, "reward": 0.1 * s * k, "kl": 0.05 * s, "ste_score_mean": 50 + s}
                                      for s in range(n)])


def test_plots_synthetic(tmp_path):
    logs, res = tmp_path / "logs", tmp_path / "results"
    for name, k in [("reinforce_beta0.1_seed0", 1.0), ("reinforce_beta0.1_seed1", 1.1),
                    ("reinforce_beta0_seed0", 1.5), ("reinforce_beta0.2_seed0", 0.8)]:
        _write_run(logs, name, k=k)
    _write_run(logs, "pilot_x")
    assert "pilot_x" not in plots.load_runs(logs)
    (res / "metrics").mkdir(parents=True)
    for tag, v in [("base", 1.0), ("tuned", 2.0)]:
        (res / "metrics" / f"{tag}.json").write_text(json.dumps(
            {"ste_score_mean": 50 * v, "avg_sentence_words_mean": 20 / v, "pct_passive_mean": 10 / v,
             "list_items_mean": v, "filler_phrases_mean": 1 / v, "complex_words_mean": 5 / v}))
    made = plots.make_all(logs, res, smooth=3, log=lambda *_: None)
    assert all(made.values())
    for f in ["reward_curves", "kl_curves", "ste_curves", "seed_mean_std", "beta_sweep", "ste_before_after"]:
        assert (res / "plots" / f"{f}.png").stat().st_size > 1000


def test_plots_empty_dirs(tmp_path):
    made = plots.make_all(tmp_path / "nologs", tmp_path / "res", log=lambda *_: None)
    assert not any(made.values())


def test_final_value():
    assert plots.final_value(list(range(100)), 0.1) == pytest.approx(94.5)
    assert plots.final_value([None]) is None


@pytest.mark.slow
def test_generate_eval_smollm(tmp_path, fixtures_dir):
    from ste_rlhf.eval.generate_eval import generate_for_tag
    cfg = load_config(overrides=["eval.max_new_tokens=16"])
    path = generate_for_tag(cfg["policy"]["name"], "tiny", cfg, prompts_path=fixtures_dir / "prompts.jsonl",
                            limit=2, out_dir=tmp_path, log=lambda *_: None)
    rows = read_jsonl(path)
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"id", "prompt", "response", "finished", "num_tokens", "model"}
        assert isinstance(r["finished"], bool) and 0 < r["num_tokens"] <= 16
        if not r["finished"]:
            assert r["num_tokens"] == 16
    # resumable: second call generates nothing new
    generate_for_tag(cfg["policy"]["name"], "tiny", cfg, prompts_path=fixtures_dir / "prompts.jsonl",
                     limit=2, out_dir=tmp_path, log=lambda *_: None)
    assert len(read_jsonl(path)) == 2
    # teacher-forcing rollout builds from saved rows
    from ste_rlhf.common.models import load_tokenizer
    ro = metrics.build_rollout(load_tokenizer(cfg["policy"]["name"]), rows, 320)
    assert ro.response_ids.shape[0] == 2 and ro.response_mask.sum() > 0
