import argparse

import pytest
import torch

from ste_rlhf.common import (RunLogger, add_config_args, append_jsonl, apply_overrides, get_device,
                            load_config, per_token_kl, read_jsonl, resolve_run_name, set_seed,
                            write_jsonl)
from ste_rlhf.common.generation import Rollout, build_response_mask


def test_load_config_matches_contract(cfg):
    assert cfg["policy"]["max_new_tokens"] == 256
    assert cfg["rl"]["max_kl"] == 15.0 and cfg["rl"]["save_steps"] == [150]
    assert cfg["rl"]["run_name"] is None
    assert cfg["serving"]["max_new_tokens_limit"] == 2048


def test_overrides_typed_and_dotted(cfg):
    out = apply_overrides(cfg, ["rl.beta_kl=0", "rl.lr=1e-5", "rl.use_baseline=false",
                                "rl.run_name=foo", "rl.save_steps=[10, 20]", "rl.max_kl=null"])
    assert out["rl"]["beta_kl"] == 0 and isinstance(out["rl"]["beta_kl"], int)
    assert out["rl"]["lr"] == 1e-5 and isinstance(out["rl"]["lr"], float)
    assert out["rl"]["use_baseline"] is False
    assert out["rl"]["run_name"] == "foo"
    assert out["rl"]["save_steps"] == [10, 20]
    assert out["rl"]["max_kl"] is None
    assert cfg["rl"]["beta_kl"] == 0.1  # original untouched


def test_overrides_reject_unknown(cfg):
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["rl.nope=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["rl.beta_kl"])


def test_repeated_set_flags_all_kept():
    p = add_config_args(argparse.ArgumentParser())
    a = p.parse_args(["--set", "rl.steps=5", "--set", "rl.beta_kl=0", "seed=3", "--set", "rl.lr=1e-4"])
    assert a.set == ["rl.steps=5", "rl.beta_kl=0", "seed=3", "rl.lr=1e-4"]
    assert a.config == "configs/v2.yaml"
    cfg = load_config(overrides=a.set)
    assert (cfg["rl"]["steps"], cfg["rl"]["beta_kl"], cfg["seed"], cfg["rl"]["lr"]) == (5, 0, 3, 1e-4)


def test_resolve_run_name(cfg):
    assert resolve_run_name(cfg) == "reinforce_beta0.1_seed0"
    assert resolve_run_name(apply_overrides(cfg, ["rl.run_name=x"])) == "x"


def test_get_device_explicit_and_auto():
    assert get_device({"device": "cpu"}).type == "cpu"
    assert get_device({"device": "auto"}).type in ("cuda", "mps", "cpu")
    assert get_device().type in ("cuda", "mps", "cpu")


def test_set_seed_reproducible():
    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    assert torch.equal(a, torch.rand(3))


def test_io_roundtrip(tmp_path):
    p = tmp_path / "sub" / "x.jsonl"
    write_jsonl(p, [{"a": 1}, {"b": "é"}])
    append_jsonl(p, {"c": 3})
    assert read_jsonl(p) == [{"a": 1}, {"b": "é"}, {"c": 3}]


def test_runlogger_replace_vs_resume(tmp_path, cfg):
    d = tmp_path / "run"
    lg = RunLogger(d, cfg)
    lg.log(1, {"loss": 0.5})
    lg.log_samples(1, ["p"], ["r"], [0.3], finished=[True])
    assert (d / "config.yaml").exists()
    assert len(read_jsonl(d / "metrics.jsonl")) == 1
    assert read_jsonl(d / "samples.jsonl")[0]["finished"] is True

    lg2 = RunLogger(d, cfg, resume=True)
    lg2.log(2, {"loss": 0.4})
    assert [r["step"] for r in read_jsonl(d / "metrics.jsonl")] == [1, 2]

    RunLogger(d, cfg)  # new run, same name: replaced
    assert not (d / "metrics.jsonl").exists()
    assert not (d / "samples.jsonl").exists()
    RunLogger(d, cfg).log(1, {"loss": 1.0})
    assert len(read_jsonl(d / "metrics.jsonl")) == 1


def test_per_token_kl_masking():
    logp = torch.tensor([[-1.0, -2.0, -3.0]])
    ref = torch.tensor([[-1.5, -1.0, -0.5]])
    mask = torch.tensor([[1, 1, 0]])
    kl = per_token_kl(logp, ref, mask)
    assert kl.shape == (1, 3)
    d = ref - logp  # k3 = exp(d) - 1 - d
    expected = (torch.exp(d) - 1 - d) * mask
    assert torch.allclose(kl, expected)
    assert kl[0, 2] == 0.0  # masked position


def test_per_token_kl_nonnegative_and_zero_for_identical():
    torch.manual_seed(0)
    logp = -torch.rand(4, 10) * 5
    ref = -torch.rand(4, 10) * 5
    mask = torch.ones(4, 10)
    assert (per_token_kl(logp, ref, mask) >= 0).all()
    assert torch.allclose(per_token_kl(logp, logp, mask), torch.zeros(4, 10))


def test_per_token_kl_unbiased_on_policy():
    # tokens sampled from p: E_p[k3] equals KL(p || q) exactly for a categorical
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.2, 0.5, 0.3])
    k3 = per_token_kl(p.log()[None], q.log()[None], torch.ones(1, 3))[0]
    assert torch.isclose((p * k3).sum(), (p * (p / q).log()).sum(), atol=1e-6)


def test_build_response_mask_finished_logic():
    eos, pad = 9, 0
    resp = torch.tensor([[5, 6, 9, 7, 7],    # EOS at 2 -> finished, junk after it masked
                         [5, 6, 7, 8, 5],    # no EOS -> hit cap
                         [9, 9, 9, 9, 9]])   # EOS immediately
    clean, mask, fin = build_response_mask(resp, eos, pad)
    assert fin == [True, False, True]
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]
    assert clean.tolist() == [[5, 6, 9, 0, 0], [5, 6, 7, 8, 5], [9, 0, 0, 0, 0]]


def test_rollout_dataclass_fields():
    f = Rollout.__dataclass_fields__
    for k in ("query_ids", "query_mask", "response_ids", "response_mask", "responses", "finished"):
        assert k in f


def test_fixture_prompts(tiny_prompts):
    assert len(tiny_prompts) == 8
