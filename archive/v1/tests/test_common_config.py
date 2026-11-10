import pytest

from minirlhf.common.config import load_config, resolve_run_name


def test_default_values():
    c = load_config()
    assert c["rl"]["beta_kl"] == 0.05 and c["policy"]["max_new_tokens"] == 48
    assert c["data"]["mix"]["hh-rlhf"] == 1.0


def test_dotted_overrides():
    c = load_config(None, ["rl.beta_kl=0", "rl.use_baseline=false", "rl.lr=1e-4", "seed=3",
                           "reward.type=distilbert", "rl.run_name=foo"])
    assert c["rl"]["beta_kl"] == 0 and c["rl"]["use_baseline"] is False
    assert c["rl"]["lr"] == pytest.approx(1e-4) and c["seed"] == 3
    assert c["reward"]["type"] == "distilbert" and c["rl"]["run_name"] == "foo"
    assert load_config()["rl"]["beta_kl"] == 0.05  # defaults untouched


def test_bad_override():
    with pytest.raises(KeyError):
        load_config(None, ["rl.nope=1"])
    with pytest.raises(ValueError):
        load_config(None, ["novalue"])


def test_run_name():
    assert resolve_run_name(load_config()) == "reinforce_beta0.05_seed0"
    assert resolve_run_name(load_config(None, ["rl.beta_kl=0", "seed=2", "rl.use_baseline=false"])) == "reinforce_beta0_seed2_nobaseline"
    assert resolve_run_name(load_config(None, ["rl.run_name=x"])) == "x"
