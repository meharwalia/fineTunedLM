"""Fast serving tests: fakes injected into the registry, no real models."""
from types import SimpleNamespace

import pytest
import torch

from ste_rlhf import serving
from webapp import model_interface as mi


class FakeScorer:
    def score(self, prompts, responses):
        return torch.tensor([1.5] * len(prompts))


@pytest.fixture
def reg(monkeypatch):
    r = serving._Registry()
    monkeypatch.setattr(serving, "REGISTRY", r)
    calls = []

    def fake_generate(model, tok, prompts, cfg, greedy=False):
        calls.append((model, list(prompts), dict(cfg), greedy))
        assert len(prompts) == 1
        finished = model != "slow"
        mask = torch.tensor([[1, 1, 1, 1] if finished else [1, 1, 1]])
        return SimpleNamespace(responses=[f"answer from {model}"], finished=[finished], response_mask=mask)

    monkeypatch.setattr(serving, "_generate", fake_generate)
    r.device, r.base, r.base_tok, r.base_name = "cpu", "base", object(), "SmolLM2-135M-Instruct"
    r.max_new_tokens_limit = 100
    return r, calls


S = mi.GenerationSettings


def test_generate_base_with_reward(reg):
    r, calls = reg
    r.reward = FakeScorer()
    text, finished, n, lat, reward = serving.generate("base", "hi", S(max_new_tokens=50))
    assert (text, finished, n, reward) == ("answer from base", True, 3, 1.5)  # EOS excluded
    assert lat >= 0
    _, _, cfg, greedy = calls[0]
    assert greedy and cfg["max_new_tokens"] == 50 and cfg["repetition_penalty"] == 1.1


def test_clamp_sampling_and_no_reward(reg):
    r, calls = reg
    _, _, _, _, reward = serving.generate("base", "hi", S(max_new_tokens=5000, greedy=False, temperature=0.5))
    assert reward is None
    _, _, cfg, greedy = calls[0]
    assert cfg["max_new_tokens"] == 100 and cfg["temperature"] == 0.5 and not greedy


def test_unfinished_counts_all_tokens(reg):
    r, _ = reg
    r.base = "slow"
    _, finished, n, _, _ = serving.generate("base", "hi", S())
    assert finished is False and n == 3


def test_tuned_unavailable(reg):
    with pytest.raises(LookupError):
        serving.generate("tuned", "hi", S())
    with pytest.raises(mi.ModelNotAvailableError):
        mi.generate_response("hi", "tuned", S())


def test_status_shape(reg):
    r, _ = reg
    st = mi.get_status()
    assert st == {"device": "cpu", "base": {"available": True, "name": "SmolLM2-135M-Instruct"},
                  "tuned": {"available": False, "name": None, "training_info": None},
                  "reward_model": {"available": False}}
    r.tuned, r.tuned_tok, r.tuned_name = "tuned", object(), "rlhf_final (run)"
    r.tuned_info = {"algo": "REINFORCE", "steps": 3}
    r.reward = FakeScorer()
    st = mi.get_status()
    assert st["tuned"] == {"available": True, "name": "rlhf_final (run)", "training_info": {"algo": "REINFORCE", "steps": 3}}
    assert st["reward_model"]["available"] is True


def test_model_interface_generate_response(reg):
    r, _ = reg
    r.reward = FakeScorer()
    resp = mi.generate_response("hi", "base", S(max_new_tokens=20))
    assert resp.model_kind == "base" and resp.model_name == "SmolLM2-135M-Instruct"
    assert resp.response == "answer from base" and resp.finished is True and resp.reward == 1.5


def test_resolve_paths_default_and_env(monkeypatch):
    for k in ("STE_SERVING_BASE", "STE_SERVING_TUNED", "STE_SERVING_REWARD"):
        monkeypatch.delenv(k, raising=False)
    cfg = {"serving": {"base_model": "org/hub-model", "tuned_path": "checkpoints/rlhf_final",
                       "reward_path": "checkpoints/reward_model"}}
    p = serving.resolve_paths(cfg)
    assert p["base"] == "org/hub-model"
    assert p["tuned"] == serving.ROOT / "checkpoints" / "rlhf_final"
    assert p["reward"] == serving.ROOT / "checkpoints" / "reward_model"
    assert (serving.ROOT / "src").is_dir()
    monkeypatch.setenv("STE_SERVING_BASE", "other/model")
    monkeypatch.setenv("STE_SERVING_TUNED", "/abs/tuned")
    monkeypatch.setenv("STE_SERVING_REWARD", "rel/reward")
    p = serving.resolve_paths(cfg)
    assert p["base"] == "other/model" and str(p["tuned"]) == "/abs/tuned"
    assert p["reward"] == serving.ROOT / "rel" / "reward"
