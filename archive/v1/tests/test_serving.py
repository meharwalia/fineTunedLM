"""Integration tests for minirlhf.serving (real models; slow)."""
from pathlib import Path

import pytest

from minirlhf import serving

pytestmark = pytest.mark.slow

TUNED = "/Users/meharwalia/RL/.claude/worktrees/agent-a95dc7e21c8583ea4/checkpoints/rlhf_final"
REWARD = "/Users/meharwalia/RL/.claude/worktrees/agent-ac28ea0c29ba88d28/checkpoints/reward_model"
needs_assets = pytest.mark.skipif(not (Path(TUNED).is_dir() and Path(REWARD).is_dir()),
                                  reason="test checkpoints not available")
MISSING = "/nonexistent/minirlhf_test_dir"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DEVICE", "cpu")
    serving.reset()
    yield
    serving.reset()


def _missing(monkeypatch):
    monkeypatch.setenv("TUNED_MODEL_PATH", MISSING + "/tuned")
    monkeypatch.setenv("REWARD_MODEL_PATH", MISSING + "/rm")


def test_base_answer_and_missing_models(monkeypatch):
    _missing(monkeypatch)
    text, n = serving.answer("base", "How can I stay focused while studying?", 16, True, 0.7)
    assert text.strip()
    assert 0 < n <= 16
    st = serving.status()
    assert st["device"] == "cpu"
    assert st["base"] == {"available": True, "name": "SmolLM2-135M-Instruct"}
    assert st["tuned"] == {"available": False, "name": None, "training_info": None}
    assert st["reward_model"] == {"available": False}
    with pytest.raises(serving.ModelUnavailable):
        serving.answer("tuned", "hi", 8, True, 0.7)
    assert serving.score("hi", text) is None


def test_sampling_path(monkeypatch):
    _missing(monkeypatch)
    text, n = serving.answer("base", "Tell me a fun fact.", 16, False, 0.7)
    assert isinstance(text, str) and n <= 16


@needs_assets
def test_tuned_and_reward(monkeypatch):
    monkeypatch.setenv("TUNED_MODEL_PATH", TUNED)
    monkeypatch.setenv("REWARD_MODEL_PATH", REWARD)
    st = serving.status()
    assert st["tuned"]["available"] is True
    assert st["tuned"]["name"] == "rlhf_final (reinforce_beta0.05_seed0)"
    assert st["tuned"]["training_info"]["algo"] == "reinforce"
    assert st["reward_model"]["available"] is True
    text, n = serving.answer("tuned", "How can I stay focused while studying?", 16, True, 0.7)
    assert text.strip() and n > 0
    assert isinstance(serving.score("How can I stay focused while studying?", text), float)


def test_relative_paths_resolve_against_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("TUNED_MODEL_PATH", raising=False)
    monkeypatch.delenv("REWARD_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    s = serving.resolve_settings()
    root = Path(serving.__file__).resolve().parents[2]
    assert (root / "webapp").is_dir()
    assert s["tuned_path"] == root / "checkpoints" / "rlhf_final"
    assert s["reward_path"] == root / "checkpoints" / "reward_model"
    monkeypatch.setenv("TUNED_MODEL_PATH", "some/rel")
    assert serving.resolve_settings()["tuned_path"] == root / "some" / "rel"
    monkeypatch.setenv("TUNED_MODEL_PATH", "/abs/path")
    assert serving.resolve_settings()["tuned_path"] == Path("/abs/path")


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("BASE_MODEL", "some/Other-Model")
    assert serving.resolve_settings()["base_model"] == "some/Other-Model"
    monkeypatch.delenv("BASE_MODEL")
    assert serving.resolve_settings()["base_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
