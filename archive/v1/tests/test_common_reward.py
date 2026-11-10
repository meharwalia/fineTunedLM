import pytest
import torch

from minirlhf.reward.factory import load_scorer
from minirlhf.reward.heuristic import HeuristicScorer


def test_deterministic_and_dtype():
    s = HeuristicScorer()
    a = s.score(["How do I boil an egg?"], ["Sure, boil the egg for ten minutes."])
    b = s.score(["How do I boil an egg?"], ["Sure, boil the egg for ten minutes."])
    assert a.dtype == torch.float32 and a.device.type == "cpu" and not a.requires_grad
    assert torch.equal(a, b) and a.shape == (1,)


def test_ordering():
    s = HeuristicScorer()
    prompt = "How do I boil an egg?"
    good = "Sure, happy to help! Boil the egg in water for ten minutes, then cool it."
    bad = "Whatever, stupid question. " + "eggs are eggs and eggs are eggs " * 15
    sc = s.score([prompt, prompt], [good, bad])
    assert sc[0] > sc[1]


def test_factory(cfg, monkeypatch):
    import sys
    import types

    assert isinstance(load_scorer(cfg, torch.device("cpu")), HeuristicScorer)
    # non-heuristic types lazily import minirlhf.reward.scorer.DistilBertScorer
    calls = {}

    class Fake:
        def __init__(self, *a):
            calls["args"] = a

    mod = types.ModuleType("minirlhf.reward.scorer")
    mod.DistilBertScorer = Fake
    monkeypatch.setitem(sys.modules, "minirlhf.reward.scorer", mod)
    cfg["reward"]["type"] = "distilbert"
    assert isinstance(load_scorer(cfg, torch.device("cpu")), Fake)
    assert calls["args"] == (cfg["reward"]["checkpoint"], torch.device("cpu"), 384, True)
