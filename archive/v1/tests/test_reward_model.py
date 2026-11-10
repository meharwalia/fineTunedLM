import json

import pytest
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, DistilBertConfig

from minirlhf.common.io import read_jsonl
from minirlhf.reward.model import bt_loss, save_checkpoint
from minirlhf.reward.scorer import DistilBertScorer
from minirlhf.reward.train import hh_pair, parse_hh_single_turn


def test_bt_loss_direction():
    good = bt_loss(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0]))
    bad = bt_loss(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))
    tie = bt_loss(torch.zeros(2), torch.zeros(2))
    assert good < tie < bad
    assert tie.item() == pytest.approx(0.6931, abs=1e-3)


def test_parse_hh():
    t = "\n\nHuman: What is 2+2?\n\nAssistant: It is 4."
    assert parse_hh_single_turn(t) == ("What is 2+2?", "It is 4.")
    multi = t + "\n\nHuman: and 3+3?\n\nAssistant: 6."
    assert parse_hh_single_turn(multi) is None
    long = "\n\nHuman: q\n\nAssistant: " + " ".join(["w"] * 130)
    assert hh_pair(long, t) is None  # prompt mismatch / long
    a = "\n\nHuman: q\n\nAssistant: good"
    b = "\n\nHuman: q\n\nAssistant: bad"
    assert hh_pair(a, b)["chosen"] == "good"
    assert hh_pair(a, long) is None


@pytest.fixture
def tiny_ckpt(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(60)] + \
            ["happy", "to", "help", "whatever", "short", "answer", "how", "do", "i", "boil", "an", "egg"]
    vf = tmp_path / "vocab.txt"
    vf.write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(vf), do_lower_case=True)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=2, n_heads=2,
                           max_position_embeddings=64, num_labels=1)
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_config(cfg)
    d = tmp_path / "ckpt"
    meta = {"base_model": "tiny", "max_length": 32, "val_accuracy": 0.9, "per_source_accuracy": {},
            "train_reward_mean": 0.5, "train_reward_std": 2.0, "data_counts": {}}
    save_checkpoint(model, tok, d, meta)
    return d


def test_meta_keys(tiny_ckpt):
    meta = json.loads((tiny_ckpt / "rm_meta.json").read_text())
    assert set(meta) == {"base_model", "max_length", "val_accuracy", "per_source_accuracy",
                         "train_reward_mean", "train_reward_std", "data_counts"}


def test_scorer_shape_dtype_and_normalisation(tiny_ckpt):
    prompts = ["how do i boil an egg"] * 5
    resps = ["happy to help", "whatever", "short answer", "an egg", "happy " * 100]
    raw = DistilBertScorer(tiny_ckpt, "cpu", 32, False)
    norm = DistilBertScorer(tiny_ckpt, "cpu", 32, True)
    s_raw, s_norm = raw.score(prompts, resps), norm.score(prompts, resps)
    assert s_raw.shape == (5,) and s_raw.dtype == torch.float32 and s_raw.device.type == "cpu"
    assert not s_raw.requires_grad
    assert torch.allclose(s_norm, (s_raw - 0.5) / 2.0, atol=1e-5)
    # batching does not change results
    small = DistilBertScorer(tiny_ckpt, "cpu", 32, False, )
    small.batch_size = 2
    assert torch.allclose(small.score(prompts, resps), s_raw, atol=1e-5)
    assert raw.score([], []).shape == (0,)


def test_factory_builds_scorer(tiny_ckpt, cfg):
    from minirlhf.reward.factory import load_scorer
    cfg["reward"].update(type="distilbert", checkpoint=str(tiny_ckpt), max_length=32)
    sc = load_scorer(cfg, torch.device("cpu"))
    assert sc.score(["how"], ["happy"]).shape == (1,)


@pytest.mark.slow
def test_overfit_fixture_pairs(fixtures_dir):
    from minirlhf.reward.train import pair_margins, train_reward_model
    rows = read_jsonl(fixtures_dir / "prefs_small.jsonl")
    assert len(rows) == 20
    model, tok, meta, _ = train_reward_model(
        rows, rows, base_model="distilbert-base-uncased", device=torch.device("cpu"),
        max_length=64, lr=5e-5, epochs=8, batch_size=10, weight_decay=0.0, log=lambda *_: None)
    margins = pair_margins(model, tok, rows, 64, torch.device("cpu"))
    assert (margins > 0).mean() == 1.0
    assert meta["val_accuracy"] == 1.0
