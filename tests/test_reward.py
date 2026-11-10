import copy
import json

import pytest
import torch
from transformers import AutoModel, AutoTokenizer, ModernBertConfig

from ste_rlhf.common import read_jsonl
from ste_rlhf.reward.model import RewardModel, bt_loss, collate, encode_pair
from ste_rlhf.reward.scorer import Scorer, load_scorer, normalize_scores

BASE = "answerdotai/ModernBERT-base"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(BASE)


def tiny_backbone(vocab):
    c = ModernBertConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, global_attn_every_n_layers=2, max_position_embeddings=256)
    return AutoModel.from_config(c)


def test_short_pair_matches_tokenizer(tok):
    ids = encode_pair(tok, "How do I reset?", "Hold the button.", 2048)
    assert ids == tok("How do I reset?", "Hold the button.")["input_ids"]


def test_truncation_keeps_head_and_tail(tok):
    prompt = "word " * 400
    resp_tokens = tok(" ".join(f"w{i}" for i in range(3000)), add_special_tokens=False)["input_ids"]
    resp = tok.decode(resp_tokens)
    ids = encode_pair(tok, prompt, resp, 512)
    assert len(ids) <= 512
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    p_len = ids.index(tok.sep_token_id) - 1
    assert p_len <= 256  # prompt truncated first
    body = ids[ids.index(tok.sep_token_id) + 1:-1]
    r = tok(resp, add_special_tokens=False)["input_ids"]
    h = len(body) // 2 + len(body) % 2
    assert body[:h] == r[:h]                      # head kept
    assert body[h:] == r[len(r) - (len(body) - h):]  # tail kept
    assert len(body) < len(r)                     # middle cut
    assert len(ids) == 512


def test_truncation_noop_when_fits(tok):
    ids = encode_pair(tok, "q " * 10, "a " * 10, 512)
    assert len(ids) == len(tok("q " * 10, "a " * 10)["input_ids"])


def test_bt_loss_sign():
    good = bt_loss(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 0.0]))
    bad = bt_loss(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 3.0]))
    assert good < bad
    assert bt_loss(torch.zeros(3), torch.zeros(3)).item() == pytest.approx(0.6931, abs=1e-3)
    c = torch.tensor([0.1], requires_grad=True)
    bt_loss(c, torch.tensor([0.5])).backward()
    assert c.grad.item() < 0  # gradient descent raises r_chosen


def test_normalisation_math():
    raw = torch.tensor([1.0, 3.0, 5.0, 7.0])
    z = normalize_scores(raw, float(raw.mean()), float(raw.std()))
    assert z.mean().abs() < 1e-6 and z.std() == pytest.approx(1.0, abs=1e-5)


def test_tiny_model_and_scorer_roundtrip(tok, tmp_path):
    torch.manual_seed(0)
    m = RewardModel(tiny_backbone(len(tok))).eval()
    meta = {"max_length": 128, "train_reward_mean": 0.5, "train_reward_std": 2.0}
    m.save(tmp_path / "rm", tok, meta)
    cfg = {"reward": {"checkpoint": str(tmp_path / "rm"), "max_length": 128, "normalize": True}}
    s = load_scorer(cfg, torch.device("cpu"))
    ps, rs = ["q1", "q2 longer", "q3"], ["a b c", "d " * 300, "e"]
    out = s.score(ps, rs)
    assert out.shape == (3,)
    ref = Scorer(m, tok, 128, 0.5, 2.0, True, micro_batch=2).score(ps, rs)
    assert torch.allclose(out, ref, atol=1e-5)  # order-independent, batching-independent
    raw = Scorer(m, tok, 128, normalize=False).score(ps, rs)
    assert torch.allclose((raw - 0.5) / 2.0, out, atol=1e-5)


def test_padding_does_not_change_score(tok):
    torch.manual_seed(0)
    m = RewardModel(tiny_backbone(len(tok))).eval()
    a = encode_pair(tok, "q", "short", 128)
    b = encode_pair(tok, "q", "much longer answer " * 10, 128)
    with torch.no_grad():
        alone = m(**collate(tok, [a]))
        both = m(**collate(tok, [a, b]))
    assert torch.allclose(alone[0], both[0], atol=1e-4)


@pytest.mark.slow
def test_train_one_epoch_and_roundtrip(fixtures_dir, tmp_path, cfg):
    from ste_rlhf.reward.train import train
    pairs = read_jsonl(fixtures_dir / "pairs_small.jsonl")
    cfg = copy.deepcopy(cfg)
    cfg["reward"]["max_length"] = 512
    cfg["reward_training"].update(epochs=1, batch_size=4, lr=3.0e-5)
    ckpt = tmp_path / "rm"
    meta = train(cfg, train_pairs=pairs, val_pairs=pairs, out_dir=ckpt, log_dir=tmp_path / "logs")
    assert (tmp_path / "logs" / "metrics.jsonl").exists()
    assert {"train_reward_mean", "train_reward_std", "val_accuracy", "val_accuracy_by_type"} <= set(meta)
    cfg["reward"]["checkpoint"] = str(ckpt)
    dev = torch.device("cpu")
    s = load_scorer(cfg, dev)
    ps = [p["prompt"] for p in pairs]
    c = s.score(ps, [p["chosen"] for p in pairs])
    r = s.score(ps, [p["rejected"] for p in pairs])
    assert c.mean() > r.mean()
    # save/load round trip is identical
    ckpt2 = tmp_path / "rm2"
    s.model.save(ckpt2, s.tokenizer, json.loads((ckpt / "rm_meta.json").read_text()))
    cfg["reward"]["checkpoint"] = str(ckpt2)
    c2 = load_scorer(cfg, dev).score(ps, [p["chosen"] for p in pairs])
    assert torch.allclose(c, c2, atol=1e-6)
