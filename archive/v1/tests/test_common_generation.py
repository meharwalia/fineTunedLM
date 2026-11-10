import pytest
import torch

from minirlhf.common.generation import generate
from minirlhf.common.kl import per_token_kl
from minirlhf.common.logprobs import response_logprobs
from minirlhf.common.models import load_policy_from

PROMPTS = ["What is the capital of France?", "Explain photosynthesis in one very long sentence please, with many details."]


def test_kl_math():
    lp, lr = torch.tensor([[-1.0, -2.0]]), torch.tensor([[-1.5, -1.0]])
    m = torch.tensor([[1, 0]])
    assert torch.allclose(per_token_kl(lp, lr, m), torch.tensor([[0.5, 0.0]]))


@pytest.mark.slow
def test_generate_shapes_and_mask(cfg, small_policy):
    model, tok = small_policy
    ro = generate(model, tok, PROMPTS, cfg["policy"], greedy=False)
    B, Lr = ro.response_ids.shape
    assert B == 2 and 1 <= Lr <= 8
    assert ro.query_ids.shape == ro.query_mask.shape and ro.query_ids.shape[0] == 2
    assert ro.response_mask.shape == (2, Lr) and len(ro.responses) == 2
    assert tok.padding_side == "left"
    # left padding: last query token is real for every row
    assert (ro.query_mask[:, -1] == 1).all()
    for i in range(B):
        m = ro.response_mask[i]
        n = int(m.sum())
        assert n >= 1 and (m[:n] == 1).all() and (m[n:] == 0).all()
        eos_pos = (ro.response_ids[i] == tok.eos_token_id).nonzero().flatten()
        if len(eos_pos):
            assert n == int(eos_pos[0]) + 1  # includes the first EOS
        else:
            assert n == Lr
        assert ro.responses[i] == ro.responses[i].strip()
        assert "<|im_end|>" not in ro.responses[i]


@pytest.mark.slow
def test_eos_mask_forced(cfg, small_policy):
    model, tok = small_policy
    g = dict(cfg["policy"], max_new_tokens=40)
    ro = generate(model, tok, ["Say hi."], g, greedy=True)
    n = int(ro.response_mask.sum())
    ids = ro.response_ids[0]
    if (ids == tok.eos_token_id).any():
        assert n == int((ids == tok.eos_token_id).nonzero()[0]) + 1
        assert (ids[n:] == tok.pad_token_id).all()


@pytest.mark.slow
def test_greedy_deterministic(cfg, small_policy):
    model, tok = small_policy
    a = generate(model, tok, PROMPTS, cfg["policy"], greedy=True)
    b = generate(model, tok, PROMPTS, cfg["policy"], greedy=True)
    assert torch.equal(a.response_ids, b.response_ids)


@pytest.mark.slow
def test_logprobs_vs_manual(cfg, small_policy):
    model, tok = small_policy
    ro = generate(model, tok, PROMPTS, cfg["policy"], greedy=False)
    with torch.no_grad():
        lp, ent = response_logprobs(model, ro, return_entropy=True)
    assert lp.shape == ro.response_ids.shape == ent.shape
    assert (ent >= 0).all()
    for i in range(2):
        q = ro.query_ids[i][ro.query_mask[i].bool()]
        n = int(ro.response_mask[i].sum())
        r = ro.response_ids[i, :n]
        ids = torch.cat([q, r])[None]
        with torch.no_grad():
            logits = model(ids).logits[0].float().log_softmax(-1)
        for t in range(n):
            manual = logits[len(q) + t - 1, r[t]]
            assert torch.allclose(lp[i, t], manual, atol=1e-3), (i, t)


@pytest.mark.slow
def test_logprobs_gradients(cfg):
    model, tok = load_policy_from("HuggingFaceTB/SmolLM2-135M-Instruct", torch.device("cpu"))
    model.train()
    ro = generate(model, tok, PROMPTS[:1], cfg["policy"])
    assert model.training
    lp = response_logprobs(model, ro)
    assert lp.requires_grad
    (lp * ro.response_mask).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.slow
def test_kl_zero_for_identical(cfg, small_policy):
    from minirlhf.common.models import load_reference
    model, tok = small_policy
    ref = load_reference(cfg, torch.device("cpu"))
    ro = generate(model, tok, PROMPTS, cfg["policy"])
    with torch.no_grad():
        lp = response_logprobs(model, ro)
        lr = response_logprobs(ref, ro)
    kl = per_token_kl(lp, lr, ro.response_mask)
    assert kl.abs().max() < 1e-3
    assert not any(p.requires_grad for p in ref.parameters())
