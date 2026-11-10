"""Slow tests with the real cached SmolLM2-135M-Instruct (run with HF_HUB_OFFLINE=1)."""
import pytest
import torch

from ste_rlhf.common import (generate, get_device, load_policy, load_reference, per_token_kl,
                            response_logprobs)
from ste_rlhf.common.generation import format_prompts

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def policy(cfg_module):
    dev = get_device(cfg_module)
    model, tok = load_policy(cfg_module, dev)
    return model, tok


@pytest.fixture(scope="module")
def cfg_module():
    from ste_rlhf.common import load_config
    return load_config()


def test_policy_train_reference_frozen(cfg_module, policy):
    model, tok = policy
    assert model.training
    assert next(model.parameters()).dtype == torch.float32
    ref = load_reference(cfg_module, get_device(cfg_module))
    assert not ref.training and not any(p.requires_grad for p in ref.parameters())
    assert tok.padding_side == "left"


def test_chat_template_system_prompt(policy):
    _, tok = policy
    text = format_prompts(tok, ["Say hi."])[0]
    print("\nCHAT TEMPLATE:\n" + text)
    assert text.rstrip().endswith("<|im_start|>assistant")
    assert "Say hi." in text
    # Documented finding: the template adds a default system turn when none is given.
    assert "<|im_start|>system" in text


def test_generate_cap_not_finished(cfg_module, policy):
    model, tok = policy
    ro = generate(model, tok, ["Explain how a car engine works in detail.",
                               "How do I set up a Python virtual environment?"],
                  {**cfg_module["policy"], "max_new_tokens": 8}, greedy=True)
    assert ro.finished == [False, False]
    assert ro.response_ids.shape[1] == 8 and ro.response_mask.sum().item() == 16
    assert model.training  # restored


def test_generate_short_question_finishes(cfg_module, policy):
    model, tok = policy
    ro = generate(model, tok, ["Say hi."], {**cfg_module["policy"], "max_new_tokens": 256}, greedy=True)
    assert ro.finished == [True]
    n = int(ro.response_mask[0].sum())
    assert ro.response_ids[0, n - 1].item() == tok.eos_token_id
    assert ro.responses[0]


def test_response_logprobs(cfg_module, policy):
    model, tok = policy
    prompts = ["Say hi.", "Explain what a firewall does in two sentences."]  # different lengths -> left padding
    ro = generate(model, tok, prompts, {**cfg_module["policy"], "max_new_tokens": 24}, greedy=True)
    with torch.no_grad():
        lp, ent = response_logprobs(model, ro, return_entropy=True)
        lp_mb = response_logprobs(model, ro, micro_batch=1)
    B, Lr = ro.response_ids.shape
    assert lp.shape == ent.shape == (B, Lr)
    assert torch.allclose(lp, lp_mb, atol=1e-3)
    assert not ent.requires_grad
    m = ro.response_mask.bool()
    # Greedy tokens are the argmax, so their logprob is high (and never below -ln(V)).
    assert lp[m].mean().item() > -1.5
    assert (ent[m] >= 0).all()
    # KL against an identical frozen reference is ~0
    ref = load_reference(cfg_module, get_device(cfg_module))
    with torch.no_grad():
        lp_ref = response_logprobs(ref, ro)
    assert per_token_kl(lp, lp_ref, ro.response_mask).abs().max().item() < 1e-3
