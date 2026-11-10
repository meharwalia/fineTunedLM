import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_policy_from(path, device):
    """Load a trainable model + tokenizer from a hub name or checkpoint dir."""
    tok = _tok(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_policy(cfg, device):
    return load_policy_from(cfg["policy"]["name"], device)


def load_reference(cfg, device):
    model, _ = load_policy_from(cfg["policy"]["name"], device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
