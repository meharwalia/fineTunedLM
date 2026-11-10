import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_policy_from(path, device):
    """Trainable model (train mode) + tokenizer, from a hub name or checkpoint dir.
    float32 everywhere (including MPS): half precision is unreliable for training there."""
    tok = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.train()
    return model, tok


def load_policy(cfg, device):
    """Returns (model, tokenizer); the model is in train mode."""
    return load_policy_from(cfg["policy"]["name"], device)


def load_reference(cfg, device):
    """Frozen copy of the base policy: eval mode, requires_grad False."""
    model, _ = load_policy_from(cfg["policy"]["name"], device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
