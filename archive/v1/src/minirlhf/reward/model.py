"""Reward model core: DistilBERT sequence classifier with one logit, Bradley-Terry loss."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

META_NAME = "rm_meta.json"


def build_model(base_model: str):
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    tok = AutoTokenizer.from_pretrained(base_model)
    return model, tok


def encode_pairs(tokenizer, prompts, responses, max_length, device=None):
    """C3 rule: sentence-pair encoding, truncation=longest_first."""
    enc = tokenizer(list(prompts), list(responses), truncation="longest_first",
                    max_length=max_length, padding=True, return_tensors="pt")
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def raw_rewards(model, tokenizer, prompts, responses, max_length, device=None) -> torch.Tensor:
    """Un-normalised logits [B] (keeps grad if enabled by the caller)."""
    device = device or next(model.parameters()).device
    enc = encode_pairs(tokenizer, prompts, responses, max_length, device)
    return model(**enc).logits[:, 0]


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry: -logsigmoid(r_c - r_r).mean()."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def save_checkpoint(model, tokenizer, out_dir, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    keys = ["base_model", "max_length", "val_accuracy", "per_source_accuracy",
            "train_reward_mean", "train_reward_std", "data_counts"]
    with open(out / META_NAME, "w") as f:
        json.dump({k: meta[k] for k in keys}, f, indent=2)


def load_meta(checkpoint) -> dict:
    with open(Path(checkpoint) / META_NAME) as f:
        return json.load(f)
