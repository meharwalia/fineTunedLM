"""DistilBertScorer: implements RewardScorer with the C3 scoring rule."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from minirlhf.common.device import get_device
from minirlhf.reward.interface import RewardScorer
from minirlhf.reward.model import encode_pairs, load_meta


class DistilBertScorer(RewardScorer):
    def __init__(self, checkpoint, device=None, max_length=None, normalize=True, batch_size=32):
        self.device = torch.device(device) if device is not None else get_device()
        meta = load_meta(checkpoint)
        self.max_length = int(max_length or meta["max_length"])
        self.normalize = bool(normalize)
        self.mean = float(meta["train_reward_mean"])
        self.std = float(meta["train_reward_std"]) or 1.0
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        assert len(prompts) == len(responses)
        out = []
        for i in range(0, len(prompts), self.batch_size):
            enc = encode_pairs(self.tokenizer, prompts[i:i + self.batch_size],
                               responses[i:i + self.batch_size], self.max_length, self.device)
            out.append(self.model(**enc).logits[:, 0].float().cpu())
        s = torch.cat(out) if out else torch.zeros(0)
        if self.normalize:
            s = (s - self.mean) / self.std
        return s.to(torch.float32)
