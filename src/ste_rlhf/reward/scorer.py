"""Reward scorer (contract C4)."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import RewardModel, collate, encode_pair, load_meta, load_reward_tokenizer


def normalize_scores(raw: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (raw - mean) / max(std, 1e-8)


class Scorer:
    def __init__(self, model: RewardModel, tokenizer, max_length: int, mean: float = 0.0,
                 std: float = 1.0, normalize: bool = True, micro_batch: int = 8):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mean, self.std = float(mean), float(std)
        self.normalize = normalize
        self.micro_batch = micro_batch
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def raw_score(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        assert len(prompts) == len(responses)
        enc = [encode_pair(self.tokenizer, p, r, self.max_length) for p, r in zip(prompts, responses)]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))  # length bucketing
        out = torch.zeros(len(enc))
        self.model.eval()
        for s in range(0, len(order), self.micro_batch):
            idx = order[s:s + self.micro_batch]
            batch = {k: v.to(self.device) for k, v in collate(self.tokenizer, [enc[i] for i in idx]).items()}
            out[idx] = self.model(**batch).float().cpu()
        return out

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Float tensor [B] (CPU), z-normalised with train stats if normalize=True."""
        r = self.raw_score(prompts, responses)
        return normalize_scores(r, self.mean, self.std) if self.normalize else r


def load_scorer(cfg: dict, device, checkpoint=None) -> Scorer:
    path = Path(checkpoint or cfg["reward"]["checkpoint"])
    meta = load_meta(path)
    model = RewardModel.load(path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Scorer(model, load_reward_tokenizer(path), max_length=int(meta.get("max_length", cfg["reward"]["max_length"])),
                  mean=meta["train_reward_mean"], std=meta["train_reward_std"],
                  normalize=bool(cfg["reward"].get("normalize", True)))
