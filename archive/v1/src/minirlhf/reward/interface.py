from abc import ABC, abstractmethod

import torch


class RewardScorer(ABC):
    @abstractmethod
    def score(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Return float32 CPU tensor [B], no grad."""
