"""Hand-written reward used only to develop the trainer before the real reward model exists."""
import re

import torch

from .interface import RewardScorer

POLITE = ("please", "thank", "thanks", "sure", "happy to", "glad", "certainly", "of course",
          "welcome", "hope this helps", "sorry")
RUDE = ("stupid", "idiot", "shut up", "dumb", "hate you", "go away", "whatever")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "it", "for", "on", "what", "how",
         "why", "do", "does", "i", "you", "me", "my", "can", "with", "that", "this", "be"}


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class HeuristicScorer(RewardScorer):
    def __init__(self, *args, **kwargs):
        pass

    def _one(self, prompt, response):
        words = _words(response)
        low = response.lower()
        s = 0.0
        s += 0.5 * min(sum(low.count(p) for p in POLITE), 3)
        s -= 1.0 * min(sum(low.count(p) for p in RUDE), 3)
        pw = {w for w in _words(prompt) if w not in _STOP}
        if pw:
            s += 2.0 * len(pw & set(words)) / len(pw)
        if len(words) > 60:
            s -= 0.05 * (len(words) - 60)
        if len(words) < 3:
            s -= 1.0
        if len(words) >= 4:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            if trigrams:
                s -= 2.0 * (1 - len(set(trigrams)) / len(trigrams))
        return s

    def score(self, prompts, responses):
        vals = [self._one(p, r) for p, r in zip(prompts, responses)]
        return torch.tensor(vals, dtype=torch.float32)
