"""ModernBERT reward model: backbone + Linear(hidden, 1) on the mean-pooled hidden states.

Pooling choice: masked mean pooling. ModernBERT's [CLS] token is only weakly trained as a
sequence summary (no next-sentence objective), and ModernBERT sentence-embedding models use
mean pooling. Mean pooling also sees the tail of long (up to 2048-token) answers directly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MAX_PROMPT_TOKENS = 256


def encode_pair(tokenizer, prompt: str, response: str, max_length: int,
                max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> list[int]:
    """[CLS] prompt [SEP] response [SEP] as token ids, at most max_length long.

    Too long: the prompt is cut to max_prompt_tokens (head kept) first, then the MIDDLE of the
    response is removed (head and tail kept, split evenly)."""
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r = tokenizer(response, add_special_tokens=False)["input_ids"]
    n_special = 3  # [CLS] prompt [SEP] response [SEP]
    avail = max_length - n_special
    if len(p) + len(r) > avail:
        p = p[:min(max_prompt_tokens, max(avail // 2, 1))]
    budget = max(avail - len(p), 1)
    if len(r) > budget:
        head = (budget + 1) // 2
        tail = budget - head
        r = r[:head] + (r[len(r) - tail:] if tail > 0 else [])
    return [tokenizer.cls_token_id] + p + [tokenizer.sep_token_id] + r + [tokenizer.sep_token_id]


def collate(tokenizer, id_lists: list[list[int]]) -> dict[str, torch.Tensor]:
    """Dynamic padding to the longest sequence in the batch."""
    L = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), L), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry: -log sigmoid(r_c - r_r), mean over pairs."""
    return -nn.functional.logsigmoid(r_chosen - r_rejected).mean()


class RewardModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        hidden_size = hidden_size or backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

    def save(self, path, tokenizer=None, meta: dict | None = None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone")
        torch.save({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, path / "head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(path / "tokenizer")
        if meta is not None:
            (path / "rm_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        model = cls(AutoModel.from_pretrained(path / "backbone", dtype=torch.float32))
        model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        return model


def load_reward_tokenizer(path_or_name):
    p = Path(str(path_or_name))
    if (p / "tokenizer").is_dir():
        return AutoTokenizer.from_pretrained(str(p / "tokenizer"))
    return AutoTokenizer.from_pretrained(str(path_or_name))


def load_meta(path) -> dict:
    return json.loads((Path(path) / "rm_meta.json").read_text())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
