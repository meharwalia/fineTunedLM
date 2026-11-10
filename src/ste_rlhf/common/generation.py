"""Prompt formatting and generation (contract C3).

SmolLM2 chat-template note: its template injects a default system message when the
messages list has no system turn (see tests/test_slow_models.py, which prints and
checks it). We pass only the user turn, so the model sees that default system text
and no STE hint.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    prompts: list                # raw prompt strings
    query_ids: torch.Tensor      # [B, Lq] left-padded
    query_mask: torch.Tensor     # [B, Lq]
    response_ids: torch.Tensor   # [B, Lr] right-padded with pad
    response_mask: torch.Tensor  # [B, Lr] 1 up to and including first EOS
    responses: list              # decoded, special tokens stripped, .strip()'d
    finished: list               # bool per row: EOS appeared before the cap


def format_prompts(tokenizer, prompts, system: str | None = None) -> list[str]:
    """`system` replaces the template's default system message (web demo only; training and
    eval leave it None)."""
    head = [{"role": "system", "content": system}] if system else []
    return [
        tokenizer.apply_chat_template(head + [{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]


def _g(gen_cfg, key, default=None):
    v = gen_cfg.get(key, default) if gen_cfg is not None else default
    return default if v is None else v


def build_response_mask(resp: torch.Tensor, eos_id: int, pad_id: int):
    """resp [B, Lr] generated ids. Returns (clean_resp, mask, finished list).
    mask covers tokens up to and including the first EOS."""
    is_eos = resp == eos_id
    has = is_eos.any(1)
    first = torch.where(has, is_eos.float().argmax(1),
                        torch.full((resp.shape[0],), resp.shape[1], device=resp.device))
    pos = torch.arange(resp.shape[1], device=resp.device)[None, :]
    mask = (pos <= first[:, None]).long()
    resp = torch.where(mask.bool(), resp, torch.full_like(resp, pad_id))
    return resp, mask, [bool(x) for x in has.tolist()]


@torch.no_grad()
def generate(model, tokenizer, prompts, gen_cfg, greedy: bool = False) -> Rollout:
    """gen_cfg keys: max_new_tokens, temperature, top_p, repetition_penalty,
    max_prompt_tokens (a full cfg['policy'] dict works)."""
    prompts = list(prompts)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = format_prompts(tokenizer, prompts, _g(gen_cfg, "system"))
    tokenizer.truncation_side = "left"  # keep the tail (generation prompt) if too long
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=int(_g(gen_cfg, "max_prompt_tokens", 320)), add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    kwargs = dict(max_new_tokens=int(_g(gen_cfg, "max_new_tokens", 512)),
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    rp = float(_g(gen_cfg, "repetition_penalty", 1.0))
    if rp != 1.0:
        kwargs["repetition_penalty"] = rp
    if greedy:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kwargs.update(do_sample=True, temperature=float(_g(gen_cfg, "temperature", 0.9)),
                      top_p=float(_g(gen_cfg, "top_p", 0.95)), top_k=0)
    was_training = model.training
    model.eval()
    out = model.generate(input_ids=input_ids, attention_mask=attn, **kwargs)
    model.train(was_training)

    resp, mask, finished = build_response_mask(out[:, input_ids.shape[1]:],
                                               tokenizer.eos_token_id, tokenizer.pad_token_id)
    responses = [tokenizer.decode(r[m.bool()], skip_special_tokens=True).strip()
                 for r, m in zip(resp, mask)]
    return Rollout(prompts=prompts, query_ids=input_ids, query_mask=attn,
                   response_ids=resp, response_mask=mask, responses=responses, finished=finished)
