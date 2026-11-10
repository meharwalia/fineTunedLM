"""Prompt formatting and generation (contract C4)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    prompts: list            # raw prompt strings
    query_ids: torch.Tensor  # [B, Lq] left-padded
    query_mask: torch.Tensor # [B, Lq]
    response_ids: torch.Tensor   # [B, Lr] right-padded with pad
    response_mask: torch.Tensor  # [B, Lr] 1 up to and including first EOS
    responses: list          # decoded texts, special tokens stripped, .strip()'d


def format_prompts(tokenizer, prompts) -> list[str]:
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]


def _g(gen_cfg, key, default=None):
    v = gen_cfg.get(key, default) if gen_cfg is not None else default
    return default if v is None else v


@torch.no_grad()
def generate(model, tokenizer, prompts, gen_cfg, greedy: bool = False) -> Rollout:
    """gen_cfg keys: max_new_tokens, temperature, top_p, repetition_penalty, max_prompt_tokens.
    A full cfg['policy'] dict works as gen_cfg."""
    prompts = list(prompts)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = format_prompts(tokenizer, prompts)
    tokenizer.truncation_side = "left"  # keep the tail (generation prompt) if too long
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=int(_g(gen_cfg, "max_prompt_tokens", 128)), add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    kwargs = dict(max_new_tokens=int(_g(gen_cfg, "max_new_tokens", 48)),
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

    resp = out[:, input_ids.shape[1]:]
    eos = tokenizer.eos_token_id
    is_eos = resp == eos
    # position of first EOS (or Lr if none); mask covers up to and including it
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((resp.shape[0],), resp.shape[1], device=resp.device))
    pos = torch.arange(resp.shape[1], device=resp.device)[None, :]
    mask = (pos <= first[:, None]).long()
    resp = torch.where(mask.bool(), resp, torch.full_like(resp, tokenizer.pad_token_id))
    responses = [tokenizer.decode(r[m.bool()], skip_special_tokens=True).strip() for r, m in zip(resp, mask)]
    return Rollout(prompts=prompts, query_ids=input_ids, query_mask=attn,
                   response_ids=resp, response_mask=mask, responses=responses)
