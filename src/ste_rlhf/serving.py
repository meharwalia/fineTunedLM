"""Serving registry (contract C7): process-wide base/tuned policies + reward scorer for the web app."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch

from ste_rlhf.common.config import load_config
from ste_rlhf.common.device import get_device
from ste_rlhf.common.generation import generate as _generate
from ste_rlhf.common.models import load_policy_from

ROOT = Path(__file__).resolve().parents[2]
REPETITION_PENALTY = 1.1

_lock = threading.Lock()


class _Registry:
    def __init__(self):
        self.device = None
        self.base = self.base_tok = None
        self.tuned = self.tuned_tok = None
        self.reward = None
        self.base_name = ""
        self.tuned_name = None
        self.tuned_info = None
        self.default_max_new_tokens = 1024
        self.max_new_tokens_limit = 2048
        self.system_prompts = {}


REGISTRY = _Registry()


def _res(v) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def resolve_paths(cfg: dict | None = None) -> dict:
    """Serving paths: config values (relative ones resolved against the project root); env overrides win.

    The base model is a hub name unless it exists as a local path (absolute, or relative to the root)."""
    cfg = cfg or load_config()
    s = cfg["serving"]
    base = os.environ.get("STE_SERVING_BASE") or s["base_model"]
    local = _res(base)
    if local.exists():
        base = str(local)
    return {"base": base,
            "tuned": _res(os.environ.get("STE_SERVING_TUNED") or s["tuned_path"]),
            "reward": _res(os.environ.get("STE_SERVING_REWARD") or s["reward_path"])}


def load(cfg: dict | None = None) -> None:
    from ste_rlhf.reward.scorer import load_scorer

    cfg = cfg or load_config()
    paths = resolve_paths(cfg)
    r = REGISTRY
    r.device = get_device(cfg)
    r.default_max_new_tokens = int(cfg["serving"].get("default_max_new_tokens", 1024))
    r.max_new_tokens_limit = int(cfg["serving"].get("max_new_tokens_limit", 2048))
    r.system_prompts = dict(cfg["serving"].get("system_prompts") or {})
    r.base, r.base_tok = load_policy_from(paths["base"], r.device)
    r.base.eval().requires_grad_(False)
    r.base_name = Path(paths["base"]).name
    r.tuned = r.tuned_tok = r.tuned_name = r.tuned_info = None
    tdir = paths["tuned"]
    if tdir.is_dir() and (tdir / "config.json").exists():
        r.tuned, r.tuned_tok = load_policy_from(str(tdir), r.device)
        r.tuned.eval().requires_grad_(False)
        info_p = tdir / "training_info.json"
        r.tuned_info = json.loads(info_p.read_text()) if info_p.exists() else None
        run = (r.tuned_info or {}).get("run_name")
        r.tuned_name = f"{tdir.name} ({run})" if run else tdir.name
    r.reward = None
    if paths["reward"].is_dir():
        r.reward = load_scorer(cfg, r.device, checkpoint=str(paths["reward"]))


def status() -> dict:
    r = REGISTRY
    return {"device": str(r.device) if r.device is not None else "cpu",
            "base": {"available": r.base is not None, "name": r.base_name},
            "tuned": {"available": r.tuned is not None, "name": r.tuned_name,
                      "training_info": r.tuned_info if r.tuned is not None else None},
            "reward_model": {"available": r.reward is not None}}


def generate(kind: str, prompt: str, settings) -> tuple:
    """Returns (response, finished, num_tokens, latency_ms, reward or None).

    Raises LookupError if the requested model is not loaded."""
    r = REGISTRY
    model, tok = (r.base, r.base_tok) if kind == "base" else (r.tuned, r.tuned_tok)
    if model is None:
        raise LookupError(f"{kind} model not available")
    cfg = {"max_new_tokens": max(1, min(int(settings.max_new_tokens), r.max_new_tokens_limit)),
           "temperature": float(settings.temperature), "repetition_penalty": REPETITION_PENALTY,
           "system": r.system_prompts.get(kind)}
    with _lock:
        t0 = time.perf_counter()
        ro = _generate(model, tok, [prompt], cfg, greedy=bool(settings.greedy))
        latency_ms = int(round((time.perf_counter() - t0) * 1000))
        text, finished = ro.responses[0], bool(ro.finished[0])
        n = int(ro.response_mask[0].sum().item()) - (1 if finished else 0)
        reward = None
        if r.reward is not None:
            with torch.no_grad():
                reward = float(r.reward.score([prompt], [text])[0])
    return text, finished, max(n, 0), latency_ms, reward
