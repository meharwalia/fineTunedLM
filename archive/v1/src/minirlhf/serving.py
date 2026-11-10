"""Model registry for serving (used by webapp/model_interface.py).

Loads the base model, the tuned model (if its directory exists) and the reward model
(if ``rm_meta.json`` exists). A missing tuned/reward model never raises.

Settings precedence: environment variables (BASE_MODEL, TUNED_MODEL_PATH,
REWARD_MODEL_PATH, DEVICE; same names as ``webapp/config.py``) override the ``serving``
section of ``configs/default.yaml``. Relative paths are resolved against the project
root (the directory containing ``webapp/``), never the current working directory.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from minirlhf.common.config import load_config
from minirlhf.common.device import get_device
from minirlhf.common.generation import generate
from minirlhf.common.models import load_policy_from

log = logging.getLogger("minirlhf.serving")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# C4 demo settings
REPETITION_PENALTY = 1.1
TOP_P = 0.95
MAX_PROMPT_TOKENS = 128


class ModelUnavailable(Exception):
    """Requested model (e.g. the tuned one) is missing or failed to load."""


def resolve_path(p) -> Path:
    """Absolute path; relative paths are resolved against the project root."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_settings(env=None) -> dict:
    """Merge the `serving` section of configs/default.yaml with environment overrides."""
    env = os.environ if env is None else env
    cfg = load_config()
    s = dict(cfg.get("serving") or {})
    base = env.get("BASE_MODEL") or s.get("base_model") or cfg["policy"]["name"]
    tuned = env.get("TUNED_MODEL_PATH") or s.get("tuned_path") or "checkpoints/rlhf_final"
    reward = env.get("REWARD_MODEL_PATH") or s.get("reward_path") or "checkpoints/reward_model"
    device = env.get("DEVICE") or cfg.get("device") or "auto"
    return {"base_model": base, "tuned_path": resolve_path(tuned),
            "reward_path": resolve_path(reward), "device": device}


@dataclass
class _Entry:
    model: object
    tokenizer: object
    name: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    training_info: Optional[dict] = None


class Registry:
    def __init__(self, settings: dict):
        self.settings = settings
        self.device = get_device({"device": settings["device"]})
        self.base: Optional[_Entry] = None
        self.tuned: Optional[_Entry] = None
        self.tuned_error: Optional[str] = None
        self.scorer = None
        self.scorer_lock = threading.Lock()
        self._load()

    def _load(self):
        s = self.settings
        base_name = s["base_model"]
        model, tok = load_policy_from(base_name, self.device)
        model.eval()
        self.base = _Entry(model, tok, str(base_name).rstrip("/").split("/")[-1])

        tdir: Path = s["tuned_path"]
        if tdir.is_dir():
            try:
                info = None
                info_path = tdir / "training_info.json"
                if info_path.is_file():
                    with open(info_path, encoding="utf-8") as f:
                        info = json.load(f)
                m, t = load_policy_from(str(tdir), self.device)
                m.eval()
                run = (info or {}).get("run_name")
                name = f"{tdir.name} ({run})" if run else tdir.name
                self.tuned = _Entry(m, t, name, training_info=info)
            except Exception as e:  # corrupt checkpoint: report unavailable, keep serving
                log.exception("failed to load tuned model from %s", tdir)
                self.tuned_error = f"{e.__class__.__name__}: {e}"
        else:
            log.info("tuned model not found at %s", tdir)

        rdir: Path = s["reward_path"]
        if (rdir / "rm_meta.json").is_file():
            try:
                from minirlhf.reward.scorer import DistilBertScorer
                self.scorer = DistilBertScorer(str(rdir), device=self.device, normalize=True)
            except Exception:
                log.exception("failed to load reward model from %s", rdir)
        else:
            log.info("reward model not found at %s", rdir)

    def entry(self, kind: str) -> _Entry:
        if kind == "base":
            e = self.base
        elif kind == "tuned":
            e = self.tuned
        else:
            raise ValueError(f"unknown model kind {kind!r}")
        if e is None:
            raise ModelUnavailable(self.tuned_error or "Model not trained yet")
        return e


_registry: Optional[Registry] = None
_registry_lock = threading.Lock()


def load(settings: Optional[dict] = None) -> Registry:
    """Load (once) and return the registry. Later calls return the same instance."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = Registry(settings or resolve_settings())
        return _registry


def reset() -> None:
    """Drop the registry (for tests); the next load() re-reads the settings."""
    global _registry
    with _registry_lock:
        _registry = None


def model_name(kind: str) -> str:
    return load().entry(kind).name


def answer(kind: str, prompt: str, max_new_tokens: int = 64, greedy: bool = True,
           temperature: float = 0.7) -> tuple[str, int]:
    """Generate one answer. Returns (text, number of generated tokens, EOS excluded)."""
    reg = load()
    e = reg.entry(kind)
    gen_cfg = {"max_new_tokens": int(max_new_tokens), "temperature": float(temperature),
               "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY,
               "max_prompt_tokens": MAX_PROMPT_TOKENS}
    with e.lock, torch.inference_mode():
        r = generate(e.model, e.tokenizer, [prompt], gen_cfg, greedy=bool(greedy))
    ids, mask = r.response_ids[0], r.response_mask[0].bool()
    n = int((mask & (ids != e.tokenizer.eos_token_id)).sum().item())
    return r.responses[0], n


def score(prompt: str, text: str) -> Optional[float]:
    """Reward-model score of (prompt, text), or None if no reward model is loaded."""
    reg = load()
    if reg.scorer is None:
        return None
    with reg.scorer_lock, torch.inference_mode():
        return float(reg.scorer.score([prompt], [text])[0].item())


def status() -> dict:
    """Exact get_status() shape of contract C1. Loads models if not yet loaded."""
    reg = load()
    return {
        "device": reg.device.type,
        "base": {"available": reg.base is not None, "name": reg.base.name},
        "tuned": {"available": reg.tuned is not None,
                  "name": reg.tuned.name if reg.tuned else None,
                  "training_info": reg.tuned.training_info if reg.tuned else None},
        "reward_model": {"available": reg.scorer is not None},
    }
