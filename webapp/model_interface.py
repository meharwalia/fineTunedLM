"""Model interface (contract C5): the ONLY module the web server calls to get answers.

Signatures, dataclasses and docstrings are the contract. The function bodies are connected by the S9 integration step
(``ste_rlhf.serving``); settings come from environment variables (see webapp/config.py)
over the ``serving`` section of configs/v2.yaml.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional

ModelKind = Literal["base", "tuned"]


@dataclass
class GenerationSettings:
    max_new_tokens: int = 1024      # 8-2048
    greedy: bool = True
    temperature: float = 0.7        # used only when greedy is False (0.1-1.5)


@dataclass
class ModelResponse:
    model_kind: ModelKind
    model_name: str                 # e.g. "SmolLM2-135M-Instruct" or "rlhf_final"
    response: str
    reward: Optional[float]         # reward-model score; None if no reward model is available
    latency_ms: int
    num_tokens: int
    finished: bool = True           # True if the model ended its answer; False if it hit the token limit


class ModelNotAvailableError(Exception):
    """Raised when the requested model is not trained or failed to load."""


def load_models() -> None:
    """Load the base model, the tuned model (if trained) and the reward model (if trained).

    Called exactly once at server startup, before any request is served.
    Must not raise if only the tuned and/or reward model is missing; in that case
    the corresponding entry in get_status() reports available=False. May raise
    if the base model cannot be loaded (the server logs it and keeps running).
    """
    from ste_rlhf import serving
    serving.load()


def get_status() -> dict:
    """Return {"device": str,
               "base":   {"available": bool, "name": str},
               "tuned":  {"available": bool, "name": str | None, "training_info": dict | None},
               "reward_model": {"available": bool}}

    The result must be JSON-serialisable, cheap to compute, and must not run generation.
    "device" is e.g. "mps", "cuda" or "cpu". "training_info" is a free-form dict about
    the tuned model's training run, or None; the web UI displays these keys when present:
    "algo" (str, e.g. "reinforce"), "steps" (int), "beta_kl" (float), "max_new_tokens" (int)
    and "dataset" (str) -- i.e. the contents of checkpoints/rlhf_final/training_info.json (C6).
    """
    from ste_rlhf import serving
    return serving.status()


def generate_response(prompt: str, model_kind: ModelKind,
                      settings: GenerationSettings) -> ModelResponse:
    """Generate one answer from the base or tuned model, score it with the reward model
    if available, and time the call. Raises ModelNotAvailableError if that model is missing.
    (Implemented by ste_rlhf.serving.)

    The prompt is already stripped and validated (1-500 chars) and settings are already
    validated (max_new_tokens 8-2048, temperature 0.1-1.5) by the server. Must be
    thread-safe: the server calls it from worker threads, once per model kind per request.
    latency_ms is the wall-clock time of the call in milliseconds, and num_tokens is the
    number of generated tokens, and finished is False when generation stopped at
    max_new_tokens instead of at end-of-answer. reward is None if no reward model is available. Any
    exception other than ModelNotAvailableError is shown to the user as "Generation failed".

    Implementation note: latency_ms times the generation only (reward scoring excluded);
    num_tokens excludes the EOS token.
    """
    from ste_rlhf import serving
    try:
        text, finished, n, latency_ms, reward = serving.generate(model_kind, prompt, settings)
    except LookupError as e:
        raise ModelNotAvailableError(str(e)) from e
    st = serving.status()
    name = st["base"]["name"] if model_kind == "base" else st["tuned"]["name"]
    return ModelResponse(model_kind=model_kind, model_name=name or model_kind, response=text, reward=reward,
                         latency_ms=latency_ms, num_tokens=n, finished=finished)
