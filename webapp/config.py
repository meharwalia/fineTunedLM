"""Web app settings read from environment variables (contract C5/C7 serving defaults).

Not used by the server itself while the model interface is unconnected; the
integration step reads these from model_interface.py.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tuned_model_path: str = "checkpoints/rlhf_final"
    reward_model_path: str = "checkpoints/reward_model"
    device: str = "auto"            # auto = cuda > mps > cpu
    max_new_tokens: int = 1024


def get_settings() -> Settings:
    d = Settings()
    try:
        max_new = int(os.environ.get("MAX_NEW_TOKENS", d.max_new_tokens))
    except ValueError:
        max_new = d.max_new_tokens
    return Settings(
        base_model=os.environ.get("BASE_MODEL", d.base_model),
        tuned_model_path=os.environ.get("TUNED_MODEL_PATH", d.tuned_model_path),
        reward_model_path=os.environ.get("REWARD_MODEL_PATH", d.reward_model_path),
        device=os.environ.get("DEVICE", d.device),
        max_new_tokens=max_new,
    )
