"""Shared library (contract C3)."""
from .config import add_config_args, apply_overrides, load_config, resolve_run_name
from .device import get_device
from .generation import Rollout, generate
from .io import append_jsonl, read_jsonl, write_jsonl
from .kl import per_token_kl
from .logging import RunLogger
from .logprobs import response_logprobs
from .models import load_policy, load_reference, load_tokenizer
from .seed import set_seed
