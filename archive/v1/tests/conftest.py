from pathlib import Path

import pytest

from minirlhf.common.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def cfg():
    c = load_config()
    c["policy"]["max_new_tokens"] = 8
    c["rl"]["batch_size"] = 2
    c["device"] = "cpu"
    return c


@pytest.fixture(scope="session")
def small_policy():
    """Real SmolLM2-135M-Instruct on CPU: (model, tokenizer)."""
    from minirlhf.common.models import load_policy_from
    import torch
    return load_policy_from("HuggingFaceTB/SmolLM2-135M-Instruct", torch.device("cpu"))
