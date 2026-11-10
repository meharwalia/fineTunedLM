import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session")
def tiny_prompts():
    return [json.loads(l)["prompt"] for l in (FIXTURES / "prompts.jsonl").read_text().splitlines() if l.strip()]


@pytest.fixture
def cfg():
    from ste_rlhf.common import load_config
    return load_config()
