import pytest

from minirlhf.common.schemas import validate_record
from minirlhf.data.generate_candidates import generate_candidates


@pytest.mark.slow
def test_generate_candidates_smoke(cfg, small_policy):
    model, tok = small_policy
    prompts = [{"id": f"p{i}", "prompt": p, "source": "x"} for i, p in enumerate(
        ["How do I boil an egg?", "Name three colors.", "Why is the sky blue?"])]
    cands, dropped = generate_candidates(cfg, prompts, model, tok, batch_size=2)
    assert len(cands) + dropped == 3
    for c in cands:
        validate_record("candidate", c)
        assert c["responses"][0] != c["responses"][1]
