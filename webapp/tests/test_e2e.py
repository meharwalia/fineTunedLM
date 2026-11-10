"""End-to-end tests with the real models through the FastAPI app (slow, needs the GPU/MPS free)."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("ste_rlhf.serving")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.server import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_compare_real_models():
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["connected"] is True
        body = c.post("/api/compare", json={"prompt": "How do I reset a home Wi-Fi router?",
                                            "max_new_tokens": 64}).json()
    base = body["base"]
    assert base["ok"] is True, base
    assert base["response"].strip() and isinstance(base["finished"], bool)
    assert isinstance(base["ste"], dict)
    tuned = body["tuned"]
    if (ROOT / "checkpoints" / "rlhf_final").is_dir():
        assert tuned["ok"] is True, tuned
        assert isinstance(tuned["finished"], bool) and isinstance(tuned["ste"], dict)
    else:
        assert tuned["ok"] is True or tuned.get("error") == "not_available" or "not_available" in str(tuned)
