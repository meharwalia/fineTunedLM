"""Tests for request validation and the error mapping. No real models, no fake answers.

The model interface functions are monkeypatched to RAISE (never to return answers), so
these tests load no model. Real-model behaviour is covered by test_e2e.py.
"""
import pytest
from fastapi.testclient import TestClient

from webapp import model_interface, server
from webapp.model_interface import ModelNotAvailableError, ModelResponse
from webapp.server import app


def _raiser(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    for name in ("load_models", "get_status", "generate_response"):
        mp.setattr(model_interface, name, _raiser(NotImplementedError("not connected")))
    with TestClient(app) as c:  # lifespan: load_models raises NotImplementedError
        yield c
    mp.undo()


@pytest.mark.parametrize("exc,error", [
    (NotImplementedError("x"), "not_connected"),
    (ModelNotAvailableError("Model not trained yet"), "not_available"),
    (RuntimeError("boom"), "failed"),
])
def test_compare_error_mapping(client, monkeypatch, exc, error):
    monkeypatch.setattr(model_interface, "generate_response", _raiser(exc))
    body = client.post("/api/compare", json={"prompt": "hi"}).json()
    for k in ("base", "tuned"):
        assert body[k]["ok"] is False
        assert body[k]["error"] == error
        assert body[k]["message"]


def test_one_failure_does_not_hide_other(client, monkeypatch):
    def gen(prompt, kind, settings):
        if kind == "tuned":
            raise ModelNotAvailableError("Model not trained yet")
        raise RuntimeError("base broke")
    monkeypatch.setattr(model_interface, "generate_response", gen)
    body = client.post("/api/compare", json={"prompt": "hi"}).json()
    assert body["base"]["error"] == "failed" and body["base"]["message"] == "base broke"
    assert body["tuned"]["error"] == "not_available"


def test_health_status_failure_is_not_connected(client, monkeypatch):
    monkeypatch.setattr(model_interface, "get_status", _raiser(RuntimeError("x")))
    assert client.get("/api/health").json() == {"connected": False, "status": None}


def test_health_not_connected(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"connected": False, "status": None}


def test_examples(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    ex = r.json()["examples"]
    assert len(ex) >= 1
    assert all(isinstance(e, str) and e.strip() for e in ex)


def test_compare_not_connected(client):
    r = client.post("/api/compare", json={"prompt": "  Why is the sky blue?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "Why is the sky blue?"
    assert body["settings"] == {"max_new_tokens": 1024, "greedy": True, "temperature": 0.7}
    for k in ("base", "tuned"):
        assert body[k]["ok"] is False
        assert body[k]["error"] == "not_connected"
        assert body[k]["message"]


@pytest.mark.parametrize("payload", [
    {"prompt": ""},
    {"prompt": "   \n\t "},
    {"prompt": "x" * 501},
    {},
    {"prompt": "hi", "max_new_tokens": 7},
    {"prompt": "hi", "max_new_tokens": 2049},
    {"prompt": "hi", "temperature": 0.05},
    {"prompt": "hi", "temperature": 1.6},
])
def test_compare_invalid(client, payload):
    assert client.post("/api/compare", json=payload).status_code == 422


def test_compare_boundaries_ok(client):
    for payload in ({"prompt": "x" * 500, "max_new_tokens": 8, "temperature": 0.1},
                    {"prompt": "hi", "max_new_tokens": 2048, "temperature": 1.5, "greedy": False}):
        assert client.post("/api/compare", json=payload).status_code == 200


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "RLHF" in r.text


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_max_new_tokens_limit(client):
    assert client.post("/api/compare", json={"prompt": "hi", "max_new_tokens": 2048}).status_code == 200
    assert client.post("/api/compare", json={"prompt": "hi", "max_new_tokens": 2049}).status_code == 422


def _fake_gen(prompt, kind, settings):
    return ModelResponse(model_kind=kind, model_name=kind, response="Do this.", reward=0.5,
                         latency_ms=3, num_tokens=3, finished=(kind == "base"))


def test_finished_and_ste_passthrough(client, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(model_interface, "generate_response", _fake_gen)
    ste_mod = types.ModuleType("ste_rlhf.ste.metrics")
    ste_mod.ste_report = lambda text: {"ste_score": 88, "words": len(text.split())}
    for name in ("ste_rlhf", "ste_rlhf.ste"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ste_rlhf.ste.metrics", ste_mod)
    body = client.post("/api/compare", json={"prompt": "hi"}).json()
    assert body["base"]["finished"] is True and body["tuned"]["finished"] is False
    assert body["base"]["ste"] == {"ste_score": 88, "words": 2}


def test_ste_null_when_module_missing(client, monkeypatch):
    import sys
    monkeypatch.setattr(model_interface, "generate_response", _fake_gen)
    monkeypatch.setitem(sys.modules, "ste_rlhf.ste.metrics", None)  # forces ImportError
    body = client.post("/api/compare", json={"prompt": "hi"}).json()
    assert body["base"]["ok"] and body["base"]["ste"] is None


def test_results_200_and_404(client, monkeypatch, tmp_path):
    f = tmp_path / "summary.json"
    monkeypatch.setattr(server, "RESULTS_PATH", f)
    r = client.get("/api/results")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    f.write_text('{"base": {"ste_score": 40}, "tuned": {"ste_score": 85}}')
    r = client.get("/api/results")
    assert r.status_code == 200 and r.json()["tuned"]["ste_score"] == 85


def test_health_connected_passthrough(client, monkeypatch):
    st = {"device": "cpu", "base": {"available": True, "name": "b"},
          "tuned": {"available": True, "name": "t",
                    "training_info": {"algo": "reinforce", "max_new_tokens": 768}},
          "reward_model": {"available": True}}
    monkeypatch.setattr(model_interface, "get_status", lambda: st)
    assert client.get("/api/health").json() == {"connected": True, "status": st}


def test_examples_v2_preferred(client, monkeypatch, tmp_path):
    f = tmp_path / "examples_v2.json"
    f.write_text('[{"question": "How do I reset a router?", "label": "Router"}]')
    monkeypatch.setattr(server, "EXAMPLES_V2_PATH", f)
    body = client.get("/api/examples").json()
    assert body["examples"] == ["How do I reset a router?"]
    assert body["selected"][0]["label"] == "Router"
