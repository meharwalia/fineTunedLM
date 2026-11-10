# Mini-RLHF web app

A small FastAPI + plain HTML/JS app that sends one question to the original model and the
RLHF-tuned (REINFORCE) model and shows both answers side by side with reward score,
latency and token count.

**Status:** the model interface is connected to the real models (see "Running with real models"
below). If it ever raises `NotImplementedError` the UI shows "not connected". There are no mock
answers anywhere.

## Run

From the project root:

```
pip install -r webapp/requirements.txt
uvicorn webapp.server:app --port 8000
# open http://localhost:8000
```

Tests: `python -m pytest webapp/tests -q` (the repo's pytest config uses `testpaths=tests`, so pass the path).

## Environment variables (`webapp/config.py`)

| Variable | Default |
|---|---|
| `BASE_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `TUNED_MODEL_PATH` | `checkpoints/rlhf_final` |
| `REWARD_MODEL_PATH` | `checkpoints/reward_model` |
| `DEVICE` | `auto` (cuda > mps > cpu) |
| `MAX_NEW_TOKENS` | `64` |

These are read via `webapp.config.get_settings()` for the integration step; the server itself does not use them.

## API

| Method | Path | Response |
|---|---|---|
| GET | `/` | the web page |
| GET | `/api/health` | `{"connected": bool, "status": {...} \| null}` (`connected` is false if `get_status()` raises anything) |
| GET | `/api/examples` | `{"examples": [str, ...]}` |
| POST | `/api/compare` | request `{"prompt" (1-500 chars after strip), "max_new_tokens" (8-256, 64), "greedy" (true), "temperature" (0.1-1.5, 0.7)}` -> `{"prompt", "settings", "base": Result, "tuned": Result}`; invalid input -> 422 |

`Result` is `{"ok": true, "model_name", "response", "reward", "latency_ms", "num_tokens"}` or
`{"ok": false, "error": "not_connected" | "not_available" | "failed", "message"}`. The two models
are called separately, so one failing never hides the other.

## Interface contract (C1)

`model_interface.py` is the only thing the server calls:
`load_models()`, `get_status()`, `generate_response(prompt, model_kind, settings)`, plus
`GenerationSettings`, `ModelResponse`, `ModelNotAvailableError`. Outcome mapping:
`NotImplementedError` -> `not_connected`; `ModelNotAvailableError` -> `not_available`
("Model not trained yet"); any other exception -> `failed` with its message. The integration step
replaces the function bodies without changing signatures. `get_status()["tuned"]["training_info"]`
may contain `algo`, `steps`, `beta_kl` (contents of `training_info.json`), which the header displays.

## Running with real models

The interface is connected: `model_interface.py` delegates to `minirlhf.serving`
(`src/minirlhf/serving.py`), a model registry that loads everything once at startup.

- **Base model**: `BASE_MODEL` (default `HuggingFaceTB/SmolLM2-135M-Instruct`), always required.
- **Tuned model**: loaded from `TUNED_MODEL_PATH` (default `<project root>/checkpoints/rlhf_final`) if
  that directory exists; `training_info.json` there is shown in the header. If missing, the tuned card
  shows "Model not trained yet" (`not_available`) and the base card still works.
- **Reward model**: loaded from `REWARD_MODEL_PATH` (default `<project root>/checkpoints/reward_model`)
  if `rm_meta.json` exists there (normalized scores per contract C3). If missing, `reward` is `null`.
- **Settings precedence**: environment variables (`BASE_MODEL`, `TUNED_MODEL_PATH`,
  `REWARD_MODEL_PATH`, `DEVICE`) override the `serving` section of `configs/default.yaml`.
  Relative paths are resolved against the project root (the directory containing `webapp/`), not the
  current directory, so `uvicorn webapp.server:app` works from anywhere.
- **Generation** uses the C4 demo settings: repetition penalty 1.1, greedy by default, or sampling with
  the requested temperature and top_p 0.95. One lock per model serialises generation; `latency_ms`
  times generation only (not reward scoring).

Example:

```
export PYTHONPATH=src            # or `pip install -e .`
TUNED_MODEL_PATH=/path/to/rlhf_final REWARD_MODEL_PATH=/path/to/reward_model \
  uvicorn webapp.server:app --port 8000
curl localhost:8000/api/health
curl -X POST localhost:8000/api/compare -H 'content-type: application/json' \
  -d '{"prompt":"How can I stay focused while studying?"}'
```

Tests: `pytest webapp/tests tests/test_serving.py` (`test_e2e.py` and `test_serving.py` load the real
small model; the e2e asset tests use checkpoint paths and are skipped if absent).
