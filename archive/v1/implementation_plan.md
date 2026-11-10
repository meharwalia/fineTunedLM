# Mini-RLHF Implementation Plan (subagents from start to finish)

Companion to [project.md](project.md), which explains *what* and *why*. This file explains *how to build it*, as a sequence of subagent tasks coordinated by a lead.

**Scope.** RLHF with **REINFORCE only**, plus a **FastAPI server and simple web frontend** that sends one question to the base model and to the RLHF-tuned model and shows both answers side by side.

**Key design rule for the web app.** The web app does **not** contain any fake or mock model service. It calls one function interface, `webapp/model_interface.py`. The web agent writes only the **function signatures** (bodies raise `NotImplementedError`); a later **integration agent** fills them in using the RLHF pipeline's code.

---

## 0. Overview

```
 WAVE 0  (blocking)       S0  Scaffold + shared library + contracts
                           │
           ┌──────────┬────┴─────┬──────────┬──────────┐
 WAVE 1    ▼          ▼          ▼          ▼          ▼
 (parallel) S1         S2         S3         S4         S5
           Data       Reward     REINFORCE  Eval       Web app
           pipeline   model      trainer    tooling    (signatures only)
           └──────────┴────┬─────┴──────────┴──────────┘
                           ▼  lead merges + runs all tests
 WAVE 2    S6 Training runner          ∥   S7 Integration agent
 (parallel) (reward model + RL runs,       (implements model_interface.py with
            export rlhf_final)              the pipeline; end-to-end web tests)
              + YOU: label 300–500 pairs
                           │
 WAVE 3    S8 Experiments (seeds, β sweep, baseline ablation, evaluation)
                           │
 WAVE 4    S9 Report + slides   →   Lead: final demo check
```

| ID | Subagent | Wave | Runs in parallel with |
|---|---|---|---|
| S0 | Scaffold + shared library | 0 | – (everything waits for it) |
| S1 | Data pipeline | 1 | S2, S3, S4, S5 |
| S2 | Reward model | 1 | S1, S3, S4, S5 |
| S3 | REINFORCE trainer | 1 | S1, S2, S4, S5 |
| S4 | Evaluation tooling | 1 | S1, S2, S3, S5 |
| S5 | Web app (frontend + FastAPI + interface signatures) | 1 | S1, S2, S3, S4 |
| S6 | Training runner | 2 | S7 |
| S7 | Integration (model interface ↔ RLHF pipeline ↔ web app) | 2 | S6 |
| S8 | Experiments | 3 | – |
| S9 | Report + slides | 4 | – |

**Why S7 can run alongside S6.** The integration agent connects the web app to the pipeline using the **base model** straight away. The tuned model appears automatically once S6 exports `checkpoints/rlhf_final/`; S7's code loads it if it exists.

**Compute caveat.** Code can be written in parallel, but training shares the Mac GPU. S6 and S8 run training jobs **one at a time** (or in separate Colab sessions). S7's short inference calls are fine alongside.

---

## 1. The lead (main session)

The lead does not write feature code. It:
1. Spawns each subagent with its brief (section 8), in its own **git worktree** (`isolation: worktree`), after S0 has made the initial commit. **Every subagent (S0–S9) runs on Sonnet**: always pass `model: "sonnet"` when spawning, and never use a `fork` subagent type (forks ignore the model setting and inherit the lead's model). Subagents that start their own helpers must also use Sonnet.
2. Reviews each subagent's report and diff, runs its tests, and merges its branch into `main`.
3. Adds config keys that subagents request (only S0's files hold shared config).
4. Asks the user for human steps (labelling, win-rate judging) at the right moment.
5. Runs the final demo check (section 7).

## 2. Ground rules for every subagent

1. **Only edit files you own** (section 3). Shared files (`src/minirlhf/common/`, `configs/default.yaml`, `tests/conftest.py`) belong to S0 and are read-only afterwards. Need a change? Say so in your report.
2. **Never change a contract** in section 4 without flagging it in your report.
3. Code runs on `mps`, `cuda` and `cpu` (auto: cuda > mps > cpu), via `common.device.get_device`.
4. Tests: fast tests with no downloads (`pytest -m "not slow"`), plus at least one `@pytest.mark.slow` smoke test with the real small model where relevant.
5. Work on your own branch (`s1-data`, `s2-reward`, …). Commit when done.
6. Finish with a short report: what was built, how to run it, test results, requested changes.

---

## 3. Repository layout and ownership

```
RL/
├── project.md, implementation_plan.md                         lead
├── .gitignore, requirements.txt, pyproject.toml               S0
├── configs/default.yaml                                       S0
├── src/minirlhf/
│   ├── common/ config device seed io schemas models
│   │           generation logprobs kl logging                 S0
│   ├── reward/ interface.py heuristic.py factory.py           S0
│   ├── reward/ model.py train.py scorer.py evaluate.py        S2
│   ├── data/   prompts generate_candidates rule_labeller
│   │           hh_rlhf label_cli build_pairs                  S1
│   ├── rl/     reinforce.py export.py                         S3
│   ├── eval/   generate_eval metrics winrate_cli plots
│   │           report_tables                                  S4
│   └── serving.py                                             S7
├── tests/  conftest.py fixtures/ test_common_*                S0
│           test_data_* (S1)  test_reward_* (S2)  test_reinforce* (S3)
│           test_eval_* (S4)  test_serving* (S7)
├── webapp/
│   ├── requirements.txt, config.py, server.py, examples.json  S5
│   ├── static/ index.html style.css app.js                    S5
│   ├── model_interface.py      signatures: S5  →  bodies: S7
│   ├── tests/test_server.py                                   S5
│   ├── tests/test_e2e.py                                      S7
│   └── README.md                                              S5 (S7 adds a "connected" section)
├── data/                                                      generated by S1 / S6
├── checkpoints/ reward_model/ <run>/ rlhf_final/              generated by S6 / S8
├── logs/                                                      generated by S6 / S8
├── results/                                                   generated by S4 tools in S6 / S8
└── report/                                                    S9
```

---

## 4. Contracts

### C1. Model interface — `webapp/model_interface.py`
The **only** thing the web server calls to get answers. S5 writes it with these exact signatures and docstrings; every function body is `raise NotImplementedError("model interface not connected yet")`. S7 replaces the bodies and must not change the signatures.

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

ModelKind = Literal["base", "tuned"]

@dataclass
class GenerationSettings:
    max_new_tokens: int = 64        # 8–256
    greedy: bool = True
    temperature: float = 0.7        # used only when greedy is False (0.1–1.5)

@dataclass
class ModelResponse:
    model_kind: ModelKind
    model_name: str                 # e.g. "SmolLM2-135M-Instruct" or "rlhf_final (reinforce_beta0.05_seed0)"
    response: str
    reward: Optional[float]         # reward-model score; None if no reward model is available
    latency_ms: int
    num_tokens: int

class ModelNotAvailableError(Exception):
    """Raised when the requested model is not trained or failed to load."""

def load_models() -> None:
    """Load the base model, the tuned model (if trained) and the reward model (if trained).
    Called once at server startup. Must not raise if only the tuned/reward model is missing."""

def get_status() -> dict:
    """Return {"device": str,
               "base":   {"available": bool, "name": str},
               "tuned":  {"available": bool, "name": str | None, "training_info": dict | None},
               "reward_model": {"available": bool}}"""

def generate_response(prompt: str, model_kind: ModelKind,
                      settings: GenerationSettings) -> ModelResponse:
    """Generate one answer from the base or tuned model, score it with the reward model
    if available, and time the call. Raises ModelNotAvailableError if that model is missing."""
```

**How the web server treats each outcome** (so the UI works before integration):
| Interface behaviour | Server/UI behaviour |
|---|---|
| `NotImplementedError` | Card shows **"Model not connected yet"**; `/api/health` reports `"connected": false` |
| `ModelNotAvailableError` | Card shows **"Model not trained yet"** (expected for `tuned` before S6 finishes) |
| Any other exception | Card shows **"Generation failed"** with the message; the other card is unaffected |
| `ModelResponse` | Card shows the answer, reward bar, latency, token count |

### C2. Web API — implemented by S5
| Method | Path | Response |
|---|---|---|
| GET | `/` | `static/index.html` |
| GET | `/api/health` | `{"connected": bool, "status": <get_status() result> \| null}` |
| GET | `/api/examples` | `{"examples": [str, ...]}` |
| POST | `/api/compare` | Request `{"prompt": str (1–500 chars), "max_new_tokens": int = 64, "greedy": bool = true, "temperature": float = 0.7}` → `{"prompt": str, "settings": {...}, "base": Result, "tuned": Result}` |

`Result = {"ok": true, "model_name", "response", "reward", "latency_ms", "num_tokens"}` or `{"ok": false, "error": "not_connected" | "not_available" | "failed", "message": str}`. Each model is called separately, so one failing never hides the other. Invalid input → `422`.

### C3. Checkpoints — written by S6/S8, read by S7
| Artifact | Location | Contents |
|---|---|---|
| Reward model | `checkpoints/reward_model/` | `save_pretrained` of `AutoModelForSequenceClassification(num_labels=1)` + tokenizer + `rm_meta.json` `{"base_model", "max_length", "val_accuracy", "per_source_accuracy", "train_reward_mean", "train_reward_std", "data_counts"}` |
| Training runs | `checkpoints/<run>/step_<N>/`, `checkpoints/<run>/final/` | `save_pretrained` policy + tokenizer |
| Tuned model for the demo | `checkpoints/rlhf_final/` | Copy of the best run's `final/` + `training_info.json` `{"base_model", "algo": "reinforce", "run_name", "steps", "beta_kl", "eval_reward_base", "eval_reward_tuned", "created"}` |

**Reward scoring rule** (S2's scorer and anything else that scores):
`inputs = tokenizer(prompt, response, truncation="longest_first", max_length=max_length)`, `score = (logits[0, 0] − train_reward_mean) / train_reward_std`.

### C4. Prompt formatting and generation (S0's `common.generation`, used by everyone)
- `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)`, left padding, `pad_token = eos_token` if missing.
- Decode only new tokens, `skip_special_tokens=True`, `.strip()`.
- Training: sampling, `temperature=0.9`, `top_p=0.95`, `max_new_tokens=48`.
- Demo/eval: greedy, `max_new_tokens=64`, `repetition_penalty=1.1`; both models always get identical settings.

### C5. Common library API (S0)
```python
get_device(cfg) -> torch.device;  set_seed(seed)
load_config(path, overrides: list[str]) -> dict          # supports --set rl.beta_kl=0
read_jsonl / write_jsonl / append_jsonl;  validate_record(kind, obj)
load_policy(cfg, device) -> (model, tokenizer)            # trainable
load_reference(cfg, device) -> model                      # frozen
load_policy_from(path, device) -> (model, tokenizer)      # checkpoints / rlhf_final
generate(model, tokenizer, prompts, gen_cfg, greedy=False) -> Rollout
    # Rollout: prompts, query_ids, query_mask, response_ids, response_mask (to EOS incl.), responses (texts)
response_logprobs(model, rollout, return_entropy=False) -> Tensor[B, Lr]
per_token_kl(logp, logp_ref, mask) -> Tensor[B, Lr]
RunLogger(run_dir, cfg).log(step, metrics) / .log_samples(step, prompts, responses, rewards)
resolve_run_name(cfg) -> "reinforce_beta{β}_seed{seed}[_nobaseline]"
```
`reward/interface.py`: `RewardScorer.score(prompts, responses) -> Tensor[B]` (CPU float32, no grad).
`reward/heuristic.py`: `HeuristicScorer` (polite words +, prompt-word overlap +, length over 60 words −, repetition −), used **only** to develop the trainer before the real reward model exists.
`reward/factory.py`: `load_scorer(cfg, device)` → `HeuristicScorer` if `reward.type == "heuristic"`, else `minirlhf.reward.scorer.DistilBertScorer(checkpoint, device, max_length, normalize)` (lazy import).

### C6. Data and log formats (JSONL)
| File | Schema |
|---|---|
| `data/prompts/{train,eval}.jsonl` | `{"id", "prompt", "source"}` |
| `data/candidates/candidates.jsonl` | `{"id", "prompt", "responses": [str, str], "gen": {...}}` |
| `data/prefs/{human,rule,hh,train,val}.jsonl` | `{"id", "prompt", "chosen", "rejected", "source": "human" \| "rule" \| "hh-rlhf"}` |
| `results/generations/<tag>.jsonl` | `{"id", "prompt", "response", "model"}` |
| `logs/<run>/metrics.jsonl` | `{"step", "reward_mean", "reward_std", "kl_mean", "return_mean", "baseline", "loss", "entropy", "resp_len_mean", "time_s"}` |
| `logs/<run>/samples.jsonl` | `{"step", "prompt", "response", "reward"}` |

### C7. Config — `configs/default.yaml` (S0)
```yaml
seed: 0
device: auto
policy:   {name: HuggingFaceTB/SmolLM2-135M-Instruct, max_prompt_tokens: 128, max_new_tokens: 48, temperature: 0.9, top_p: 0.95}
reward:   {type: heuristic, base_model: distilbert-base-uncased, checkpoint: checkpoints/reward_model, max_length: 384, normalize: true}
data:     {prompts_dir: data/prompts, candidates_path: data/candidates/candidates.jsonl, prefs_dir: data/prefs,
           n_train_prompts: 2000, n_eval_prompts: 200, hh_rlhf_n: 3000, val_fraction: 0.1,
           mix: {human: 1.0, rule: 1.0, hh-rlhf: 1.0}}
reward_training: {lr: 3.0e-5, epochs: 2, batch_size: 16, weight_decay: 0.01}
rl:       {run_name: null, steps: 300, batch_size: 16, lr: 1.0e-5, beta_kl: 0.05, grad_clip: 1.0,
           use_baseline: true, baseline_ema: 0.9, normalize_advantages: true,
           log_every: 1, sample_every: 10, save_every: 50}
eval:     {greedy: true, max_new_tokens: 64, repetition_penalty: 1.1, out_dir: results}
serving:  {base_model: HuggingFaceTB/SmolLM2-135M-Instruct, tuned_path: checkpoints/rlhf_final,
           reward_path: checkpoints/reward_model}
```

---

## 5. Subagent tasks

### Wave 0

#### S0 — Scaffold and shared library
- `git init`; Python 3.11/3.12 venv in `.venv`; pinned `requirements.txt` (`torch`, `transformers`, `datasets`, `pyyaml`, `matplotlib`, `tqdm`, `pytest`); `pyproject.toml` (package `minirlhf`, src layout, pytest marker `slow`); `pip install -e .`; `.gitignore` (`.venv/ data/ checkpoints/ logs/ results/ __pycache__/`).
- `configs/default.yaml` exactly as C7; everything in C5; fixtures (`prompts_small`, `candidates_small`, `prefs_small`, `metrics_small` JSONL); `tests/conftest.py` with a tiny-settings `cfg` fixture and a slow `small_policy` fixture.
- Tests: config overrides, schema validation, generation shapes and EOS mask, log-prob gather vs manual, KL = 0 for identical models, heuristic scorer determinism.
- Initial commit on `main`.
- **Done when:** fast and slow tests pass on the Mac.

### Wave 1 (all five in parallel, each in its own worktree)

#### S1 — Data pipeline (`src/minirlhf/data/`)
| Module | Output |
|---|---|
| `prompts.py`: Dolly-15k rows without context (open_qa, general_qa, brainstorming, creative_writing), ≤ 40 words, + optional `data/prompts/handwritten.txt`; dedupe; seeded split | `data/prompts/train.jsonl`, `eval.jsonl` |
| `generate_candidates.py`: 2 sampled responses per prompt via `common.generation`; drop identical/empty | `data/candidates/candidates.jsonl` |
| `rule_labeller.py`: politeness, on-topic overlap, > 80 words penalty, repetition/gibberish penalty; skip near-ties | `data/prefs/rule.jsonl` |
| `hh_rlhf.py`: single-turn only, both answers ≤ 120 words, sample `hh_rlhf_n` | `data/prefs/hh.jsonl` |
| `label_cli.py`: terminal A/B labeller, random left/right order, keys `1`/`2`/`s`/`q`, resumable | `data/prefs/human.jsonl` |
| `build_pairs.py`: merge by `data.mix`, dedupe, stratified split | `data/prefs/train.jsonl`, `val.jsonl` |
- **Done when:** tests pass (hh parser, rule ordering, split/dedupe, label order mapping) and every module runs end-to-end (candidates capped at 50 for the smoke run).

#### S2 — Reward model (`reward/model.py`, `train.py`, `scorer.py`, `evaluate.py`)
- Sentence-pair DistilBERT, Bradley–Terry loss `−logsigmoid(r_c − r_r).mean()`, AdamW + warmup/decay, pairwise val accuracy per epoch, keep best, save in **C3** format.
- `DistilBertScorer(checkpoint, device, max_length, normalize)` implements `RewardScorer` with the C3 scoring rule.
- `evaluate.py`: overall + per-source accuracy, margin histogram.
- Develops on an hh-rlhf subset (`train.py --source hh`) since S1's files may not exist yet.
- **Done when:** tests pass (loss direction, scorer shape/normalisation, overfits 20 fixture pairs) and val accuracy ≥ 0.60 on the hh subset.

#### S3 — REINFORCE trainer (`rl/reinforce.py`, `rl/export.py`)
One step: sample prompts → `generate` → `scores = scorer.score(...)` → `logp` (grad) and `logp_ref` → `kl = per_token_kl(...).sum(-1)` → `R = scores − β·kl` → baseline `b ← ema·b + (1−ema)·mean(R)` (if `use_baseline`) → `A = R − b`, normalised by batch std → `loss = −(A.detach() · Σ_t logp_t).mean()` → backward, clip, AdamW → log/sample/checkpoint per C3/C6.
- CLI: `python -m minirlhf.rl.reinforce --config configs/default.yaml --set rl.beta_kl=0.05`
- `export.py --run <name>`: copy `final/` to `checkpoints/rlhf_final/` + write `training_info.json`.
- Develops with `reward.type: heuristic`.
- **Done when:** tests pass (advantage sign on synthetic tensors, baseline EMA, export format) and a 100-step heuristic run shows rising reward with bounded KL.

#### S4 — Evaluation tooling (`src/minirlhf/eval/`)
| Module | Output |
|---|---|
| `generate_eval.py --model base\|<path> --tag <name>` (C4 demo settings) | `results/generations/<tag>.jsonl` |
| `metrics.py`: mean/std reward, length, distinct-1/2, KL vs base | `results/metrics/<tag>.json`, `summary.md` |
| `winrate_cli.py`: blind A/B between two tags | `results/winrate/<a>_vs_<b>.json` (win rate + 95% CI) |
| `plots.py`: reward/KL curves, mean ± std over seeds, β-sweep reward-vs-KL, baseline ablation | `results/plots/*.png` |
| `report_tables.py`: side-by-side samples | `results/samples.md` |
- **Done when:** tests pass (distinct-n, CI maths, plots on fixture metrics) and the chain runs on the base model.

#### S5 — Web app (`webapp/`), signatures only, no mock
1. `webapp/model_interface.py`: exactly the C1 signatures, dataclasses and docstrings; all function bodies `raise NotImplementedError("model interface not connected yet")`. **No fake answers, no canned responses, no mock service.**
2. `webapp/server.py` (FastAPI):
   - calls `model_interface.load_models()` in the `lifespan` handler; if it raises `NotImplementedError`, logs "not connected" and keeps running.
   - `/api/health`, `/api/examples`, `/api/compare` per **C2**; calls `generate_response` once for `"base"` and once for `"tuned"` with identical settings, mapping each outcome per the C1 table. Plain `def` endpoint (runs in a worker thread).
   - serves `static/`. Run with `uvicorn webapp.server:app --port 8000`.
3. `webapp/static/` (plain HTML/CSS/JS, no build step):
   - header with title + status from `/api/health` (connected?, device, models available, training info such as "REINFORCE, 300 steps, β = 0.05")
   - textarea + **Compare** button (Ctrl/Cmd+Enter), example chips, collapsible settings (max tokens, greedy, temperature)
   - two cards side by side: **"Original model"** and **"After RLHF (REINFORCE)"**, each with response, reward bar, latency, token count; higher reward highlighted; clear states for *not connected*, *not trained yet*, *failed*, and loading
   - responsive (cards stack on narrow screens), light/dark via `prefers-color-scheme`, button disabled while a request runs
4. `webapp/tests/test_server.py`: request validation (`422`), examples endpoint, and the **unconnected** behaviour (health reports `connected: false`; compare returns `ok: false, error: "not_connected"` for both cards). These need no models.
5. `webapp/requirements.txt` (pinned `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `pytest`), `webapp/README.md`, `webapp/examples.json` (6–8 questions).
- **Done when:** the server starts, the page loads and shows "Model not connected yet" on both cards, and tests pass.

### Wave 2 (after the lead merges Wave 1)

#### S6 — Training runner (long-running jobs, run in the background)
| Step | Who | Action |
|---|---|---|
| 6.1 | S6 | Build prompts, generate candidates, rule labels, hh subset |
| 6.2 | **You** | `python -m minirlhf.data.label_cli`: label 300–500 pairs |
| 6.3 | S6 (during 6.2) | Reward model v1 on `hh` + `rule` |
| 6.4 | S6 (after 6.2) | Reward model v2 including `human`; keep the better one; report per-source accuracy |
| 6.5 | S6 | `reward.type: distilbert`; 20-step smoke run; full run (β = 0.05, seed 0); inspect samples for reward hacking |
| 6.6 | S6 | Evaluate base vs tuned on held-out prompts; `export` to `checkpoints/rlhf_final/` |
- S6 may fix small bugs in pipeline code it hits, but must list every change in its report.
- **Done when:** `checkpoints/reward_model/` and `checkpoints/rlhf_final/` exist in C3 format and the tuned model beats the base on held-out mean reward.

#### S7 — Integration agent (in parallel with S6)
1. `src/minirlhf/serving.py`: a small model registry on top of the pipeline:
   - loads the base model with `load_policy_from(serving.base_model)`; the tuned model from `serving.tuned_path` **if it exists** (reads `training_info.json`); the reward model via `DistilBertScorer` **if** `rm_meta.json` exists
   - `answer(kind, prompt, max_new_tokens, greedy, temperature) -> (text, num_tokens)` using `common.generation.generate` with C4 demo settings; one lock per model; `torch.inference_mode()`
   - `score(prompt, text) -> float | None`
2. `webapp/model_interface.py`: replace each `NotImplementedError` body with calls into `minirlhf.serving` (signatures unchanged). `generate_response` times the call, scores the answer, and raises `ModelNotAvailableError` for a missing tuned model.
3. Tests:
   - `tests/test_serving.py` (slow): base answer is non-empty; tuned missing → not available; scoring returns `None` without a reward model.
   - `webapp/tests/test_e2e.py` (slow): start the app with `TestClient`; `/api/health` is connected; `/api/compare` returns a real base answer; tuned card is `not_available` before training and `ok` once `checkpoints/rlhf_final/` exists (test skips that part if the folder is missing).
4. Add a "Running with real models" section to `webapp/README.md`; add `minirlhf` (editable) to the web app's install instructions.
- **Done when:** with only the base model, the page shows a real base answer and "Model not trained yet" for the tuned card; after S6's export, a server restart shows both answers with reward scores.

### Wave 3

#### S8 — Experiments
| Experiment | Runs |
|---|---|
| Seeds | `reinforce_beta0.05_seed{0,1,2}` |
| β sweep + reward hacking | `reinforce_beta{0,0.02,0.2}_seed0`; extreme β = 0 samples → `results/hacking.md` |
| Baseline ablation | `reinforce_beta0.05_seed0_nobaseline` |
| Evaluation | metrics for base and every final checkpoint; plots; `results/samples.md`; set up win-rate sessions (**you** label: tuned vs base) |
- Re-export the best run to `checkpoints/rlhf_final/` if it beats S6's.
- **Done when:** all plots and tables in `results/` exist and `results/metrics/summary.md` compares every run.

### Wave 4

#### S9 — Report and slides (`report/`)
- `report/report.md` following project.md §13: problem, idea, MDP framing, pipeline, reward model, REINFORCE, results (plots, tables, hacking examples, win rate), web demo screenshots, course-concept mapping (project.md §10), limitations, future work (actor-critic/PPO, DPO).
- Slide outline (`report/slides.md`) with one figure per slide.

---

## 6. Human steps (the lead asks you at these points)

| When | What | Time |
|---|---|---|
| Wave 2, step 6.2 | Label 300–500 response pairs in the terminal | ~1–2 hours |
| Wave 3 | Blind win-rate labelling: tuned vs base on ~100 eval prompts | ~30 minutes |
| Wave 4 | Review the report and slides | – |

---

## 7. Final demo check (lead)

1. All branches merged; `pytest` (fast + slow) and `webapp` tests pass.
2. `uvicorn webapp.server:app --port 8000`; open `http://localhost:8000`.
3. Run the example questions and 5 held-out prompts: both cards answer, reward scores shown, tuned scores higher on average.
4. Screenshots into `report/`.

---

## 8. Subagent briefs (copy-paste)

Spawn every subagent with `model: "sonnet"` (fresh agent, not a fork).

Every brief starts with: *"Read project.md and implementation_plan.md (sections 2–4 are binding). Work only on files your ID owns in section 3, on your own branch. Finish with a report: what you built, how to run it, test results, requested changes."*

- **S0:** "You are S0. Implement Wave 0 (section 5, S0) exactly, following contracts C4–C7. Make the initial commit on `main`."
- **S1:** "You are S1. Implement the data pipeline (section 5, S1) following C6. Produce real data files that validate."
- **S2:** "You are S2. Implement the reward model (section 5, S2). Save in the C3 format and implement `DistilBertScorer` with the exact C5 constructor and the C3 scoring rule. Train on an hh-rlhf subset until S1's data exists."
- **S3:** "You are S3. Implement the REINFORCE trainer and export script (section 5, S3) using only the common library and `load_scorer` with `reward.type: heuristic`. Show a 100-step run with rising reward."
- **S4:** "You are S4. Implement the evaluation tooling (section 5, S4). Demonstrate it on the base model and fixture metrics."
- **S5:** "You are S5. Build the web app (section 5, S5). Write `webapp/model_interface.py` with exactly the C1 signatures and `NotImplementedError` bodies; do not create any mock, fake or canned model service. The server and UI must handle every C1 outcome, and the page must work in the 'not connected' state. Never import from `src/minirlhf/`."
- **S6:** "You are S6. Run Wave 2 training (section 5, S6). Run long jobs in the background. Stop and ask the lead when human labelling (6.2) is needed. Export the best model to `checkpoints/rlhf_final/` in the C3 format."
- **S7:** "You are S7. Integrate the model interface (section 5, S7): implement `src/minirlhf/serving.py` and fill in the bodies of `webapp/model_interface.py` without changing its signatures. It must work with only the base model and pick up the tuned and reward models when their checkpoints exist."
- **S8:** "You are S8. Run the experiments in section 5, S8, one training job at a time. Produce all plots and tables in `results/`."
- **S9:** "You are S9. Write the report and slide outline (section 5, S9) from the files in `results/` and project.md."

---

## 9. Definition of done

- [ ] Reward model val accuracy ≥ 0.60, reported per data source
- [ ] REINFORCE improves mean reward on **held-out** prompts vs the base model, across 3 seeds
- [ ] β sweep plot, documented reward-hacking example at β = 0, baseline-ablation plot
- [ ] Blind human win rate: tuned vs base
- [ ] `checkpoints/rlhf_final/` exported with `training_info.json`
- [ ] Web app: `model_interface.py` connected; one question in, base and RLHF answers side by side with reward scores; all tests pass
- [ ] Report and slides with demo screenshots and the course-concept mapping
