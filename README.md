# Fine-Tuning a Small Language Model for Technical Work with RLHF

This project uses Reinforcement Learning from Human Feedback (RLHF) to fine-tune a small language model, SmolLM2-135M-Instruct, for technical work. The goal is a model that answers technical and how-to questions clearly and precisely, the way a good technical manual would: short sentences, numbered steps for procedures, consistent technical terms, and no vague wording or filler.

The writing target is **ASD-STE100 Simplified Technical English (STE)**, the standard used in aircraft maintenance manuals. Every rule the model is trained toward is checked by code, so "clearer" is a number, not an opinion.

## Models and data

**Trained in this project** (hosted on the Hugging Face Hub):

| Model | Link |
|---|---|
| Tuned language model (`rlhf_final`) | [YOUR_HF_USERNAME/smollm2-135m-ste-rlhf](https://huggingface.co/YOUR_HF_USERNAME/smollm2-135m-ste-rlhf) |
| Reward model (`reward_model`) | [YOUR_HF_USERNAME/modernbert-ste-reward-model](https://huggingface.co/YOUR_HF_USERNAME/modernbert-ste-reward-model) |

**Used in this project:**

| What | Role | Link |
|---|---|---|
| SmolLM2-135M-Instruct | Base language model that gets fine-tuned | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| ModernBERT-base | Backbone of the reward model | [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) |
| HelpSteer2 | Source of the questions and original answers (CC-BY-4.0) | [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| ASD-STE100 | Writing standard the model is trained toward | [asd-ste100.org](https://www.asd-ste100.org/) |

**Used in version 1** (`archive/v1/`):

| What | Role | Link |
|---|---|---|
| DistilBERT | Reward model backbone | [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) |
| hh-rlhf | Preference data for the reward model | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| Dolly 15k | Prompts for RL training | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |

## Results

Measured on 200 held-out questions that were never used in training:

| | Original model | Tuned model |
|---|---|---|
| STE score (0–100) | 72.4 | **92.1** |
| Answers passing ≥ 7 of 8 STE rules | 28% | **86.5%** |
| Average words per sentence | 17.2 | **11.1** |
| Passive sentences | 10.7% | **7.5%** |
| Reward-model score | −0.70 | **+0.05** |
| Answers that finish within the token limit | 96% | 91% |

Three training seeds gave nearly the same curves, so the result does not depend on one lucky run. The full tables, plots and analysis are in [results/experiments.md](results/experiments.md).

## How it works

RLHF here has three steps:

1. **Build preference data.** Take real questions and answers from the [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset. For each one, write a version of the answer that follows STE. The STE version is "chosen" and the original is "rejected". Add hard negatives: shuffled, off-topic, looping or cut-off answers that the reward model must learn to score low.
2. **Train a reward model.** A ModernBERT-base classifier learns to give the chosen answer a higher score than the rejected one. It now acts as an automatic judge of clear writing.
3. **Fine-tune with reinforcement learning.** The language model answers training questions, the reward model scores each answer, and **REINFORCE** updates the model to make high-scoring answers more likely. A **KL penalty** (β = 0.1) keeps the model close to the original so that it does not drift into nonsense that tricks the reward model. A safety stop ends training if the drift gets too large.

The tuned model learns *how* to write (shorter sentences, lists, plain words). It does not learn new facts: RLHF on a model this size changes style, not knowledge.

### The 8 STE rules (checked by code)

| Rule | Check |
|---|---|
| R1 | Sentences are short: 20 words or fewer for steps, 25 for other text |
| R2 | Active voice: at most 10% passive sentences |
| R3 | One instruction per sentence |
| R4 | Steps and items go in a list, not in a paragraph |
| R5 | Paragraphs are short: at most 3 sentences or 6 lines |
| R6 | No vague words ("some", "various", "a number of") |
| R7 | Simple words instead of complex ones ("use", not "utilize") |
| R8 | No filler phrases ("It is important to note that…") |

## Architecture

```
                          configs/v2.yaml  (all settings)
                                   │
 ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌────────────┐
 │ 1. Data      │──►│ 2. Reward    │──►│ 3. RL         │──►│ 4. Eval      │   │ 5. Web app │
 │ data/        │   │ reward/      │   │ rl/           │   │ eval/        │   │ webapp/    │
 │ prompts →    │   │ ModernBERT   │   │ REINFORCE     │   │ 200 held-out │   │ base vs.   │
 │ STE pairs    │   │ scores text  │   │ + KL penalty  │   │ questions    │   │ tuned      │
 └──────────────┘   └──────┬───────┘   └──────┬────────┘   └──────────────┘   └─────▲──────┘
                           │                  │                                     │
                           ▼                  ▼                                     │
              checkpoints/reward_model   checkpoints/rlhf_final ────────────────────┘

        ste/  (rule checker): used by data checks, evaluation and the web app badges
```

### Directory layout

```
fineTunedLM/
├── configs/v2.yaml           Every setting: models, data sizes, training, eval, serving
├── src/ste_rlhf/             The Python package
│   ├── common/               Shared helpers: config, model loading, generation, log-probs, KL, seeding
│   ├── data/                 Prompt selection, splits, rewrite batches, preference pairs
│   ├── ste/                  STE rule checker (R1–R8) and word lists
│   ├── reward/               Reward model: training, scoring, robustness probe
│   ├── rl/                   REINFORCE trainer and export of the final model
│   ├── eval/                 Held-out generation, metrics, plots, samples, human win-rate tool
│   └── serving.py            Loads base, tuned and reward models for the web app
├── webapp/                   FastAPI server + plain HTML/CSS/JS front end
├── tests/                    Unit tests, one file per stage
├── data/                     Prompts, STE rewrites and preference pairs (ready to use)
├── checkpoints/              Trained models (not in git: download from the Hub, see below)
├── results/                  Evaluation outputs, plots and experiments.md
├── report/                   Screenshots for the report
└── archive/v1/               Version 1 (generic "better answers" goal), kept for reference
```

### Checkpoints

Model weights are not stored in git. The two models the demo needs are on the Hugging Face Hub (see [Models and data](#models-and-data)); the experiment runs below were kept locally.

| Folder | What it is |
|---|---|
| `checkpoints/rlhf_final` | **The tuned model** (β = 0.1, seed 1). Used by the web app |
| `checkpoints/reward_model` | **The reward model** used for training and in the web app |
| `checkpoints/beta0.1_seed{0,1,2}` | Main runs with three seeds, each with `final/` and `step_150/` |
| `checkpoints/beta0_seed0` | Control run with no KL penalty (shows reward over-optimisation) |
| `checkpoints/pilot_*` | Short pilot runs used to choose the answer length and batch size |
| `checkpoints/reward_model_nodegen`, `_degen150` | Reward models trained without or with fewer hard negatives (ablation) |

## How to run

All commands run from the project root. You need Python 3.11+. Training works on Apple Silicon (MPS), CUDA or CPU; `device: auto` picks the best one.

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The first run downloads SmolLM2-135M-Instruct and ModernBERT-base from Hugging Face.

### 2. Run the demo

Download the two trained models into `checkpoints/`:

```bash
hf download YOUR_HF_USERNAME/smollm2-135m-ste-rlhf --local-dir checkpoints/rlhf_final
```

```bash
hf download YOUR_HF_USERNAME/modernbert-ste-reward-model --local-dir checkpoints/reward_model
```

Then start the web app:

```bash
uvicorn webapp.server:app --port 8000
```

Open http://localhost:8000, ask a question, and compare the original model with the tuned model side by side. Each answer shows its reward score and STE badges (sentence length, passive voice, list items).

You can override model paths with environment variables: `BASE_MODEL`, `TUNED_MODEL_PATH`, `REWARD_MODEL_PATH` and `DEVICE`.

### 3. Score any text against the STE rules

```bash
python -m ste_rlhf.ste.cli --text "It is important to note that the file is opened by the program." --show-fails
```

### 4. Reproduce training

The preference data is already in `data/`, so you can start from the reward model. Every command reads `configs/v2.yaml`. Override any value with `--set key=value`.

```bash
python -m ste_rlhf.reward.train
```

```bash
python -m ste_rlhf.rl.reinforce --set rl.run_name=beta0.1_seed1 rl.beta_kl=0.1 seed=1
```

```bash
python -m ste_rlhf.rl.export --run beta0.1_seed1
```

These steps train the reward model into `checkpoints/reward_model`, train the policy into `checkpoints/<run_name>/`, and copy the chosen run to `checkpoints/rlhf_final`. The other runs in the results use the same command with a different `rl.run_name`, `rl.beta_kl` and `seed`.

To rebuild the data from scratch, the modules in `src/ste_rlhf/data/` run in this order: `select_prompts` → `splits` → `writer_batches` → (write the STE rewrites for each batch) → `collect_writes` → `build_pairs`. In this project an LLM (Claude Sonnet) wrote the STE rewrites, and each one had to pass the rule checker.

### 5. Evaluate

```bash
python -m ste_rlhf.eval.generate_eval --model HuggingFaceTB/SmolLM2-135M-Instruct --tag base
```

```bash
python -m ste_rlhf.eval.generate_eval --model checkpoints/rlhf_final --tag tuned
```

```bash
python -m ste_rlhf.eval.metrics --tags base tuned
```

```bash
python -m ste_rlhf.eval.plots
```

```bash
python -m ste_rlhf.eval.winrate_cli --a tuned --b base
```

These generate answers on the 200 held-out questions, compute the metrics tables, draw the training plots, and (optionally) run a blind side-by-side human comparison in the terminal. Output goes to `results/`.

### 6. Run the tests

```bash
python -m pytest -m "not slow" tests webapp/tests
```

Drop `-m "not slow"` to also run the tests that load the real models.

## Version 1

`archive/v1/` holds the first version of the project. It aimed for generally "better" answers, using a DistilBERT reward model trained on hh-rlhf data and Dolly prompts (links above). Only its code and docs are kept. Its main lesson shaped v2: RLHF changes *how* a small model writes much more than *what* it knows, so v2 chose one clear, measurable writing style. See [archive/v1/project.md](archive/v1/project.md).
