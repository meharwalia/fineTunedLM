# Mini-RLHF: Improving a Small LLM's Responses with Reinforcement Learning from Human Feedback

> A college project for **UCS760 – Edge AI and Robotics: Reinforcement Learning and Conversational AI**.
> Goal: reproduce the RLHF pipeline that ChatGPT-style models use, scaled down so it runs on a laptop or free Colab, and show the result in a simple web app.

---

## 1. Project summary

**One sentence.** We take a small pre-trained language model, improve the quality of its responses using Reinforcement Learning from Human Feedback (RLHF) with the REINFORCE algorithm, and build a web app that shows the original and improved model's answers side by side.

**30-second version.** Language models learn to produce *plausible* text, not necessarily *good* text. Writing perfect answers for a model to copy is expensive, but saying which of two answers is better is easy. So we collect preference pairs, train a small BERT model to predict those preferences as a score, and use that score as the reward in a reinforcement learning loop. The language model is the agent and each token is an action. REINFORCE, a policy-gradient algorithm, adjusts the model so that higher-scoring responses become more likely, and a KL penalty keeps it from drifting into nonsense that games the reward. A web app sends the same question to the original and the RLHF-tuned model so the difference is visible.

**Core idea for the report.** *The reward model turns human opinions into a number, and policy gradients turn that number into weight updates.*

### Deliverables
1. **RLHF pipeline**: data collection → reward model → REINFORCE fine-tuning → evaluation.
2. **Web demo**: a FastAPI server plus a simple frontend. The user types a question; it goes to the **base model** and the **RLHF-tuned model**, and both answers (with their reward scores) appear side by side.

### Constraints
- No training of an LLM from scratch. We use an **existing small pre-trained model**.
- Must run on a personal Mac (Apple Silicon `mps` backend) or free Google Colab (T4 GPU).
- Small budget: models under ~150M parameters, short generations (30–60 tokens), small batches (8–16).

### Defaults (can be changed)
| Decision | Default | Alternative |
|---|---|---|
| Definition of "quality" | **Helpful + concise + polite** answers | Positive sentiment (simplest), conciseness only |
| Hardware | Mac with `mps`, falling back to CPU | Google Colab T4 |
| RL algorithm | **REINFORCE with baseline** | (PPO / actor-critic as future work) |

---

## 2. Course material reviewed

All slides and documents in this folder were read. Relevance to the project:

| File(s) | Relevant? | Used for |
|---|---|---|
| 1. intro_RL, 2. Bellman Optimality Equation, 3. MDP_detailed, 4. MDP(review) | ✅ | Framing RLHF as an MDP / contextual bandit, value functions, returns |
| 5. Dynamic Programming | ◐ | Contrast: why model-based planning is impossible here |
| 6. Monte Carlo | ✅ | Monte Carlo returns: the reward arrives only at the end of a response |
| 7. Temporal Difference | ◐ | Contrast: why we use MC returns instead of TD bootstrapping |
| 8. Function Approximation | ✅ | Neural networks as policy and reward model |
| 9. Policy Gradient Methods | ✅✅ | **Core algorithm: Monte Carlo REINFORCE**; actor-critic and PPO as future work |
| 10. Deep RL, 11. DQN | ✅ | Deep policy networks, advantage `(return − baseline)`; DQN as contrast |
| postMidSem: 13. NLP, 14. Attention & Transformers, 15. BERT, UNIT 4 | ✅ | Tokenisation, decoder transformers (GPT), BERT fine-tuned with one output layer = reward model; conversational AI framing for the web demo |
| GPU-Architecture (both) | ◐ | Batched training and inference on GPU/MPS |
| 16. UCS760_experiment (lab list) | ◐ | Course uses Python, OpenAI Gym, Q-learning, SARSA, MC, DP, NLTK |
| 11/12. ASR, 16. NMT, 17. TTS, Robotic Manipulators | ❌ | Not used (apart from attention background) |

The Policy Gradient lecture gives REINFORCE as pseudocode. This project applies exactly that algorithm to a real language model.

---

## 3. How LLM fine-tuning works

An LLM is a large function with millions of numbers called **weights**. Given text, it outputs a probability for every possible next token:

```
"The capital of France is"  →  Paris: 0.82,  Lyon: 0.03,  the: 0.02, ...
```

**Fine-tuning means nudging those weights** so the probabilities shift toward outputs we want. The architecture doesn't change; we keep training an existing model on a new signal. Each training step:

1. Run text through the model to get its probabilities.
2. Compute a **loss**, a number for how wrong it was.
3. **Backpropagation** works out the direction to move every weight to reduce the loss.
4. **Gradient descent** moves the weights a tiny step in that direction.

What differs between kinds of fine-tuning is **where the loss comes from**:

| Type | Loss signal | In words |
|---|---|---|
| Supervised fine-tuning (SFT) | Cross-entropy against a correct answer | "Here's the right answer: copy it." |
| RL fine-tuning (RLHF) | Reward × log-probability of what the model wrote | "You wrote this, it scored well: do more like it." |

SFT needs someone to write the perfect answer. RLHF only needs someone to say **which of two answers is better**, which is much cheaper. That's why RLHF exists.

---

## 4. Models

| Role | Model | Trained? | Job |
|---|---|---|---|
| **Policy model** | `HuggingFaceTB/SmolLM2-135M-Instruct` (135M) | ✅ Yes, this is the model we improve | Writes responses |
| **Reference model** | Frozen copy of the starting policy | ❌ Frozen | The "leash": the KL penalty measures drift from it. Also the "normal LLM" in the web demo |
| **Reward model** | `distilbert-base-uncased` + one-number output head | Trained once, then frozen | Judges responses and outputs a quality score |

**Why these models**
- **SmolLM2-135M-Instruct** is about GPT-2 size, so it trains on a laptop. It's already instruction-tuned, so it *answers* prompts and the SFT stage of RLHF is done for us.
- **DistilBERT**: the BERT lecture notes that BERT "can be fine-tuned with just one additional output layer". That's exactly a reward model: text in, one score out.
- From the Transformers/BERT slides: GPT-style models are **unidirectional decoders** (generate text); BERT is a **bidirectional encoder** (understands and judges text). RLHF uses one of each.

**Backup choice.** If SmolLM2 causes problems, swap in GPT-2 small (124M) or distilgpt2 (82M). This is an *alternative*, not an extra model in the system.

---

## 5. Data

### 5.1 Prompts (the "states")
A few hundred to a few thousand short questions, from:
- public datasets: `databricks/databricks-dolly-15k`, `Anthropic/hh-rlhf`, and/or
- 100–200 hand-written prompts on one theme (study help, customer support, …).

Split into **train** prompts (used in the RL loop) and **eval** prompts (never trained on, used for before/after comparison).

### 5.2 Preference pairs (the "human feedback")
Each record: *(prompt, chosen response, rejected response)*. Three sources, combined:

1. **Own labels (the real "HF").** The policy generates 2 responses per prompt; a terminal script shows A vs. B and we press 1 or 2. Target: 300–500 labels.
2. **Existing human preference data.** `Anthropic/hh-rlhf` has ~160k real human chosen/rejected pairs. Use a 2–5k subset (short, single-turn examples only).
3. **Rule/AI labels (RLAIF), to scale cheaply.** Rules such as "prefer the answer that is on-topic, polite and under 80 words" label thousands of pairs automatically.

Each pair records its `source` (`human`, `rule`, `hh-rlhf`) so we can report how each source affects reward-model accuracy.

---

## 6. RLHF as a reinforcement learning problem

### 6.1 MDP formulation
| MDP element | In the LLM |
|---|---|
| State `s_t` | Prompt + tokens generated so far |
| Action `a_t` | Next token, from a vocabulary of ~49k (discrete) |
| Policy `π_θ(a\|s)` | The LLM's softmax over the vocabulary |
| Transition | Deterministic: append the token |
| Episode | One full response, ending at end-of-sequence or a length limit (episodic, γ = 1) |
| Reward | Reward-model score at the **end** only (sparse) − KL penalty |

Alternative view (from the MDP lecture, "bandits are MDPs with one state"): RLHF can be treated as a **contextual bandit** where the whole response is a single action that receives one reward. With γ = 1 and a single terminal reward, REINFORCE's return is the same for every token of a response, so both views give the same update.

### 6.2 What the reward is
The **reward is one number** saying how good a response is (higher = better). The reward model produces it:

```
Prompt:  "How do I make my code run faster?"
Resp. A: "Profile it first to find the slow part, then optimise that."  → +2.1
Resp. B: "idk lol just buy a better computer"                           → −1.4
```

The scores have no units; only the ordering matters. The return actually used in RL is:

```
R = reward-model score − β · KL(π_θ ‖ π_ref)
```

Without the KL penalty the policy finds strange text that fools the reward model ("great great great helpful!!!"). This failure is called **reward hacking**.

### 6.3 Training the reward model (Bradley–Terry)
For each pair, score both responses and push the chosen one above the rejected one:

```
loss = −log σ( r(prompt, chosen) − r(prompt, rejected) )
```

Metric: **pairwise accuracy** on held-out pairs (fraction where chosen scores higher). About 65–75% is acceptable for this project.

### 6.4 REINFORCE: how the reward model's score changes the policy
The reward model never edits the policy's weights directly. Its score becomes a **multiplier on the learning signal**. Monte Carlo REINFORCE (Lecture 9):

```
θ ← θ + α · (R − b) · ∇θ Σ_t log π_θ(token_t | context_t)
```

- `∇ log π(token)` is the direction that makes that token more likely.
- `R` is the response's return (reward-model score − KL penalty), a **Monte Carlo** return because it's only known once the response is finished.
- `b` is a **baseline**: a running average of recent returns. Subtracting it doesn't change the expected gradient but greatly reduces variance (the lecture's "choose extensions to reduce variance").
- `A = R − b` is the **advantage**: how much better than usual this response was.

| Situation | Advantage | Effect |
|---|---|---|
| Better than usual | + | Every token in that response becomes **more likely** |
| Worse than usual | − | Every token becomes **less likely** |
| About usual | ≈ 0 | Little change |

Example: return +2.1, baseline +0.5, so the advantage is +1.6. Gradient descent nudges the weights so "Profile it first…" becomes slightly more probable for similar prompts.

**One training step**
1. Sample a batch of training prompts.
2. Policy generates a response for each (sampling with temperature = exploration).
3. Reward model scores each response.
4. Compute KL vs. the reference model; return `R = score − β·KL`.
5. Update baseline `b ← 0.9·b + 0.1·mean(R)`; advantage `A = R − b` (normalised by batch std).
6. Loss `= −mean(A · Σ_t log π_θ(token_t))`; backpropagate; clip gradients; optimiser step.

**Future work (not implemented):** actor-critic / PPO (learned critic instead of a running-average baseline, clipped updates) and DPO (no RL loop).

---

## 7. Pipeline

```
 STAGE 1 — Starting model
 ┌──────────────────────────┐
 │ SmolLM2-135M-Instruct    │──── copy & freeze ───▶  Reference model
 │ (already instruction-    │                         (= "normal LLM" in demo)
 │  tuned = SFT done)       │
 └────────────┬─────────────┘
              │
 STAGE 2 — Learn what "good" means
              ▼
   Prompts ──▶ SmolLM2 writes 2 answers each
              ▼
   You (or rules / hh-rlhf) pick the better one   ← the "human feedback"
              ▼
   ┌──────────────────────────┐
   │ Train DistilBERT reward  │   loss: chosen must score > rejected
   │ model on the pairs       │
   └────────────┬─────────────┘
                │ (frozen judge)
 STAGE 3 — REINFORCE loop, repeat a few hundred times
                ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 1. Policy answers a batch of prompts (sampling)         │
   │ 2. Reward model scores each answer        → reward      │
   │ 3. Subtract KL penalty vs. reference model → return R   │
   │ 4. Advantage = R − running-average baseline             │
   │ 5. Update: make high-advantage answers more likely      │
   └─────────────────────────────────────────────────────────┘
                ▼
 STAGE 4 — Evaluate
   Held-out prompts, before vs. after: mean reward, KL,
   own A/B judgement (win rate), sample answers side by side
                ▼
 STAGE 5 — Web demo
   Browser ──▶ FastAPI ──┬──▶ Base model       ──┐
                         └──▶ RLHF-tuned model ──┴──▶ both answers + reward scores
```

---

## 8. Web demo

**Purpose.** Show the effect of RLHF visually: one question in, two answers out, side by side.

| Part | Design |
|---|---|
| **Backend** | FastAPI. It never loads models itself: it calls one function interface, `webapp/model_interface.py` (`load_models`, `get_status`, `generate_response(prompt, model_kind, settings)`). The web app is built against the signatures; an integration step later connects them to the RLHF pipeline's base model, tuned model and reward model. No mock model service. |
| **Endpoint** | `POST /api/compare` with `{"prompt": "..."}` → base answer, tuned answer, each with reward score and generation time (each model's result reported separately, so one failing doesn't hide the other) |
| **Fairness** | Both models get identical generation settings (greedy by default) so the only difference is the RLHF training |
| **Frontend** | One static HTML page with plain JavaScript (no build step): a text box, a "Compare" button, example-question chips, two answer cards ("Original model" vs "After RLHF") with a reward bar each |
| **Before integration / training** | Before the interface is connected, both cards say "Model not connected yet". After connection but before training, the tuned card says "Model not trained yet". The server runs in every state. |
| **Run** | `uvicorn webapp.server:app --port 8000`, then open `http://localhost:8000` |

---

## 9. Evaluation and experiments

| Experiment | What it shows |
|---|---|
| **Before vs. after** on held-out prompts | Mean reward, win rate judged by us, example responses |
| **Seeds** (3 runs at the default β) | How stable REINFORCE is; mean ± std reward curves |
| **β sweep** (e.g. 0, 0.02, 0.05, 0.2) | Trade-off between reward gain and KL drift |
| **Reward hacking demo** (β = 0) | Policy exploits the reward model; why the KL term exists |
| **Baseline ablation** (with vs. without baseline) | Variance reduction from the baseline (Lecture 9) |
| **Reward model accuracy by data source** | Value of human vs. rule vs. hh-rlhf labels |
| **Live demo** | Web app comparing base and tuned answers |

Metrics logged every step: mean reward, mean KL, loss, response length, entropy.

---

## 10. Concepts from the RL coursework used

| Lecture | Where it appears |
|---|---|
| **1. Intro to RL** | Agent = LLM, environment = prompts + reward model; reward hypothesis; policy as the agent's behaviour function |
| **2. Bellman equations** | Return definition with γ = 1 for episodic tasks; the baseline approximates the value V(s) of a prompt |
| **3–4. MDPs** | Token-level MDP; bandit = MDP with one state → contextual-bandit view of RLHF |
| **5. Dynamic Programming** | Contrast: DP needs a full model and sweeps all states (~49k^50 here), hence model-free RL |
| **6. Monte Carlo** | The reward arrives at the end of the episode → Monte Carlo return; exploration by sampling with temperature instead of ε-greedy |
| **7. Temporal Difference** | Contrast: TD bootstraps from a value estimate; with a single terminal reward and short episodes, MC returns are simple and unbiased. TD/actor-critic is future work |
| **8. Function Approximation** | Policy and reward model are neural networks trained by SGD; non-stationary data explains why RL fine-tuning needs small learning rates |
| **9. Policy Gradients** | **Monte Carlo REINFORCE** exactly as in the lecture pseudocode; `∇ log π · (G − b)`; variance reduction with a baseline; KL as regularisation |
| **10. Deep RL** | RL + DL; deep policy network; advantage = return − baseline (the A3C slide's `q_t − V(s)` idea) |
| **11. DQN** | Contrast: argmax over 49k tokens per step is impractical, so policy-based; frozen reference model ≈ frozen target network |
| **13–15. NLP, Transformers, BERT** | Tokenisation, decoder transformer as policy, BERT + one output layer as reward model; conversational AI (query in, response out) in the web demo |
| **GPU lecture** | Batched generation and training on GPU/MPS |

**Summary line for the report:** RLHF here is Monte Carlo REINFORCE with a baseline on a token-level MDP, with a learned BERT reward model in place of a hand-coded reward and a KL penalty as regularisation.

---

## 11. Technology stack

| Purpose | Tool |
|---|---|
| Language | Python 3.11 or 3.12 in a virtual environment (safer for ML wheels than the system 3.14) |
| Deep learning | PyTorch (`mps` on Apple Silicon, `cuda` on Colab, CPU fallback) |
| Models and tokenisers | Hugging Face `transformers` |
| Datasets | Hugging Face `datasets` |
| Web server | FastAPI + Uvicorn |
| Frontend | Plain HTML, CSS and JavaScript (no framework, no build step) |
| Config | YAML |
| Plots | matplotlib |
| Tests | pytest (+ FastAPI `TestClient`) |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Training too slow on laptop | Short generations, small batches, Colab fallback, distilgpt2 as a smaller policy |
| Reward hacking | KL penalty; inspect samples every N steps; cap response length |
| Weak reward model (near 50% accuracy) | More/cleaner pairs; hh-rlhf subset; simpler quality definition |
| REINFORCE is high-variance / unstable | Baseline, advantage normalisation, low learning rate, gradient clipping, larger batches |
| Policy and reward model use different tokenisers | Reward model always receives decoded **text**, never policy token ids |
| Web demo slow on first request | Load models at startup; short max length; greedy decoding |

---

## 13. Presentation outline

1. **Problem.** LLMs aren't aligned with what people consider a good answer.
2. **Idea.** Learn from comparisons instead of correct answers.
3. **RL framing.** The MDP table.
4. **Pipeline.** The diagram in section 7.
5. **Algorithms.** Bradley–Terry reward model, REINFORCE with baseline, KL penalty.
6. **Results.** Reward before vs. after, β sweep, reward-hacking demo, sample responses.
7. **Live demo.** The web app comparing both models.
8. **Course link.** The concept table in section 10.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Policy** | The model that chooses actions (here: the LLM choosing tokens) |
| **Reference model** | Frozen copy of the starting policy, used for the KL penalty and as the "normal LLM" in the demo |
| **Reward model** | Network that maps (prompt, response) → one quality score |
| **Return** | Total reward for an episode; here the reward-model score minus the KL penalty |
| **Baseline** | Running average of recent returns, subtracted to reduce variance |
| **Advantage** | Return − baseline: how much better than usual a response was |
| **KL divergence** | Measure of how far the policy's probabilities have drifted from the reference model's |
| **REINFORCE** | Monte Carlo policy-gradient algorithm: increase log-probability of actions in proportion to their return |
| **Bradley–Terry loss** | Pairwise loss that trains the reward model to rank chosen above rejected |
| **Reward hacking** | Policy exploiting flaws in the reward model instead of genuinely improving |
| **SFT** | Supervised fine-tuning on example answers |
| **RLAIF** | RL from AI (or rule-based) feedback instead of human labels |

See [implementation_plan.md](implementation_plan.md) for the build plan.
