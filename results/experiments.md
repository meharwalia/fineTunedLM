# v2 runs and evaluation results

Evaluation: 200 held-out HelpSteer2 prompts (never used in training or reward-model data). Greedy decoding, repetition penalty 1.1, up to 1024 tokens, which matches the web demo. Generated 2026-11-10.

## 1. The four training runs

All runs use SmolLM2-135M-Instruct, REINFORCE, the ModernBERT reward model, 256-token training cap, lr 1e-5, batch 8 × grad-accum 2, up to 300 steps, KL safety stop at mean KL > 15 over 20 steps.

| Run | Purpose | Stopped | Reward (first 20 → last 20 steps) | STE in training | KL at end | Finish rate | Entropy |
|---|---|---|---|---|---|---|---|
| β=0.1, seed 0 | Main run | Step 190, KL stop | −0.83 → +0.02 | 75.0 → 92.2 | 15.3 | 0.29 → 0.46 | 2.30 → 1.44 |
| β=0.1, seed 1 | Repeatability | Step 197, KL stop | −0.86 → +0.14 | 75.6 → 91.6 | 15.2 | 0.36 → 0.58 | 2.37 → 1.56 |
| β=0.1, seed 2 | Repeatability | Step 197, KL stop | −0.80 → +0.06 | 76.9 → 92.9 | 15.1 | 0.31 → 0.53 | 2.35 → 1.47 |
| β=0 | Control: no KL penalty | Ran all 300 (stop disabled) | −0.85 → +0.41 | 74.7 → 91.0 | 23.0 | 0.28 → 0.19 | 2.31 → 1.40 |

- **Seeds agree.** All three β=0.1 runs follow nearly the same reward and KL curves (`plots/seed_mean_std.png`), and each stopped at step 190–197. The result is not luck of one seed.
- **Every β=0.1 run hit the KL stop.** The penalty (β=0.1) slows drift but does not stop it within 300 steps; the hard stop is what ends training.
- **β=0 chases reward further.** It reaches the highest training reward (+0.41), with KL rising to 23–30. Its answers get longer and finish less often (0.19), the classic sign of over-optimising the reward.

## 2. Held-out evaluation (200 prompts)

| Model | STE | % pass (≥7/8 rules) | Avg sentence words | % passive | RM score | KL vs base | Finish rate | Answers that loop* | Distinct-2 |
|---|---|---|---|---|---|---|---|---|---|
| Base | 72.4 | 28% | 17.2 | 10.7 | −0.70 | 0 | 96% | 7 | 0.64 |
| Seed 0, step 150 | 88.1 | 74% | 10.8 | 6.8 | −0.02 | 20.8 | 83% | 35 | 0.58 |
| Seed 0, final | 90.2 | 80% | 9.7 | 6.5 | +0.02 | 30.2 | 70% | 69 | 0.53 |
| **Seed 1, final (demo model)** | **92.1** | **86.5%** | 11.1 | 7.5 | +0.05 | 25.9 | **91%** | **30** | **0.61** |
| Seed 2, final | 93.8 | 91.5% | 8.1 | 3.9 | −0.03 | 31.9 | 63% | 67 | 0.49 |
| β=0, final | 88.6 | 76.5% | 8.7 | 4.6 | +0.23 | 40.2 | 80% | 61 | 0.52 |

\* Loop = the same non-trivial line appears 3+ times in one answer. Distinct-2 = share of unique word pairs (lower = more repetition).

**Seed 1 vs base, per prompt:** STE higher on 157/200, equal on 34, lower on 9. The reward model prefers the tuned answer on 172/200.

### Why seed 1 is the demo model
It has the best balance: a high STE score (92.1, 86.5% passing), the fewest loops of any tuned model (30/200), 91% of answers finish on their own, and the lowest repetition among the tuned models. Seed 2 scores slightly higher on STE but loops more than twice as often and finishes only 63% of answers. Exported to `checkpoints/rlhf_final`.

### Step-150 vs final (seed 0)
Stopping earlier halves the looping (69 → 35) and raises the finish rate (70% → 83%) at a cost of 2 STE points. More steps buy a little more style and a lot more degeneration.

## 3. Reward hacking and failure modes

- **Looping lists.** The main failure. The tuned models learned that numbered lists score well, and some answers repeat one item ("Demon Baphomet's Equipment" × 20, "Ses Email: …" × 10). Looping grows with KL: base 7 → step-150 35 → final 69 → β=0 61.
- **β=0 games the reward model hardest.** It has the highest RM score (+0.23) but a lower STE score than the β=0.1 finals and the most drift (KL 40). The reward model rates its answers well even when they are not better STE: the gap between the reward model and the true goal opens as soon as the KL leash is removed.
- **Style, not knowledge.** STE form improves a lot; facts do not. A 135M model invents details (wrong salaries, wrong recipes, wrong code) in both the base and tuned versions. RLHF changes how it writes, not what it knows.
- **What held up.** The reward-model fixes before training (shuffled, off-topic, repeated and truncated negatives) worked. No run collapsed into gibberish, one-word answers or off-topic text, which the first reward model would have rewarded.

## Files
- `generations/<tag>.jsonl`: all 1,200 answers.
- `metrics/summary.md`, `metrics/<tag>.json`: full metric tables.
- `plots/`: reward, KL and STE curves; seed mean ± std; β comparison; STE before/after.
- `demo_candidates.md`: 18 candidate demo examples for review.
- `samples_adhoc/s1_vs_base_scored.json`: per-prompt scores for seed 1 vs base.

Note: `metrics/summary.json` also holds a `tuned` entry, a copy of `s1_final`, which the web app's results panel reads. Re-add it if metrics are regenerated.
