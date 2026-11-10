# S8 RL pilot report (16 GB Apple-silicon MPS, SmolLM2-135M, real reward model)

All runs: 12 optimizer steps requested, rl.save_steps=[], sample_every=4, prompts from rl_train.jsonl.
**All runs used rl.logprob_micro_batch=2.** With the default micro-batch 4, cap 512 OOM'd on step 1 (MPS "Insufficient Memory").
Logs: logs/pilot_<name>/, stdout logs/pilot_pilot_<name>.out, status logs/pilot_status.txt.

| Run | Answers/step | Steps done | s/step mean (excl. step 1) [min-max] | Peak mem (reported) | resp_len | finished | reward 1-3 -> last 3 | kl_mean end | STE score 1-3 -> last 3 | 300 steps |
|---|---|---|---|---|---|---|---|---|---|---|
| A cap512 (mb2) | 16 | 1 (stalled on step 2, killed after >5 min) | step 1 = 36 s | 12.2 GB | 260 | 0.88 | -0.58 (step 1) | 0.0 | 79.7 | not viable here (swap thrash; mb4 OOM) |
| B cap384 (mb2) | 16 | 10 (OOM crash on step 11) | 63 s mean, median 59 [18-126]; clean steps 1-2: ~19 s | 6.7 GB (5.7 steady) | 272 | 0.74 | -0.53 -> -0.49 | -1.04 | 73.7 -> 78.6 | ~1.6 h if clean (19 s); 5.3 h at the observed 63 s |
| C cap256 (mb2) | 16 | 12 OK | 14.3 [11.8-15.6] | 4.9 GB | 216 | 0.35 | -0.44 -> -0.43 | -0.84 | 78.4 -> 79.7 | **1.2 h** |
| D cap384 b8 x a1 (mb2) | 8 | 12 OK | 12.0 [10.1-16.3] | 6.7 GB | 297 | 0.68 | -0.48 -> -0.65 | -1.03 | 76.0 -> 70.8 | 1.0 h (but only 8 answers/step) |

Notes on the table
- B's step times were erratic (18 s, 18 s, then 126, 117, 59, ... s) and it eventually OOM'd, while D (same cap, half the batch per step) was steady at about 12 s. Heavy macOS swap was visible (5 GB swap used). Another cap384/accum2 step is about 2x D, so ~20-25 s clean, but the machine cannot hold that reliably. A also showed 12 GB peak, so the 16 GB unified memory is the limit.
- A uvicorn process (webapp.server, port 8000, idle, pid 19376) was running throughout; it is not a GPU job, and I did not stop it. It and other desktop apps contribute to memory pressure.
- Only 12 steps at lr 1e-5: reward and STE-score changes are inside noise (reward -0.53 -> -0.49 for B, flat for C, down for D). Nothing here shows learning yet, and nothing shows breakage. 12 steps is too short to conclude.
- **Flag for the lead: kl_mean is negative and drifts to about -1 to -2 in every run.** An unbiased sampled KL estimate should be about >= 0. This suggests the rollout sampling distribution (temperature/top_p/repetition_penalty in generate) differs from the plain policy logprobs used by response_logprobs, or the sign/estimator in per_token_kl. The beta*KL term is therefore currently a bonus for drifting. Worth checking before the 300-step runs (step 1 KL is ~5e-6, correct).
- finished_rate drops sharply with the cap: 0.88 at 512 (step 1), ~0.7 at 384, ~0.35 at 256. At cap 256 most answers are truncated, so the RM sees cut-off text, and RL could learn to be shorter or stop mid-thought.

## Sample answers (run B cap384, last sampled step 8; first 300 chars)
1. (reward -0.91, not finished) "I can assist you with creating an AIDA framework to develop a product campaign, focusing on the target audience of young people and TikTok. The follow-up content will focus on defining the target audience, specifying the desired outcomes of the campaign, and creating the overall direction for the pr"
2. (reward +0.03, not finished) "Welcome to San Francisco, the Golden State, the proud home of the Sun-Santa Angels – my nerdy friends! Here are some must-do activities and experiences for nerdy tourists:\n\n1. **Mysteries and Strange Events:** San Francisco is home to many infamous ghostly apparitions and unexplained phenomena. Be p"
Assessment: fluent, on-topic English, not garbage. Content is shallow with hallucinated details (Sun-Santa Angels, Rocky Mountains in SF), typical of a 135M model at temperature sampling.

## 135M vs 360M (greedy, 384 tokens, first 3 eval prompts, same MPS device)
SmolLM2-360M-Instruct was not cached; downloaded it (HF_HUB_DISABLE_XET=1; it is now in ~/.cache/huggingface/hub).

| Prompt | 135M | 360M |
|---|---|---|
| 1 AI painting of automobiles | 2.3 s: "To visualize the painting process of automobiles using AI, you could start by creating a series of interactive visualizations that allow users to explore different aspects of AI-driven painting. Here'" | 4.9 s: "Sure, I can help you with that. Here are some ways you can visualize the use of AI in the painting process of automobiles:\n\n1. **AI-Generated Images**: You can use AI to generate images of cars that a" |
| 2 Excel yes/no count | 1.8 s: "To count all the yes/no Share Prompt in Excel, you can use the following formula:\n\n=COUNTIF(A:A,IF(A:A=\"Yes\",A:A=\"No\",A:A=\"Yes\"))\n\nHere's how it works:\n\n1. The formula starts with the asterisk (*) to " | 1.8 s: "To count the number of yes/no responses in an Excel formula, you can use the following formula:\n\n=COUNTIF(A1:A10,\"Yes\")\n\nThis formula will count the number of cells in the range A1:A10 that contain th" |
| 3 autonomous cell tower | 2.1 s: "A completely autonomous smart cell tower that had no climb technology would be a marvel of modern engineering. Here's a possible scenario:\n\nImagine a tower that's designed to be self-sufficient, with " | 4.9 s: "A completely autonomous smart cell tower that had no climb technology would consist of a network of interconnected, self-sustaining, and highly efficient communication systems. Here's a possible break" |

360M is noticeably more correct (valid COUNTIF vs a nonsense nested IF in 135M) and ~1.5-2.5x slower to generate. 135M is acceptable for the project's purpose (STE style shift is measurable even with weak content) but its factual quality is poor; 360M RL does not fit this 16 GB machine in practice given 135M cap384 already struggles, so I did not pilot 360M RL.

## Recommendation
- **Use cap 256 (or 320 if tolerable), accum 2, batch 8, logprob_micro_batch 2.** It is the only 16-answer config that ran cleanly: 14.3 s/step, 4.9 GB, so about 1.2 h per 300-step run (about 7 runs in the S10 list = ~8.5 h). Cap 384 at 16 answers/step is unstable here (OOM at step 11, swap-driven 60-125 s steps), and cap 512 does not fit (mb4 OOM, mb2 stalled at 12 GB).
- Fallback if you want longer answers: cap 384 with accum 1 (8 answers/step, 12 s/step, 1.0 h/300 steps, 6.7 GB, stable) at the cost of noisier gradients; consider lr or more steps to compensate. Close other apps / stop the port-8000 uvicorn server before long runs to reduce swap.
- The cap-256 truncation (finished ~0.35) is the main cost; evaluation at cap 1024 is a separate, generation-only job and fits fine. Make sure the RM and STE metric are applied to truncated text consistently in train and eval, or the policy may learn to shorten answers.
- Fix/check the negative KL before the sweep (see flag above).
- 135M answers: fluent but shallow and sometimes wrong; good enough for a demonstrable style/RM-score shift, not for factual quality. Say so in the report.
