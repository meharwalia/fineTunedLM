"""REINFORCE with a KL penalty and an EMA baseline.

    R = score(x, y) - beta * sum_t (logp_t - logp_ref_t)
    A = (R - b) / (std + 1e-8)          (b = EMA of batch-mean R, init from first batch)
    loss = -(A.detach() * sum_t logp_t).mean() / grad_accum

Run: python -m ste_rlhf.rl.reinforce [--config ...] [--set KEY=VALUE ...]
The STE score is logged only; it never enters the reward.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from ste_rlhf.common import (RunLogger, add_config_args, generate, get_device, load_config,
                             load_policy, load_reference, per_token_kl, read_jsonl,
                             resolve_run_name, response_logprobs, set_seed)

PROMPTS_PATH = "data/prompts/rl_train.jsonl"
FALLBACK_PROMPTS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "prompts.jsonl"


def update_baseline(baseline, batch_mean: float, ema: float) -> float:
    """EMA baseline; initialised from the first batch mean."""
    if baseline is None:
        return float(batch_mean)
    return ema * baseline + (1.0 - ema) * float(batch_mean)


def compute_advantages(returns: torch.Tensor, baseline, normalize: bool) -> torch.Tensor:
    """A = R - baseline (batch mean if no baseline); optionally divided by the
    micro-batch std of R (+1e-8)."""
    adv = returns - (baseline if baseline is not None else returns.mean())
    if normalize:
        adv = adv / (returns.std(unbiased=False) + 1e-8)
    return adv


def reinforce_loss(advantages: torch.Tensor, logp: torch.Tensor, mask: torch.Tensor,
                   grad_accum: int = 1) -> torch.Tensor:
    """-(A.detach() * sum_t logp_t).mean() / grad_accum"""
    seq_logp = (logp * mask.to(logp.dtype)).sum(1)
    return -(advantages.detach() * seq_logp).mean() / grad_accum


def should_stop_kl(kl_history, max_kl, window) -> bool:
    """True if max_kl is set and the mean of the last `window` step KLs exceeds it."""
    if max_kl is None or not kl_history:
        return False
    window = max(1, int(window))
    if len(kl_history) < window:
        return False
    recent = kl_history[-window:]
    return sum(recent) / len(recent) > float(max_kl)


def _empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def gpu_mem_gb(device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _ste_scores(texts):
    """STE score per text (logging only), or None entries if the STE module is missing."""
    try:
        from ste_rlhf.ste.metrics import ste_report
    except ImportError:
        return [None] * len(texts)
    return [float(ste_report(t)["ste_score"]) for t in texts]


def load_prompts(cfg):
    path = Path(PROMPTS_PATH)
    if not path.exists():
        print(f"WARNING: {PROMPTS_PATH} not found; falling back to {FALLBACK_PROMPTS}")
        path = FALLBACK_PROMPTS
    return [r["prompt"] for r in read_jsonl(path)]


def _save(model, tok, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def train(cfg, scorer=None, prompts=None, run_name=None, root=".", device=None):
    """Returns dict(run_name, steps_done, stopped_early, kl_history)."""
    rl = cfg["rl"]
    root = Path(root)
    device = device or get_device(cfg)
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    run = run_name or resolve_run_name(cfg)
    log_dir, ckpt_dir = root / "logs" / run, root / "checkpoints" / run
    logger = RunLogger(log_dir, cfg)

    if prompts is None:
        prompts = load_prompts(cfg)
    if scorer is None:
        from ste_rlhf.reward.scorer import load_scorer
        scorer = load_scorer(cfg, device)

    policy, tok = load_policy(cfg, device)
    ref = load_reference(cfg, device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(rl["lr"]))
    beta, accum = float(rl["beta_kl"]), int(rl["grad_accum"])
    use_baseline = bool(rl.get("use_baseline", True))
    save_steps = set(rl.get("save_steps") or [])
    mb = rl.get("logprob_micro_batch")

    baseline, kl_hist, stopped, steps_done = None, [], None, 0
    for step in range(1, int(rl["steps"]) + 1):
        t0 = time.time()
        acc = {k: [] for k in ("reward", "kl", "ret", "ent", "len", "fin", "ste")}
        loss_total = 0.0
        sample_rows = None
        policy.train()
        for _ in range(accum):
            batch = rng.sample(prompts, min(int(rl["batch_size"]), len(prompts)))
            roll = generate(policy, tok, batch, cfg["policy"], greedy=False)
            mask = roll.response_mask.to(device)
            with torch.no_grad():
                rewards = scorer.score(batch, roll.responses).float().to(device)
                logp_ref = response_logprobs(ref, roll, micro_batch=mb)
            logp, ent = response_logprobs(policy, roll, return_entropy=True, micro_batch=mb)
            kl = per_token_kl(logp.detach(), logp_ref, mask).sum(1)
            ret = rewards - beta * kl
            if use_baseline and baseline is None:
                baseline = update_baseline(None, ret.mean().item(), float(rl["baseline_ema"]))
            adv = compute_advantages(ret, baseline, bool(rl.get("normalize_advantages", True)))
            loss = reinforce_loss(adv, logp, mask, accum)
            loss.backward()
            if use_baseline:
                baseline = update_baseline(baseline, ret.mean().item(), float(rl["baseline_ema"]))

            n_tok = mask.sum().clamp(min=1)
            loss_total += loss.item()
            acc["reward"] += rewards.tolist()
            acc["kl"] += kl.tolist()
            acc["ret"] += ret.tolist()
            acc["ent"].append(((ent * mask).sum() / n_tok).item())
            acc["len"] += mask.sum(1).float().tolist()
            acc["fin"] += [float(f) for f in roll.finished]
            ste = _ste_scores(roll.responses)
            acc["ste"] += ste
            if sample_rows is None:
                sample_rows = (batch, roll.responses, rewards.tolist(), roll.finished, ste)
            del roll, mask, rewards, logp_ref, logp, ent, kl, ret, adv, loss
            _empty_cache(device)

        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(rl["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        _empty_cache(device)

        mean = lambda xs: sum(xs) / len(xs)
        ste_vals = [s for s in acc["ste"] if s is not None]
        r = torch.tensor(acc["reward"])
        row = dict(reward_mean=r.mean().item(), reward_std=r.std(unbiased=False).item(),
                   kl_mean=mean(acc["kl"]), return_mean=mean(acc["ret"]),
                   baseline=baseline, loss=loss_total, entropy=mean(acc["ent"]),
                   resp_len_mean=mean(acc["len"]), finished_rate=mean(acc["fin"]),
                   ste_score_mean=mean(ste_vals) if ste_vals else None,
                   time_s=time.time() - t0, gpu_mem_gb=gpu_mem_gb(device))
        if step % int(rl.get("log_every", 1)) == 0:
            logger.log(step, row)
        if sample_rows and (step == 1 or step % int(rl["sample_every"]) == 0):
            p, resp, rw, fin, st = sample_rows
            logger.log_samples(step, p[:4], resp[:4], rw[:4], finished=fin[:4], ste_score=st[:4])
        print(f"[{run}] step {step}/{rl['steps']} reward {row['reward_mean']:.3f} "
              f"kl {row['kl_mean']:.2f} len {row['resp_len_mean']:.0f} "
              f"fin {row['finished_rate']:.2f} {row['time_s']:.1f}s mem {row['gpu_mem_gb']:.1f}GB",
              flush=True)
        steps_done = step
        kl_hist.append(row["kl_mean"])

        if step in save_steps:
            _save(policy, tok, ckpt_dir / f"step_{step}")
        if should_stop_kl(kl_hist, rl.get("max_kl"), rl.get("max_kl_window", 20)):
            stopped = {"stopped_at_step": step,
                       "reason": f"mean KL over last {rl.get('max_kl_window', 20)} steps "
                                 f"exceeded max_kl={rl['max_kl']}"}
            (log_dir / "stopped_early.json").write_text(json.dumps(stopped, indent=2))
            print(f"[{run}] KL early stop at step {step}")
            break

    _save(policy, tok, ckpt_dir / "final")
    return {"run_name": run, "steps_done": steps_done, "stopped_early": stopped,
            "kl_history": kl_hist}


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
