"""REINFORCE trainer (project.md 6.4): sample, score, KL-penalised return, baseline, policy gradient."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from minirlhf.common import (RunLogger, generate, get_device, load_config, load_policy,
                             load_reference, per_token_kl, read_jsonl, resolve_run_name,
                             response_logprobs, set_seed)
from minirlhf.reward.factory import load_scorer

FALLBACK_PROMPTS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "prompts_small.jsonl"


def _gpu_mem_gb(device) -> float:
    """Memory held by the GPU allocator (driver-level on MPS), for diagnosing leaks."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def update_baseline(baseline, mean_return: float, ema: float):
    """EMA baseline; initialised with the first batch mean."""
    if baseline is None:
        return float(mean_return)
    return ema * baseline + (1.0 - ema) * float(mean_return)


def compute_advantages(returns: torch.Tensor, baseline, normalize: bool) -> torch.Tensor:
    """A = R - b (b = 0 when baseline is None), optionally divided by (std + 1e-8)."""
    adv = returns - (0.0 if baseline is None else baseline)
    if normalize:
        adv = adv / (adv.std(unbiased=False) + 1e-8)
    return adv


def reinforce_loss(logp: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """-(A.detach() * sum_t logp_t).mean() over the batch."""
    seq_logp = (logp * mask.to(logp.dtype)).sum(-1)
    return -(advantages.detach() * seq_logp).mean()


def load_train_prompts(cfg) -> list[str]:
    path = Path(cfg["data"]["prompts_dir"]) / "train.jsonl"
    if not path.exists():
        print(f"[reinforce] {path} not found, falling back to {FALLBACK_PROMPTS}")
        path = FALLBACK_PROMPTS
    return [r["prompt"] for r in read_jsonl(path)]


def sample_prompts(prompts, n, rng: random.Random):
    if len(prompts) >= n:
        return rng.sample(prompts, n)
    return [rng.choice(prompts) for _ in range(n)]


def _save(model, tok, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def train(cfg, log_root="logs", ckpt_root="checkpoints"):
    rl = cfg["rl"]
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = get_device(cfg)
    run = resolve_run_name(cfg)
    logger = RunLogger(Path(log_root) / run, cfg)
    ckpt_dir = Path(ckpt_root) / run

    model, tok = load_policy(cfg, device)
    ref = load_reference(cfg, device)
    scorer = load_scorer(cfg, device)
    prompts_pool = load_train_prompts(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(rl["lr"]))
    beta, baseline = float(rl["beta_kl"]), None
    kl_hist: list[float] = []
    steps, bs = int(rl["steps"]), int(rl["batch_size"])
    model.train()
    print(f"[reinforce] run={run} device={device} steps={steps} batch={bs} beta={beta}")

    for step in range(1, steps + 1):
        t0 = time.time()
        prompts = sample_prompts(prompts_pool, bs, rng)
        ro = generate(model, tok, prompts, cfg["policy"], greedy=False)
        scores = scorer.score(prompts, ro.responses).float().to(device)
        mask = ro.response_mask.to(device).float()

        with torch.no_grad():
            logp_ref = response_logprobs(ref, ro)
        logp, ent = response_logprobs(model, ro, return_entropy=True)
        kl = per_token_kl(logp.detach(), logp_ref, mask).sum(-1)
        returns = scores - beta * kl
        if rl["use_baseline"]:
            baseline = update_baseline(baseline, returns.mean().item(), float(rl["baseline_ema"]))
        adv = compute_advantages(returns, baseline if rl["use_baseline"] else None,
                                 bool(rl["normalize_advantages"]))
        loss = reinforce_loss(logp, mask, adv)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(rl["grad_clip"]))
        opt.step()

        if step % int(rl["log_every"]) == 0 or step == steps:
            ent_mean = ((ent.detach() * mask).sum() / mask.sum().clamp(min=1)).item()
            row = logger.log(step, dict(
                reward_mean=scores.mean().item(), reward_std=scores.std(unbiased=False).item(),
                kl_mean=kl.mean().item(), return_mean=returns.mean().item(),
                baseline=baseline if baseline is not None else 0.0, loss=loss.item(),
                entropy=ent_mean, resp_len_mean=mask.sum(-1).mean().item(), time_s=time.time() - t0,
                gpu_mem_gb=_gpu_mem_gb(device)))
            print(f"[{step}/{steps}] reward={row['reward_mean']:.3f} kl={row['kl_mean']:.3f} "
                  f"len={row['resp_len_mean']:.1f} ent={row['entropy']:.2f} {row['time_s']:.1f}s", flush=True)
        if step % int(rl["sample_every"]) == 0 or step == 1:
            k = min(4, bs)
            logger.log_samples(step, prompts[:k], ro.responses[:k], scores[:k].tolist())
        if int(rl["save_every"]) > 0 and step % int(rl["save_every"]) == 0 and step != steps:
            _save(model, tok, ckpt_dir / f"step_{step}")
        # Safeguard (like PPO's target_kl): once the reward plateaus, normalised
        # advantages are mostly noise and the policy can random-walk away from the
        # reference until it collapses. Stop when the windowed mean KL gets too high.
        kl_hist.append(kl.mean().item())
        max_kl, win = rl.get("max_kl"), int(rl.get("max_kl_window", 20))
        stop = max_kl is not None and len(kl_hist) >= win and sum(kl_hist[-win:]) / win > float(max_kl)
        del logp, ent, logp_ref, loss, ro
        # Response shapes vary every step; release cached blocks so the MPS
        # allocator doesn't grow until it hits unified-memory pressure.
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        if stop:
            msg = (f"mean KL over last {win} steps = {sum(kl_hist[-win:]) / win:.2f} "
                   f"> max_kl {max_kl}; stopping early at step {step}")
            print(f"[reinforce] {msg}", flush=True)
            with open(Path(logger.run_dir) / "stopped_early.json", "w") as f:
                json.dump({"stopped_at_step": step, "reason": msg}, f)
            break

    _save(model, tok, ckpt_dir / "final")
    print(f"[reinforce] saved {ckpt_dir / 'final'}")
    return run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    a = ap.parse_args(argv)
    train(load_config(a.config, a.overrides))


if __name__ == "__main__":
    main()
