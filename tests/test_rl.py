import json
import math

import pytest
import torch
import yaml

from ste_rlhf.rl.export import export
from ste_rlhf.rl.reinforce import (compute_advantages, reinforce_loss, should_stop_kl, train,
                                   update_baseline)


class FakeScorer:
    """Test-only scorer: shorter answers and answers containing '1.' score higher."""

    def score(self, prompts, responses):
        return torch.tensor([-len(r) / 100.0 + (1.0 if "1." in r else 0.0) for r in responses])


def test_baseline_update():
    assert update_baseline(None, 2.0, 0.9) == 2.0
    assert math.isclose(update_baseline(2.0, 4.0, 0.9), 0.9 * 2.0 + 0.1 * 4.0)


def test_advantage_normalisation():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a = compute_advantages(r, 2.5, normalize=True)
    assert abs(a.mean().item()) < 1e-6
    assert math.isclose(a.std(unbiased=False).item(), 1.0, rel_tol=1e-4)
    raw = compute_advantages(r, 2.0, normalize=False)
    assert torch.allclose(raw, r - 2.0)
    # no baseline: centred on the batch mean
    assert abs(compute_advantages(r, None, False).mean().item()) < 1e-6


def test_loss_sign_and_gradient_direction():
    # logits param -> logp; the higher-advantage sample must get its logp pushed up
    # by a gradient-descent step, the lower-advantage sample pushed down.
    logp = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]], requires_grad=True)
    mask = torch.ones(2, 2)
    adv = torch.tensor([1.0, -1.0])
    loss = reinforce_loss(adv, logp, mask, grad_accum=1)
    loss.backward()
    assert logp.grad[0].sum() < 0   # descent increases logp of the good sample
    assert logp.grad[1].sum() > 0   # and decreases it for the bad one
    # grad_accum scaling and mask
    logp2 = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    l2 = reinforce_loss(torch.tensor([1.0]), logp2, torch.tensor([[1, 0]]), grad_accum=2)
    assert math.isclose(l2.item(), 0.5)  # -(1 * -1)/2
    l2.backward()
    assert logp2.grad[0, 1].item() == 0.0


def test_advantage_not_differentiated():
    adv = torch.tensor([1.0], requires_grad=True)
    logp = torch.tensor([[-1.0]], requires_grad=True)
    reinforce_loss(adv, logp, torch.ones(1, 1)).backward()
    assert adv.grad is None


def test_kl_early_stop_decision():
    assert not should_stop_kl([100.0], None, 1)
    assert not should_stop_kl([], 1.0, 1)
    assert should_stop_kl([20.0], 15.0, 1)
    assert not should_stop_kl([20.0, 1.0], 15.0, 2)      # mean 10.5
    assert should_stop_kl([1.0, 30.0, 20.0], 15.0, 2)    # mean 25
    assert not should_stop_kl([30.0], 15.0, 2)           # window not yet full
    assert should_stop_kl([0.0], -1, 1)


def test_export_fake_run(tmp_path):
    run = "fake_run"
    (tmp_path / "checkpoints" / run / "final").mkdir(parents=True)
    (tmp_path / "checkpoints" / run / "final" / "model.safetensors").write_text("w")
    old = tmp_path / "checkpoints" / "rlhf_final"
    old.mkdir()
    (old / "stale.txt").write_text("x")
    logd = tmp_path / "logs" / run
    logd.mkdir(parents=True)
    (logd / "metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step": 7}\n')
    cfg = {"policy": {"name": "base/m", "max_new_tokens": 512},
           "rl": {"beta_kl": 0.1}, "reward": {"checkpoint": "checkpoints/reward_model"}}
    (logd / "config.yaml").write_text(yaml.safe_dump(cfg))
    info = export(run, root=tmp_path, eval_reward_base=0.1, eval_ste_tuned=80.0)
    dst = tmp_path / "checkpoints" / "rlhf_final"
    assert (dst / "model.safetensors").exists() and not (dst / "stale.txt").exists()
    saved = json.loads((dst / "training_info.json").read_text())
    assert saved == info
    assert saved["steps"] == 7 and saved["beta_kl"] == 0.1 and saved["max_new_tokens"] == 512
    assert saved["run_name"] == run and saved["algo"] == "REINFORCE"
    assert saved["reward_model"] == "checkpoints/reward_model"
    assert saved["dataset"] == "helpsteer2+ste_rewrites"
    assert saved["eval_reward_base"] == 0.1 and saved["eval_reward_tuned"] is None
    assert saved["created"].endswith("+00:00")


def _small_cfg(cfg, **rl):
    cfg["rl"].update(steps=2, batch_size=2, grad_accum=2, sample_every=1, save_steps=[1],
                     logprob_micro_batch=2, **rl)
    cfg["policy"]["max_new_tokens"] = 16
    return cfg


@pytest.mark.slow
def test_smoke_run(cfg, tiny_prompts, tmp_path):
    cfg = _small_cfg(cfg, max_kl=None)
    out = train(cfg, scorer=FakeScorer(), prompts=tiny_prompts, run_name="smoke", root=tmp_path)
    assert out["steps_done"] == 2 and out["stopped_early"] is None
    rows = [json.loads(l) for l in (tmp_path / "logs/smoke/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    for k in ("reward_mean", "reward_std", "kl_mean", "return_mean", "baseline", "loss", "entropy",
              "resp_len_mean", "finished_rate", "ste_score_mean", "time_s", "gpu_mem_gb"):
        assert k in rows[0]
    assert (tmp_path / "logs/smoke/samples.jsonl").exists()
    assert (tmp_path / "checkpoints/smoke/final/config.json").exists()
    assert (tmp_path / "checkpoints/smoke/step_1").is_dir()


@pytest.mark.slow
def test_early_stop(cfg, tiny_prompts, tmp_path):
    cfg = _small_cfg(cfg, max_kl=-1, max_kl_window=1)
    cfg["rl"]["steps"] = 5
    out = train(cfg, scorer=FakeScorer(), prompts=tiny_prompts, run_name="stop", root=tmp_path)
    assert out["steps_done"] == 1
    info = json.loads((tmp_path / "logs/stop/stopped_early.json").read_text())
    assert info["stopped_at_step"] == 1 and "reason" in info
    assert (tmp_path / "checkpoints/stop/final").is_dir()
