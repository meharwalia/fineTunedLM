import copy
import json
import random

import pytest
import torch

from minirlhf.rl.export import export_run
from minirlhf.rl.reinforce import (compute_advantages, reinforce_loss, sample_prompts, train,
                                   update_baseline)


class Toy(torch.nn.Module):
    def __init__(self, v=6):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3, v))  # 3 positions

    def logp(self, ids):  # ids [B,3]
        lp = torch.log_softmax(self.logits, -1)
        return torch.stack([lp[t][ids[:, t]] for t in range(3)], 1)


def test_positive_advantage_increases_logprob():
    m = Toy()
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.ones(2, 3)
    adv = torch.tensor([1.0, -1.0])
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    before = m.logp(ids).detach().sum(-1)
    reinforce_loss(m.logp(ids), mask, adv).backward()
    opt.step()
    after = m.logp(ids).detach().sum(-1)
    assert after[0] > before[0] and after[1] < before[1]


def test_mask_zeroes_gradient():
    m = Toy()
    ids = torch.tensor([[1, 2, 3]])
    reinforce_loss(m.logp(ids), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([1.0])).backward()
    assert m.logits.grad[1].abs().sum() == 0 and m.logits.grad[0].abs().sum() > 0


def test_baseline_ema():
    b = update_baseline(None, 2.0, 0.9)
    assert b == 2.0
    assert update_baseline(b, 0.0, 0.9) == pytest.approx(1.8)


def test_advantages():
    r = torch.tensor([1.0, 2.0, 3.0, 6.0])
    assert torch.allclose(compute_advantages(r, 2.0, False), r - 2.0)
    assert torch.allclose(compute_advantages(r, None, False), r)
    a = compute_advantages(r, 2.0, True)
    assert a.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(torch.sign(a), torch.sign(r - 2.0))


def test_sample_prompts_with_replacement():
    out = sample_prompts(["a", "b"], 5, random.Random(0))
    assert len(out) == 5 and set(out) <= {"a", "b"}


def _dummy_run(tmp_path, run="r1"):
    final = tmp_path / "checkpoints" / run / "final"
    final.mkdir(parents=True)
    (final / "config.json").write_text("{}")
    (final / "model.safetensors").write_text("x")
    return tmp_path / "checkpoints"


def test_export_layout(tmp_path, cfg):
    root = _dummy_run(tmp_path)
    (root / "rlhf_final").mkdir()
    (root / "rlhf_final" / "stale.txt").write_text("old")
    dst, info = export_run("r1", cfg, ckpt_root=root, log_root=tmp_path / "logs", eval_reward_tuned=1.5)
    assert dst == root / "rlhf_final" and not (dst / "stale.txt").exists()
    assert (dst / "model.safetensors").exists()
    on_disk = json.loads((dst / "training_info.json").read_text())
    assert set(on_disk) == {"base_model", "algo", "run_name", "steps", "beta_kl",
                            "eval_reward_base", "eval_reward_tuned", "created"}
    assert on_disk["algo"] == "reinforce" and on_disk["eval_reward_base"] is None
    assert on_disk["eval_reward_tuned"] == 1.5 and on_disk["run_name"] == "r1"


@pytest.mark.slow
def test_train_smoke(tmp_path, cfg):
    cfg["rl"].update(steps=3, batch_size=2, save_every=2, sample_every=1, run_name="smoke")
    cfg["policy"]["max_new_tokens"] = 8
    cfg["data"]["prompts_dir"] = str(tmp_path / "none")
    run = train(cfg, log_root=tmp_path / "logs", ckpt_root=tmp_path / "ckpt")
    rows = [json.loads(l) for l in (tmp_path / "logs" / run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {"step", "reward_mean", "reward_std", "kl_mean", "return_mean", "baseline", "loss",
            "entropy", "resp_len_mean", "time_s"} <= set(rows[0])
    assert (tmp_path / "logs" / run / "samples.jsonl").exists()
    assert (tmp_path / "ckpt" / run / "final" / "config.json").exists()
    assert (tmp_path / "ckpt" / run / "step_2").is_dir()


@pytest.mark.slow
def test_early_stop_on_kl(tmp_path, cfg):
    """With max_kl below any reachable KL, training stops after the window and still saves final."""
    import json as _json
    from minirlhf.rl.reinforce import train
    cfg = copy.deepcopy(cfg)
    cfg["rl"].update(steps=10, batch_size=2, max_kl=-1.0, max_kl_window=2, save_every=0,
                     run_name="early", lr=1e-3)
    cfg["policy"]["max_new_tokens"] = 8
    train(cfg, log_root=str(tmp_path / "logs"), ckpt_root=str(tmp_path / "ckpt"))
    info = _json.loads((tmp_path / "logs" / "early" / "stopped_early.json").read_text())
    assert info["stopped_at_step"] == 2
    assert (tmp_path / "ckpt" / "early" / "final").is_dir()
