"""Copy a run's final checkpoint to checkpoints/rlhf_final/ and write training_info.json (C3)."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from minirlhf.common import load_config


def export_run(run, cfg, ckpt_root="checkpoints", log_root="logs", eval_reward_base=None, eval_reward_tuned=None):
    ckpt_root = Path(ckpt_root)
    src = ckpt_root / run / "final"
    if not src.is_dir():
        raise FileNotFoundError(f"no final checkpoint at {src}")
    dst = ckpt_root / "rlhf_final"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    steps = int(cfg["rl"]["steps"])
    beta = cfg["rl"]["beta_kl"]
    run_cfg = Path(log_root) / run / "config.yaml"
    if run_cfg.exists():  # prefer the config the run actually used
        rc = load_config(run_cfg)
        steps, beta = int(rc["rl"]["steps"]), rc["rl"]["beta_kl"]
    metrics_path = Path(log_root) / run / "metrics.jsonl"
    if metrics_path.exists():  # steps actually completed (early stopping may end a run sooner)
        lines = metrics_path.read_text().strip().splitlines()
        if lines:
            steps = int(json.loads(lines[-1])["step"])
    info = {"base_model": cfg["policy"]["name"], "algo": "reinforce", "run_name": run, "steps": steps,
            "beta_kl": float(beta), "eval_reward_base": eval_reward_base, "eval_reward_tuned": eval_reward_tuned,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(dst / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return dst, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--eval-reward-base", type=float, default=None)
    ap.add_argument("--eval-reward-tuned", type=float, default=None)
    a = ap.parse_args(argv)
    dst, info = export_run(a.run, load_config(a.config, a.overrides),
                           eval_reward_base=a.eval_reward_base, eval_reward_tuned=a.eval_reward_tuned)
    print(f"exported to {dst}\n{json.dumps(info, indent=2)}")


if __name__ == "__main__":
    main()
