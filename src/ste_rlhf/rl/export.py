"""Copy checkpoints/<run>/final to checkpoints/rlhf_final and write training_info.json (C6).

Run: python -m ste_rlhf.rl.export --run <run> [--eval-reward-base x --eval-reward-tuned y
                                              --eval-ste-base a --eval-ste-tuned b]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


def export(run, root=".", eval_reward_base=None, eval_reward_tuned=None,
           eval_ste_base=None, eval_ste_tuned=None):
    root = Path(root)
    src = root / "checkpoints" / run / "final"
    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist")
    log_dir = root / "logs" / run
    cfg = yaml.safe_load((log_dir / "config.yaml").read_text())
    steps = 0
    mpath = log_dir / "metrics.jsonl"
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            if line.strip():
                steps = json.loads(line)["step"]
    dst = root / "checkpoints" / "rlhf_final"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    info = {
        "base_model": cfg["policy"]["name"], "algo": "REINFORCE", "run_name": run,
        "steps": steps, "beta_kl": cfg["rl"]["beta_kl"],
        "max_new_tokens": cfg["policy"]["max_new_tokens"],
        "reward_model": cfg["reward"]["checkpoint"], "dataset": "helpsteer2+ste_rewrites",
        "eval_reward_base": eval_reward_base, "eval_reward_tuned": eval_reward_tuned,
        "eval_ste_base": eval_ste_base, "eval_ste_tuned": eval_ste_tuned,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dst / "training_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    for k in ("eval-reward-base", "eval-reward-tuned", "eval-ste-base", "eval-ste-tuned"):
        ap.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args(argv)
    info = export(a.run, a.root, a.eval_reward_base, a.eval_reward_tuned,
                  a.eval_ste_base, a.eval_ste_tuned)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
