import json
from pathlib import Path

import yaml

from .io import append_jsonl
from .schemas import validate_record


class RunLogger:
    """Writes logs/<run>/metrics.jsonl and samples.jsonl (+ config.yaml snapshot)."""

    def __init__(self, run_dir, cfg=None, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        if not resume:
            # A new run with an existing name replaces the old logs instead of
            # appending to them (appending mixes steps from different runs).
            self.metrics_path.unlink(missing_ok=True)
            self.samples_path.unlink(missing_ok=True)
        if cfg is not None:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

    def log(self, step, metrics: dict):
        row = {"step": int(step), **{k: (float(v) if v is not None else None) for k, v in metrics.items()}}
        append_jsonl(self.metrics_path, row)
        return row

    def log_samples(self, step, prompts, responses, rewards):
        for p, r, w in zip(prompts, responses, rewards):
            row = {"step": int(step), "prompt": p, "response": r, "reward": float(w)}
            validate_record("sample", row)
            append_jsonl(self.samples_path, row)
