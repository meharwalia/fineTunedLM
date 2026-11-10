"""YAML config loading with dotted overrides (``rl.beta_kl=0``)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "v2.yaml"


def _parse_value(text: str):
    # YAML scalar typing: 0 -> int, 1.0e-5 -> float, true -> bool, null -> None.
    # PyYAML reads "1e-5" as a string, so retry as float.
    val = yaml.safe_load(text)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass
    return val


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config section {p!r} in override {item!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path=None, overrides=None) -> dict:
    path = Path(path) if path else DEFAULT_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def add_config_args(parser):
    """Adds --config and --set. action="extend" keeps ALL repeated --set flags."""
    parser.add_argument("--config", default="configs/v2.yaml", help="YAML config path")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set rl.beta_kl=0.05 rl.steps=10")
    return parser


def resolve_run_name(cfg: dict) -> str:
    rl = cfg["rl"]
    if rl.get("run_name"):
        return rl["run_name"]
    name = f"reinforce_beta{float(rl['beta_kl']):g}_seed{cfg['seed']}"
    if not rl.get("use_baseline", True):
        name += "_nobaseline"
    return name
