import torch


def get_device(cfg=None) -> torch.device:
    """auto: cuda > mps > cpu; otherwise the explicit name in cfg['device']."""
    want = (cfg or {}).get("device", "auto") if isinstance(cfg, dict) else "auto"
    if want and want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
