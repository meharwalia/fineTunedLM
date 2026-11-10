from .heuristic import HeuristicScorer


def load_scorer(cfg, device):
    r = cfg["reward"]
    if r["type"] == "heuristic":
        return HeuristicScorer()
    from minirlhf.reward.scorer import DistilBertScorer  # lazy: written by S2
    return DistilBertScorer(r["checkpoint"], device, r["max_length"], r["normalize"])
