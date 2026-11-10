"""Record validation for the JSONL formats in contract C6."""
from numbers import Real

_NUM = (int, float)
PREF_SOURCES = ("human", "rule", "hh-rlhf")

# kind -> {field: type or tuple of types}
_SCHEMAS = {
    "prompt": {"id": str, "prompt": str, "source": str},
    "candidate": {"id": str, "prompt": str, "responses": list, "gen": dict},
    "pref": {"id": str, "prompt": str, "chosen": str, "rejected": str, "source": str},
    "generation": {"id": str, "prompt": str, "response": str, "model": str},
    "metrics": {k: _NUM for k in ("step", "reward_mean", "reward_std", "kl_mean", "return_mean",
                                  "baseline", "loss", "entropy", "resp_len_mean", "time_s")},
    "sample": {"step": int, "prompt": str, "response": str, "reward": _NUM},
}


def validate_record(kind: str, obj: dict) -> dict:
    """Return obj if valid, else raise ValueError."""
    if kind not in _SCHEMAS:
        raise ValueError(f"unknown record kind {kind!r}; known: {sorted(_SCHEMAS)}")
    if not isinstance(obj, dict):
        raise ValueError(f"{kind}: record must be a dict")
    for field, typ in _SCHEMAS[kind].items():
        if field not in obj:
            raise ValueError(f"{kind}: missing field {field!r}")
        v = obj[field]
        if isinstance(v, bool) or not isinstance(v, typ):
            raise ValueError(f"{kind}: field {field!r} has wrong type {type(v).__name__}")
    if kind == "candidate":
        if len(obj["responses"]) != 2 or not all(isinstance(r, str) for r in obj["responses"]):
            raise ValueError("candidate: 'responses' must be a list of exactly two strings")
    if kind == "pref" and obj["source"] not in PREF_SOURCES:
        raise ValueError(f"pref: source must be one of {PREF_SOURCES}, got {obj['source']!r}")
    return obj
