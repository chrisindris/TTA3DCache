from __future__ import annotations

from math import log


def normalized_entropy(distribution: dict[str, float]) -> float:
    values = [max(0.0, float(value)) for value in distribution.values()]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    probs = [value / total for value in values if value > 0.0]
    entropy = -sum(prob * log(prob) for prob in probs)
    return entropy / log(len(probs)) if len(probs) > 1 else 0.0
