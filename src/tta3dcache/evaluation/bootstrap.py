from __future__ import annotations

from random import Random


def paired_bootstrap_delta(base: list[int], adapted: list[int], *, num_samples: int = 1000, seed: int = 42) -> tuple[float, tuple[float, float]]:
    if len(base) != len(adapted):
        raise ValueError("base and adapted must be the same length")
    if not base:
        return 0.0, (0.0, 0.0)
    rng = Random(seed)
    deltas: list[float] = []
    for _ in range(num_samples):
        indices = [rng.randrange(len(base)) for _ in range(len(base))]
        delta = sum(adapted[index] - base[index] for index in indices) / len(indices)
        deltas.append(delta)
    deltas.sort()
    mean_delta = sum(adapted[i] - base[i] for i in range(len(base))) / len(base)
    lower = deltas[int(0.025 * (len(deltas) - 1))]
    upper = deltas[int(0.975 * (len(deltas) - 1))]
    return mean_delta, (lower, upper)
