from __future__ import annotations

from ..view_sampling.diversity import jaccard_overlap


def overlap_penalty(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    return jaccard_overlap(left, right)
