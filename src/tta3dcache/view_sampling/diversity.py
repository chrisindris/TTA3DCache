from __future__ import annotations

from itertools import combinations


def jaccard_overlap(left: tuple[str, ...] | list[str], right: tuple[str, ...] | list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def mean_pairwise_overlap(view_ids: tuple[str, ...] | list[str]) -> float:
    items = list(view_ids)
    if len(items) < 2:
        return 0.0
    overlaps = [jaccard_overlap((left,), (right,)) for left, right in combinations(items, 2)]
    return sum(overlaps) / len(overlaps)


def diversity_score(view_ids: tuple[str, ...] | list[str]) -> float:
    return max(0.0, min(1.0, 1.0 - mean_pairwise_overlap(view_ids)))
