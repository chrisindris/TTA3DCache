from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Iterable

from .diversity import diversity_score, jaccard_overlap


@dataclass(frozen=True, slots=True)
class CandidateViewSet:
    candidate_id: str
    view_ids: tuple[str, ...]
    generation_type: str
    cdviews_scores: tuple[float, ...]
    diversity_score: float
    overlap_with_base: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CdViewsSelection:
    ranked_view_ids: list[str]
    ranked_scores: list[float]
    base_view_ids: list[str]
    raw_metadata: dict[str, Any]


def _deduplicate(candidate_sets: Iterable[CandidateViewSet]) -> list[CandidateViewSet]:
    seen: set[tuple[str, ...]] = set()
    unique: list[CandidateViewSet] = []
    for candidate in candidate_sets:
        key = tuple(sorted(candidate.view_ids))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _scores_for_views(ranked_view_ids: list[str], ranked_scores: list[float], view_ids: tuple[str, ...]) -> tuple[float, ...]:
    lookup = dict(zip(ranked_view_ids, ranked_scores, strict=False))
    return tuple(float(lookup.get(view_id, 0.0)) for view_id in view_ids)


def _make_candidate(candidate_id: str, generation_type: str, view_ids: tuple[str, ...], base_view_ids: list[str], ranked_view_ids: list[str], ranked_scores: list[float], metadata: dict[str, Any]) -> CandidateViewSet:
    return CandidateViewSet(
        candidate_id=candidate_id,
        view_ids=view_ids,
        generation_type=generation_type,
        cdviews_scores=_scores_for_views(ranked_view_ids, ranked_scores, view_ids),
        diversity_score=diversity_score(view_ids),
        overlap_with_base=jaccard_overlap(view_ids, base_view_ids),
        metadata=metadata,
    )


def build_candidate_view_sets(selection: CdViewsSelection, *, num_sets: int, views_per_set: int, diversity_lambda: float, include_random_control: bool, seed: int, max_ranked_pool: int = 16) -> list[CandidateViewSet]:
    rng = Random(seed)
    ranked_view_ids = selection.ranked_view_ids[:max_ranked_pool]
    ranked_scores = selection.ranked_scores[:max_ranked_pool]
    base_view_ids = selection.base_view_ids[:views_per_set]
    base_tuple = tuple(base_view_ids[:views_per_set])
    candidates: list[CandidateViewSet] = []

    if base_tuple:
        candidates.append(_make_candidate("base_default", "cdviews_default", base_tuple, selection.base_view_ids, ranked_view_ids, ranked_scores, {"rank": 0}))

    if ranked_view_ids:
        top_critical = tuple(ranked_view_ids[:views_per_set])
        candidates.append(_make_candidate("top_critical", "top_critical", top_critical, selection.base_view_ids, ranked_view_ids, ranked_scores, {"rank": 1}))

    if ranked_view_ids:
        pool = list(dict.fromkeys(ranked_view_ids))
        greedy: list[str] = []
        while len(greedy) < min(views_per_set, len(pool)):
            best_view = None
            best_score = float("-inf")
            for view_id in pool:
                if view_id in greedy:
                    continue
                criticality = ranked_scores[ranked_view_ids.index(view_id)] if view_id in ranked_view_ids else 0.0
                dissimilarity = 1.0 - jaccard_overlap(tuple(greedy), (view_id,)) if greedy else 1.0
                score = criticality + diversity_lambda * dissimilarity
                if score > best_score:
                    best_view = view_id
                    best_score = score
            if best_view is None:
                break
            greedy.append(best_view)
        if greedy:
            candidates.append(_make_candidate("diversity_heavy", "diversity_heavy", tuple(greedy), selection.base_view_ids, ranked_view_ids, ranked_scores, {"rank": 2}))

    if base_tuple:
        fallback_ranked = [view_id for view_id in ranked_view_ids if view_id not in base_tuple]
        for index, removed in reversed(list(enumerate(base_tuple))):
            replacement = fallback_ranked[index % len(fallback_ranked)] if fallback_ranked else removed
            view_ids = list(base_tuple)
            view_ids[index] = replacement
            candidates.append(_make_candidate(f"leave_one_out_{index}", "leave_one_out", tuple(view_ids), selection.base_view_ids, ranked_view_ids, ranked_scores, {"removed_view": removed, "replacement_view": replacement}))

    neighbors = selection.raw_metadata.get("nearby_view_ids", {})
    jitter_ids = list(base_tuple)
    for index, view_id in enumerate(jitter_ids):
        nearby = neighbors.get(view_id, []) if isinstance(neighbors, dict) else []
        if nearby:
            jitter_ids[index] = nearby[0]
    if tuple(jitter_ids) != base_tuple and jitter_ids:
        candidates.append(_make_candidate("nearby_frame_jitter", "nearby_frame_jitter", tuple(jitter_ids), selection.base_view_ids, ranked_view_ids, ranked_scores, {"source": "metadata"}))

    if include_random_control and ranked_view_ids:
        random_pool = list(dict.fromkeys(ranked_view_ids))
        rng.shuffle(random_pool)
        random_view_ids = tuple(random_pool[:views_per_set])
        if random_view_ids:
            candidates.append(_make_candidate("random_control", "random_control", random_view_ids, selection.base_view_ids, ranked_view_ids, ranked_scores, {"seed": seed}))

    unique_candidates = _deduplicate(candidates)
    return unique_candidates[:num_sets]
