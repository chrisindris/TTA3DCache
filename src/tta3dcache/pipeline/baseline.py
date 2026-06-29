from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..answers.confidence import ConfidenceEstimator
from ..answers.equivalence import AnswerEquivalenceResolver
from ..cache.answer_cache import AnswerCache
from ..cache.entries import CacheEntry
from ..cache.prototype_cache import PrototypeCache
from ..config import TTA3DCacheConfig
from ..integration.cdviews_adapter import CdViewsAdapter, selection_from_prepared_example
from ..integration.vlm_runner import VLMRunner
from ..scoring.consistency import agreement_score
from ..scoring.fusion import FusionResult, fuse_answers
from ..view_sampling.candidate_sets import CandidateViewSet, build_candidate_view_sets


@dataclass(slots=True)
class PipelineResult:
    run_id: str
    scene_id: str
    question_id: str
    question: str
    base_answer: str
    final_answer: str
    adapted: bool
    candidate_sets: list[CandidateViewSet]
    candidate_answers: list[dict[str, Any]]
    cache_scores: dict[str, float]
    prototype_scores: dict[str, float]
    fusion: FusionResult
    vlm_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)


def run_baseline(example: Any, cfg: TTA3DCacheConfig, *, cdviews_adapter: CdViewsAdapter, vlm_runner: VLMRunner, run_id: str = "baseline") -> PipelineResult:
    prepared = cdviews_adapter.prepare_example(example)
    selection = selection_from_prepared_example(prepared)
    resolver = AnswerEquivalenceResolver(mode=cfg.answers.equivalence_mode, threshold=cfg.answers.embedding_threshold, canonicalize_spatial_terms=cfg.answers.canonicalize_spatial_terms)
    confidence_estimator = ConfidenceEstimator(source_priority=cfg.confidence.source_priority)
    base_view_ids = tuple(prepared.base_view_ids[: cfg.cdviews.views_per_set] or prepared.ranked_view_ids[: cfg.cdviews.views_per_set])
    base_response = vlm_runner.generate(question=prepared.question, view_ids=base_view_ids, prompt_type="short_answer", candidate_id="base_default", metadata=prepared.metadata)
    base_answer = resolver.normalized(base_response.answer)
    base_confidence = confidence_estimator.estimate(model_probability=base_response.confidence, agreement=1.0)

    candidate_sets = build_candidate_view_sets(selection, num_sets=cfg.candidate_sets.num_sets, views_per_set=cfg.cdviews.views_per_set, diversity_lambda=cfg.candidate_sets.diversity_lambda, include_random_control=cfg.candidate_sets.include_random_control, seed=cfg.seed, max_ranked_pool=cfg.candidate_sets.max_ranked_pool)
    candidate_answers: list[dict[str, Any]] = []
    answer_cache = AnswerCache(max_size=cfg.cache.max_size, minimum_confidence=cfg.confidence.minimum_cache_confidence, similarity_temperature=cfg.cache.similarity_temperature, similarity_threshold=cfg.cache.similarity_threshold)
    prototype_cache = PrototypeCache(momentum=cfg.prototype.momentum, minimum_support=cfg.prototype.minimum_support)

    for candidate in candidate_sets:
        response = vlm_runner.generate(question=prepared.question, view_ids=candidate.view_ids, prompt_type="short_answer", candidate_id=candidate.candidate_id, metadata=candidate.metadata)
        normalized_answer = resolver.normalized(response.answer)
        semantic_cluster_id = resolver.cluster_id(response.answer)
        confidence = confidence_estimator.estimate(model_probability=response.confidence, agreement=agreement_score([normalized_answer, base_answer]))
        uncertainty = 1.0 - confidence
        entry = CacheEntry(
            key=None,
            value=normalized_answer,
            confidence=confidence,
            uncertainty=uncertainty,
            diversity=candidate.diversity_score,
            overlap=candidate.overlap_with_base,
            metadata={"candidate_id": candidate.candidate_id, "generation_type": candidate.generation_type, **candidate.metadata},
        )
        answer_cache.update(entry)
        if cfg.prototype.enabled:
            prototype_cache.update(entry)
        candidate_answers.append(
            {
                "candidate_id": candidate.candidate_id,
                "raw_answer": response.answer,
                "normalized_answer": normalized_answer,
                "semantic_cluster_id": semantic_cluster_id,
                "raw_response": response.raw_response,
                "rationale": response.rationale,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "view_ids": candidate.view_ids,
                "diversity_score": candidate.diversity_score,
                "overlap_score": candidate.overlap_with_base,
                "cdviews_score": sum(candidate.cdviews_scores) / max(1, len(candidate.cdviews_scores)),
                "prompt_type": "short_answer",
                "feature": None,
                "metadata": {"generation_type": candidate.generation_type, **candidate.metadata},
            }
        )

    cache_scores = answer_cache.score_answers()
    prototype_scores = prototype_cache.score_answers(None) if cfg.prototype.enabled else {}
    fusion = fuse_answers(
        base_answer=base_answer,
        base_confidence=base_confidence,
        cache_scores=cache_scores,
        prototype_scores=prototype_scores,
        lambda_base=cfg.fusion.lambda_base,
        lambda_cache=cfg.fusion.lambda_cache,
        lambda_proto=cfg.fusion.lambda_proto,
        lambda_instability=cfg.fusion.lambda_instability,
        fallback_entropy_threshold=cfg.fusion.fallback_entropy_threshold,
    )
    final_answer = fusion.answer
    adapted = final_answer != base_answer
    return PipelineResult(
        run_id=run_id,
        scene_id=prepared.scene_id,
        question_id=prepared.question_id,
        question=prepared.question,
        base_answer=base_answer,
        final_answer=final_answer,
        adapted=adapted,
        candidate_sets=candidate_sets,
        candidate_answers=candidate_answers,
        cache_scores=cache_scores,
        prototype_scores=prototype_scores,
        fusion=fusion,
        vlm_calls=1 + len(candidate_sets),
        metadata={"config": cfg.to_dict(), **prepared.metadata},
    )
