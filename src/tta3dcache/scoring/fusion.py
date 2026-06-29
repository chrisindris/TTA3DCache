from __future__ import annotations

from dataclasses import dataclass

from .consistency import answer_instability
from .uncertainty import normalized_entropy


@dataclass(slots=True)
class FusionResult:
    answer: str
    scores: dict[str, float]
    entropy: float
    fallback_used: bool


def fuse_answers(*, base_answer: str, base_confidence: float, cache_scores: dict[str, float], prototype_scores: dict[str, float], lambda_base: float, lambda_cache: float, lambda_proto: float, lambda_instability: float, fallback_entropy_threshold: float) -> FusionResult:
    answers = set(cache_scores) | set(prototype_scores) | {base_answer}
    if not answers:
        return FusionResult(answer=base_answer, scores={base_answer: base_confidence}, entropy=0.0, fallback_used=True)

    scores: dict[str, float] = {}
    for answer in answers:
        base_support = base_confidence if answer == base_answer else 0.0
        cache_support = cache_scores.get(answer, 0.0)
        proto_support = prototype_scores.get(answer, 0.0)
        instability = answer_instability(cache_scores).get(answer, 1.0)
        scores[answer] = lambda_base * base_support + lambda_cache * cache_support + lambda_proto * proto_support - lambda_instability * instability

    entropy = normalized_entropy(scores)
    if entropy > fallback_entropy_threshold:
        return FusionResult(answer=base_answer, scores=scores, entropy=entropy, fallback_used=True)

    def tie_key(answer: str) -> tuple[float, float, float, str]:
        return (scores[answer], base_confidence if answer == base_answer else 0.0, prototype_scores.get(answer, 0.0), answer)

    best_answer = max(answers, key=tie_key)
    return FusionResult(answer=best_answer, scores=scores, entropy=entropy, fallback_used=False)
