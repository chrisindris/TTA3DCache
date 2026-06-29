from __future__ import annotations

from collections import Counter


def agreement_score(answers: list[str]) -> float:
    if not answers:
        return 0.0
    counts = Counter(answers)
    return max(counts.values()) / len(answers)


def answer_instability(answer_weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, weight) for weight in answer_weights.values())
    if total <= 0.0:
        return {answer: 1.0 for answer in answer_weights}
    return {answer: 1.0 - (max(0.0, weight) / total) for answer, weight in answer_weights.items()}
