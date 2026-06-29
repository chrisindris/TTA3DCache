from __future__ import annotations

from dataclasses import dataclass


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip() == target.strip()


def normalized_exact_match(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == target.strip().lower()


@dataclass(slots=True)
class AccuracySummary:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total
