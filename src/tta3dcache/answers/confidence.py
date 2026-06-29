from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class ConfidenceEstimator:
    source_priority: tuple[str, ...] = ("model_probability", "agreement", "self_report")

    def estimate(self, *, model_probability: float | None = None, agreement: float | None = None, self_report: Any = None) -> float:
        for source in self.source_priority:
            if source == "model_probability" and model_probability is not None:
                return clamp_probability(model_probability)
            if source == "agreement" and agreement is not None:
                return clamp_probability(agreement)
            if source == "self_report" and self_report is not None:
                parsed = self._parse_self_report(self_report)
                if parsed is not None:
                    return clamp_probability(parsed)
        return 0.0

    def _parse_self_report(self, self_report: Any) -> float | None:
        if isinstance(self_report, dict):
            value = self_report.get("confidence")
            if isinstance(value, str):
                return {"low": 0.25, "medium": 0.5, "high": 0.8}.get(value.lower())
            if isinstance(value, (int, float)):
                return float(value)
        if isinstance(self_report, (int, float)):
            return float(self_report)
        return None
