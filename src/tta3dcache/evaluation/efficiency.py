from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EfficiencySummary:
    mean_vlm_calls: float
    mean_runtime_seconds: float
