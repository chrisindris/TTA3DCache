from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PruningPolicy:
    name: str = "confidence_diversity"
