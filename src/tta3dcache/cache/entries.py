from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class CacheEntry:
    key: torch.Tensor | None
    value: str
    confidence: float
    uncertainty: float
    diversity: float
    overlap: float
    metadata: dict[str, Any] = field(default_factory=dict)
