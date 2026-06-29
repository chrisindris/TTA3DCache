from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .entries import CacheEntry


def _normalize(vector: torch.Tensor | None) -> torch.Tensor | None:
    if vector is None:
        return None
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        return None
    flat = vector.float().reshape(-1)
    if torch.linalg.norm(flat).item() <= 1e-12:
        return None
    return F.normalize(flat, dim=0)


@dataclass(slots=True)
class PrototypeState:
    vector: torch.Tensor
    support: int = 0
    mean_weight: float = 0.0


@dataclass(slots=True)
class PrototypeCache:
    momentum: float = 0.9
    minimum_support: int = 2
    prototypes: dict[str, PrototypeState] = field(default_factory=dict)

    def clear(self) -> None:
        self.prototypes.clear()

    def update(self, entry: CacheEntry) -> bool:
        vector = _normalize(entry.key)
        if vector is None:
            return False
        state = self.prototypes.get(entry.value)
        evidence_weight = max(0.0, min(1.0, float(entry.confidence))) * max(0.0, min(1.0, float(entry.diversity)))
        if state is None:
            self.prototypes[entry.value] = PrototypeState(vector=vector, support=1, mean_weight=evidence_weight)
            return True
        updated = self.momentum * state.vector + (1.0 - self.momentum) * vector
        updated = _normalize(updated)
        if updated is None:
            updated = vector
        state.vector = updated
        state.support += 1
        state.mean_weight = (state.mean_weight * (state.support - 1) + evidence_weight) / state.support
        return True

    def score_answers(self, query_key: torch.Tensor | None) -> dict[str, float]:
        if query_key is None:
            return {}
        query = _normalize(query_key)
        if query is None:
            return {}
        scores: dict[str, float] = {}
        for answer, state in self.prototypes.items():
            if state.support < self.minimum_support:
                continue
            similarity = float(F.cosine_similarity(query, state.vector, dim=0).clamp(-1.0, 1.0).item())
            scores[answer] = similarity * torch.log1p(torch.tensor(float(state.support))).item() * max(0.0, state.mean_weight)
        return scores
