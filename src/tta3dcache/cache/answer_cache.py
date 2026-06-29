from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .entries import CacheEntry


def _safe_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    if left.ndim == 1:
        left = left.unsqueeze(0)
    if right.ndim == 1:
        right = right.unsqueeze(0)
    size = max(left.shape[-1], right.shape[-1])
    if left.shape[-1] != size:
        left = F.pad(left, (0, size - left.shape[-1]))
    if right.shape[-1] != size:
        right = F.pad(right, (0, size - right.shape[-1]))
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).mean().clamp(-1.0, 1.0).item())


@dataclass(slots=True)
class AnswerCache:
    max_size: int = 32
    minimum_confidence: float = 0.35
    similarity_temperature: float = 1.0
    similarity_threshold: float = 0.75
    entries: list[CacheEntry] = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()

    def update(self, entry: CacheEntry) -> bool:
        if not torch.isfinite(torch.tensor(entry.confidence)) or entry.confidence < self.minimum_confidence:
            return False
        self.entries.append(entry)
        self._prune()
        return True

    def aggregate_query_key(self) -> torch.Tensor | None:
        usable = [entry for entry in self.entries if entry.key is not None]
        if not usable:
            return None
        weights = torch.tensor([self._entry_weight(entry) for entry in usable], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)
        keys = torch.stack([entry.key.float().reshape(-1) for entry in usable], dim=0)
        return F.normalize((keys * weights.unsqueeze(1)).sum(dim=0), dim=0)

    def score_answers(self, query_key: torch.Tensor | None = None) -> dict[str, float]:
        scores: dict[str, float] = {}
        for entry in self.entries:
            weight = self._entry_weight(entry)
            if query_key is not None and entry.key is not None:
                similarity = _safe_cosine_similarity(query_key, entry.key)
                weight *= float(torch.exp(torch.tensor(self.similarity_temperature * similarity)).item())
            scores[entry.value] = scores.get(entry.value, 0.0) + weight
        return scores

    def negative_score_answers(self, query_key: torch.Tensor | None = None) -> dict[str, float]:
        return {}

    def _entry_weight(self, entry: CacheEntry) -> float:
        confidence = max(0.0, min(1.0, float(entry.confidence)))
        diversity = max(0.0, min(1.0, float(entry.diversity)))
        overlap = max(0.0, min(1.0, float(entry.overlap)))
        return confidence * diversity * (1.0 - overlap + 1e-8)

    def _prune(self) -> None:
        if len(self.entries) <= self.max_size:
            return
        self.entries.sort(key=self._entry_weight, reverse=True)
        kept: list[CacheEntry] = []
        for entry in self.entries:
            if len(kept) >= self.max_size:
                break
            if any(entry.value == other.value and entry.key is not None and other.key is not None and _safe_cosine_similarity(entry.key, other.key) >= self.similarity_threshold for other in kept):
                continue
            kept.append(entry)
        self.entries[:] = kept[: self.max_size]
