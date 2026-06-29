from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import torch
import torch.nn.functional as F

from .normalization import normalize_open_answer


class TextEncoder(Protocol):
    def encode(self, text: str) -> torch.Tensor:
        ...


@lru_cache(maxsize=4096)
def _fallback_embedding(text: str, dim: int = 128) -> tuple[float, ...]:
    vector = [0.0] * dim
    for index, token in enumerate(text.split()):
        bucket = (hash(token) + index * 131) % dim
        vector[bucket] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


@dataclass(slots=True)
class AnswerEquivalenceResolver:
    mode: str = "exact_normalized"
    threshold: float = 0.85
    canonicalize_spatial_terms: bool = True
    encoder: TextEncoder | None = None
    _embedding_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def normalized(self, text: str) -> str:
        return normalize_open_answer(text, canonicalize_spatial_terms=self.canonicalize_spatial_terms)

    def cluster_id(self, text: str) -> str:
        normalized = self.normalized(text)
        if self.mode == "embedding_cluster":
            return self._embedding_cluster_id(normalized)
        return normalized

    def equivalent(self, left: str, right: str) -> bool:
        left_norm = self.normalized(left)
        right_norm = self.normalized(right)
        if self.mode == "embedding_cluster":
            return self._embedding_similarity(left_norm, right_norm) >= self.threshold
        return left_norm == right_norm

    def _embedding_cluster_id(self, normalized: str) -> str:
        if self.encoder is None:
            embedding = torch.tensor(_fallback_embedding(normalized), dtype=torch.float32)
        else:
            embedding = self.encoder.encode(normalized).detach().float().cpu()
        self._embedding_cache[normalized] = embedding
        quantized = torch.round(embedding * 1000).to(torch.int64).tolist()
        return "emb:" + ",".join(str(value) for value in quantized)

    def _embedding_similarity(self, left: str, right: str) -> float:
        left_embedding = self._embedding_cache.get(left)
        if left_embedding is None:
            left_embedding = torch.tensor(_fallback_embedding(left), dtype=torch.float32)
            self._embedding_cache[left] = left_embedding
        right_embedding = self._embedding_cache.get(right)
        if right_embedding is None:
            right_embedding = torch.tensor(_fallback_embedding(right), dtype=torch.float32)
            self._embedding_cache[right] = right_embedding
        if left_embedding.numel() != right_embedding.numel():
            size = max(left_embedding.numel(), right_embedding.numel())
            left_embedding = F.pad(left_embedding, (0, size - left_embedding.numel()))
            right_embedding = F.pad(right_embedding, (0, size - right_embedding.numel()))
        return float(F.cosine_similarity(left_embedding, right_embedding, dim=0).clamp(-1.0, 1.0).item())
