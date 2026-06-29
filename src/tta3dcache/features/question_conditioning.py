from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _normalize_safe(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return F.normalize(tensor.float(), dim=0)


@dataclass(slots=True)
class QuestionConditionedFeatureEncoder:
    mode: str = "concat"
    alpha: float = 0.5

    def encode(self, question_embedding: torch.Tensor, visual_feature: torch.Tensor) -> torch.Tensor:
        question_embedding = _normalize_safe(question_embedding)
        visual_feature = _normalize_safe(visual_feature)
        if self.mode == "gated" and question_embedding.shape == visual_feature.shape:
            gate = torch.full_like(question_embedding, self.alpha)
            return _normalize_safe(gate * visual_feature + (1.0 - gate) * question_embedding)
        if question_embedding.shape == visual_feature.shape:
            return _normalize_safe(self.alpha * question_embedding + (1.0 - self.alpha) * visual_feature)
        concatenated = torch.cat([question_embedding.reshape(-1), visual_feature.reshape(-1)], dim=0)
        return _normalize_safe(concatenated)
