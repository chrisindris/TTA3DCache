from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class PooledViewEncoder:
    pooling: str = "weighted_mean"
    normalize: bool = True

    def pool(self, features: Iterable[torch.Tensor], weights: Iterable[float] | None = None) -> torch.Tensor:
        tensors = [feature.float() for feature in features]
        if not tensors:
            return torch.zeros(1, dtype=torch.float32)
        stacked = torch.stack([tensor.reshape(-1) for tensor in tensors], dim=0)
        if weights is None:
            if self.pooling == "max":
                pooled = stacked.max(dim=0).values
            else:
                pooled = stacked.mean(dim=0)
        else:
            weight_tensor = torch.tensor(list(weights), dtype=stacked.dtype, device=stacked.device)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
            if self.pooling == "max":
                pooled = stacked.max(dim=0).values
            else:
                pooled = (stacked * weight_tensor.unsqueeze(1)).sum(dim=0)
        if self.normalize:
            pooled = F.normalize(pooled, dim=0)
        return pooled
