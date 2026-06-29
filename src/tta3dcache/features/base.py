from __future__ import annotations

from typing import Protocol

import torch


class ViewFeatureEncoder(Protocol):
    def encode_view(self, view: torch.Tensor) -> torch.Tensor:
        ...

    def encode_question(self, question: str) -> torch.Tensor:
        ...
