from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class PreparedExample:
    scene_id: str
    question_id: str
    question: str
    raw_example: dict[str, Any]
    ranked_view_ids: list[str]
    ranked_scores: list[float]
    base_view_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    def prepare_example(self, example: Any) -> PreparedExample:
        ...
