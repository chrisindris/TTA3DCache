from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..view_sampling.candidate_sets import CdViewsSelection
from .dataset_adapter import PreparedExample


class CdViewsAdapter(Protocol):
    def prepare_example(self, example: Any) -> PreparedExample:
        ...


@dataclass(slots=True)
class StaticCdViewsAdapter:
    ranked_view_ids_key: str = "ranked_view_ids"
    ranked_scores_key: str = "ranked_scores"
    base_view_ids_key: str = "base_view_ids"
    metadata_key: str = "metadata"

    def prepare_example(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(
            scene_id=str(example["scene_id"]),
            question_id=str(example["question_id"]),
            question=str(example["question"]),
            raw_example=dict(example),
            ranked_view_ids=list(example.get(self.ranked_view_ids_key, [])),
            ranked_scores=[float(value) for value in example.get(self.ranked_scores_key, [])],
            base_view_ids=list(example.get(self.base_view_ids_key, [])),
            metadata=dict(example.get(self.metadata_key, {})),
        )


def selection_from_prepared_example(example: PreparedExample) -> CdViewsSelection:
    return CdViewsSelection(
        ranked_view_ids=example.ranked_view_ids,
        ranked_scores=example.ranked_scores,
        base_view_ids=example.base_view_ids,
        raw_metadata=example.metadata,
    )
