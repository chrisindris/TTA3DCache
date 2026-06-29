from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VLMResponse:
    raw_response: str
    answer: str
    confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VLMRunner(Protocol):
    def generate(self, *, question: str, view_ids: tuple[str, ...], prompt_type: str, candidate_id: str, metadata: dict[str, Any] | None = None) -> VLMResponse:
        ...


@dataclass(slots=True)
class MockVLMRunner:
    answer_map: dict[tuple[str, ...], tuple[str, float]]
    default_answer: str = "unknown"

    def generate(self, *, question: str, view_ids: tuple[str, ...], prompt_type: str, candidate_id: str, metadata: dict[str, Any] | None = None) -> VLMResponse:
        answer, confidence = self.answer_map.get(tuple(view_ids), (self.default_answer, 0.5))
        return VLMResponse(raw_response=answer, answer=answer, confidence=confidence, rationale=None, metadata={"prompt_type": prompt_type, "candidate_id": candidate_id, **(metadata or {})})
