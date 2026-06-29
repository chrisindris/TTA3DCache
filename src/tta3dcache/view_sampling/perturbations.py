from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ViewNeighborhood:
    neighbors: dict[str, tuple[str, ...]]

    def nearby(self, view_id: str) -> tuple[str, ...]:
        return self.neighbors.get(view_id, ())
