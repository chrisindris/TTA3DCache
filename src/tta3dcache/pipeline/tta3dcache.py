from __future__ import annotations

from typing import Any

from ..config import TTA3DCacheConfig
from ..integration.cdviews_adapter import CdViewsAdapter
from ..integration.vlm_runner import VLMRunner
from .baseline import PipelineResult, run_baseline


def run_mvp(example: Any, cfg: TTA3DCacheConfig, *, cdviews_adapter: CdViewsAdapter, vlm_runner: VLMRunner, run_id: str = "tta3dcache") -> PipelineResult:
    return run_baseline(example, cfg, cdviews_adapter=cdviews_adapter, vlm_runner=vlm_runner, run_id=run_id)
