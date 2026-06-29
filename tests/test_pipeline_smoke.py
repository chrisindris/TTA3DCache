from __future__ import annotations

import unittest

from tta3dcache.config import load_config
from tta3dcache.integration.cdviews_adapter import StaticCdViewsAdapter
from tta3dcache.integration.vlm_runner import MockVLMRunner
from tta3dcache.pipeline.tta3dcache import run_mvp


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_downweights_overlapping_wrong_answers(self) -> None:
        cfg = load_config("configs/full_tta3dcache.yaml", {"candidate_sets": {"num_sets": 5}, "cache": {"enabled": True}, "prototype": {"enabled": True}, "fusion": {"fallback_entropy_threshold": 1.0}})
        example = {
            "scene_id": "scene-1",
            "question_id": "q-1",
            "question": "What object is on the table?",
            "ranked_view_ids": ["v1", "v2", "v3", "v4", "v5"],
            "ranked_scores": [0.99, 0.98, 0.97, 0.4, 0.39],
            "base_view_ids": ["v1", "v2", "v3", "v4"],
            "metadata": {"nearby_view_ids": {"v3": ["v5"]}},
        }
        answer_map = {
            ("v1", "v2", "v3", "v4"): ("wrong answer", 0.4),
            ("v5", "v2", "v3", "v4"): ("wrong answer", 0.4),
            ("v1", "v5", "v3", "v4"): ("wrong answer", 0.4),
            ("v1", "v2", "v5", "v4"): ("correct answer", 0.95),
            ("v1", "v2", "v3", "v5"): ("correct answer", 0.95),
        }
        result = run_mvp(example, cfg, cdviews_adapter=StaticCdViewsAdapter(), vlm_runner=MockVLMRunner(answer_map, default_answer="wrong answer"), run_id="smoke")
        self.assertEqual(result.final_answer, "correct answer")
        self.assertGreaterEqual(result.vlm_calls, 1)


if __name__ == "__main__":
    unittest.main()
