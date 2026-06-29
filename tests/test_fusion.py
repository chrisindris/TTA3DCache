from __future__ import annotations

import unittest

from tta3dcache.scoring.fusion import fuse_answers


class FusionTests(unittest.TestCase):
    def test_majority_vote_mode_matches_hand_computed_results(self) -> None:
        result = fuse_answers(base_answer="lamp", base_confidence=0.4, cache_scores={"lamp": 2.0, "chair": 1.0}, prototype_scores={}, lambda_base=0.0, lambda_cache=1.0, lambda_proto=0.0, lambda_instability=0.0, fallback_entropy_threshold=1.0)
        self.assertEqual(result.answer, "lamp")

    def test_diversity_weighting_can_prevent_duplicate_dominance(self) -> None:
        result = fuse_answers(base_answer="wrong", base_confidence=0.3, cache_scores={"wrong": 1.0, "correct": 1.5}, prototype_scores={"correct": 0.5}, lambda_base=0.5, lambda_cache=1.0, lambda_proto=1.0, lambda_instability=0.25, fallback_entropy_threshold=1.0)
        self.assertEqual(result.answer, "correct")

    def test_high_entropy_triggers_fallback(self) -> None:
        result = fuse_answers(base_answer="lamp", base_confidence=0.4, cache_scores={"lamp": 1.0, "chair": 1.0}, prototype_scores={}, lambda_base=1.0, lambda_cache=1.0, lambda_proto=0.0, lambda_instability=0.0, fallback_entropy_threshold=0.1)
        self.assertEqual(result.answer, "lamp")
        self.assertTrue(result.fallback_used)

    def test_ties_are_deterministic(self) -> None:
        first = fuse_answers(base_answer="lamp", base_confidence=0.5, cache_scores={"chair": 1.0, "lamp": 1.0}, prototype_scores={}, lambda_base=1.0, lambda_cache=1.0, lambda_proto=0.0, lambda_instability=0.0, fallback_entropy_threshold=1.0)
        second = fuse_answers(base_answer="lamp", base_confidence=0.5, cache_scores={"chair": 1.0, "lamp": 1.0}, prototype_scores={}, lambda_base=1.0, lambda_cache=1.0, lambda_proto=0.0, lambda_instability=0.0, fallback_entropy_threshold=1.0)
        self.assertEqual(first.answer, second.answer)

    def test_base_answer_remains_when_cache_empty(self) -> None:
        result = fuse_answers(base_answer="lamp", base_confidence=0.5, cache_scores={}, prototype_scores={}, lambda_base=1.0, lambda_cache=1.0, lambda_proto=1.0, lambda_instability=0.0, fallback_entropy_threshold=1.0)
        self.assertEqual(result.answer, "lamp")


if __name__ == "__main__":
    unittest.main()
