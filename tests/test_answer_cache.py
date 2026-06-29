from __future__ import annotations

import unittest

import torch

from tta3dcache.cache.answer_cache import AnswerCache
from tta3dcache.cache.entries import CacheEntry


class AnswerCacheTests(unittest.TestCase):
    def test_rejects_low_confidence_entries(self) -> None:
        cache = AnswerCache(minimum_confidence=0.5)
        updated = cache.update(CacheEntry(key=torch.tensor([1.0]), value="lamp", confidence=0.2, uncertainty=0.8, diversity=1.0, overlap=0.0))
        self.assertFalse(updated)
        self.assertEqual(cache.entries, [])

    def test_respects_max_size(self) -> None:
        cache = AnswerCache(max_size=2, minimum_confidence=0.0)
        for index in range(3):
            cache.update(CacheEntry(key=torch.tensor([float(index)]), value=f"answer-{index}", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        self.assertLessEqual(len(cache.entries), 2)

    def test_aggregates_equivalent_answers(self) -> None:
        cache = AnswerCache(minimum_confidence=0.0)
        cache.update(CacheEntry(key=torch.tensor([1.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        cache.update(CacheEntry(key=torch.tensor([0.9]), value="lamp", confidence=0.8, uncertainty=0.2, diversity=0.8, overlap=0.1))
        scores = cache.score_answers()
        self.assertGreater(scores["lamp"], 0.0)

    def test_overlap_lowers_effective_weight(self) -> None:
        cache = AnswerCache(minimum_confidence=0.0)
        cache.update(CacheEntry(key=torch.tensor([1.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        cache.update(CacheEntry(key=torch.tensor([1.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.9))
        scores = cache.score_answers()
        self.assertLess(scores["lamp"], 2.0)

    def test_empty_cache_returns_empty_scores(self) -> None:
        cache = AnswerCache()
        self.assertEqual(cache.score_answers(), {})

    def test_clear_resets_state(self) -> None:
        cache = AnswerCache(minimum_confidence=0.0)
        cache.update(CacheEntry(key=torch.tensor([1.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        cache.clear()
        self.assertEqual(cache.entries, [])


if __name__ == "__main__":
    unittest.main()
