from __future__ import annotations

import unittest

import torch

from tta3dcache.cache.entries import CacheEntry
from tta3dcache.cache.prototype_cache import PrototypeCache


class PrototypeCacheTests(unittest.TestCase):
    def test_normalized_prototypes(self) -> None:
        cache = PrototypeCache(momentum=0.5, minimum_support=1)
        cache.update(CacheEntry(key=torch.tensor([1.0, 0.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        state = cache.prototypes["lamp"]
        self.assertAlmostEqual(float(torch.linalg.norm(state.vector).item()), 1.0, places=5)

    def test_deterministic_ema(self) -> None:
        cache = PrototypeCache(momentum=0.5, minimum_support=1)
        first = CacheEntry(key=torch.tensor([1.0, 0.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0)
        second = CacheEntry(key=torch.tensor([0.0, 1.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0)
        cache.update(first)
        cache.update(second)
        state = cache.prototypes["lamp"]
        self.assertAlmostEqual(float(torch.linalg.norm(state.vector).item()), 1.0, places=5)

    def test_minimum_support_behavior(self) -> None:
        cache = PrototypeCache(momentum=0.9, minimum_support=2)
        cache.update(CacheEntry(key=torch.tensor([1.0, 0.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        self.assertEqual(cache.score_answers(torch.tensor([1.0, 0.0])), {})

    def test_no_nans_for_zero_or_degenerate_features(self) -> None:
        cache = PrototypeCache(momentum=0.9, minimum_support=1)
        updated = cache.update(CacheEntry(key=torch.tensor([0.0, 0.0]), value="lamp", confidence=1.0, uncertainty=0.0, diversity=1.0, overlap=0.0))
        self.assertFalse(updated)


if __name__ == "__main__":
    unittest.main()
