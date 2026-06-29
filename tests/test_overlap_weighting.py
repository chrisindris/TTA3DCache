from __future__ import annotations

import unittest

from tta3dcache.scoring.consistency import agreement_score
from tta3dcache.scoring.uncertainty import normalized_entropy
from tta3dcache.view_sampling.diversity import jaccard_overlap


class OverlapWeightingTests(unittest.TestCase):
    def test_jaccard_overlap(self) -> None:
        self.assertAlmostEqual(jaccard_overlap(("v1", "v2"), ("v2", "v3")), 1 / 3)

    def test_agreement_score(self) -> None:
        self.assertAlmostEqual(agreement_score(["a", "a", "b"]), 2 / 3)

    def test_entropy(self) -> None:
        self.assertLess(normalized_entropy({"a": 0.9, "b": 0.1}), 1.0)


if __name__ == "__main__":
    unittest.main()
