from __future__ import annotations

import unittest

from tta3dcache.answers.equivalence import AnswerEquivalenceResolver
from tta3dcache.answers.normalization import normalize_open_answer


class AnswerNormalizationTests(unittest.TestCase):
    def test_equivalent_spatial_phrasings_group_together(self) -> None:
        resolver = AnswerEquivalenceResolver()
        self.assertEqual(resolver.normalized("on the left side of the table"), resolver.normalized("left of the table"))
        self.assertEqual(resolver.normalized("to the table's left"), resolver.normalized("left of the table"))

    def test_opposite_relations_remain_distinct(self) -> None:
        resolver = AnswerEquivalenceResolver()
        self.assertNotEqual(resolver.normalized("left"), resolver.normalized("right"))

    def test_articles_and_punctuation_are_removed_safely(self) -> None:
        self.assertEqual(normalize_open_answer("The, lamp!"), "lamp")

    def test_number_normalization_behaves_predictably(self) -> None:
        self.assertEqual(normalize_open_answer("two chairs"), "2 chairs")


if __name__ == "__main__":
    unittest.main()
