from __future__ import annotations

import unittest

from tta3dcache.view_sampling.candidate_sets import CdViewsSelection, build_candidate_view_sets


class CandidateSetTests(unittest.TestCase):
    def test_deterministic_under_fixed_seed(self) -> None:
        selection = CdViewsSelection(ranked_view_ids=["v1", "v2", "v3", "v4", "v5"], ranked_scores=[0.9, 0.8, 0.7, 0.6, 0.5], base_view_ids=["v1", "v2", "v3", "v4"], raw_metadata={})
        left = build_candidate_view_sets(selection, num_sets=5, views_per_set=4, diversity_lambda=0.5, include_random_control=True, seed=7)
        right = build_candidate_view_sets(selection, num_sets=5, views_per_set=4, diversity_lambda=0.5, include_random_control=True, seed=7)
        self.assertEqual([candidate.view_ids for candidate in left], [candidate.view_ids for candidate in right])

    def test_no_duplicate_view_sets(self) -> None:
        selection = CdViewsSelection(ranked_view_ids=["v1", "v2", "v3", "v4"], ranked_scores=[0.9, 0.8, 0.7, 0.6], base_view_ids=["v1", "v2", "v3", "v4"], raw_metadata={})
        candidates = build_candidate_view_sets(selection, num_sets=8, views_per_set=4, diversity_lambda=0.5, include_random_control=False, seed=0)
        self.assertEqual(len({candidate.view_ids for candidate in candidates}), len(candidates))

    def test_default_set_is_preserved(self) -> None:
        selection = CdViewsSelection(ranked_view_ids=["v1", "v2", "v3", "v4"], ranked_scores=[0.9, 0.8, 0.7, 0.6], base_view_ids=["v1", "v2", "v3", "v4"], raw_metadata={})
        candidates = build_candidate_view_sets(selection, num_sets=2, views_per_set=4, diversity_lambda=0.5, include_random_control=False, seed=0)
        self.assertEqual(candidates[0].generation_type, "cdviews_default")

    def test_leave_one_out_changes_exactly_one_element_where_possible(self) -> None:
        selection = CdViewsSelection(ranked_view_ids=["v1", "v2", "v3", "v4", "v5"], ranked_scores=[0.9, 0.8, 0.7, 0.6, 0.5], base_view_ids=["v1", "v2", "v3", "v4"], raw_metadata={})
        candidates = build_candidate_view_sets(selection, num_sets=5, views_per_set=4, diversity_lambda=0.5, include_random_control=False, seed=0)
        leave_one_out = next(candidate for candidate in candidates if candidate.generation_type == "leave_one_out")
        self.assertEqual(len(set(leave_one_out.view_ids) - set(selection.base_view_ids)), 1)


if __name__ == "__main__":
    unittest.main()
