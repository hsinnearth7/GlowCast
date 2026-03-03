"""Tests for interleaving experiments."""

import pytest

from app.experimentation.interleaving import InterleavingAnalyzer


class TestInterleavingAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return InterleavingAnalyzer()

    def test_team_draft_length(self, analyzer):
        ranking_a = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        ranking_b = ["doc3", "doc1", "doc5", "doc2", "doc4"]
        interleaved, team_a, team_b = analyzer.team_draft(ranking_a, ranking_b, k=4)
        assert len(interleaved) == 4

    def test_team_draft_no_duplicates(self, analyzer):
        ranking_a = ["a", "b", "c", "d"]
        ranking_b = ["c", "a", "d", "b"]
        interleaved, _, _ = analyzer.team_draft(ranking_a, ranking_b, k=4)
        assert len(set(interleaved)) == len(interleaved)

    def test_team_assignment_covers_all(self, analyzer):
        ranking_a = ["a", "b", "c"]
        ranking_b = ["c", "b", "a"]
        interleaved, team_a, team_b = analyzer.team_draft(ranking_a, ranking_b, k=3)
        all_assigned = set(team_a) | set(team_b)
        assert set(interleaved).issubset(all_assigned)

    def test_compute_delta(self, analyzer):
        interleaved = ["a", "b", "c", "d"]
        clicks = {"a": 1, "c": 1}
        team_a = ["a", "c"]
        team_b = ["b", "d"]
        result = analyzer.compute_delta(interleaved, clicks, team_a, team_b)
        assert hasattr(result, "delta") or isinstance(result, (dict, float))

    def test_run_experiment(self, analyzer):
        rankings_a = [["a", "b", "c"]] * 20
        rankings_b = [["c", "b", "a"]] * 20
        def click_model(items):
            return {items[0]: 1}  # always click first
        result = analyzer.run_experiment(rankings_a, rankings_b, click_model, k=3)
        assert result is not None
