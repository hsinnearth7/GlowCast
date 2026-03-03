"""Tests for MLflow experiment tracking."""

from unittest.mock import MagicMock, patch

import pytest

from app.mlops.mlflow_tracker import ExperimentTracker


class TestExperimentTracker:
    @pytest.fixture
    def tracker(self):
        return ExperimentTracker(experiment_name="test_experiment")

    def test_start_run(self, tracker):
        with tracker.start_run("test_run_1", params={"lr": 0.01}):
            tracker.log_metrics({"mape": 0.12, "rmse": 8.3})

    def test_log_metrics(self, tracker):
        with tracker.start_run("test_run_2"):
            tracker.log_metrics({"mape": 0.15})

    def test_promote_champion_better(self, tracker):
        # Patch MlflowClient to avoid registry errors for non-registered models
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []
        with patch("app.mlops.mlflow_tracker.MlflowClient", return_value=mock_client):
            result = tracker.promote_champion(
                "test_model",
                challenger_metrics={"mape": 0.10},
                champion_metrics={"mape": 0.15},
            )
        assert result == "promoted"

    def test_retain_champion_worse(self, tracker):
        result = tracker.promote_champion(
            "test_model",
            challenger_metrics={"mape": 0.20},
            champion_metrics={"mape": 0.15},
        )
        assert result == "retained"

    def test_promote_champion_equal(self, tracker):
        result = tracker.promote_champion(
            "test_model",
            challenger_metrics={"mape": 0.15},
            champion_metrics={"mape": 0.15},
        )
        assert result == "retained"
