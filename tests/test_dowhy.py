"""Tests for DoWhy causal inference pipeline."""

import numpy as np
import pandas as pd
import pytest

from app.causal.dowhy_pipeline import DoWhyPipeline


class TestDoWhyPipeline:
    @pytest.fixture
    def causal_data(self):
        rng = np.random.default_rng(42)
        n = 500
        confounder = rng.normal(0, 1, n)
        treatment = (confounder + rng.normal(0, 0.5, n) > 0).astype(float)
        outcome = 2.0 * treatment + 1.5 * confounder + rng.normal(0, 0.5, n)
        return pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "confounder": confounder,
        })

    def test_build_model(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        pipeline.build_model(causal_data)
        assert pipeline is not None

    def test_identify(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        pipeline.build_model(causal_data)
        pipeline.identify()

    def test_estimate(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        pipeline.build_model(causal_data)
        pipeline.identify()
        result = pipeline.estimate()
        assert "ate" in result
        # True ATE is 2.0; should be close
        assert abs(result["ate"] - 2.0) < 1.0

    def test_refute(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        pipeline.build_model(causal_data)
        pipeline.identify()
        pipeline.estimate()
        refutations = pipeline.refute()
        assert isinstance(refutations, list)
        assert len(refutations) >= 1

    def test_run_pipeline(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        result = pipeline.run_pipeline(causal_data)
        assert "ate" in result or "estimate" in result

    def test_placebo_treatment_refutation(self, causal_data):
        pipeline = DoWhyPipeline(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"],
        )
        pipeline.build_model(causal_data)
        pipeline.identify()
        pipeline.estimate()
        refutations = pipeline.refute(methods=["placebo_treatment"])
        assert len(refutations) >= 1
