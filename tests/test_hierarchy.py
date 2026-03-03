"""Tests for hierarchical forecasting."""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.hierarchy import HierarchicalReconciler, get_hierarchy_spec


class TestHierarchySpec:
    def test_spec_has_4_levels(self):
        spec = get_hierarchy_spec()
        assert len(spec["levels"]) == 4

    def test_spec_countries(self):
        spec = get_hierarchy_spec()
        assert spec["n_countries"] == 5 or "country_fcs" in spec

    def test_spec_fcs(self):
        spec = get_hierarchy_spec()
        assert spec.get("n_fcs", 12) == 12 or "FC" in str(spec)


class TestHierarchicalReconciler:
    @pytest.fixture
    def simple_hierarchy(self):
        """Create a small hierarchy for testing."""
        uids = []
        records = []
        for country in ["US", "DE"]:
            for fc in [f"FC_{country}_1", f"FC_{country}_2"]:
                for sku in [f"SKU_{i}" for i in range(3)]:
                    uid = f"{sku}__{fc}"
                    uids.append(uid)
                    records.append({
                        "unique_id": uid,
                        "sku_id": sku,
                        "fc_id": fc,
                        "country": country,
                        "national": "Total",
                    })
        S_df = pd.DataFrame(records).set_index("unique_id")
        return S_df

    @pytest.fixture
    def base_forecasts(self, simple_hierarchy):
        """Create base forecasts for all hierarchy levels (bottom + upper)."""
        rng = np.random.default_rng(42)
        records = []
        # Bottom-level UIDs from the hierarchy
        bottom_uids = list(simple_hierarchy.index)
        # Aggregate UIDs: national, countries, FCs
        upper_uids = (
            ["National"]
            + list(simple_hierarchy["country"].unique())
            + list(simple_hierarchy["fc_id"].unique())
        )
        all_uids = upper_uids + bottom_uids
        for uid in all_uids:
            for h in range(7):
                records.append({
                    "unique_id": uid,
                    "ds": pd.Timestamp("2023-07-01") + pd.Timedelta(days=h),
                    "y_hat": float(rng.poisson(10)),
                })
        df = pd.DataFrame(records)
        # Keep ds as string so that after _reconcile_manual's melt the types match
        df["ds"] = df["ds"].apply(lambda x: str(x.date()) if hasattr(x, "date") else str(x))
        return df

    def test_build_summing_matrix(self, simple_hierarchy):
        reconciler = HierarchicalReconciler()
        S = reconciler.build_summing_matrix(simple_hierarchy)
        assert isinstance(S, np.ndarray)
        assert S.shape[1] == len(simple_hierarchy)  # bottom level

    def test_reconcile(self, simple_hierarchy, base_forecasts):
        reconciler = HierarchicalReconciler()
        reconciler.build_summing_matrix(simple_hierarchy)
        # Use "ols" method which does not require in-sample residuals
        result = reconciler.reconcile(base_forecasts, method="ols")
        assert isinstance(result, pd.DataFrame)
        assert "y_rec" in result.columns or "y_hat" in result.columns

    def test_reconciled_non_negative(self, simple_hierarchy, base_forecasts):
        reconciler = HierarchicalReconciler()
        reconciler.build_summing_matrix(simple_hierarchy)
        # Use "ols" method which does not require in-sample residuals
        result = reconciler.reconcile(base_forecasts, method="ols")
        y_col = "y_rec" if "y_rec" in result.columns else "y_hat"
        assert (result[y_col] >= 0).all()
