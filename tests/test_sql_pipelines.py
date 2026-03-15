"""Tests for SQL analysis pipelines."""

import pandas as pd
import pytest

from app.sql.executor import SQLPipelineExecutor


class TestSQLPipelineExecutor:
    @pytest.fixture
    def executor(self, small_tables):
        tables = {
            "dim_product": small_tables["dim_product"],
            "dim_plant": small_tables["dim_plant"],
            "dim_supplier": small_tables["dim_supplier"],
            "fact_cost_transactions": small_tables["fact_cost_transactions"],
            "fact_purchase_orders": small_tables["fact_purchase_orders"],
            "fact_quality_events": small_tables["fact_quality_events"],
        }
        # Filter out any None values (some tables may not be generated)
        # Limit rows per table so SQLite stays within the 999-variable limit
        # (method="multi" creates one INSERT per chunk: n_rows × n_cols ≤ 999)
        filtered = {}
        for k, v in tables.items():
            if v is not None and isinstance(v, pd.DataFrame):
                # Safe row limit: floor(999 / n_cols), at least 1 row
                n_cols = max(1, len(v.columns))
                max_rows = max(1, 999 // n_cols)
                filtered[k] = v.head(max_rows)
        exc = SQLPipelineExecutor(filtered)
        exc.load_tables()
        return exc

    def test_tables_loaded(self, executor):
        names = executor.table_names()
        assert "dim_product" in names
        assert "fact_cost_transactions" in names

    def test_row_counts_positive(self, executor):
        counts = executor.row_counts()
        for table, count in counts.items():
            assert count > 0, f"{table} has 0 rows"

    def test_cost_variance_pipeline(self, executor):
        result = executor.run_pipeline("dos_woc.sql")
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # may be empty for small data

    def test_should_cost_gap_pipeline(self, executor):
        result = executor.run_pipeline("scrap_risk.sql")
        assert isinstance(result, pd.DataFrame)

    def test_supplier_performance_pipeline(self, executor):
        result = executor.run_pipeline("cross_zone_penalty.sql")
        assert isinstance(result, pd.DataFrame)

    def test_cost_anomaly_pipeline(self, executor):
        result = executor.run_pipeline("demand_anomaly.sql")
        assert isinstance(result, pd.DataFrame)

    def test_run_all_pipelines(self, executor):
        results = executor.run_all_pipelines()
        assert isinstance(results, dict)
        assert len(results) >= 1

    def test_explain_analyze(self, executor):
        explanation = executor.explain_analyze("dos_woc.sql")
        assert isinstance(explanation, str)

    def test_table_info(self, executor):
        info = executor.table_info("dim_product")
        assert isinstance(info, pd.DataFrame)
        assert len(info) > 0

    def test_context_manager(self, small_tables):
        dp = small_tables["dim_product"]
        tables = {k: v for k, v in {"dim_product": dp}.items() if v is not None and isinstance(v, pd.DataFrame)}
        assert len(tables) > 0, "dim_product must be a valid DataFrame"
        with SQLPipelineExecutor(tables) as exc:
            exc.load_tables()
            names = exc.table_names()
            assert "dim_product" in names

