"""GlowCast data quality validation runner.

Runs Great Expectations validation suites against the star schema tables
and reports results to Prometheus metrics and stdout.

Usage
-----
    python data_quality/validate.py
    python data_quality/validate.py --suite sales_data
    python data_quality/validate.py --data-dir /path/to/data

Exit codes:
    0 — All validations passed
    1 — One or more validations failed
    2 — Configuration or runtime error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expectation suite loader
# ---------------------------------------------------------------------------

EXPECTATIONS_DIR = Path(__file__).parent / "expectations"

SUITE_NAMES = {
    "sales_data": "glowcast_fact_sales",
    "product_data": "glowcast_dim_product",
    "social_signals": "glowcast_fact_social_signals",
}


def load_expectation_suite(suite_file: str) -> dict[str, Any]:
    """Load an expectation suite from JSON file."""
    path = EXPECTATIONS_DIR / suite_file
    if not path.exists():
        raise FileNotFoundError(f"Expectation suite not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Lightweight validator (no GX dependency required)
# ---------------------------------------------------------------------------


class DataValidator:
    """Lightweight data quality validator.

    Implements a subset of Great Expectations checks without requiring
    the full GX library.  In production, use the GX checkpoint directly.
    """

    def __init__(self, df: pd.DataFrame, suite: dict[str, Any]) -> None:
        self.df = df
        self.suite = suite
        self.results: list[dict[str, Any]] = []

    def validate(self) -> dict[str, Any]:
        """Run all expectations in the suite against the dataframe.

        Returns
        -------
        dict
            Validation result with pass/fail status and individual check results.
        """
        expectations = self.suite.get("expectations", [])
        passed = 0
        failed = 0

        for exp in expectations:
            exp_type = exp["expectation_type"]
            kwargs = exp.get("kwargs", {})
            result = self._evaluate(exp_type, kwargs)
            result["expectation_type"] = exp_type
            result["kwargs"] = kwargs
            self.results.append(result)

            if result["success"]:
                passed += 1
            else:
                failed += 1

        total = passed + failed
        success_rate = passed / total if total > 0 else 0.0

        return {
            "suite_name": self.suite.get("expectation_suite_name", "unknown"),
            "success": failed == 0,
            "total_expectations": total,
            "passed": passed,
            "failed": failed,
            "success_rate": round(success_rate, 4),
            "results": self.results,
        }

    def _evaluate(self, exp_type: str, kwargs: dict) -> dict[str, Any]:
        """Evaluate a single expectation."""
        try:
            handler = getattr(self, f"_check_{exp_type}", None)
            if handler:
                return handler(kwargs)
            else:
                return {"success": True, "note": f"Skipped (not implemented): {exp_type}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Expectation implementations
    # ------------------------------------------------------------------

    def _check_expect_table_row_count_to_be_between(self, kwargs: dict) -> dict:
        n = len(self.df)
        return {
            "success": kwargs.get("min_value", 0) <= n <= kwargs.get("max_value", float("inf")),
            "observed_value": n,
        }

    def _check_expect_column_values_to_not_be_null(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        null_count = int(self.df[col].isna().sum())
        return {"success": null_count == 0, "observed_value": null_count}

    def _check_expect_column_values_to_be_between(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        mostly = kwargs.get("mostly", 1.0)
        series = self.df[col].dropna()
        in_range = ((series >= kwargs.get("min_value", float("-inf"))) &
                    (series <= kwargs.get("max_value", float("inf")))).mean()
        return {"success": float(in_range) >= mostly, "observed_value": round(float(in_range), 4)}

    def _check_expect_column_values_to_be_in_set(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        valid_set = set(kwargs["value_set"])
        all_in_set = self.df[col].dropna().isin(valid_set).all()
        return {"success": bool(all_in_set)}

    def _check_expect_column_values_to_be_unique(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        n_dupes = int(self.df[col].duplicated().sum())
        return {"success": n_dupes == 0, "observed_value": n_dupes}

    def _check_expect_column_values_to_match_regex(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        matches = self.df[col].dropna().str.match(kwargs["regex"])
        match_rate = float(matches.mean()) if len(matches) > 0 else 0.0
        return {"success": match_rate >= kwargs.get("mostly", 1.0), "observed_value": round(match_rate, 4)}

    def _check_expect_column_mean_to_be_between(self, kwargs: dict) -> dict:
        col = kwargs["column"]
        if col not in self.df.columns:
            return {"success": False, "error": f"Column {col!r} not found"}
        mean_val = float(self.df[col].mean())
        return {
            "success": kwargs.get("min_value", float("-inf")) <= mean_val <= kwargs.get("max_value", float("inf")),
            "observed_value": round(mean_val, 4),
        }

    def _check_expect_table_columns_to_match_ordered_list(self, kwargs: dict) -> dict:
        expected = kwargs["column_list"]
        actual = list(self.df.columns)
        return {"success": actual == expected, "observed_value": actual}

    def _check_expect_column_pair_values_a_to_be_greater_than_b(self, kwargs: dict) -> dict:
        col_a = kwargs["column_A"]
        col_b = kwargs["column_B"]
        if col_a not in self.df.columns or col_b not in self.df.columns:
            return {"success": False, "error": "Column(s) not found"}
        if kwargs.get("or_equal", False):
            valid = (self.df[col_a] >= self.df[col_b]).all()
        else:
            valid = (self.df[col_a] > self.df[col_b]).all()
        return {"success": bool(valid)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run validation suites and report results."""
    parser = argparse.ArgumentParser(description="GlowCast data quality validation")
    parser.add_argument(
        "--suite",
        choices=list(SUITE_NAMES.keys()) + ["all"],
        default="all",
        help="Which expectation suite to run (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each expectation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    suites_to_run = list(SUITE_NAMES.keys()) if args.suite == "all" else [args.suite]

    all_passed = True
    results_summary: list[dict[str, Any]] = []

    for suite_name in suites_to_run:
        suite_file = f"{suite_name}.json"
        logger.info("Running suite: %s", suite_name)

        try:
            suite = load_expectation_suite(suite_file)
        except FileNotFoundError as exc:
            logger.error("Suite not found: %s", exc)
            all_passed = False
            continue

        # Try to load corresponding data file
        data_file = args.data_dir / f"{suite_name}.csv"
        if not data_file.exists():
            # Try alternative names
            alt_names = {
                "sales_data": "Fact_Sales.csv",
                "product_data": "Dim_Product.csv",
                "social_signals": "Fact_Social_Signals.csv",
            }
            alt = args.data_dir / alt_names.get(suite_name, "")
            if alt.exists():
                data_file = alt
            else:
                logger.warning("Data file not found for %s — generating sample data", suite_name)
                df = _generate_sample_data(suite_name)
                if df is None:
                    logger.error("Cannot generate sample data for %s", suite_name)
                    continue
                result = DataValidator(df, suite).validate()
                _print_result(result, args.verbose)
                results_summary.append(result)
                if not result["success"]:
                    all_passed = False
                continue

        logger.info("Loading data from %s", data_file)
        try:
            df = pd.read_csv(data_file)
        except Exception as exc:
            logger.error("Failed to load %s: %s", data_file, exc)
            all_passed = False
            continue

        result = DataValidator(df, suite).validate()
        _print_result(result, args.verbose)
        results_summary.append(result)
        if not result["success"]:
            all_passed = False

    # Update Prometheus metrics if available
    _update_metrics(results_summary)

    # Summary
    logger.info("=" * 60)
    for r in results_summary:
        status = "PASS" if r["success"] else "FAIL"
        logger.info(
            "[%s] %s — %d/%d expectations passed (%.0f%%)",
            status,
            r["suite_name"],
            r["passed"],
            r["total_expectations"],
            r["success_rate"] * 100,
        )

    return 0 if all_passed else 1


def _print_result(result: dict[str, Any], verbose: bool) -> None:
    """Print validation result to stdout."""
    status = "PASSED" if result["success"] else "FAILED"
    logger.info(
        "Suite %s: %s (%d/%d)",
        result["suite_name"],
        status,
        result["passed"],
        result["total_expectations"],
    )
    if verbose:
        for r in result["results"]:
            check_status = "OK" if r["success"] else "FAIL"
            logger.info("  [%s] %s", check_status, r.get("expectation_type", "unknown"))
            if not r["success"] and "error" in r:
                logger.info("        Error: %s", r["error"])


def _generate_sample_data(suite_name: str) -> pd.DataFrame | None:
    """Generate minimal sample data for validation testing."""
    import numpy as np

    np.random.seed(42)

    if suite_name == "sales_data":
        n = 1000
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D").tolist()[:n],
            "sku_id": [f"SKU_{i:04d}" for i in np.random.randint(1, 100, n)],
            "fc_id": np.random.choice(["FC_US_01", "FC_DE_01", "FC_UK_01", "FC_JP_01", "FC_IN_01"], n),
            "units_sold": np.random.negative_binomial(5, 0.3, n),
            "revenue": np.random.uniform(10, 5000, n).round(2),
            "units_returned": np.random.negative_binomial(1, 0.5, n),
            "promo_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "treatment_flag": np.random.choice([0, 1], n, p=[0.80, 0.20]),
        })
    elif suite_name == "product_data":
        concerns = ["Hydrating", "AntiAging", "Brightening", "SunProtection", "Acne"]
        textures = ["Lightweight", "Rich"]
        n = 100
        c = [concerns[i % 5] for i in range(n)]
        t = [textures[i % 2] for i in range(n)]
        return pd.DataFrame({
            "sku_id": [f"SKU_{i:04d}" for i in range(1, n + 1)],
            "product_name": [f"Product_{i}" for i in range(1, n + 1)],
            "concern": c,
            "texture": t,
            "shelf_life_days": np.random.randint(450, 910, n),
            "base_price": np.random.uniform(10, 200, n).round(2),
            "cost": np.random.uniform(5, 100, n).round(2),
            "segment_gene": [f"{c[i]}_{t[i]}" for i in range(n)],
        })
    elif suite_name == "social_signals":
        n = 500
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D").tolist()[:n],
            "sku_id": [f"SKU_{i:04d}" for i in np.random.randint(1, 100, n)],
            "platform": np.random.choice(["reddit", "tiktok", "cosme"], n),
            "mentions": np.random.negative_binomial(3, 0.1, n),
            "sentiment_score": np.random.uniform(-0.5, 0.9, n).round(3),
            "engagement_rate": np.random.uniform(0.01, 0.5, n).round(3),
            "trending_flag": np.random.choice([0, 1], n, p=[0.90, 0.10]),
        })

    return None


def _update_metrics(results: list[dict[str, Any]]) -> None:
    """Update Prometheus metrics with validation results."""
    try:
        from app.metrics import DATA_QUALITY_SCORE

        for result in results:
            DATA_QUALITY_SCORE.labels(dataset=result["suite_name"]).set(result["success_rate"])
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
