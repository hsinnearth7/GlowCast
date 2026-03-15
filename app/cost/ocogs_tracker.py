"""OCOGS (Overall Cost of Goods Sold) Tracker — monitors actual vs. budgeted cost over time.

Tracks cost trends per SKU/plant/supplier and integrates with CUPED
for measuring cost-reduction experiment effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CostVarianceResult:
    """Cost variance analysis for a period."""

    period_start: str
    period_end: str
    total_actual: float
    total_budget: float
    variance_pct: float
    favorable: bool
    by_category: dict[str, float]


class OCOGSTracker:
    """Tracks actual vs. budgeted OCOGS and analyzes cost trends.

    Parameters
    ----------
    cost_transactions : pd.DataFrame
        Historical cost transactions (Fact_Cost_Transactions).
    products : pd.DataFrame
        Product master with target costs (Dim_Product).
    """

    def __init__(
        self,
        cost_transactions: pd.DataFrame,
        products: pd.DataFrame,
    ):
        self.transactions = cost_transactions.copy()
        self.products = products

        if "transaction_date" in self.transactions.columns:
            self.transactions["transaction_date"] = pd.to_datetime(
                self.transactions["transaction_date"]
            )

    def compute_variance(
        self,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> CostVarianceResult:
        """Compute cost variance between actual and target (budget) for a period.

        Parameters
        ----------
        period_start, period_end : str, optional
            ISO date strings. Defaults to full data range.
        """
        df = self.transactions
        if period_start:
            df = df[df["transaction_date"] >= pd.Timestamp(period_start)]
        if period_end:
            df = df[df["transaction_date"] <= pd.Timestamp(period_end)]

        if df.empty:
            return CostVarianceResult(
                period_start=period_start or "",
                period_end=period_end or "",
                total_actual=0.0,
                total_budget=0.0,
                variance_pct=0.0,
                favorable=True,
                by_category={},
            )

        # Merge with product targets
        merged = df.merge(
            self.products[["sku_id", "target_cost", "category"]],
            on="sku_id",
            how="left",
        )

        total_actual = float((merged["total_unit_cost"] * merged["volume"]).sum())
        total_budget = float((merged["target_cost"].fillna(merged["total_unit_cost"]) * merged["volume"]).sum())

        variance = (total_actual - total_budget) / total_budget if total_budget > 0 else 0.0

        # Variance by category
        by_cat = {}
        if "category" in merged.columns:
            for cat, group in merged.groupby("category"):
                cat_actual = float((group["total_unit_cost"] * group["volume"]).sum())
                cat_budget = float((group["target_cost"].fillna(group["total_unit_cost"]) * group["volume"]).sum())
                by_cat[cat] = round((cat_actual - cat_budget) / cat_budget if cat_budget > 0 else 0.0, 4)

        p_start = period_start or str(df["transaction_date"].min().date())
        p_end = period_end or str(df["transaction_date"].max().date())

        return CostVarianceResult(
            period_start=p_start,
            period_end=p_end,
            total_actual=round(total_actual, 2),
            total_budget=round(total_budget, 2),
            variance_pct=round(variance, 4),
            favorable=variance <= 0,
            by_category=by_cat,
        )

    def trend_analysis(
        self, freq: str = "ME", lookback_months: int = 12
    ) -> pd.DataFrame:
        """Analyze monthly cost trends.

        Parameters
        ----------
        freq : str
            Pandas frequency string (default monthly).
        lookback_months : int
            Number of months to include.

        Returns
        -------
        pd.DataFrame
            Columns: period, avg_unit_cost, total_cost, volume, cost_change_pct
        """
        df = self.transactions.copy()
        if lookback_months and not df.empty:
            cutoff = df["transaction_date"].max() - pd.DateOffset(months=lookback_months)
            df = df[df["transaction_date"] >= cutoff]

        if df.empty:
            return pd.DataFrame(columns=["period", "avg_unit_cost", "total_cost", "volume", "cost_change_pct"])

        grouped = df.groupby(pd.Grouper(key="transaction_date", freq=freq)).agg(
            avg_unit_cost=("total_unit_cost", "mean"),
            total_cost=("total_unit_cost", lambda x: (x * df.loc[x.index, "volume"]).sum()),
            volume=("volume", "sum"),
        ).reset_index()

        grouped.rename(columns={"transaction_date": "period"}, inplace=True)
        grouped["cost_change_pct"] = grouped["avg_unit_cost"].pct_change()

        return grouped

    def flag_outliers(self, z_threshold: float = 2.5) -> pd.DataFrame:
        """Flag cost transactions with Z-score above threshold.

        Returns
        -------
        pd.DataFrame
            Outlier transactions with z_score column.
        """
        if self.transactions.empty:
            return pd.DataFrame()

        df = self.transactions.copy()
        mean_cost = df["total_unit_cost"].mean()
        std_cost = df["total_unit_cost"].std()

        if std_cost == 0 or np.isnan(std_cost):
            df["z_score"] = 0.0
        else:
            df["z_score"] = (df["total_unit_cost"] - mean_cost) / std_cost

        return df[df["z_score"].abs() > z_threshold].sort_values(
            "z_score", ascending=False, key=abs
        )
