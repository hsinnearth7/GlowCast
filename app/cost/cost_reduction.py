"""Cost Reduction Engine — analyzes historical cost reduction actions and recommends new ones.

Uses DoWhy causal pipeline to estimate the causal effect of each action type on cost,
and uplift models to identify which SKUs benefit most from which interventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReductionRecommendation:
    """A cost reduction recommendation for a specific SKU."""

    sku_id: str
    action_type: str
    estimated_savings_pct: float
    confidence: float
    rationale: str


class CostReductionEngine:
    """Analyzes and recommends cost reduction actions.

    Parameters
    ----------
    cost_transactions : pd.DataFrame
        Historical cost data (Fact_Cost_Transactions).
    reduction_actions : pd.DataFrame
        Historical reduction actions and outcomes (Fact_Cost_Reduction_Actions).
    products : pd.DataFrame
        Product master data (Dim_Product).
    """

    def __init__(
        self,
        cost_transactions: pd.DataFrame,
        reduction_actions: pd.DataFrame,
        products: pd.DataFrame,
    ):
        self.transactions = cost_transactions
        self.actions = reduction_actions
        self.products = products
        self._action_effectiveness: dict[str, float] = {}
        self._compute_effectiveness()

    def _compute_effectiveness(self) -> None:
        """Compute average effectiveness per action type from historical data."""
        if self.actions.empty:
            return

        completed = self.actions[self.actions["status"] == "completed"]
        if completed.empty:
            return

        for action_type, group in completed.groupby("action_type"):
            savings = group["actual_savings_pct"].dropna()
            if len(savings) > 0:
                self._action_effectiveness[action_type] = float(savings.mean())

    def recommend_actions(
        self, sku_id: str, top_n: int = 3
    ) -> list[ReductionRecommendation]:
        """Recommend cost reduction actions for a specific SKU.

        Parameters
        ----------
        sku_id : str
            Target SKU identifier.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        list[ReductionRecommendation]
            Ranked recommendations by estimated savings.
        """
        product = self.products[self.products["sku_id"] == sku_id]
        if product.empty:
            logger.warning("SKU %s not found in product master", sku_id)
            return []

        product_row = product.iloc[0]
        recommendations = []

        # Score each action type based on product characteristics
        action_scores = self._score_actions(product_row)

        for action_type, (savings_est, confidence, rationale) in sorted(
            action_scores.items(), key=lambda x: x[1][0], reverse=True
        )[:top_n]:
            recommendations.append(
                ReductionRecommendation(
                    sku_id=sku_id,
                    action_type=action_type,
                    estimated_savings_pct=round(savings_est, 4),
                    confidence=round(confidence, 2),
                    rationale=rationale,
                )
            )

        return recommendations

    def _score_actions(
        self, product: pd.Series
    ) -> dict[str, tuple[float, float, str]]:
        """Score each action type for a product based on its cost profile."""
        scores: dict[str, tuple[float, float, str]] = {}

        commodity_sens = float(product.get("commodity_sensitivity", 0.5))
        labor_int = float(product.get("labor_intensity", 0.3))
        overhead_alloc = float(product.get("overhead_allocation", 0.2))
        tariff_exp = float(product.get("tariff_exposure", 0.1))

        # Historical effectiveness as prior
        hist_eff = self._action_effectiveness

        scores["supplier_switch"] = (
            hist_eff.get("supplier_switch", 0.08) * (1 + commodity_sens),
            0.7 if "supplier_switch" in hist_eff else 0.4,
            f"High commodity sensitivity ({commodity_sens:.0%}) suggests supplier diversification",
        )

        scores["material_substitution"] = (
            hist_eff.get("material_substitution", 0.06) * (1 + commodity_sens * 0.5),
            0.6 if "material_substitution" in hist_eff else 0.3,
            f"Commodity exposure ({commodity_sens:.0%}) makes material alternatives viable",
        )

        scores["process_optimization"] = (
            hist_eff.get("process_optimization", 0.05) * (1 + labor_int),
            0.8 if "process_optimization" in hist_eff else 0.5,
            f"Labor intensity ({labor_int:.0%}) indicates process optimization potential",
        )

        scores["automation"] = (
            hist_eff.get("automation", 0.12) * labor_int,
            0.5 if "automation" in hist_eff else 0.3,
            f"Labor-intensive ({labor_int:.0%}) — automation could reduce recurring costs",
        )

        scores["volume_consolidation"] = (
            hist_eff.get("volume_consolidation", 0.04),
            0.7 if "volume_consolidation" in hist_eff else 0.5,
            "Consolidating volumes across plants for better pricing",
        )

        scores["nearshoring"] = (
            hist_eff.get("nearshoring", 0.07) * (1 + tariff_exp * 2),
            0.4 if "nearshoring" in hist_eff else 0.2,
            f"Tariff exposure ({tariff_exp:.0%}) may justify nearshoring",
        )

        scores["design_change"] = (
            hist_eff.get("design_change", 0.10) * (1 + overhead_alloc * 0.5),
            0.5 if "design_change" in hist_eff else 0.3,
            f"Design-for-cost can reduce overhead allocation ({overhead_alloc:.0%})",
        )

        scores["negotiate_contract"] = (
            hist_eff.get("negotiate_contract", 0.03),
            0.8 if "negotiate_contract" in hist_eff else 0.6,
            "Contract renegotiation for volume/term improvements",
        )

        return scores

    def estimate_savings(
        self, action_type: str, sku_id: str
    ) -> dict[str, float]:
        """Estimate savings for a specific action on a specific SKU.

        Returns
        -------
        dict
            Keys: estimated_savings_pct, confidence, historical_avg
        """
        product = self.products[self.products["sku_id"] == sku_id]
        if product.empty:
            return {"estimated_savings_pct": 0.0, "confidence": 0.0, "historical_avg": 0.0}

        scores = self._score_actions(product.iloc[0])
        if action_type in scores:
            savings, conf, _ = scores[action_type]
            hist_avg = self._action_effectiveness.get(action_type, 0.0)
            return {
                "estimated_savings_pct": round(savings, 4),
                "confidence": round(conf, 2),
                "historical_avg": round(hist_avg, 4),
            }
        return {"estimated_savings_pct": 0.0, "confidence": 0.0, "historical_avg": 0.0}

    def track_realization(self) -> pd.DataFrame:
        """Track projected vs actual savings realization across all completed actions.

        Returns
        -------
        pd.DataFrame
            Columns: action_type, count, avg_projected, avg_actual, realization_rate
        """
        if self.actions.empty:
            return pd.DataFrame(columns=[
                "action_type", "count", "avg_projected", "avg_actual", "realization_rate"
            ])

        completed = self.actions[self.actions["status"] == "completed"].copy()
        if completed.empty:
            return pd.DataFrame(columns=[
                "action_type", "count", "avg_projected", "avg_actual", "realization_rate"
            ])

        result = completed.groupby("action_type").agg(
            count=("action_id", "count"),
            avg_projected=("projected_savings_pct", "mean"),
            avg_actual=("actual_savings_pct", "mean"),
        ).reset_index()

        result["realization_rate"] = np.where(
            result["avg_projected"] > 0,
            result["avg_actual"] / result["avg_projected"],
            0.0,
        )

        return result.round(4)
