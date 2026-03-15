"""Make-vs-Buy Calculator — compares internal manufacturing cost vs. external supplier quotes.

Evaluates the economic trade-off between in-house production and outsourcing,
considering cost, quality, lead time, capacity utilization, and strategic factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MakeVsBuyResult:
    """Result of a make-vs-buy analysis for a single SKU."""

    sku_id: str
    make_cost: float
    buy_cost: float
    cost_advantage: str  # "make" or "buy"
    cost_delta_pct: float
    quality_score_make: float
    quality_score_buy: float
    lead_time_make: int
    lead_time_buy: int
    recommendation: str  # "make", "buy", or "review"
    composite_score_make: float
    composite_score_buy: float
    breakeven_volume: int | None


class MakeVsBuyCalculator:
    """Compares in-house manufacturing vs outsourcing across multiple dimensions.

    Parameters
    ----------
    products : pd.DataFrame
        Product master (Dim_Product).
    plants : pd.DataFrame
        Plant data (Dim_Plant).
    supplier_quotes : pd.DataFrame
        Supplier quotes (Fact_Supplier_Quotes).
    quality_events : pd.DataFrame
        Quality inspection data (Fact_Quality_Events).
    weights : dict, optional
        Dimension weights for composite scoring.
    """

    DEFAULT_WEIGHTS = {
        "cost": 0.35,
        "quality": 0.30,
        "lead_time": 0.20,
        "strategic": 0.15,
    }

    def __init__(
        self,
        products: pd.DataFrame,
        plants: pd.DataFrame,
        supplier_quotes: pd.DataFrame,
        quality_events: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.products = products
        self.plants = plants
        self.quotes = supplier_quotes
        self.quality_events = quality_events if quality_events is not None else pd.DataFrame()
        self.weights = weights or self.DEFAULT_WEIGHTS

    def analyze(self, sku_id: str, plant_id: str) -> MakeVsBuyResult:
        """Run make-vs-buy analysis for a specific SKU at a specific plant.

        Parameters
        ----------
        sku_id : str
            Product SKU identifier.
        plant_id : str
            Plant for in-house manufacturing.

        Returns
        -------
        MakeVsBuyResult
            Comprehensive comparison result.
        """
        product = self.products[self.products["sku_id"] == sku_id]
        if product.empty:
            raise ValueError(f"SKU {sku_id} not found")
        product_row = product.iloc[0]

        plant = self.plants[self.plants["plant_id"] == plant_id]
        if plant.empty:
            raise ValueError(f"Plant {plant_id} not found")
        plant_row = plant.iloc[0]

        # Make cost (internal manufacturing)
        base_cost = float(product_row["base_unit_cost"])
        labor_rate = float(plant_row["labor_rate_hourly"])
        overhead_rate = float(plant_row["overhead_rate"])
        labor_int = float(product_row["labor_intensity"])

        make_cost = base_cost * (1 + labor_int * (labor_rate / 15.0 - 1) + overhead_rate)

        # Buy cost (best supplier quote)
        sku_quotes = self.quotes[self.quotes["sku_id"] == sku_id]
        if sku_quotes.empty:
            buy_cost = base_cost * 1.15  # Default: 15% premium if no quotes
            best_supplier = product_row.get("primary_supplier", "Unknown")
            buy_lead_time = 14
        else:
            latest_quotes = sku_quotes.sort_values("quote_date").groupby("supplier_id").last()
            best_quote = latest_quotes.loc[latest_quotes["quoted_price"].idxmin()]
            buy_cost = float(best_quote["quoted_price"])
            best_supplier = str(best_quote.name)
            buy_lead_time = int(best_quote["lead_time_days"])

        # Quality scores
        quality_make = self._compute_quality_score(sku_id, plant_id=plant_id)
        quality_buy = self._compute_quality_score(sku_id, supplier_id=best_supplier if not sku_quotes.empty else None)

        # Lead times
        make_lead_time = max(1, int(5 + labor_int * 10))  # Estimated internal lead time

        # Composite scoring
        cost_advantage = "make" if make_cost <= buy_cost else "buy"
        cost_delta = (buy_cost - make_cost) / min(make_cost, buy_cost) if min(make_cost, buy_cost) > 0 else 0

        w = self.weights
        # Normalize scores to 0-1 (higher is better)
        max_cost = max(make_cost, buy_cost)
        cost_score_make = 1 - (make_cost / max_cost) if max_cost > 0 else 0.5
        cost_score_buy = 1 - (buy_cost / max_cost) if max_cost > 0 else 0.5

        max_lt = max(make_lead_time, buy_lead_time)
        lt_score_make = 1 - (make_lead_time / max_lt) if max_lt > 0 else 0.5
        lt_score_buy = 1 - (buy_lead_time / max_lt) if max_lt > 0 else 0.5

        # Strategic score: in-house gives more control
        strategic_make = 0.7
        strategic_buy = 0.4

        composite_make = (
            w["cost"] * cost_score_make
            + w["quality"] * quality_make
            + w["lead_time"] * lt_score_make
            + w["strategic"] * strategic_make
        )
        composite_buy = (
            w["cost"] * cost_score_buy
            + w["quality"] * quality_buy
            + w["lead_time"] * lt_score_buy
            + w["strategic"] * strategic_buy
        )

        # Recommendation
        if composite_make > composite_buy * 1.05:
            recommendation = "make"
        elif composite_buy > composite_make * 1.05:
            recommendation = "buy"
        else:
            recommendation = "review"

        # Breakeven volume (simplified)
        fixed_overhead = base_cost * overhead_rate * 100  # rough fixed cost
        if buy_cost > make_cost and (buy_cost - make_cost) > 0:
            breakeven_vol = int(fixed_overhead / (buy_cost - make_cost))
        elif make_cost > buy_cost and (make_cost - buy_cost) > 0:
            breakeven_vol = int(fixed_overhead / (make_cost - buy_cost))
        else:
            breakeven_vol = None

        return MakeVsBuyResult(
            sku_id=sku_id,
            make_cost=round(make_cost, 4),
            buy_cost=round(buy_cost, 4),
            cost_advantage=cost_advantage,
            cost_delta_pct=round(cost_delta, 4),
            quality_score_make=round(quality_make, 3),
            quality_score_buy=round(quality_buy, 3),
            lead_time_make=make_lead_time,
            lead_time_buy=buy_lead_time,
            recommendation=recommendation,
            composite_score_make=round(composite_make, 4),
            composite_score_buy=round(composite_buy, 4),
            breakeven_volume=breakeven_vol,
        )

    def _compute_quality_score(
        self, sku_id: str, plant_id: str | None = None, supplier_id: str | None = None
    ) -> float:
        """Compute quality score from historical quality events."""
        if self.quality_events.empty:
            return 0.92  # Default quality score

        df = self.quality_events[self.quality_events["sku_id"] == sku_id]
        if plant_id:
            df = df[df["plant_id"] == plant_id]
        if supplier_id:
            df = df[df["supplier_id"] == supplier_id]

        if df.empty:
            return 0.92

        return float(1 - df["defect_rate"].mean())

    def sensitivity_analysis(
        self, sku_id: str, plant_id: str, cost_range: float = 0.20, steps: int = 5
    ) -> pd.DataFrame:
        """Run sensitivity analysis on make-vs-buy over a cost range.

        Parameters
        ----------
        sku_id, plant_id : str
            Target SKU and plant.
        cost_range : float
            +/- percentage range to vary costs.
        steps : int
            Number of steps in each direction.

        Returns
        -------
        pd.DataFrame
            Sensitivity results with columns: cost_change_pct, make_cost, buy_cost, recommendation.
        """
        base_result = self.analyze(sku_id, plant_id)
        results = []

        for pct in np.linspace(-cost_range, cost_range, 2 * steps + 1):
            adjusted_make = base_result.make_cost * (1 + pct)
            adjusted_buy = base_result.buy_cost * (1 + pct * 0.5)  # Buy cost less sensitive

            rec = "make" if adjusted_make < adjusted_buy else "buy"
            results.append({
                "cost_change_pct": round(pct, 3),
                "make_cost": round(adjusted_make, 4),
                "buy_cost": round(adjusted_buy, 4),
                "recommendation": rec,
            })

        return pd.DataFrame(results)
