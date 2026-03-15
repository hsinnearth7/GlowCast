"""Should-Cost Model — decomposes product cost into constituent elements and benchmarks against targets.

The Should-Cost approach answers: "What *should* this product cost given its BOM,
labor content, overhead allocation, and market commodity prices?"

Integration:
    - Uses DoWhy causal graphs to identify significant cost drivers
    - Feeds cost gaps into CostReductionEngine for action recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Decomposed cost structure for a single SKU."""

    sku_id: str
    raw_material_cost: float
    labor_cost: float
    overhead_cost: float
    logistics_cost: float
    tariff_cost: float
    total_should_cost: float
    current_actual_cost: float
    gap_pct: float  # (actual - should) / should
    cost_elements: dict[str, float] = field(default_factory=dict)


class ShouldCostModel:
    """Decomposes and benchmarks product costs against should-cost targets.

    Parameters
    ----------
    commodity_prices : pd.DataFrame
        Current commodity price index (Fact_Commodity_Prices).
    plant_data : pd.DataFrame
        Plant labor/overhead rates (Dim_Plant).
    supplier_data : pd.DataFrame
        Supplier price premiums (Dim_Supplier).
    """

    def __init__(
        self,
        commodity_prices: pd.DataFrame,
        plant_data: pd.DataFrame,
        supplier_data: pd.DataFrame,
    ):
        self.commodity_prices = commodity_prices
        self.plant_data = plant_data
        self.supplier_data = supplier_data
        self._commodity_index = self._build_commodity_index()

    def _build_commodity_index(self) -> dict[str, float]:
        """Build latest commodity price index lookup."""
        if self.commodity_prices.empty:
            return {}
        latest = (
            self.commodity_prices
            .sort_values("price_date")
            .groupby("commodity")
            .last()
        )
        return latest["price_index"].to_dict()

    def decompose(self, product: pd.Series, plant_id: str) -> CostBreakdown:
        """Decompose a product's cost into should-cost elements.

        Parameters
        ----------
        product : pd.Series
            A row from Dim_Product with cost attributes.
        plant_id : str
            Manufacturing plant ID for labor/overhead rates.

        Returns
        -------
        CostBreakdown
            Full cost decomposition with gap analysis.
        """
        base_cost = float(product["base_unit_cost"])
        commodity_sens = float(product["commodity_sensitivity"])
        labor_int = float(product["labor_intensity"])
        overhead_alloc = float(product["overhead_allocation"])
        tariff_exp = float(product["tariff_exposure"])

        # Get plant rates
        plant = self.plant_data[self.plant_data["plant_id"] == plant_id]
        if plant.empty:
            labor_rate = 15.0
            overhead_rate = 0.30
        else:
            labor_rate = float(plant.iloc[0]["labor_rate_hourly"])
            overhead_rate = float(plant.iloc[0]["overhead_rate"])

        # Get commodity price multiplier
        commodity = product.get("commodity_exposure", "Steel")
        commodity_idx = self._commodity_index.get(commodity, 1.0)
        commodity_multiplier = 1.0 + (commodity_idx - 1.0) * commodity_sens

        # Should-cost decomposition
        raw_material = base_cost * commodity_sens * commodity_multiplier
        labor = base_cost * labor_int * (labor_rate / 15.0)  # Normalized to reference rate
        overhead = base_cost * overhead_alloc * (1 + overhead_rate)
        logistics = base_cost * 0.05  # ~5% logistics baseline
        tariff = base_cost * tariff_exp

        total_should = raw_material + labor + overhead + logistics + tariff
        actual = float(product.get("target_cost", total_should * 1.1))
        gap = (actual - total_should) / total_should if total_should > 0 else 0.0

        return CostBreakdown(
            sku_id=str(product["sku_id"]),
            raw_material_cost=round(raw_material, 4),
            labor_cost=round(labor, 4),
            overhead_cost=round(overhead, 4),
            logistics_cost=round(logistics, 4),
            tariff_cost=round(tariff, 4),
            total_should_cost=round(total_should, 4),
            current_actual_cost=round(actual, 4),
            gap_pct=round(gap, 4),
            cost_elements={
                "raw_material": round(raw_material, 4),
                "labor": round(labor, 4),
                "overhead": round(overhead, 4),
                "logistics": round(logistics, 4),
                "tariff": round(tariff, 4),
            },
        )

    def decompose_batch(
        self, products: pd.DataFrame, plant_id: str
    ) -> list[CostBreakdown]:
        """Decompose costs for multiple products."""
        results = []
        for _, row in products.iterrows():
            results.append(self.decompose(row, plant_id))
        return results

    def benchmark(self, breakdowns: list[CostBreakdown]) -> pd.DataFrame:
        """Benchmark should-cost vs actual across products.

        Returns a DataFrame with gap analysis sorted by largest gaps.
        """
        records = []
        for bd in breakdowns:
            records.append({
                "sku_id": bd.sku_id,
                "should_cost": bd.total_should_cost,
                "actual_cost": bd.current_actual_cost,
                "gap_pct": bd.gap_pct,
                "gap_abs": bd.current_actual_cost - bd.total_should_cost,
                "largest_element": max(bd.cost_elements, key=bd.cost_elements.get),
            })
        df = pd.DataFrame(records)
        return df.sort_values("gap_pct", ascending=False).reset_index(drop=True)

    def identify_gaps(
        self, breakdowns: list[CostBreakdown], threshold: float = 0.10
    ) -> list[CostBreakdown]:
        """Identify products where actual cost exceeds should-cost by more than threshold."""
        return [bd for bd in breakdowns if bd.gap_pct > threshold]
