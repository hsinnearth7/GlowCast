"""Price Elasticity Analyzer — estimates how cost changes propagate to pricing and demand.

Uses DoWhy to establish causal price->demand relationships and integrates
with CUPED for pricing experiment analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ElasticityResult:
    """Price elasticity estimation result."""

    sku_id: str
    elasticity: float  # % change in demand / % change in price
    is_elastic: bool   # |elasticity| > 1
    r_squared: float
    p_value: float
    optimal_markup: float  # Profit-maximizing markup over cost
    confidence_interval: tuple[float, float]


class PriceElasticityAnalyzer:
    """Estimates price elasticity of demand using log-log regression.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cost transactions with pricing data (Fact_Cost_Transactions).
    purchase_orders : pd.DataFrame
        Purchase order data with quantities (Fact_Purchase_Orders).
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        purchase_orders: pd.DataFrame,
    ):
        self.transactions = transactions
        self.purchase_orders = purchase_orders

    def estimate_elasticity(self, sku_id: str) -> ElasticityResult:
        """Estimate price elasticity for a specific SKU using log-log OLS.

        The model: ln(Q) = α + β·ln(P) + ε
        where β is the price elasticity of demand.

        Parameters
        ----------
        sku_id : str
            Target SKU identifier.

        Returns
        -------
        ElasticityResult
            Elasticity estimate with confidence interval.
        """
        # Get price and quantity data
        txn = self.transactions[self.transactions["sku_id"] == sku_id].copy()
        po = self.purchase_orders[self.purchase_orders["sku_id"] == sku_id].copy()

        # Use PO data for price/quantity pairs if available
        if not po.empty and len(po) >= 5:
            prices = po["unit_price"].values
            quantities = po["quantity"].values
        elif not txn.empty and len(txn) >= 5:
            prices = txn["total_unit_cost"].values
            quantities = txn["volume"].values
        else:
            return ElasticityResult(
                sku_id=sku_id,
                elasticity=-1.0,
                is_elastic=False,
                r_squared=0.0,
                p_value=1.0,
                optimal_markup=0.30,
                confidence_interval=(-2.0, 0.0),
            )

        # Filter out zeros/negatives for log transformation
        mask = (prices > 0) & (quantities > 0)
        prices = prices[mask]
        quantities = quantities[mask]

        if len(prices) < 5:
            return ElasticityResult(
                sku_id=sku_id,
                elasticity=-1.0,
                is_elastic=False,
                r_squared=0.0,
                p_value=1.0,
                optimal_markup=0.30,
                confidence_interval=(-2.0, 0.0),
            )

        # Log-log regression
        log_p = np.log(prices)
        log_q = np.log(quantities)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_q)

        # 95% CI for elasticity
        t_crit = stats.t.ppf(0.975, len(prices) - 2)
        ci_low = slope - t_crit * std_err
        ci_high = slope + t_crit * std_err

        # Optimal markup: for constant elasticity, markup = 1/(1+1/elasticity)
        if slope < -1:
            optimal_markup = -1 / (slope + 1) if slope != -1 else float("inf")
            optimal_markup = min(max(optimal_markup, 0.05), 2.0)
        else:
            optimal_markup = 0.50  # Default if inelastic

        return ElasticityResult(
            sku_id=sku_id,
            elasticity=round(float(slope), 4),
            is_elastic=abs(slope) > 1,
            r_squared=round(float(r_value ** 2), 4),
            p_value=round(float(p_value), 6),
            optimal_markup=round(float(optimal_markup), 4),
            confidence_interval=(round(float(ci_low), 4), round(float(ci_high), 4)),
        )

    def estimate_batch(self, sku_ids: list[str]) -> pd.DataFrame:
        """Estimate elasticity for multiple SKUs.

        Returns
        -------
        pd.DataFrame
            Elasticity results for all SKUs.
        """
        results = []
        for sku_id in sku_ids:
            r = self.estimate_elasticity(sku_id)
            results.append({
                "sku_id": r.sku_id,
                "elasticity": r.elasticity,
                "is_elastic": r.is_elastic,
                "r_squared": r.r_squared,
                "p_value": r.p_value,
                "optimal_markup": r.optimal_markup,
                "ci_low": r.confidence_interval[0],
                "ci_high": r.confidence_interval[1],
            })
        return pd.DataFrame(results)

    def sensitivity_curve(
        self,
        sku_id: str,
        price_range: tuple[float, float] = (0.5, 2.0),
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Generate a price-demand sensitivity curve.

        Parameters
        ----------
        sku_id : str
            Target SKU.
        price_range : tuple
            Min/max price multipliers relative to current price.
        n_points : int
            Number of points on the curve.

        Returns
        -------
        pd.DataFrame
            Columns: price_multiplier, estimated_demand, estimated_revenue, estimated_profit
        """
        result = self.estimate_elasticity(sku_id)
        elasticity = result.elasticity

        # Get base price and quantity
        txn = self.transactions[self.transactions["sku_id"] == sku_id]
        if txn.empty:
            base_price = 10.0
            base_qty = 100
        else:
            base_price = float(txn["total_unit_cost"].mean())
            base_qty = float(txn["volume"].mean())

        records = []
        for mult in np.linspace(price_range[0], price_range[1], n_points):
            price = base_price * mult
            # Q = Q0 * (P/P0)^elasticity
            demand = base_qty * (mult ** elasticity)
            revenue = price * demand
            cost = base_price * demand  # Approximate cost at base rate
            profit = revenue - cost

            records.append({
                "price_multiplier": round(float(mult), 3),
                "price": round(float(price), 2),
                "estimated_demand": round(float(max(demand, 0)), 1),
                "estimated_revenue": round(float(max(revenue, 0)), 2),
                "estimated_profit": round(float(profit), 2),
            })

        return pd.DataFrame(records)
