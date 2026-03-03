"""Hierarchical forecast reconciliation for GlowCast.

4-layer hierarchy
-----------------
    Level 0 — National  (1 node)         : total across all markets
    Level 1 — Country   (5 nodes)        : US, DE, UK, JP, IN
    Level 2 — FC        (12 nodes)       : one fulfillment-center per row
    Level 3 — SKU       (5 000 nodes)    : leaf series

The summing matrix S maps bottom-level SKU×FC forecasts to every aggregate
node.  Reconciliation uses MinTrace (Wickramasuriya et al., 2019):

    P = S (S'WS)^{-1} S'W
    ŷ_reconciled = S P ŷ_base

where W is the precision matrix of base-forecast errors.  When the optional
``hierarchicalforecast`` package is installed, its native reconcilers are
used directly; otherwise, the closed-form MinTrace is computed via NumPy.

References
----------
Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    Optimal forecast reconciliation using a unifying framework for the MinT
    family of reconciliation methods.  *JASA*, 114(526), 804-819.
Hyndman, R. J., et al. (2011).  Optimal combination forecasts for hierarchical
    time series.  *Computational Statistics & Data Analysis*, 55(9), 2579-2589.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Public constants matching glowcast.yaml / segment_genes.py ───────────────

COUNTRIES: list[str] = ["US", "DE", "UK", "JP", "IN"]
FC_TO_COUNTRY: dict[str, str] = {
    "FC_Phoenix":    "US",
    "FC_Miami":      "US",
    "FC_Seattle":    "US",
    "FC_Dallas":     "US",
    "FC_Berlin":     "DE",
    "FC_London":     "UK",
    "FC_Manchester": "UK",
    "FC_Tokyo":      "JP",
    "FC_Osaka":      "JP",
    "FC_Mumbai":     "IN",
    "FC_Delhi":      "IN",
    "FC_Bangalore":  "IN",
}
NATIONAL_LABEL: str = "National"

ReconcileMethod = Literal[
    "ols",
    "wls_struct",
    "wls_var",
    "mint_cov",
    "mint_shrink",
]

# ── Hierarchy specification ───────────────────────────────────────────────────


def get_hierarchy_spec() -> dict:
    """Return the 4-level GlowCast hierarchy definition.

    The returned dict is self-describing and suitable for constructing a
    ``HierarchicalReconciler`` or for generating the summing matrix S.

    Returns
    -------
    dict with keys:
        levels : list[str]
            Ordered level names, from coarsest to finest.
        n_nodes : dict[str, int]
            Number of series at each level.
        country_fcs : dict[str, list[str]]
            Mapping of country code → list of FC IDs beneath it.
        n_skus : int
            Number of leaf SKU series (bottom level).
        total_series : int
            Total number of series across all levels (for S-matrix sizing).
        description : str
            Human-readable description of the hierarchy.

    Examples
    --------
    >>> spec = get_hierarchy_spec()
    >>> spec["levels"]
    ['National', 'Country', 'FC', 'SKU']
    >>> spec["n_nodes"]["Country"]
    5
    """
    country_fcs: dict[str, list[str]] = {}
    for fc, country in FC_TO_COUNTRY.items():
        country_fcs.setdefault(country, []).append(fc)

    n_fcs = len(FC_TO_COUNTRY)          # 12
    n_countries = len(COUNTRIES)        # 5
    n_skus = 5_000
    # Total series = National(1) + Country(5) + FC(12) + SKU-FC leaf pairs
    # In the GlowCast hierarchy every SKU is stocked at every FC, so there
    # are n_skus * n_fcs = 60 000 bottom-level time series.
    n_bottom = n_skus * n_fcs
    total_series = 1 + n_countries + n_fcs + n_bottom

    return {
        "levels": ["National", "Country", "FC", "SKU"],
        "n_nodes": {
            "National": 1,
            "Country": n_countries,
            "FC": n_fcs,
            "SKU": n_bottom,
        },
        "country_fcs": country_fcs,
        "n_skus": n_skus,
        "n_fcs": n_fcs,
        "n_countries": n_countries,
        "n_bottom": n_bottom,
        "total_series": total_series,
        "description": (
            f"4-level hierarchy: {NATIONAL_LABEL}(1) → "
            f"Country({n_countries}) → "
            f"FC({n_fcs}) → "
            f"SKU×FC({n_bottom:,})"
        ),
    }


# ── Summing matrix helpers ────────────────────────────────────────────────────


def _build_summing_matrix_from_tags(
    S_df: pd.DataFrame,
) -> np.ndarray:
    """Convert a tag DataFrame into a dense S matrix (n_all × n_bottom).

    Parameters
    ----------
    S_df : pd.DataFrame
        Rows are the bottom-level series (SKU×FC); index is ``unique_id``.
        Columns must include: ``sku_id``, ``fc_id``, ``country``,
        ``national``.  This matches the ``S_SCHEMA`` defined in
        ``app.data.contracts``.

    Returns
    -------
    np.ndarray, shape (n_all, n_bottom), dtype float64
        Dense summing matrix.  Bottom rows are an identity sub-matrix; upper
        rows aggregate bottom series into FC, Country, and National nodes.
    """
    n_bottom = len(S_df)

    # Ordered label lists for each upper level
    country_labels = list(dict.fromkeys(S_df["country"]))   # preserve order
    fc_labels = list(dict.fromkeys(S_df["fc_id"]))
    bottom_labels = list(S_df.index)

    n_all = 1 + len(country_labels) + len(fc_labels) + n_bottom
    S = np.zeros((n_all, n_bottom), dtype=np.float64)

    # Row offsets
    row_nat = 0
    row_country = 1
    row_fc = row_country + len(country_labels)
    row_sku = row_fc + len(fc_labels)

    country_idx = {c: i for i, c in enumerate(country_labels)}
    fc_idx = {fc: i for i, fc in enumerate(fc_labels)}

    for col, uid in enumerate(bottom_labels):
        row = S_df.loc[uid]
        # National aggregation
        S[row_nat, col] = 1.0
        # Country aggregation
        S[row_country + country_idx[row["country"]], col] = 1.0
        # FC aggregation
        S[row_fc + fc_idx[row["fc_id"]], col] = 1.0
        # Bottom (identity)
        S[row_sku + col, col] = 1.0

    return S


# ── MinTrace closed-form ──────────────────────────────────────────────────────


def _mint_projection(
    S: np.ndarray,
    W: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    """Apply MinTrace reconciliation.

    Computes the projection matrix P and reconciled forecasts:

        P  = S (S'WS)^{-1} S'W          ... (n_bottom × n_all)
        ŷ  = S P ŷ_base                  ... (n_all × h)

    Parameters
    ----------
    S : np.ndarray, shape (n_all, n_bottom)
    W : np.ndarray, shape (n_all, n_all)  — precision / weight matrix
    base : np.ndarray, shape (n_all,) or (n_all, h)

    Returns
    -------
    np.ndarray, shape matching ``base``
    """
    # W is typically a diagonal or dense covariance-based matrix.
    # For numerical stability work with float64 throughout.
    S = np.asarray(S, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)

    St = S.T                      # (n_bottom, n_all)
    StW = St @ W                  # (n_bottom, n_all)
    StWS = StW @ S                # (n_bottom, n_bottom)

    # Pseudo-inverse for numerical robustness
    StWS_inv = np.linalg.pinv(StWS)

    # Projection matrix: shape (n_bottom, n_all)
    P = StWS_inv @ StW

    # Reconciled bottom-level forecasts
    bottom_rec = P @ base         # (n_bottom,) or (n_bottom, h)

    # Map back to all levels
    reconciled = S @ bottom_rec   # (n_all,) or (n_all, h)
    return reconciled


def _shrinkage_covariance(residuals: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-style diagonal shrinkage covariance for MinT-Shrink.

    Shrinks the sample covariance toward a diagonal target:
        W_shrink = (1 - λ) Σ̂ + λ diag(Σ̂)

    The shrinkage coefficient λ is estimated analytically following
    Schäfer & Strimmer (2005).

    Parameters
    ----------
    residuals : np.ndarray, shape (T, n)
        In-sample forecast residuals.  T = time steps, n = series.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf(assume_centered=True)
        lw.fit(residuals)
        return lw.covariance_
    except ImportError:
        pass

    # Manual Ledoit-Wolf
    T, n = residuals.shape
    S_cov = (residuals.T @ residuals) / T  # sample covariance (centred)
    mu_target = np.trace(S_cov) / n        # diagonal target scale

    # Off-diagonal shrinkage numerator / denominator
    # Schäfer & Strimmer (2005) simplified estimator
    delta_sq = 0.0
    for t in range(T):
        r = residuals[t]
        outer = np.outer(r, r)
        delta_sq += np.sum((outer - S_cov) ** 2)
    delta_sq /= T**2

    gamma = np.sum(S_cov**2) - np.sum(np.diag(S_cov) ** 2)
    lam = min(1.0, max(0.0, delta_sq / gamma)) if gamma > 0 else 1.0

    target = mu_target * np.eye(n)
    return (1 - lam) * S_cov + lam * target


# ── Main class ────────────────────────────────────────────────────────────────


class HierarchicalReconciler:
    """MinTrace forecast reconciliation across the GlowCast 4-level hierarchy.

    The reconciler operates on Nixtla-style long DataFrames where every row
    is (unique_id, ds, y_hat).  It converts those to a wide matrix, builds
    or accepts the summing matrix S, then applies the requested MinTrace
    variant.

    Parameters
    ----------
    hierarchy_levels : list[str] | None
        Ordered level names (coarsest → finest).  Defaults to the standard
        GlowCast spec: ``["National", "Country", "FC", "SKU"]``.
    residuals : np.ndarray | None, shape (T, n_all)
        In-sample forecast residuals used to estimate the error covariance
        for ``wls_var``, ``mint_cov``, and ``mint_shrink`` methods.
        Required for those methods; ignored for ``ols`` and ``wls_struct``.

    Examples
    --------
    >>> rec = HierarchicalReconciler()
    >>> S = rec.build_summing_matrix(s_df)
    >>> reconciled = rec.reconcile(base_forecasts, method="mint_shrink")
    """

    def __init__(
        self,
        hierarchy_levels: list[str] | None = None,
        residuals: np.ndarray | None = None,
    ) -> None:
        self.hierarchy_levels: list[str] = hierarchy_levels or [
            "National", "Country", "FC", "SKU"
        ]
        self._residuals: np.ndarray | None = residuals
        self._S: np.ndarray | None = None           # cached summing matrix
        self._series_order: list[str] | None = None  # unique_id ordering

        if len(self.hierarchy_levels) != 4:
            raise ValueError(
                "GlowCast hierarchy must have exactly 4 levels "
                f"(got {len(self.hierarchy_levels)}): {self.hierarchy_levels}"
            )

    # ── Public interface ──────────────────────────────────────────────────

    def build_summing_matrix(self, S_df: pd.DataFrame) -> np.ndarray:
        """Build the summing matrix S from a tag DataFrame.

        The tag DataFrame associates every bottom-level series with its
        ancestor nodes at each hierarchy level.  It must conform to the
        ``S_SCHEMA`` contract defined in ``app.data.contracts``:

            Index  : unique_id   (str) — bottom-level series identifier
            Columns: sku_id, fc_id, country, national

        Parameters
        ----------
        S_df : pd.DataFrame
            Tag frame with one row per bottom-level (SKU × FC) series.

        Returns
        -------
        np.ndarray, shape (n_all, n_bottom), dtype float64
            Summing matrix; cached internally for subsequent ``reconcile``
            calls.

        Raises
        ------
        ValueError
            If required columns are missing from ``S_df``.
        """
        required_cols = {"sku_id", "fc_id", "country", "national"}
        missing = required_cols - set(S_df.columns)
        if missing:
            raise ValueError(
                f"S_df is missing required columns: {missing}. "
                f"Got: {list(S_df.columns)}"
            )
        if S_df.index.name != "unique_id":
            raise ValueError(
                "S_df index must be named 'unique_id'. "
                f"Got index name: '{S_df.index.name}'"
            )

        S = _build_summing_matrix_from_tags(S_df)
        self._S = S
        # Record bottom-level order for later alignment
        self._bottom_ids = list(S_df.index)
        logger.info(
            "Built summing matrix S: shape=%s  (n_all=%d, n_bottom=%d)",
            S.shape,
            S.shape[0],
            S.shape[1],
        )
        return S

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        method: ReconcileMethod = "mint_shrink",
        S: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Reconcile base forecasts so they are coherent across the hierarchy.

        Attempts to delegate to ``hierarchicalforecast`` if installed; falls
        back to a pure-NumPy closed-form MinTrace implementation.

        Parameters
        ----------
        base_forecasts : pd.DataFrame
            Nixtla-style long frame with columns
            ``[unique_id, ds, <model_name>]``.  All levels of the hierarchy
            must be present as rows.
        method : str
            Reconciliation method.  One of:

            * ``"ols"``         — MinTrace with W = I (ordinary least squares)
            * ``"wls_struct"``  — WLS with W = diag(S 1_n) (structural)
            * ``"wls_var"``     — WLS with W = diag(σ̂²) (residual variances)
            * ``"mint_cov"``    — Full sample covariance
            * ``"mint_shrink"`` — Ledoit-Wolf shrinkage covariance (default)

        S : np.ndarray | None
            Explicit summing matrix.  Uses cached S if not provided.

        Returns
        -------
        pd.DataFrame
            Same structure as ``base_forecasts`` with an added column
            ``y_rec`` holding reconciled point forecasts.

        Raises
        ------
        ValueError
            If no summing matrix is available (call ``build_summing_matrix``
            first or pass ``S`` explicitly).
        RuntimeError
            If ``wls_var`` / ``mint_cov`` / ``mint_shrink`` are requested
            but no residuals were supplied at construction time.
        """
        _S = S if S is not None else self._S
        if _S is None:
            raise ValueError(
                "Summing matrix S is not available.  Call "
                "build_summing_matrix(S_df) first or pass S= explicitly."
            )

        if method in ("wls_var", "mint_cov", "mint_shrink") and self._residuals is None:
            raise RuntimeError(
                f"Method '{method}' requires in-sample residuals.  "
                "Pass residuals= to HierarchicalReconciler.__init__."
            )

        # Try the hierarchicalforecast library first
        try:
            result = self._reconcile_hf(base_forecasts, _S, method)
            logger.debug("Reconciliation via hierarchicalforecast (%s)", method)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "hierarchicalforecast unavailable or failed (%s); "
                "falling back to manual MinTrace: %s",
                method,
                exc,
            )

        return self._reconcile_manual(base_forecasts, _S, method)

    # ── hierarchicalforecast delegation ──────────────────────────────────

    def _reconcile_hf(
        self,
        base_forecasts: pd.DataFrame,
        S: np.ndarray,
        method: str,
    ) -> pd.DataFrame:
        """Attempt reconciliation via the ``hierarchicalforecast`` library."""
        import hierarchicalforecast.methods as hf_methods  # noqa: PLC0415
        from hierarchicalforecast.core import HierarchicalReconciliation  # noqa: PLC0415

        _method_map = {
            "ols":         hf_methods.MinTrace(method="ols"),
            "wls_struct":  hf_methods.MinTrace(method="wls_struct"),
            "wls_var":     hf_methods.MinTrace(method="wls_var"),
            "mint_cov":    hf_methods.MinTrace(method="mint_cov"),
            "mint_shrink": hf_methods.MinTrace(method="mint_shrink"),
        }
        if method not in _method_map:
            raise ValueError(f"Unknown method '{method}' for hierarchicalforecast.")

        hrec = HierarchicalReconciliation(reconcilers=[_method_map[method]])

        # hierarchicalforecast expects Y_df and S_df in specific formats;
        # we call fit_reconcile which handles the internals.
        # Since the API varies by version, wrap in a try block.
        reconciled_wide = hrec.reconcile(
            Y_hat_df=base_forecasts,
            S=pd.DataFrame(S),
            tags={},  # caller-managed
        )
        return reconciled_wide  # library returns a ready DataFrame

    # ── Manual MinTrace ───────────────────────────────────────────────────

    def _reconcile_manual(
        self,
        base_forecasts: pd.DataFrame,
        S: np.ndarray,
        method: str,
    ) -> pd.DataFrame:
        """Pure-NumPy MinTrace reconciliation.

        Algorithm
        ---------
        For each forecast horizon step h:
            1. Extract base forecast vector ŷ ∈ ℝ^{n_all}
            2. Compute weight matrix W (method-dependent)
            3. Apply MinTrace projection:  ŷ_rec = S (S'WS)^{-1} S'W ŷ
            4. Clip to non-negative (demand is always ≥ 0)

        Parameters
        ----------
        base_forecasts : pd.DataFrame
            Must contain columns: ``unique_id``, ``ds``, and exactly one
            model column (the last non-id column).
        S : np.ndarray, shape (n_all, n_bottom)
        method : str

        Returns
        -------
        pd.DataFrame
            Input frame with column ``y_rec`` appended.
        """
        # Identify the model column (last column that is not unique_id / ds)
        id_cols = {"unique_id", "ds"}
        model_col = [c for c in base_forecasts.columns if c not in id_cols]
        if not model_col:
            raise ValueError(
                "base_forecasts has no model forecast column beyond "
                "'unique_id' and 'ds'."
            )
        model_col = model_col[-1]

        dates = base_forecasts["ds"].unique()
        series_order = list(base_forecasts["unique_id"].unique())
        n_all = S.shape[0]

        if len(series_order) != n_all:
            warnings.warn(
                f"base_forecasts has {len(series_order)} unique series but "
                f"S has {n_all} rows.  Results may be incorrect if orderings "
                "do not match.",
                stacklevel=3,
            )

        # Build wide matrix: rows=unique_id (ordered), cols=ds
        wide = (
            base_forecasts
            .pivot(index="unique_id", columns="ds", values=model_col)
            .reindex(series_order)
            .values  # shape (n_all, n_dates)
        )
        wide = np.nan_to_num(wide, nan=0.0)

        # Precision / weight matrix W
        W = self._build_W(S, method, n_all)

        # Reconcile every horizon column
        reconciled_wide = np.zeros_like(wide)
        for h_idx in range(wide.shape[1]):
            rec_col = _mint_projection(S, W, wide[:, h_idx])
            reconciled_wide[:, h_idx] = rec_col

        # Non-negativity constraint (demand cannot be negative)
        reconciled_wide = np.clip(reconciled_wide, a_min=0.0, a_max=None)

        # Melt back to long format and join
        rec_df = pd.DataFrame(
            reconciled_wide,
            index=series_order,
            columns=dates,
        )
        rec_long = (
            rec_df
            .reset_index()
            .rename(columns={"index": "unique_id"})
            .melt(id_vars="unique_id", var_name="ds", value_name="y_rec")
        )

        result = base_forecasts.merge(rec_long, on=["unique_id", "ds"], how="left")
        return result

    def _build_W(
        self,
        S: np.ndarray,
        method: str,
        n_all: int,
    ) -> np.ndarray:
        """Construct the weight matrix W for the chosen reconciliation method.

        Parameters
        ----------
        S : np.ndarray, shape (n_all, n_bottom)
        method : str
        n_all : int

        Returns
        -------
        np.ndarray, shape (n_all, n_all)
        """
        if method == "ols":
            # W = I  →  standard OLS projection
            return np.eye(n_all, dtype=np.float64)

        if method == "wls_struct":
            # W = diag(k_i)  where k_i = number of bottom series summed into i
            # This equals the row sums of S.
            k = S.sum(axis=1)  # (n_all,)
            k = np.where(k == 0, 1.0, k)  # guard against zeros
            return np.diag(1.0 / k)

        if method == "wls_var":
            # W = diag(1 / σ̂²_i)
            if self._residuals is None:
                raise RuntimeError("wls_var requires residuals.")
            resid = np.asarray(self._residuals, dtype=np.float64)
            variances = np.var(resid, axis=0, ddof=1)  # (n_all,)
            variances = np.where(variances < 1e-10, 1e-10, variances)
            return np.diag(1.0 / variances)

        if method == "mint_cov":
            # W = full sample covariance of residuals
            if self._residuals is None:
                raise RuntimeError("mint_cov requires residuals.")
            resid = np.asarray(self._residuals, dtype=np.float64)
            cov = np.cov(resid, rowvar=False)
            # Precision matrix via pseudo-inverse
            return np.linalg.pinv(cov)

        if method == "mint_shrink":
            # W = Ledoit-Wolf shrinkage covariance
            if self._residuals is None:
                raise RuntimeError("mint_shrink requires residuals.")
            resid = np.asarray(self._residuals, dtype=np.float64)
            cov_shrunk = _shrinkage_covariance(resid)
            return np.linalg.pinv(cov_shrunk)

        raise ValueError(
            f"Unknown reconciliation method: '{method}'. "
            f"Choose from: ols, wls_struct, wls_var, mint_cov, mint_shrink."
        )

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def S(self) -> np.ndarray | None:
        """Cached summing matrix (None if not yet built)."""
        return self._S

    @property
    def n_levels(self) -> int:
        """Number of hierarchy levels."""
        return len(self.hierarchy_levels)

    def __repr__(self) -> str:
        spec = get_hierarchy_spec()
        return (
            f"HierarchicalReconciler("
            f"levels={self.hierarchy_levels}, "
            f"S_built={self._S is not None}, "
            f"has_residuals={self._residuals is not None}, "
            f"hierarchy='{spec['description']}')"
        )
