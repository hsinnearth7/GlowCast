"""GlowCast FeatureStore — dual-mode offline batch + online serving.

Design
------
The store follows the **AP > CP eventual consistency model** common in
production feature platforms (e.g. Feast, Tecton):

* **Offline store** (``materialize_offline``) is the *authoritative source of
  truth*.  It computes every feature deterministically from raw DataFrames
  using strict ``shift(1)`` guards to prevent target leakage, and is the only
  path used to build training datasets.

* **Online store** (``update_online`` / ``get_online_features``) is a
  best-effort, low-latency projection of the most recent row.  It is
  populated by the serving pipeline after each inference cycle.  Because
  writes go to both stores independently, **the two stores may be
  transiently inconsistent** (Availability > Consistency).  Callers must
  never use online features for model training.

Feature Groups
--------------
cost_features
    Lag 1/7/14/28 days, rolling mean/std over 7/28 days, day-of-week,
    month — all shifted by 1 day to prevent leakage.

commodity_features
    ``social_momentum_t3`` — commodity price momentum lagged 3 days to
    reflect the empirically observed T-3 leading indicator in the
    GlowCast synthetic data generator.

climate_features
    ``temperature``, ``humidity`` — daily weather covariates joined by
    (unique_id, ds) after extracting the plant region.

inventory_features
    ``days_to_expiry`` — days remaining until nearest batch expiry,
    computed from Fact_Purchase_Orders when provided.

Usage
-----
    fs = FeatureStore()
    features = fs.materialize_offline(Y_df, X_commodity=commodity_df, X_climate=climate_df)
    train_df  = fs.get_training_features(start_date="2023-01-01")

    # Serving path
    fs.update_online("SKU_0001__PLANT_SZ", {"lag_1": 42.0, ...})
    row = fs.get_online_features("SKU_0001__PLANT_SZ")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    # Lag windows (days)
    "lag_windows": [1, 7, 14, 28],
    # Rolling statistic windows (days)
    "rolling_windows": [7, 28],
    # Social momentum lag (T-3 leading indicator documented in data_generator.py)
    "social_lag_days": 3,
    # Online store TTL (seconds) — informational; eviction is caller's responsibility
    "online_ttl_seconds": 300,
    # Offline store TTL (hours) — informational; triggers a re-materialisation warning
    "offline_ttl_hours": 24,
}


class FeatureStore:
    """Dual-mode feature store for GlowCast cost analytics.

    Parameters
    ----------
    config:
        Optional override dictionary merged on top of ``_DEFAULT_CONFIG``.
        Accepted keys mirror ``monitoring.feature_store`` in glowcast.yaml.

    Attributes
    ----------
    _offline_store : pd.DataFrame or None
        Materialised feature matrix.  ``None`` until ``materialize_offline``
        is called.
    _online_store : dict[str, dict]
        Keyed by ``unique_id``; holds the latest feature snapshot for
        online serving.
    _last_materialised : datetime or None
        UTC timestamp of the most recent ``materialize_offline`` call.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self._offline_store: pd.DataFrame | None = None
        self._online_store: dict[str, dict[str, Any]] = {}
        self._last_materialised: datetime | None = None

    # ------------------------------------------------------------------
    # Public — offline path
    # ------------------------------------------------------------------

    def materialize_offline(
        self,
        Y_df: pd.DataFrame,
        X_social: pd.DataFrame | None = None,
        X_climate: pd.DataFrame | None = None,
        X_inventory: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute all features and persist them in the offline store.

        This is the **single source of truth** for training.  All features
        are computed with a ``shift(1)`` guard so no information from the
        target day leaks into the feature vector.

        Parameters
        ----------
        Y_df:
            Nixtla-format demand DataFrame with columns
            ``[unique_id, ds, y]``.
        X_social:
            Optional commodity momentum DataFrame.  Expected columns:
            ``[ds, net_momentum]``.  If the DataFrame contains a
            ``commodity`` column it will be ignored (momentum is already
            aggregated per signal date in Fact_Commodity_Prices).
        X_climate:
            Optional climate DataFrame.  Expected columns:
            ``[ds, region, temperature_celsius, humidity_pct]``.  Joined
            to Y_df by extracting the region from ``unique_id`` (format
            ``SKU_XXXX__FC_<country>_<region>``).
        X_inventory:
            Optional inventory DataFrame (Fact_Purchase_Orders format)
            with columns ``[snapshot_date, sku_id, plant_id, expiry_date]``.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ``unique_id`` and ``ds`` as the first two
            columns, followed by all computed features.  Rows with all-NaN
            features (i.e. the first ``max_lag`` rows per series) are
            retained but flagged — callers should drop them before training.

        Notes
        -----
        AP > CP consistency note: this method overwrites ``_offline_store``
        atomically.  If the caller simultaneously serves online predictions,
        the online store retains its previous snapshot until explicitly
        updated via ``update_online``.
        """
        Y_df = self._validate_Y(Y_df)

        # ---- Demand features (core) ------------------------------------
        features = self._compute_demand_features(Y_df)

        # ---- Social features -------------------------------------------
        if X_social is not None:
            features = self._join_social_features(features, X_social)
        else:
            features["social_momentum_t3"] = np.nan

        # ---- Climate features ------------------------------------------
        if X_climate is not None:
            features = self._join_climate_features(features, X_climate)
        else:
            features["temperature"] = np.nan
            features["humidity"] = np.nan

        # ---- Inventory features ----------------------------------------
        if X_inventory is not None:
            features = self._join_inventory_features(features, X_inventory)
        else:
            features["days_to_expiry"] = np.nan

        self._offline_store = features.reset_index(drop=True)
        self._last_materialised = datetime.utcnow()

        logger.info(
            "FeatureStore materialised offline store: %d rows, %d features, timestamp=%s",
            len(self._offline_store),
            self._offline_store.shape[1] - 2,  # exclude unique_id, ds
            self._last_materialised.isoformat(),
        )
        return self._offline_store.copy()

    def get_training_features(
        self,
        unique_ids: list[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Retrieve a filtered slice of the offline feature store.

        Parameters
        ----------
        unique_ids:
            Optional list of series identifiers to include.  If ``None``
            all series are returned.
        start_date:
            Inclusive lower bound on the ``ds`` column.
        end_date:
            Inclusive upper bound on the ``ds`` column.

        Returns
        -------
        pd.DataFrame
            Filtered copy of the offline feature matrix.

        Raises
        ------
        RuntimeError
            If ``materialize_offline`` has not been called yet.
        """
        if self._offline_store is None:
            raise RuntimeError(
                "Offline store is empty. Call materialize_offline() first."
            )

        df = self._offline_store.copy()

        if unique_ids is not None:
            df = df[df["unique_id"].isin(unique_ids)]

        if start_date is not None:
            df = df[df["ds"] >= pd.Timestamp(start_date)]

        if end_date is not None:
            df = df[df["ds"] <= pd.Timestamp(end_date)]

        logger.debug("get_training_features returned %d rows.", len(df))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public — online path
    # ------------------------------------------------------------------

    def update_online(self, unique_id: str, features: dict[str, Any]) -> None:
        """Write a single-row feature snapshot to the online store.

        This is a point-in-time update: the dictionary completely replaces
        any previously stored snapshot for ``unique_id``.  It is the
        caller's responsibility to ensure the snapshot is consistent with
        the latest inference cycle.

        Parameters
        ----------
        unique_id:
            Series identifier (e.g. ``"SKU_0001__FC_US_EAST"``).
        features:
            Mapping of feature name → scalar value.  Values may be
            ``float``, ``int``, or ``None`` (representing missing).

        Notes
        -----
        AP > CP consistency note: the online store is updated independently
        of the offline store.  A brief window of inconsistency exists
        between the moment the offline store is re-materialised and the
        moment all ``update_online`` calls complete.  This is expected and
        intentional; the offline store remains the authoritative training
        source throughout.
        """
        self._online_store[unique_id] = {
            **features,
            "_updated_at": datetime.utcnow().isoformat(),
        }
        logger.debug("update_online: %s — %d features written.", unique_id, len(features))

    def get_online_features(self, unique_id: str) -> dict[str, Any] | None:
        """Retrieve the latest online feature snapshot for a single series.

        Parameters
        ----------
        unique_id:
            Series identifier.

        Returns
        -------
        dict or None
            Feature snapshot dict, or ``None`` if no snapshot exists.
        """
        snapshot = self._online_store.get(unique_id)
        if snapshot is None:
            logger.debug("get_online_features: cache miss for %s.", unique_id)
        return snapshot

    # ------------------------------------------------------------------
    # Private — feature computation
    # ------------------------------------------------------------------

    def _compute_demand_features(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-series demand features per series.

        All features are computed **group-wise** (per ``unique_id``) and
        use ``shift(1)`` so that only information available *before* the
        target date is included.  This is a strict no-leakage guarantee.

        Features produced
        -----------------
        lag_1, lag_7, lag_14, lag_28
            Target value lagged by 1, 7, 14, 28 days.
        rolling_mean_7, rolling_mean_28
            Rolling arithmetic mean (window 7 / 28 days) of the shifted
            series.
        rolling_std_7, rolling_std_28
            Rolling standard deviation (window 7 / 28 days) of the
            shifted series.
        day_of_week
            Integer 0 (Monday) … 6 (Sunday) derived from ``ds``.
        month
            Integer 1 … 12 derived from ``ds``.

        Parameters
        ----------
        Y_df:
            Validated Nixtla-format DataFrame ``[unique_id, ds, y]``.

        Returns
        -------
        pd.DataFrame
            Original columns plus all demand features.
        """
        lag_windows: list[int] = self._config["lag_windows"]
        rolling_windows: list[int] = self._config["rolling_windows"]

        records: list[pd.DataFrame] = []

        for _uid, grp in Y_df.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds").copy()

            # Shift y by 1 day — the fundamental leakage guard.
            y_shifted = grp["y"].shift(1)

            # ---- Lag features ------------------------------------------
            for lag in lag_windows:
                grp[f"lag_{lag}"] = grp["y"].shift(lag)

            # ---- Rolling features (computed on the shifted series) ------
            for window in rolling_windows:
                grp[f"rolling_mean_{window}"] = (
                    y_shifted.rolling(window=window, min_periods=1).mean()
                )
                grp[f"rolling_std_{window}"] = (
                    y_shifted.rolling(window=window, min_periods=2).std()
                )

            # ---- Calendar features (no leakage risk) -------------------
            grp["day_of_week"] = grp["ds"].dt.dayofweek
            grp["month"] = grp["ds"].dt.month

            records.append(grp)

        return pd.concat(records, ignore_index=True)

    def _join_social_features(
        self, features: pd.DataFrame, X_social: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach social momentum with a T-3 lag to prevent leakage.

        The GlowCast data generator documents a T-3 leading indicator
        relationship between commodity price signals and cost changes.  We honour
        this by lagging ``net_momentum`` by ``social_lag_days`` (default 3)
        when joining to the feature matrix.

        Parameters
        ----------
        features:
            Current feature matrix (output of ``_compute_demand_features``).
        X_social:
            Commodity momentum DataFrame.  Must contain ``ds`` and
            ``net_momentum`` columns.  Optionally contains ``concern``.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ``social_momentum_t3`` column appended.
        """
        lag = self._config["social_lag_days"]

        social = X_social[["ds", "net_momentum"]].copy()
        social["ds"] = pd.to_datetime(social["ds"])

        # Aggregate to daily granularity if multiple rows per date
        social = social.groupby("ds", as_index=False)["net_momentum"].mean()

        # Apply the T-3 lag: shift momentum forward by `lag` days
        social = social.sort_values("ds")
        social["social_momentum_t3"] = social["net_momentum"].shift(lag)
        social = social[["ds", "social_momentum_t3"]]

        features = features.merge(social, on="ds", how="left")
        logger.debug("Joined social features; non-null rows: %d", features["social_momentum_t3"].notna().sum())
        return features

    def _join_climate_features(
        self, features: pd.DataFrame, X_climate: pd.DataFrame
    ) -> pd.DataFrame:
        """Join temperature and humidity covariates to the feature matrix.

        The join key is ``(ds, region)``.  The region is extracted from
        ``unique_id`` using the convention ``SKU_XXXX__FC_<country>_<region>``.

        Parameters
        ----------
        features:
            Current feature matrix.
        X_climate:
            Climate DataFrame with columns
            ``[ds, region, temperature_celsius, humidity_pct]``.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ``temperature`` and ``humidity`` columns.
        """
        climate = X_climate[["ds", "region", "temperature_celsius", "humidity_pct"]].copy()
        climate["ds"] = pd.to_datetime(climate["ds"])
        climate = climate.rename(
            columns={"temperature_celsius": "temperature", "humidity_pct": "humidity"}
        )

        # Extract region from unique_id: "SKU_0001__FC_US_EAST" → "US_EAST"
        features["_region"] = (
            features["unique_id"]
            .str.extract(r"FC_([A-Z]{2}_[A-Z_]+)$", expand=False)
            .fillna("UNKNOWN")
        )

        features = features.merge(
            climate, left_on=["ds", "_region"], right_on=["ds", "region"], how="left"
        )
        features = features.drop(columns=["_region", "region"], errors="ignore")
        logger.debug(
            "Joined climate features; temperature non-null: %d",
            features["temperature"].notna().sum(),
        )
        return features

    def _join_inventory_features(
        self, features: pd.DataFrame, X_inventory: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute days_to_expiry from the nearest expiring batch.

        For each (sku_id, fc_id, snapshot_date) the method finds the
        minimum ``expiry_date`` across all open batches, then computes
        ``days_to_expiry = expiry_date - snapshot_date``.

        Parameters
        ----------
        features:
            Current feature matrix.
        X_inventory:
            Purchase-order-format DataFrame with columns
            ``[snapshot_date, sku_id, plant_id, expiry_date]``.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ``days_to_expiry`` column (float, NaN when
            no inventory record exists).
        """
        inv = X_inventory[["snapshot_date", "sku_id", "fc_id", "expiry_date"]].copy()
        inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
        inv["expiry_date"] = pd.to_datetime(inv["expiry_date"])

        nearest = (
            inv.groupby(["snapshot_date", "sku_id", "fc_id"], as_index=False)["expiry_date"]
            .min()
        )
        nearest["days_to_expiry"] = (
            (nearest["expiry_date"] - nearest["snapshot_date"]).dt.days.astype(float)
        )

        # Extract sku_id and fc_id from unique_id
        features["_sku"] = features["unique_id"].str.extract(r"^(SKU_\d+)", expand=False)
        features["_fc"] = features["unique_id"].str.extract(r"__(FC_\S+)$", expand=False)

        features = features.merge(
            nearest[["snapshot_date", "sku_id", "fc_id", "days_to_expiry"]],
            left_on=["ds", "_sku", "_fc"],
            right_on=["snapshot_date", "sku_id", "fc_id"],
            how="left",
        )
        features = features.drop(
            columns=["_sku", "_fc", "snapshot_date", "sku_id", "fc_id"], errors="ignore"
        )
        return features

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_Y(Y_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and coerce the Nixtla Y DataFrame.

        Parameters
        ----------
        Y_df:
            Input DataFrame.  Must contain ``unique_id``, ``ds``, ``y``.

        Returns
        -------
        pd.DataFrame
            Coerced copy with ``ds`` as ``datetime64[ns]`` and ``y`` as
            ``float64``.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required = {"unique_id", "ds", "y"}
        missing = required - set(Y_df.columns)
        if missing:
            raise ValueError(f"Y_df is missing required columns: {missing}")

        Y_df = Y_df.copy()
        Y_df["ds"] = pd.to_datetime(Y_df["ds"])
        Y_df["y"] = Y_df["y"].astype(float)
        return Y_df
