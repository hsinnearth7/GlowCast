"""Forecast I/O contracts for GlowCast.

Defines typed dataclasses for model inputs and outputs so that every
forecasting component in the pipeline shares a common interface.

Schema alignment
----------------
- ``ForecastInput.Y_df``        follows Nixtla's long format: [unique_id, ds, y]
- ``ForecastInput.X_df``        follows Nixtla's exogenous format: [unique_id, ds, <features>]
- ``ForecastOutput.predictions`` columns: [ds, y_hat, y_lo, y_hi]  (no unique_id —
  that lives at the envelope level so the DataFrame stays narrow and
  easy to concatenate across series)

Usage
-----
    from app.forecasting.contracts import ForecastInput, ForecastOutput

    inp = ForecastInput(
        unique_id="SKU_1001__FC_US_EAST",
        Y_df=y_df,          # columns: unique_id, ds, y
        horizon=14,
    )

    out = ForecastOutput(
        unique_id=inp.unique_id,
        predictions=pred_df,    # columns: ds, y_hat, y_lo, y_hi
        model_name="lightgbm",
        elapsed_ms=123.4,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastInput:
    """All data needed by any ForecastModel to fit and predict.

    Parameters
    ----------
    unique_id:
        Series identifier, e.g. ``"SKU_1001__FC_US_EAST"``.
        Must be consistent with the ``unique_id`` column in ``Y_df``
        and ``X_df`` (if provided).
    Y_df:
        Target time series in Nixtla long format.
        Required columns: ``unique_id``, ``ds`` (datetime), ``y`` (float >= 0).
        Must be sorted by ``ds`` ascending, no missing dates expected by the
        model (callers should resample / zero-fill before passing).
    X_df:
        Optional exogenous features in Nixtla long format.
        Required columns: ``unique_id``, ``ds`` plus any feature columns.
        GlowCast features include: ``social_momentum``, ``temperature``,
        ``humidity``.  When provided, must cover both the training window
        **and** the forecast horizon (i.e., len(X_df) >= len(Y_df) + horizon).
    horizon:
        Number of future steps (days) to forecast. Must be >= 1.
    """

    unique_id: str
    Y_df: pd.DataFrame
    horizon: int
    X_df: pd.DataFrame | None = field(default=None)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.Y_df is None or len(self.Y_df) == 0:
            raise ValueError("Y_df must be a non-empty DataFrame")

        required_y_cols = {"unique_id", "ds", "y"}
        missing = required_y_cols - set(self.Y_df.columns)
        if missing:
            raise ValueError(f"Y_df is missing required columns: {missing}")

        if self.X_df is not None:
            required_x_cols = {"unique_id", "ds"}
            missing_x = required_x_cols - set(self.X_df.columns)
            if missing_x:
                raise ValueError(f"X_df is missing required columns: {missing_x}")

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return len(self.Y_df)

    @property
    def last_date(self) -> pd.Timestamp:
        """Last observed date in the training series."""
        return pd.Timestamp(self.Y_df["ds"].max())

    @property
    def future_dates(self) -> pd.DatetimeIndex:
        """DatetimeIndex of the ``horizon`` forecast dates (daily frequency)."""
        return pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=self.horizon,
            freq="D",
        )


@dataclass
class ForecastOutput:
    """Structured result returned by every ForecastModel.

    Parameters
    ----------
    unique_id:
        Series identifier, mirrors ``ForecastInput.unique_id``.
    predictions:
        DataFrame with exactly the following columns:

        - ``ds``    – forecast date (datetime64[ns])
        - ``y_hat`` – point forecast (float)
        - ``y_lo``  – lower prediction-interval bound (float, may be NaN
                      when the model does not produce intervals)
        - ``y_hi``  – upper prediction-interval bound (float, may be NaN)

        Must have ``horizon`` rows, ordered by ``ds`` ascending.
    model_name:
        Human-readable model identifier, e.g. ``"lightgbm"`` or
        ``"routing_ensemble[lightgbm]"``.
    elapsed_ms:
        Wall-clock time in milliseconds for fit + predict combined, used
        for latency monitoring in the MLOps layer.
    """

    unique_id: str
    predictions: pd.DataFrame
    model_name: str
    elapsed_ms: float

    def __post_init__(self) -> None:
        required_cols = {"ds", "y_hat", "y_lo", "y_hi"}
        if self.predictions is None:
            raise ValueError("predictions DataFrame cannot be None")
        missing = required_cols - set(self.predictions.columns)
        if missing:
            raise ValueError(f"predictions is missing required columns: {missing}")
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must be >= 0, got {self.elapsed_ms}")

    @property
    def horizon(self) -> int:
        """Number of forecasted steps."""
        return len(self.predictions)

    def to_flat_df(self) -> pd.DataFrame:
        """Return predictions with ``unique_id`` and ``model_name`` columns prepended.

        Useful for concatenating results across many series into a single
        result table that can be written to a database or evaluated.

        Returns
        -------
        pd.DataFrame
            Columns: ``unique_id``, ``model``, ``ds``, ``y_hat``, ``y_lo``, ``y_hi``.
        """
        df = self.predictions.copy()
        df.insert(0, "unique_id", self.unique_id)
        df.insert(1, "model", self.model_name)
        return df[["unique_id", "model", "ds", "y_hat", "y_lo", "y_hi"]]
