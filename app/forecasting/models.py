"""GlowCast forecasting models — ABC + Strategy + Factory pattern.

Follows the ChainInsight ForecastModel interface convention:
- ``ForecastModel``         Abstract base class (ABC)
- Concrete models           NaiveMovingAverage, SARIMAXForecaster,
                            XGBoostForecaster, LightGBMForecaster,
                            ChronosForecaster, RoutingEnsemble
- ``ForecastModelFactory``  Registry-based factory; reads hyperparameters
                            from ``configs/glowcast.yaml`` via
                            ``app.settings.get_model_config()``.

Optional-dependency strategy
-----------------------------
Heavy libraries (statsmodels, xgboost, lightgbm, chronos) are imported
inside try/except blocks so that the core module is always importable.
Each model that cannot find its backend falls back to a simpler strategy
and logs a warning via the standard ``logging`` module.

Feature engineering (XGBoost / LightGBM)
-----------------------------------------
Lag features   : lag_1, lag_7, lag_14, lag_28
Rolling stats  : rolling_mean_7, rolling_std_7, rolling_mean_28, rolling_std_28
Calendar       : day_of_week, month, day_of_year
GlowCast-spec  : social_momentum, temperature, humidity  (when in X_train)
All lagged by one step via ``.shift(1)`` to prevent target leakage.

Usage
-----
    from app.forecasting.models import ForecastModelFactory

    factory  = ForecastModelFactory()
    model    = factory.create("lightgbm")
    fitted   = model.fit(Y_train, X_train)
    pred_df  = fitted.predict(h=14, X_future=X_future)
    # pred_df columns: ds, y_hat, y_lo, y_hi
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.settings import get_model_config

logger = logging.getLogger(__name__)

# ── Optional heavy-dependency imports ────────────────────────────────────────

try:
    import statsmodels.api as sm  # noqa: F401
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels not installed — SARIMAXForecaster will fall back to a "
        "simple Ridge AR model. Install via: pip install statsmodels"
    )

try:
    import xgboost as xgb  # noqa: F401
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost not installed — XGBoostForecaster will fall back to Ridge "
        "regression. Install via: pip install xgboost"
    )

try:
    import lightgbm as lgb  # noqa: F401
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    logger.warning(
        "lightgbm not installed — LightGBMForecaster will fall back to Ridge "
        "regression. Install via: pip install lightgbm"
    )

try:
    from chronos import ChronosPipeline  # noqa: F401
    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False
    logger.warning(
        "chronos-forecasting not installed — ChronosForecaster will fall back "
        "to NaiveMovingAverage. Install via: pip install chronos-forecasting"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_prediction_interval(
    y_hat: np.ndarray,
    residual_std: float,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric Gaussian prediction interval.

    Parameters
    ----------
    y_hat:
        Point forecast array.
    residual_std:
        In-sample residual standard deviation used as a noise proxy.
        Must be >= 0; if 0 a small epsilon is used to avoid degenerate bounds.
    coverage:
        Nominal coverage, default 0.90.

    Returns
    -------
    y_lo, y_hi: np.ndarray pair clipped to [0, inf).
    """
    from scipy import stats  # local import — scipy is a core dep

    alpha = 1.0 - coverage
    z = float(stats.norm.ppf(1 - alpha / 2))
    std = max(residual_std, 1e-6)
    horizon = len(y_hat)
    # Uncertainty widens with forecast horizon (√h scaling)
    horizon_factor = np.sqrt(np.arange(1, horizon + 1))
    margin = z * std * horizon_factor
    y_lo = np.clip(y_hat - margin, 0, None)
    y_hi = y_hat + margin
    return y_lo, y_hi


def _build_prediction_df(
    future_dates: pd.DatetimeIndex,
    y_hat: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
) -> pd.DataFrame:
    """Package forecast arrays into the standard output DataFrame."""
    return pd.DataFrame(
        {
            "ds": future_dates,
            "y_hat": y_hat.astype(float),
            "y_lo": y_lo.astype(float),
            "y_hi": y_hi.astype(float),
        }
    )


def _future_dates_from_series(y_series: pd.Series, h: int) -> pd.DatetimeIndex:
    """Infer forecast dates from a sorted datetime index."""
    last = pd.Timestamp(y_series.index[-1])
    return pd.date_range(start=last + pd.Timedelta(days=1), periods=h, freq="D")


# ── Feature engineering ────────────────────────────────────────────────────────

# GlowCast-specific exogenous column names (sourced from Fact_Social_Signals
# and Dim_Weather, joined before being passed as X_train / X_future).
_GLOWCAST_EXOG_COLS = ("social_momentum", "temperature", "humidity")

_LAG_DAYS = (1, 7, 14, 28)
_ROLLING_WINDOWS = (7, 28)


def _engineer_features(
    y: pd.Series,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for tree-based models.

    All features are shifted by 1 day relative to the target to prevent
    look-ahead (data leakage).  The first ``max(_LAG_DAYS)`` rows will
    contain NaN and must be dropped before training.

    Parameters
    ----------
    y:
        Target series indexed by datetime (daily frequency, sorted ascending).
    X:
        Optional exogenous DataFrame indexed by the same datetime index.
        GlowCast columns ``social_momentum``, ``temperature``, ``humidity``
        are included when present.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same index as ``y``, NaNs in early rows.
    """
    df = pd.DataFrame(index=y.index)

    # Lag features — shift(1) means "yesterday's value at today's row"
    for lag in _LAG_DAYS:
        df[f"lag_{lag}"] = y.shift(lag)

    # Rolling statistics — computed on already-shifted series so no leakage
    y_shifted = y.shift(1)
    for window in _ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = y_shifted.rolling(window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = y_shifted.rolling(window, min_periods=1).std().fillna(0)

    # Calendar features
    df["day_of_week"] = y.index.dayofweek.astype(np.int8)
    df["month"] = y.index.month.astype(np.int8)
    df["day_of_year"] = y.index.day_of_year.astype(np.int16)

    # GlowCast-specific exogenous columns
    if X is not None:
        for col in _GLOWCAST_EXOG_COLS:
            if col in X.columns:
                # Re-index onto y's index to handle misaligned joins, then shift
                aligned = X[col].reindex(y.index)
                df[col] = aligned.shift(1)

    return df


def _engineer_features_future(
    y_train: pd.Series,
    h: int,
    X_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a feature matrix for the forecast horizon using recursive lag filling.

    Because true future values of ``y`` are unknown, lag features for steps
    beyond ``lag_1`` are filled iteratively using the most recent available
    observation (a common production approximation for multi-step direct
    forecasting).

    Parameters
    ----------
    y_train:
        Full training series (datetime-indexed, sorted ascending).
    h:
        Forecast horizon in days.
    X_future:
        Optional exogenous DataFrame covering exactly the ``h`` future dates.
        GlowCast columns are included when present.

    Returns
    -------
    pd.DataFrame
        Feature matrix of shape (h, n_features).
    """
    last_date = pd.Timestamp(y_train.index[-1])
    future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq="D")
    rows: list[dict[str, Any]] = []

    # Extend the known series with a placeholder of NaN for future steps
    # We'll fill lags using the tail of the training series.
    extended = y_train.copy()

    for step, date in enumerate(future_idx):
        row: dict[str, Any] = {}

        # Lag features: use the last `lag` values from the extended series
        for lag in _LAG_DAYS:
            idx = -(lag - step) if lag > step else None
            if idx is not None and abs(idx) <= len(extended):
                row[f"lag_{lag}"] = float(extended.iloc[idx - 1]) if idx < 0 else np.nan
            else:
                row[f"lag_{lag}"] = float(extended.iloc[-(lag - step)])

        # Rolling stats — approximate from tail of training series
        tail_7 = extended.iloc[-7:]
        tail_28 = extended.iloc[-28:]
        row["rolling_mean_7"] = float(tail_7.mean()) if len(tail_7) > 0 else 0.0
        row["rolling_std_7"] = float(tail_7.std(ddof=0)) if len(tail_7) > 1 else 0.0
        row["rolling_mean_28"] = float(tail_28.mean()) if len(tail_28) > 0 else 0.0
        row["rolling_std_28"] = float(tail_28.std(ddof=0)) if len(tail_28) > 1 else 0.0

        # Calendar
        row["day_of_week"] = date.dayofweek
        row["month"] = date.month
        row["day_of_year"] = date.day_of_year

        # GlowCast exogenous — future values are known (social schedules, weather forecasts)
        if X_future is not None:
            for col in _GLOWCAST_EXOG_COLS:
                if col in X_future.columns:
                    try:
                        row[col] = float(X_future.loc[date, col])
                    except KeyError:
                        row[col] = 0.0

        rows.append(row)

    return pd.DataFrame(rows, index=future_idx)


# ── Abstract base class ────────────────────────────────────────────────────────


class ForecastModel(ABC):
    """Abstract base class for all GlowCast forecasting models.

    Implements the *Strategy* design pattern: each concrete subclass encapsulates
    a specific forecasting algorithm and can be swapped transparently through
    the ``ForecastModelFactory``.

    Contract
    --------
    ``fit`` must return ``self`` to allow method chaining::

        preds = model.fit(Y_train, X_train).predict(h=14, X_future=X_future)

    ``predict`` must return a DataFrame with columns:
        ``ds`` (datetime64[ns]), ``y_hat`` (float), ``y_lo`` (float), ``y_hi`` (float)

    Both ``X_train`` and ``X_future`` are optional; models that do not
    consume exogenous features must accept and silently ignore them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this model (used as registry key)."""

    @abstractmethod
    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "ForecastModel":
        """Fit the model on historical data.

        Parameters
        ----------
        Y_train:
            Target DataFrame in Nixtla long format:
            columns ``unique_id``, ``ds``, ``y``.
            Callers are responsible for filtering to a single series before
            calling fit.
        X_train:
            Optional exogenous DataFrame aligned with ``Y_train``.
            Columns ``unique_id``, ``ds`` plus feature columns.

        Returns
        -------
        self
            The fitted model instance (enables chaining).
        """

    @abstractmethod
    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate a point forecast + prediction interval for the next ``h`` steps.

        Parameters
        ----------
        h:
            Forecast horizon (number of future daily steps). Must be >= 1.
        X_future:
            Optional exogenous DataFrame for the forecast horizon.
            Must have at least ``h`` rows aligned to the future dates.

        Returns
        -------
        pd.DataFrame
            Columns: ``ds``, ``y_hat``, ``y_lo``, ``y_hi``.
            Shape: (h, 4), ordered by ``ds`` ascending.
        """

    # ── Convenience helpers available to all subclasses ──────────────────────

    @staticmethod
    def _extract_y_series(Y_df: pd.DataFrame) -> pd.Series:
        """Extract a datetime-indexed Series from a Nixtla-format DataFrame.

        Sorts by ``ds``, converts ``ds`` to datetime, and sets it as index.

        Parameters
        ----------
        Y_df:
            DataFrame with at minimum columns ``ds`` and ``y``.

        Returns
        -------
        pd.Series
            Float series indexed by ``pd.DatetimeIndex``, sorted ascending.
        """
        df = Y_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").set_index("ds")
        return df["y"].astype(float)

    @staticmethod
    def _extract_x_aligned(
        X_df: pd.DataFrame,
        y_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Extract exogenous features aligned to a given datetime index.

        Parameters
        ----------
        X_df:
            DataFrame with at minimum column ``ds``.
        y_index:
            Target datetime index to align to.

        Returns
        -------
        pd.DataFrame
            Feature columns indexed by ``y_index``, NaN where data is missing.
        """
        df = X_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").set_index("ds")
        feature_cols = [c for c in df.columns if c != "unique_id"]
        return df[feature_cols].reindex(y_index)


# ── Concrete model 1: NaiveMovingAverage ────────────────────────────────────


class NaiveMovingAverage(ForecastModel):
    """Rolling window moving average baseline.

    The point forecast for each horizon step is the mean of the last
    ``window`` observations.  Prediction intervals are derived from the
    rolling standard deviation scaled by a Gaussian z-score with √h
    horizon uncertainty expansion.

    Parameters
    ----------
    window:
        Look-back window in days. Defaults to 30 (matches ``naive_ma30``
        config block in ``glowcast.yaml``).
    coverage:
        Nominal prediction-interval coverage. Default 0.90.
    """

    def __init__(self, window: int = 30, coverage: float = 0.90) -> None:
        self._window = window
        self._coverage = coverage
        self._fitted_mean: float | None = None
        self._fitted_std: float | None = None
        self._last_dates: pd.DatetimeIndex | None = None

    @property
    def name(self) -> str:
        return "naive_ma30"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "NaiveMovingAverage":
        y = self._extract_y_series(Y_train)
        tail = y.iloc[-self._window :]
        self._fitted_mean = float(tail.mean())
        self._fitted_std = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
        self._last_dates = y.index
        logger.debug(
            "NaiveMovingAverage fitted: window=%d, mean=%.3f, std=%.3f",
            self._window,
            self._fitted_mean,
            self._fitted_std,
        )
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._fitted_mean is None or self._last_dates is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        future_dates = _future_dates_from_series(
            pd.Series(index=self._last_dates, dtype=float), h
        )
        y_hat = np.full(h, self._fitted_mean)
        y_lo, y_hi = _make_prediction_interval(y_hat, self._fitted_std, self._coverage)
        return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)


# ── Concrete model 2: SARIMAXForecaster ──────────────────────────────────────


class SARIMAXForecaster(ForecastModel):
    """SARIMAX forecaster wrapping ``statsmodels.tsa.statespace.SARIMAX``.

    Hyperparameters are sourced from the ``model.sarimax`` block in
    ``glowcast.yaml``.  When statsmodels is unavailable the model falls back
    to a simple Ridge auto-regression with lags [1, 7, 14].

    Parameters
    ----------
    order:
        ARIMA (p, d, q) order tuple. Default (1, 1, 1).
    seasonal_order:
        Seasonal (P, D, Q, m) order tuple. Default (1, 1, 1, 7) for weekly.
    enforce_stationarity:
        Passed to ``SARIMAX``. Default False.
    enforce_invertibility:
        Passed to ``SARIMAX``. Default False.
    coverage:
        Prediction-interval coverage. Default 0.90.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        coverage: float = 0.90,
    ) -> None:
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._enforce_stationarity = enforce_stationarity
        self._enforce_invertibility = enforce_invertibility
        self._coverage = coverage

        # Populated by fit()
        self._result: Any = None  # SARIMAXResultsWrapper or None
        self._fallback_model: Ridge | None = None
        self._fallback_lags: int = 14
        self._y_train: pd.Series | None = None
        self._using_fallback: bool = not _STATSMODELS_AVAILABLE

    @property
    def name(self) -> str:
        return "sarimax"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "SARIMAXForecaster":
        y = self._extract_y_series(Y_train)
        self._y_train = y

        exog = None
        if X_train is not None and _STATSMODELS_AVAILABLE:
            x_aligned = self._extract_x_aligned(X_train, y.index)
            glowcast_cols = [c for c in _GLOWCAST_EXOG_COLS if c in x_aligned.columns]
            if glowcast_cols:
                exog = x_aligned[glowcast_cols].values

        if _STATSMODELS_AVAILABLE:
            try:
                import statsmodels.api as sm

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sarimax = sm.tsa.SARIMAX(
                        y.values,
                        exog=exog,
                        order=self._order,
                        seasonal_order=self._seasonal_order,
                        enforce_stationarity=self._enforce_stationarity,
                        enforce_invertibility=self._enforce_invertibility,
                    )
                    self._result = sarimax.fit(disp=False)
                logger.debug("SARIMAXForecaster fitted via statsmodels")
                self._using_fallback = False
                return self
            except Exception as exc:
                logger.warning(
                    "SARIMAX fit failed (%s); falling back to Ridge AR", exc
                )
                self._using_fallback = True

        # Fallback: Ridge AR with lags [1, 7, 14]
        self._using_fallback = True
        lags = [1, 7, 14]
        max_lag = max(lags)
        if len(y) <= max_lag:
            lags = [1]
            max_lag = 1

        X_ar = np.column_stack([y.values[max_lag - lag : len(y) - lag] for lag in lags])
        y_ar = y.values[max_lag:]
        self._fallback_model = Ridge(alpha=1.0).fit(X_ar, y_ar)
        self._fallback_lags = max_lag
        logger.debug("SARIMAXForecaster fitted via Ridge AR fallback")
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._y_train is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        future_dates = _future_dates_from_series(self._y_train, h)

        if not self._using_fallback and self._result is not None:
            # statsmodels path
            exog_fc = None
            if X_future is not None:
                x_fc = self._extract_x_aligned(X_future, future_dates)
                glowcast_cols = [c for c in _GLOWCAST_EXOG_COLS if c in x_fc.columns]
                if glowcast_cols:
                    exog_fc = x_fc[glowcast_cols].fillna(0).values

            alpha = 1.0 - self._coverage
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = self._result.get_forecast(steps=h, exog=exog_fc)
                    y_hat = np.clip(fc.predicted_mean, 0, None)
                    ci = fc.conf_int(alpha=alpha)
                    y_lo = np.clip(ci.iloc[:, 0].values, 0, None)
                    y_hi = ci.iloc[:, 1].values
                return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)
            except Exception as exc:
                logger.warning("SARIMAX predict failed (%s); using Ridge fallback", exc)

        # Fallback Ridge AR recursive prediction
        if self._fallback_model is None:
            raise RuntimeError("Fallback Ridge AR model was not fitted")

        lags = [1, 7, 14]
        if self._fallback_lags == 1:
            lags = [1]
        history = list(self._y_train.values)
        predictions = []
        for _ in range(h):
            max_lag = max(lags)
            if len(history) < max_lag:
                features = [history[-1]] * len(lags)
            else:
                features = [history[-lag] for lag in lags]
            y_next = float(self._fallback_model.predict([features])[0])
            y_next = max(0.0, y_next)
            predictions.append(y_next)
            history.append(y_next)

        y_hat = np.array(predictions)
        residual_std = float(np.std(self._y_train.values[-30:])) if len(self._y_train) >= 2 else 1.0
        y_lo, y_hi = _make_prediction_interval(y_hat, residual_std, self._coverage)
        return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)


# ── Concrete model 3: XGBoostForecaster ──────────────────────────────────────


class XGBoostForecaster(ForecastModel):
    """XGBoost forecaster with GlowCast feature engineering.

    Uses a direct multi-step strategy: a single XGBoost regressor is trained
    on the full feature set and then applied step-by-step over the horizon
    using recursive lag filling for future steps.

    GlowCast-specific features (``social_momentum``, ``temperature``,
    ``humidity``) are included when present in ``X_train`` / ``X_future``.

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, min_child_weight,
    subsample, colsample_bytree, reg_alpha, reg_lambda:
        XGBoost hyperparameters. Defaults match ``glowcast.yaml``.
    coverage:
        Prediction-interval coverage. Default 0.90.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        coverage: float = 0.90,
    ) -> None:
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        self._coverage = coverage
        self._model: Any = None
        self._fallback_model: Ridge | None = None
        self._y_train: pd.Series | None = None
        self._x_train: pd.DataFrame | None = None
        self._feature_cols: list[str] = []
        self._residual_std: float = 1.0
        self._using_fallback: bool = not _XGBOOST_AVAILABLE

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "XGBoostForecaster":
        y = self._extract_y_series(Y_train)
        self._y_train = y

        x_aligned: pd.DataFrame | None = None
        if X_train is not None:
            x_aligned = self._extract_x_aligned(X_train, y.index)
        self._x_train = x_aligned

        features = _engineer_features(y, x_aligned)
        self._feature_cols = list(features.columns)

        # Drop rows with NaN (early window with insufficient lag history)
        combined = features.copy()
        combined["__target__"] = y
        combined = combined.dropna()

        X_fit = combined[self._feature_cols].values
        y_fit = combined["__target__"].values

        # In-sample residuals for PI estimation
        if _XGBOOST_AVAILABLE:
            try:
                import xgboost as xgb

                self._model = xgb.XGBRegressor(**self._params)
                self._model.fit(X_fit, y_fit)
                y_pred_train = self._model.predict(X_fit)
                self._residual_std = float(np.std(y_fit - y_pred_train))
                self._using_fallback = False
                logger.debug(
                    "XGBoostForecaster fitted: n_train=%d, residual_std=%.3f",
                    len(y_fit),
                    self._residual_std,
                )
                return self
            except Exception as exc:
                logger.warning("XGBoost fit failed (%s); falling back to Ridge", exc)
                self._using_fallback = True

        # Ridge fallback
        self._using_fallback = True
        self._fallback_model = Ridge(alpha=1.0)
        self._fallback_model.fit(X_fit, y_fit)
        y_pred_fallback = self._fallback_model.predict(X_fit)
        self._residual_std = float(np.std(y_fit - y_pred_fallback))
        logger.debug("XGBoostForecaster fitted via Ridge fallback")
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._y_train is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        future_dates = _future_dates_from_series(self._y_train, h)

        x_fut_aligned: pd.DataFrame | None = None
        if X_future is not None:
            x_fut_aligned = self._extract_x_aligned(X_future, future_dates)

        feat_future = _engineer_features_future(self._y_train, h, x_fut_aligned)

        # Align feature columns: fill any missing cols with 0, drop extras
        for col in self._feature_cols:
            if col not in feat_future.columns:
                feat_future[col] = 0.0
        feat_future = feat_future[self._feature_cols].fillna(0)

        X_pred = feat_future.values
        active_model = self._model if not self._using_fallback else self._fallback_model
        if active_model is None:
            raise RuntimeError("No fitted model available")

        y_hat = np.clip(active_model.predict(X_pred).astype(float), 0, None)
        y_lo, y_hi = _make_prediction_interval(y_hat, self._residual_std, self._coverage)
        return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)


# ── Concrete model 4: LightGBMForecaster ─────────────────────────────────────


class LightGBMForecaster(ForecastModel):
    """LightGBM forecaster with GlowCast feature engineering.

    Mirrors ``XGBoostForecaster`` in feature engineering and prediction
    strategy (direct + recursive lag fill), but uses LightGBM-specific
    hyperparameters including ``num_leaves`` and ``min_child_samples``.

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, num_leaves,
    min_child_samples, subsample, colsample_bytree,
    reg_alpha, reg_lambda:
        LightGBM hyperparameters. Defaults match ``glowcast.yaml``.
    coverage:
        Prediction-interval coverage. Default 0.90.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        coverage: float = 0.90,
    ) -> None:
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._coverage = coverage
        self._model: Any = None
        self._fallback_model: Ridge | None = None
        self._y_train: pd.Series | None = None
        self._x_train: pd.DataFrame | None = None
        self._feature_cols: list[str] = []
        self._residual_std: float = 1.0
        self._using_fallback: bool = not _LIGHTGBM_AVAILABLE

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "LightGBMForecaster":
        y = self._extract_y_series(Y_train)
        self._y_train = y

        x_aligned: pd.DataFrame | None = None
        if X_train is not None:
            x_aligned = self._extract_x_aligned(X_train, y.index)
        self._x_train = x_aligned

        features = _engineer_features(y, x_aligned)
        self._feature_cols = list(features.columns)

        combined = features.copy()
        combined["__target__"] = y
        combined = combined.dropna()

        X_fit = combined[self._feature_cols].values
        y_fit = combined["__target__"].values

        if _LIGHTGBM_AVAILABLE:
            try:
                import lightgbm as lgb

                self._model = lgb.LGBMRegressor(**self._params)
                self._model.fit(X_fit, y_fit)
                y_pred_train = self._model.predict(X_fit)
                self._residual_std = float(np.std(y_fit - y_pred_train))
                self._using_fallback = False
                logger.debug(
                    "LightGBMForecaster fitted: n_train=%d, residual_std=%.3f",
                    len(y_fit),
                    self._residual_std,
                )
                return self
            except Exception as exc:
                logger.warning("LightGBM fit failed (%s); falling back to Ridge", exc)
                self._using_fallback = True

        # Ridge fallback
        self._using_fallback = True
        self._fallback_model = Ridge(alpha=1.0)
        self._fallback_model.fit(X_fit, y_fit)
        y_pred_fallback = self._fallback_model.predict(X_fit)
        self._residual_std = float(np.std(y_fit - y_pred_fallback))
        logger.debug("LightGBMForecaster fitted via Ridge fallback")
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._y_train is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        future_dates = _future_dates_from_series(self._y_train, h)

        x_fut_aligned: pd.DataFrame | None = None
        if X_future is not None:
            x_fut_aligned = self._extract_x_aligned(X_future, future_dates)

        feat_future = _engineer_features_future(self._y_train, h, x_fut_aligned)

        for col in self._feature_cols:
            if col not in feat_future.columns:
                feat_future[col] = 0.0
        feat_future = feat_future[self._feature_cols].fillna(0)

        X_pred = feat_future.values
        active_model = self._model if not self._using_fallback else self._fallback_model
        if active_model is None:
            raise RuntimeError("No fitted model available")

        y_hat = np.clip(active_model.predict(X_pred).astype(float), 0, None)
        y_lo, y_hi = _make_prediction_interval(y_hat, self._residual_std, self._coverage)
        return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)


# ── Concrete model 5: ChronosForecaster ──────────────────────────────────────


class ChronosForecaster(ForecastModel):
    """Zero-shot forecaster wrapping Amazon Chronos-Bolt-Small.

    Uses ``ChronosPipeline`` from the ``chronos-forecasting`` package to run
    inference without any fine-tuning.  When the package is unavailable the
    model falls back to ``NaiveMovingAverage`` and emits a warning.

    Parameters
    ----------
    model_id:
        HuggingFace model ID. Default ``"amazon/chronos-bolt-small"``
        (matches ``glowcast.yaml``).
    context_length:
        Number of historical observations passed as context to the model.
        Default 512.
    prediction_length:
        Maximum horizon the model can produce in one pass. Forecasts longer
        than this are tiled/extended with the naive fallback.
        Default 14.
    coverage:
        Prediction-interval coverage for the quantile-based PI. Default 0.90.
    device:
        Torch device string. Default ``"cpu"`` for CI/CD compatibility;
        override with ``"cuda"`` in production GPU environments.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        context_length: int = 512,
        prediction_length: int = 14,
        coverage: float = 0.90,
        device: str = "cpu",
    ) -> None:
        self._model_id = model_id
        self._context_length = context_length
        self._prediction_length = prediction_length
        self._coverage = coverage
        self._device = device

        self._pipeline: Any = None
        self._fallback: NaiveMovingAverage | None = None
        self._y_train: pd.Series | None = None
        self._using_fallback: bool = not _CHRONOS_AVAILABLE

    @property
    def name(self) -> str:
        return "chronos2_zs"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "ChronosForecaster":
        y = self._extract_y_series(Y_train)
        self._y_train = y

        if _CHRONOS_AVAILABLE:
            try:
                import torch
                from chronos import ChronosPipeline

                self._pipeline = ChronosPipeline.from_pretrained(
                    self._model_id,
                    device_map=self._device,
                    torch_dtype=torch.float32,
                )
                self._using_fallback = False
                logger.debug(
                    "ChronosForecaster loaded model '%s' on device '%s'",
                    self._model_id,
                    self._device,
                )
                return self
            except Exception as exc:
                logger.warning(
                    "ChronosPipeline init failed (%s); falling back to NaiveMovingAverage",
                    exc,
                )
                self._using_fallback = True

        # Fallback
        self._using_fallback = True
        self._fallback = NaiveMovingAverage(window=30, coverage=self._coverage)
        self._fallback.fit(Y_train, X_train)
        logger.debug("ChronosForecaster fitted via NaiveMovingAverage fallback")
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._y_train is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        if self._using_fallback:
            if self._fallback is None:
                raise RuntimeError("Fallback model was not initialised")
            return self._fallback.predict(h, X_future)

        # Chronos inference path
        try:
            import torch

            context = self._y_train.values[-self._context_length :]
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            alpha = 1.0 - self._coverage
            low_q = alpha / 2
            high_q = 1.0 - alpha / 2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quantile_preds, mean_preds = self._pipeline.predict_quantiles(
                    context=context_tensor,
                    prediction_length=min(h, self._prediction_length),
                    quantile_levels=[low_q, 0.5, high_q],
                )

            # quantile_preds shape: (batch=1, n_quantiles, prediction_length)
            y_lo_raw = quantile_preds[0, 0, :h].numpy()
            y_hat_raw = quantile_preds[0, 1, :h].numpy()
            y_hi_raw = quantile_preds[0, 2, :h].numpy()

            # If h > prediction_length, extend with naive
            if h > self._prediction_length:
                extra = h - self._prediction_length
                last_val = float(y_hat_raw[-1])
                last_std = float(np.std(y_hat_raw)) if len(y_hat_raw) > 1 else 1.0
                extra_hat = np.full(extra, last_val)
                extra_lo, extra_hi = _make_prediction_interval(extra_hat, last_std, self._coverage)
                y_hat_raw = np.concatenate([y_hat_raw, extra_hat])
                y_lo_raw = np.concatenate([y_lo_raw, extra_lo])
                y_hi_raw = np.concatenate([y_hi_raw, extra_hi])

            y_hat = np.clip(y_hat_raw, 0, None)
            y_lo = np.clip(y_lo_raw, 0, None)
            y_hi = y_hi_raw
            future_dates = _future_dates_from_series(self._y_train, h)
            return _build_prediction_df(future_dates, y_hat, y_lo, y_hi)

        except Exception as exc:
            logger.warning(
                "Chronos predict failed (%s); switching to NaiveMovingAverage", exc
            )
            fallback = NaiveMovingAverage(window=30, coverage=self._coverage)
            # Re-use the training data stored at fit time
            y_df = self._y_train.reset_index()
            y_df.columns = pd.Index(["ds", "y"])
            y_df["unique_id"] = "fallback"
            fallback.fit(y_df)
            return fallback.predict(h, X_future)


# ── Concrete model 6: RoutingEnsemble ────────────────────────────────────────


class RoutingEnsemble(ForecastModel):
    """Routing ensemble that dispatches to a specialist model per series regime.

    Routing logic (evaluated at fit time):
    - **cold_start** (< ``cold_start_threshold_days`` observations)
      → ``ChronosForecaster`` (zero-shot, no training required)
    - **intermittent** (coefficient of variation > ``intermittent_threshold_cv``)
      → ``SARIMAXForecaster`` (handles sporadic demand better)
    - **mature** (all other series)
      → ``LightGBMForecaster`` (best accuracy on dense, stable series)

    Sub-models are created via ``ForecastModelFactory`` so their
    hyperparameters are always sourced from ``glowcast.yaml``.

    Parameters
    ----------
    cold_start_threshold_days:
        Maximum history length (in days) to classify a series as cold-start.
        Default 60 (matches ``glowcast.yaml``).
    intermittent_threshold_cv:
        Coefficient of variation above which a series is classified as
        intermittent.  Default 1.5.
    """

    def __init__(
        self,
        cold_start_threshold_days: int = 60,
        intermittent_threshold_cv: float = 1.5,
    ) -> None:
        self._cold_start_threshold = cold_start_threshold_days
        self._cv_threshold = intermittent_threshold_cv
        self._active_model: ForecastModel | None = None
        self._regime: str = "unknown"

    @property
    def name(self) -> str:
        return "routing_ensemble"

    @property
    def active_regime(self) -> str:
        """The regime detected at fit time: 'cold_start', 'intermittent', or 'mature'."""
        return self._regime

    @property
    def active_model_name(self) -> str:
        """Name of the sub-model currently in use."""
        return self._active_model.name if self._active_model else "none"

    def _classify_regime(self, y: pd.Series) -> str:
        """Classify the time series into a demand regime.

        Parameters
        ----------
        y:
            Datetime-indexed float series of historical demand.

        Returns
        -------
        str
            One of ``'cold_start'``, ``'intermittent'``, ``'mature'``.
        """
        n_obs = len(y)
        if n_obs < self._cold_start_threshold:
            return "cold_start"

        mean_demand = float(y.mean())
        std_demand = float(y.std(ddof=1)) if n_obs > 1 else 0.0
        cv = std_demand / mean_demand if mean_demand > 1e-9 else float("inf")

        if cv > self._cv_threshold:
            return "intermittent"

        return "mature"

    def fit(
        self,
        Y_train: pd.DataFrame,
        X_train: pd.DataFrame | None = None,
    ) -> "RoutingEnsemble":
        y = self._extract_y_series(Y_train)
        self._regime = self._classify_regime(y)

        factory = ForecastModelFactory()
        if self._regime == "cold_start":
            self._active_model = factory.create("chronos2_zs")
        elif self._regime == "intermittent":
            self._active_model = factory.create("sarimax")
        else:
            self._active_model = factory.create("lightgbm")

        self._active_model.fit(Y_train, X_train)
        logger.info(
            "RoutingEnsemble: regime='%s' → sub-model='%s'",
            self._regime,
            self._active_model.name,
        )
        return self

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._active_model is None:
            raise RuntimeError("RoutingEnsemble must be fitted before calling predict()")
        return self._active_model.predict(h, X_future)


# ── Factory ───────────────────────────────────────────────────────────────────


class ForecastModelFactory:
    """Registry-based factory for GlowCast forecasting models.

    Implements the *Factory* design pattern:  callers request models by name
    string and the factory is responsible for instantiation and
    hyperparameter injection from ``glowcast.yaml``.

    Registry
    --------
    The ``_registry`` maps canonical name strings to model classes.
    Custom models can be registered at runtime via ``register()``.

    Usage
    -----
        factory = ForecastModelFactory()

        # Create a single model with default config-driven hyperparameters
        model = factory.create("lightgbm")

        # Create all models at once
        models = factory.create_all()

        # List registered model names
        print(factory.available_models())

        # Register a custom model
        factory.register("my_model", MyCustomModel)
    """

    _registry: dict[str, type[ForecastModel]] = {
        "naive_ma30": NaiveMovingAverage,
        "sarimax": SARIMAXForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "chronos2_zs": ChronosForecaster,
        "routing_ensemble": RoutingEnsemble,
    }

    def register(self, name: str, model_class: type[ForecastModel]) -> None:
        """Register a custom model class under a given name.

        Parameters
        ----------
        name:
            Registry key. Must not conflict with existing entries unless
            intentional override is desired.
        model_class:
            A concrete subclass of ``ForecastModel``.
        """
        if not issubclass(model_class, ForecastModel):
            raise TypeError(
                f"{model_class.__name__} must be a subclass of ForecastModel"
            )
        self._registry[name] = model_class
        logger.debug("ForecastModelFactory: registered '%s' → %s", name, model_class.__name__)

    def available_models(self) -> list[str]:
        """Return a sorted list of all registered model names.

        Returns
        -------
        list[str]
            Alphabetically sorted registry keys.
        """
        return sorted(self._registry.keys())

    def create(self, name: str, **kwargs: Any) -> ForecastModel:
        """Instantiate a model by name, merging config defaults with ``kwargs``.

        Hyperparameters are loaded from the matching ``model.<name>`` block in
        ``glowcast.yaml``.  Any ``kwargs`` provided to this call take
        precedence over the config file values (caller overrides).

        Parameters
        ----------
        name:
            Registry key, e.g. ``"lightgbm"``, ``"xgboost"``, etc.
        **kwargs:
            Optional keyword arguments that override config-file defaults.

        Returns
        -------
        ForecastModel
            Fully instantiated (but not yet fitted) model.

        Raises
        ------
        KeyError
            If ``name`` is not present in the registry.
        """
        if name not in self._registry:
            raise KeyError(
                f"Unknown model '{name}'. Available: {self.available_models()}"
            )

        # Load config defaults for this model (may be empty dict)
        try:
            cfg_defaults = get_model_config(name)
        except Exception:
            cfg_defaults = {}

        # Config file uses lists for SARIMAX order tuples — convert to tuples
        for key in ("order", "seasonal_order"):
            if key in cfg_defaults and isinstance(cfg_defaults[key], list):
                cfg_defaults[key] = tuple(cfg_defaults[key])

        # Caller kwargs take precedence over config
        merged_params = {**cfg_defaults, **kwargs}

        model_class = self._registry[name]

        # Filter to only valid constructor parameters to avoid TypeError
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_params = {k: v for k, v in merged_params.items() if k in valid_params}

        logger.debug(
            "ForecastModelFactory.create('%s') with params: %s", name, filtered_params
        )
        return model_class(**filtered_params)

    def create_all(self) -> dict[str, ForecastModel]:
        """Instantiate every registered model with config-driven hyperparameters.

        Returns
        -------
        dict[str, ForecastModel]
            Mapping of model name → unfitted model instance.
            Keys match ``available_models()``.
        """
        return {name: self.create(name) for name in self.available_models()}
