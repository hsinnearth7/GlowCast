"""Explainability module for GlowCast — SHAP and LIME wrappers.

Provides production-quality, sklearn-compatible wrappers around SHAP and LIME
with graceful fallbacks when the optional packages are unavailable.

Classes
-------
SHAPExplainer
    Global feature attribution using SHAP values.  Selects TreeExplainer for
    tree-based models (LightGBM, XGBoost, RandomForest, ExtraTrees, GradientBoosting)
    and falls back to KernelExplainer for all other estimators.  When ``shap``
    itself is not installed, permutation importance from scikit-learn is used
    as a lightweight substitute.

LIMEExplainer
    Local surrogate explanation for individual predictions using LIME tabular
    explainer.  Falls back to a Gaussian-perturbation linear fit when ``lime``
    is not installed.

compare_explanations
    Module-level utility that produces a side-by-side DataFrame of the top-N
    features ranked by both SHAP mean-|shap| and LIME mean-|weight|.

Notes
-----
Tree model detection is performed by inspecting ``type(model).__name__`` and
``type(model).__mro__`` for known substrings so that both native LightGBM
estimators (``LGBMRegressor``) and pipeline-wrapped variants are handled.

References
----------
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. NeurIPS 2017.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
  Explaining the predictions of any classifier. KDD 2016.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: shap
# ---------------------------------------------------------------------------
try:
    import shap as _shap

    _SHAP_AVAILABLE = True
    logger.debug("shap package available — using SHAP explainers.")
except ImportError:
    _shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False
    logger.warning(
        "shap package not installed.  SHAPExplainer will fall back to "
        "sklearn permutation importance.  Install with: pip install shap"
    )

# ---------------------------------------------------------------------------
# Optional dependency: lime
# ---------------------------------------------------------------------------
try:
    import lime as _lime
    import lime.lime_tabular as _lime_tabular

    _LIME_AVAILABLE = True
    logger.debug("lime package available — using LIME explainers.")
except ImportError:
    _lime = None  # type: ignore[assignment]
    _lime_tabular = None  # type: ignore[assignment]
    _LIME_AVAILABLE = False
    logger.warning(
        "lime package not installed.  LIMEExplainer will fall back to "
        "Gaussian-perturbation importance.  Install with: pip install lime"
    )

# ---------------------------------------------------------------------------
# Protocol — minimal sklearn-compatible estimator interface
# ---------------------------------------------------------------------------

_TREE_MODEL_KEYWORDS: tuple[str, ...] = (
    "lgbm",
    "lightgbm",
    "xgb",
    "xgboost",
    "randomforest",
    "extratrees",
    "gradientboosting",
    "decisiontree",
    "histgradientboosting",
)


@runtime_checkable
class _PredictProto(Protocol):
    """Minimal duck-type for sklearn predict interface."""

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


def _is_tree_model(model: Any) -> bool:
    """Return True if *model* is a known tree-based estimator.

    Detection is based on the class name (and full MRO names) so that both
    native and pipeline-wrapped estimators are matched.

    Parameters
    ----------
    model:
        Any sklearn-compatible estimator object.

    Returns
    -------
    bool
        ``True`` if the model appears to be a tree-based estimator.
    """
    names = " ".join(cls.__name__.lower() for cls in type(model).__mro__)
    return any(kw in names for kw in _TREE_MODEL_KEYWORDS)


# ===========================================================================
# SHAPExplainer
# ===========================================================================


class SHAPExplainer:
    """Global feature attribution for sklearn-compatible models via SHAP.

    Uses ``shap.TreeExplainer`` when the model is a tree-based estimator
    (LightGBM, XGBoost, Random Forest, GradientBoosting, etc.) and
    ``shap.KernelExplainer`` for all other models.  When the ``shap``
    package is not installed, falls back to scikit-learn's
    ``permutation_importance`` for a model-agnostic approximation.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator implementing ``predict``.
    feature_names:
        Ordered list of feature names corresponding to the columns of X.

    Attributes
    ----------
    model:
        The wrapped estimator.
    feature_names:
        Stored feature names list.
    _shap_values:
        Cached SHAP values array (set after the first
        :meth:`compute_shap_values` call).
    _X_cache:
        Cached input array used to compute ``_shap_values``.

    Examples
    --------
    >>> explainer = SHAPExplainer(model=lgbm_model, feature_names=feat_names)
    >>> sv = explainer.compute_shap_values(X_test)
    >>> importance = explainer.feature_importance()
    >>> print(importance.head())
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        if not hasattr(model, "predict"):
            raise TypeError(
                f"model must implement a predict() method; got {type(model).__name__}."
            )
        if not feature_names:
            raise ValueError("feature_names must be a non-empty list of strings.")

        self.model = model
        self.feature_names: list[str] = list(feature_names)
        self._shap_values: NDArray[np.float64] | None = None
        self._X_cache: NDArray[np.float64] | None = None
        self._explainer: Any = None  # shap Explainer object (if shap available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute SHAP values for the given input matrix.

        If ``shap`` is available, uses ``TreeExplainer`` for tree models and
        ``KernelExplainer`` (with 100-sample background) for others.  When
        ``shap`` is unavailable, approximates feature attributions via
        permutation importance repeated across each sample.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.  Must match
            the number of features used during model training.

        Returns
        -------
        NDArray[np.float64]
            SHAP value matrix of shape ``(n_samples, n_features)``.  Each
            entry ``[i, j]`` is the SHAP attribution of feature ``j`` for
            sample ``i``.

        Raises
        ------
        ValueError
            If ``X`` is empty or has a feature dimension mismatch with
            ``self.feature_names``.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        self._validate_X(X_arr)

        if _SHAP_AVAILABLE:
            shap_values = self._compute_shap_native(X_arr)
        else:
            logger.info("shap unavailable — computing permutation-based attribution.")
            shap_values = self._compute_permutation_attribution(X_arr)

        self._shap_values = shap_values
        self._X_cache = X_arr
        return shap_values

    def feature_importance(self) -> pd.DataFrame:
        """Return features ranked by mean absolute SHAP value.

        Must be called after :meth:`compute_shap_values`.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["feature", "mean_abs_shap"]`` sorted
            descending by ``mean_abs_shap``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_shap_values` has not been called yet.
        """
        if self._shap_values is None:
            raise RuntimeError(
                "No SHAP values available.  Call compute_shap_values(X) first."
            )

        mean_abs = np.mean(np.abs(self._shap_values), axis=0)
        df = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        )
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_shap_native(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use the shap library to compute SHAP values.

        Parameters
        ----------
        X:
            Validated feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        NDArray[np.float64]
            SHAP values, shape ``(n_samples, n_features)``.
        """
        if self._explainer is None:
            if _is_tree_model(self.model):
                logger.debug(
                    "SHAPExplainer: using TreeExplainer for %s.",
                    type(self.model).__name__,
                )
                self._explainer = _shap.TreeExplainer(self.model)
            else:
                # KernelExplainer requires a background dataset — use up to 100 samples
                n_bg = min(100, X.shape[0])
                background = X[:n_bg]
                logger.debug(
                    "SHAPExplainer: using KernelExplainer (background n=%d) for %s.",
                    n_bg,
                    type(self.model).__name__,
                )
                self._explainer = _shap.KernelExplainer(
                    self.model.predict, background
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._explainer.shap_values(X)

        # Multi-output models return a list; take the first output
        if isinstance(raw, list):
            raw = raw[0]

        return np.asarray(raw, dtype=np.float64)

    def _compute_permutation_attribution(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Approximate feature attribution via column-permutation.

        For each feature ``j``, measures the mean absolute change in
        predictions when that column is randomly shuffled.  The result is
        broadcast to a per-sample matrix so the return shape matches the
        SHAP convention ``(n_samples, n_features)``.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        NDArray[np.float64]
            Attribution matrix, shape ``(n_samples, n_features)``.
        """
        rng = np.random.default_rng(42)
        n_samples, n_features = X.shape
        base_pred = self.model.predict(X)
        attributions = np.zeros((n_samples, n_features), dtype=np.float64)

        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred = self.model.predict(X_perm)
            # Per-sample attribution = absolute change in prediction
            attributions[:, j] = np.abs(base_pred - perm_pred)

        return attributions

    def _validate_X(self, X: NDArray[np.float64]) -> None:
        """Raise ValueError for empty arrays or feature-count mismatches.

        Parameters
        ----------
        X:
            Array to validate, shape ``(n_samples, n_features)``.
        """
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}."
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} features but feature_names has "
                f"{len(self.feature_names)} entries."
            )


# ===========================================================================
# LIMEExplainer
# ===========================================================================


class LIMEExplainer:
    """Local surrogate explanation for individual predictions via LIME.

    Uses ``lime.lime_tabular.LimeTabularExplainer`` when the ``lime`` package
    is installed.  Falls back to a Gaussian-perturbation linear-fit approach
    when ``lime`` is unavailable.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator implementing ``predict``
        (regression) or ``predict_proba`` (classification).
    feature_names:
        Ordered list of feature names matching the model's training features.
    mode:
        ``"regression"`` (default) or ``"classification"``.  Controls which
        prediction method is used internally by LIME.

    Examples
    --------
    >>> explainer = LIMEExplainer(model=lgbm_model, feature_names=feat_names)
    >>> weights = explainer.explain_instance(X_test[0])
    >>> print(weights)
    {'social_momentum': 0.42, 'temperature': -0.18, ...}
    """

    _VALID_MODES: frozenset[str] = frozenset({"regression", "classification"})
    _N_PERTURBATIONS: int = 500   # samples used in the fallback perturbation
    _N_LIME_FEATURES: int = 10    # top features returned by LIME

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        mode: str = "regression",
    ) -> None:
        if not hasattr(model, "predict"):
            raise TypeError(
                f"model must implement a predict() method; got {type(model).__name__}."
            )
        if not feature_names:
            raise ValueError("feature_names must be a non-empty list of strings.")
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self._VALID_MODES)}, got '{mode}'."
            )

        self.model = model
        self.feature_names: list[str] = list(feature_names)
        self.mode: str = mode
        self._lime_explainer: Any = None  # instantiated lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_instance(self, x: NDArray[np.float64]) -> dict[str, float]:
        """Return LIME feature weights for a single prediction.

        Fits a local linear surrogate model in the neighbourhood of *x* using
        either the ``lime`` library or Gaussian perturbation.

        Parameters
        ----------
        x:
            Single sample feature vector, shape ``(n_features,)``.

        Returns
        -------
        dict[str, float]
            Mapping of ``{feature_name: weight}`` for the top-N features
            (default top-10), sorted descending by absolute weight.

        Raises
        ------
        ValueError
            If *x* has the wrong number of features.
        """
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if x_arr.shape[0] != len(self.feature_names):
            raise ValueError(
                f"x has {x_arr.shape[0]} features but feature_names has "
                f"{len(self.feature_names)} entries."
            )

        if _LIME_AVAILABLE:
            return self._explain_lime_native(x_arr)
        else:
            logger.info("lime unavailable — using Gaussian-perturbation fallback.")
            return self._explain_perturbation(x_arr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_lime_explainer(self, x: NDArray[np.float64]) -> Any:
        """Lazily construct the LimeTabularExplainer.

        Uses a synthetic training matrix of shape ``(200, n_features)``
        built by adding Gaussian noise around *x* so the explainer has a
        data distribution to anchor on even without the original training set.

        Parameters
        ----------
        x:
            Reference sample used to estimate feature variance.

        Returns
        -------
        lime.lime_tabular.LimeTabularExplainer
        """
        n_features = len(self.feature_names)
        rng = np.random.default_rng(0)
        # Build a small synthetic reference dataset centred on x
        noise_scale = np.maximum(np.abs(x) * 0.1, 0.01)
        synth_data = x + rng.normal(scale=noise_scale, size=(200, n_features))

        return _lime_tabular.LimeTabularExplainer(
            training_data=synth_data,
            feature_names=self.feature_names,
            mode=self.mode,
            random_state=42,
            verbose=False,
        )

    def _explain_lime_native(self, x: NDArray[np.float64]) -> dict[str, float]:
        """Delegate explanation to lime.lime_tabular.

        Parameters
        ----------
        x:
            Single sample, shape ``(n_features,)``.

        Returns
        -------
        dict[str, float]
            Feature name → LIME weight mapping.
        """
        if self._lime_explainer is None:
            self._lime_explainer = self._build_lime_explainer(x)

        predict_fn = (
            self.model.predict_proba
            if self.mode == "classification"
            else self.model.predict
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanation = self._lime_explainer.explain_instance(
                data_row=x,
                predict_fn=predict_fn,
                num_features=self._N_LIME_FEATURES,
                num_samples=self._N_PERTURBATIONS,
            )

        # Convert list of (feature_label, weight) tuples to dict
        raw_weights: list[tuple[str, float]] = explanation.as_list()
        return {feat: float(w) for feat, w in raw_weights}

    def _explain_perturbation(self, x: NDArray[np.float64]) -> dict[str, float]:
        """Gaussian-perturbation linear surrogate fallback.

        Generates ``_N_PERTURBATIONS`` neighbours of *x* by adding
        independent Gaussian noise to each feature, obtains model predictions,
        then fits a weighted ordinary-least-squares regression (weights
        = exponential kernel of L2 distance from *x*).

        Parameters
        ----------
        x:
            Single sample, shape ``(n_features,)``.

        Returns
        -------
        dict[str, float]
            Feature name → coefficient mapping, top-10 by absolute value.
        """
        rng = np.random.default_rng(42)
        n_features = len(self.feature_names)
        n_perturb = self._N_PERTURBATIONS

        noise_scale = np.maximum(np.abs(x) * 0.2, 0.01)
        X_perturb = x + rng.normal(scale=noise_scale, size=(n_perturb, n_features))
        # Include the original point
        X_perturb = np.vstack([x[np.newaxis, :], X_perturb])

        y_perturb = np.asarray(self.model.predict(X_perturb), dtype=np.float64)

        # Exponential kernel weights: closer samples count more
        distances = np.linalg.norm(X_perturb - x, axis=1)
        kernel_width = np.sqrt(n_features) * 0.25
        sample_weights = np.exp(-(distances**2) / (2 * kernel_width**2))

        # Weighted least-squares via normal equations
        W = np.diag(sample_weights)
        XtW = X_perturb.T @ W
        try:
            coeffs = np.linalg.lstsq(XtW @ X_perturb, XtW @ y_perturb, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_features, dtype=np.float64)

        # Return top features by absolute coefficient
        ranked_idx = np.argsort(np.abs(coeffs))[::-1][: self._N_LIME_FEATURES]
        return {
            self.feature_names[j]: float(coeffs[j])
            for j in ranked_idx
        }


# ===========================================================================
# compare_explanations
# ===========================================================================


def compare_explanations(
    shap_explainer: SHAPExplainer,
    lime_explainer: LIMEExplainer,
    X_sample: NDArray[np.float64],
    top_n: int = 10,
) -> pd.DataFrame:
    """Produce a side-by-side top-feature comparison of SHAP and LIME.

    Computes SHAP values for the entire sample matrix and LIME weights for
    each row, then aggregates both into mean absolute values and returns a
    merged DataFrame ordered by SHAP rank.

    Parameters
    ----------
    shap_explainer:
        A fitted (or unfitted) :class:`SHAPExplainer`.  SHAP values will be
        computed (or re-computed) for ``X_sample``.
    lime_explainer:
        A :class:`LIMEExplainer` instance.  ``explain_instance`` is called
        for every row in ``X_sample``.
    X_sample:
        Feature matrix used for the comparison, shape
        ``(n_samples, n_features)``.
    top_n:
        Number of top features to include in the output (default 10).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``"feature"``           – feature name
        - ``"shap_mean_abs"``     – mean absolute SHAP value across samples
        - ``"lime_mean_abs"``     – mean absolute LIME weight across samples
        - ``"shap_rank"``         – rank by SHAP importance (1 = most important)
        - ``"lime_rank"``         – rank by LIME importance (1 = most important)

        Sorted ascending by ``shap_rank``, limited to ``top_n`` rows.

    Raises
    ------
    ValueError
        If ``top_n`` is not a positive integer, or if ``X_sample`` is empty.

    Examples
    --------
    >>> df = compare_explanations(shap_exp, lime_exp, X_test[:50], top_n=10)
    >>> print(df.to_string(index=False))
    """
    if top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n}.")

    X_arr = np.asarray(X_sample, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError("X_sample must be a non-empty 2-D array.")

    # ---- SHAP ----
    shap_vals = shap_explainer.compute_shap_values(X_arr)  # (n, p)
    shap_mean_abs: NDArray[np.float64] = np.mean(np.abs(shap_vals), axis=0)

    # ---- LIME ----
    n_features = len(lime_explainer.feature_names)
    lime_abs_matrix = np.zeros((X_arr.shape[0], n_features), dtype=np.float64)
    feature_index = {name: i for i, name in enumerate(lime_explainer.feature_names)}

    for row_idx in range(X_arr.shape[0]):
        weights = lime_explainer.explain_instance(X_arr[row_idx])
        for feat_name, w in weights.items():
            # LIME may return composite labels like "feature <= 0.5"; extract clean name
            clean_name = _clean_lime_label(feat_name, lime_explainer.feature_names)
            if clean_name in feature_index:
                lime_abs_matrix[row_idx, feature_index[clean_name]] = abs(w)

    lime_mean_abs: NDArray[np.float64] = np.mean(lime_abs_matrix, axis=0)

    # ---- Merge ----
    feature_names = shap_explainer.feature_names
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_mean_abs": shap_mean_abs,
            "lime_mean_abs": lime_mean_abs,
        }
    )
    df["shap_rank"] = df["shap_mean_abs"].rank(ascending=False, method="min").astype(int)
    df["lime_rank"] = df["lime_mean_abs"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("shap_rank").head(top_n).reset_index(drop=True)

    return df[["feature", "shap_mean_abs", "lime_mean_abs", "shap_rank", "lime_rank"]]


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _clean_lime_label(label: str, feature_names: list[str]) -> str:
    """Extract the plain feature name from a LIME composite label.

    LIME sometimes emits labels like ``"social_momentum <= 0.35"`` or
    ``"1.2 < temperature <= 3.4"``.  This function checks whether any known
    feature name appears as a substring and returns the first match.

    Parameters
    ----------
    label:
        Raw LIME label string.
    feature_names:
        Known feature names to search for.

    Returns
    -------
    str
        Matched feature name, or the original *label* if no match is found.
    """
    for name in feature_names:
        if name in label:
            return name
    return label
