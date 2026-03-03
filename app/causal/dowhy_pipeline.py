"""DoWhy causal inference pipeline for GlowCast.

Implements the canonical 4-step DoWhy workflow:

    1. Model   — declare a causal DAG from domain knowledge
    2. Identify — symbolically derive an estimable expression (estimand)
    3. Estimate — fit a statistical estimator for the Average Treatment Effect
    4. Refute   — challenge the estimate with falsification tests

When the ``dowhy`` package is available the real library is used.  If it is
not installed (e.g. in a lightweight CI environment) a manual implementation
using ``sklearn`` and ``scipy`` provides a drop-in fallback with identical
public API and semantics.

Reference
---------
Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal
Inference. arXiv preprint arXiv:2011.04216.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

# ── Optional DoWhy import ─────────────────────────────────────────────────────
try:
    import dowhy  # noqa: F401 (used for availability check)
    from dowhy import CausalModel

    _DOWHY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DOWHY_AVAILABLE = False
    warnings.warn(
        "dowhy is not installed.  DoWhyPipeline will use a manual sklearn/scipy "
        "fallback.  Install dowhy>=0.11 for the full structural causal model "
        "workflow including symbolic identification.",
        ImportWarning,
        stacklevel=2,
    )

# ── Optional sklearn imports (always available per pyproject.toml) ────────────
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_REFUTATION_METHODS: list[str] = [
    "random_common_cause",
    "placebo_treatment",
    "data_subset",
]

_DOWHY_REFUTER_MAP: dict[str, str] = {
    "random_common_cause": "random_common_cause",
    "placebo_treatment": "placebo_treatment_refuter",
    "data_subset": "data_subset_refuter",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────


class DoWhyPipeline:
    """End-to-end structural causal model pipeline using the DoWhy framework.

    Follows the 4-step DoWhy methodology:

    1. **Model**  — build a causal DAG from domain knowledge (``build_model``)
    2. **Identify** — derive a non-parametric identification expression (``identify``)
    3. **Estimate** — fit a statistical estimator for the ATE (``estimate``)
    4. **Refute**  — stress-test the estimate with falsification tests (``refute``)

    The pipeline seamlessly delegates to the real ``dowhy`` library when it is
    installed, and transparently falls back to a manual ``sklearn`` + ``scipy``
    implementation otherwise.  Both paths expose identical return types so
    downstream code requires no branching.

    Parameters
    ----------
    treatment : str
        Column name of the binary or continuous treatment variable (e.g.
        ``"promotion"``).
    outcome : str
        Column name of the outcome variable (e.g. ``"units_sold"``).
    common_causes : list[str]
        Column names of observed confounders that are parents of *both* the
        treatment and the outcome.  These are used to build the backdoor
        adjustment set.
    instruments : list[str] | None
        Column names of instrumental variables — variables that affect
        treatment but have no direct path to the outcome.  Optional; required
        only when using IV estimators.

    Attributes
    ----------
    _model : CausalModel | None
        Internal DoWhy CausalModel (set after ``build_model``).
    _estimand : IdentifiedEstimand | None
        Identified estimand (set after ``identify``).
    _estimate : CausalEstimate | None
        Causal estimate (set after ``estimate``).
    _data : pd.DataFrame | None
        Training data stored for refutation.

    Examples
    --------
    >>> pipe = DoWhyPipeline(
    ...     treatment="promotion",
    ...     outcome="units_sold",
    ...     common_causes=["price", "seasonality", "fc_id"],
    ... )
    >>> results = pipe.run_pipeline(data=df)
    >>> print(f"ATE = {results['ate']:.4f}")
    """

    def __init__(
        self,
        treatment: str,
        outcome: str,
        common_causes: list[str],
        instruments: list[str] | None = None,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes: list[str] = list(common_causes)
        self.instruments: list[str] = list(instruments) if instruments else []

        # Internal state
        self._data: pd.DataFrame | None = None
        self._model: Any = None           # CausalModel (dowhy) or sentinel
        self._estimand: Any = None        # IdentifiedEstimand or sentinel
        self._estimate: Any = None        # CausalEstimate or scalar ATE
        self._ate: float | None = None
        self._ci: tuple[float, float] | None = None

        logger.info(
            "DoWhyPipeline initialised",
            extra={
                "treatment": treatment,
                "outcome": outcome,
                "common_causes": common_causes,
                "instruments": instruments,
                "backend": "dowhy" if _DOWHY_AVAILABLE else "manual-sklearn",
            },
        )

    # ── Step 1: Model ─────────────────────────────────────────────────────────

    def build_model(self, data: pd.DataFrame) -> "DoWhyPipeline":
        """Step 1 — Declare the causal DAG from domain knowledge.

        Constructs a directed acyclic graph (DAG) encoding the assumed causal
        structure among treatment, outcome, common causes, and (optionally)
        instruments.  The graph is expressed in GML / DOT notation and handed
        to DoWhy's ``CausalModel``.

        Parameters
        ----------
        data : pd.DataFrame
            Observational dataset.  Must contain columns for ``treatment``,
            ``outcome``, and all ``common_causes`` (and ``instruments`` if
            provided).

        Returns
        -------
        DoWhyPipeline
            Returns ``self`` to enable method chaining.

        Raises
        ------
        ValueError
            If required columns are missing from ``data``.
        """
        required = {self.treatment, self.outcome, *self.common_causes, *self.instruments}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        self._data = data.copy()

        if _DOWHY_AVAILABLE:
            self._model = self._build_dowhy_model(data)
        else:
            self._model = "_manual"  # sentinel: use fallback path

        logger.info("Step 1/4 — causal model built", extra={"backend": "dowhy" if _DOWHY_AVAILABLE else "manual"})
        return self

    def _build_dowhy_model(self, data: pd.DataFrame) -> "CausalModel":
        """Construct a DoWhy CausalModel with an explicit GML graph."""
        # Build GML graph string
        nodes = [self.treatment, self.outcome, *self.common_causes, *self.instruments]
        node_lines = "\n".join(f'  node [ id "{n}" label "{n}" ]' for n in nodes)

        # Edges: common_causes → treatment, common_causes → outcome
        edge_lines_parts: list[str] = []
        for cc in self.common_causes:
            edge_lines_parts.append(f'  edge [ source "{cc}" target "{self.treatment}" ]')
            edge_lines_parts.append(f'  edge [ source "{cc}" target "{self.outcome}" ]')
        # treatment → outcome
        edge_lines_parts.append(f'  edge [ source "{self.treatment}" target "{self.outcome}" ]')
        # instruments → treatment
        for iv in self.instruments:
            edge_lines_parts.append(f'  edge [ source "{iv}" target "{self.treatment}" ]')

        gml_graph = "graph [\n" + node_lines + "\n" + "\n".join(edge_lines_parts) + "\n]"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CausalModel(
                data=data,
                treatment=self.treatment,
                outcome=self.outcome,
                graph=gml_graph,
                instruments=self.instruments if self.instruments else None,
            )
        return model

    # ── Step 2: Identify ──────────────────────────────────────────────────────

    def identify(self) -> "DoWhyPipeline":
        """Step 2 — Symbolically identify the causal estimand.

        Applies the backdoor criterion (Pearl, 2009) to derive an estimable
        expression for the causal effect P(Y | do(T)) from the observed
        distribution P(Y, T, W) where W are the adjustment variables.

        Returns
        -------
        DoWhyPipeline
            Returns ``self`` to enable method chaining.

        Raises
        ------
        RuntimeError
            If ``build_model`` has not been called first.
        """
        if self._model is None:
            raise RuntimeError("Call build_model(data) before identify().")

        if _DOWHY_AVAILABLE and self._model != "_manual":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._estimand = self._model.identify_effect(proceed_when_unidentifiable=True)
            logger.info("Step 2/4 — estimand identified (DoWhy)", extra={"estimand": str(self._estimand)[:200]})
        else:
            # Manual: backdoor adjustment via OLS
            self._estimand = "_backdoor_ols"
            logger.info("Step 2/4 — estimand identified (manual: backdoor OLS)")

        return self

    # ── Step 3: Estimate ──────────────────────────────────────────────────────

    def estimate(self, method: str = "backdoor.linear_regression") -> dict[str, Any]:
        """Step 3 — Estimate the Average Treatment Effect (ATE).

        Fits a statistical model consistent with the identified estimand and
        returns the point estimate plus a 95 % confidence interval.

        Parameters
        ----------
        method : str
            DoWhy estimation method string.  The default
            ``"backdoor.linear_regression"`` corresponds to OLS with the
            backdoor adjustment set as covariates.  Other valid values include
            ``"backdoor.propensity_score_matching"``,
            ``"iv.instrumental_variable"``, etc.

        Returns
        -------
        dict[str, Any]
            ``ate``  — point estimate of the Average Treatment Effect.
            ``ci_lower`` — lower bound of the 95 % confidence interval.
            ``ci_upper`` — upper bound of the 95 % confidence interval.
            ``method`` — the estimation method used.
            ``backend`` — ``"dowhy"`` or ``"manual"``.

        Raises
        ------
        RuntimeError
            If ``identify`` has not been called first.
        """
        if self._estimand is None:
            raise RuntimeError("Call identify() before estimate().")

        if _DOWHY_AVAILABLE and self._estimand != "_backdoor_ols":
            result = self._estimate_dowhy(method)
        else:
            result = self._estimate_manual()

        self._ate = result["ate"]
        self._ci = (result["ci_lower"], result["ci_upper"])

        logger.info(
            "Step 3/4 — ATE estimated",
            extra={"ate": self._ate, "ci": self._ci, "method": method},
        )
        return result

    def _estimate_dowhy(self, method: str) -> dict[str, Any]:
        """Delegate estimation to DoWhy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate = self._model.estimate_effect(
                self._estimand,
                method_name=method,
                confidence_intervals=True,
            )
        self._estimate = estimate

        ate = float(estimate.value)
        # DoWhy stores CIs as (lower, upper) or as attribute
        try:
            ci_lower, ci_upper = estimate.get_confidence_intervals()
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
        except Exception:
            # Fall back: bootstrap 95 % CI from residuals
            ci_lower, ci_upper = self._bootstrap_ci(ate)

        return {"ate": ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "method": method, "backend": "dowhy"}

    def _estimate_manual(self) -> dict[str, Any]:
        """Manual OLS backdoor estimator.

        Regresses outcome on treatment + common_causes, then reads off the
        coefficient of the treatment variable as the ATE.  Standard errors are
        estimated via heteroskedasticity-robust (HC3) sandwich estimator
        approximated through residual bootstrapping.
        """
        assert self._data is not None  # guaranteed by build_model
        data = self._data

        feature_cols = self.common_causes + self.instruments
        X_feat = data[feature_cols].values if feature_cols else np.zeros((len(data), 0))
        T = data[self.treatment].values.reshape(-1, 1)
        Y = data[self.outcome].values

        # Design matrix: [confounders, treatment]
        if X_feat.shape[1] > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            X_design = np.hstack([X_scaled, T])
        else:
            X_design = T

        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_design, Y)

        # Treatment coefficient is always the last column
        ate = float(ols.coef_[-1])
        ci_lower, ci_upper = self._bootstrap_ci(ate)

        self._estimate = ols
        return {
            "ate": ate, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "method": "backdoor.linear_regression", "backend": "manual",
        }

    def _bootstrap_ci(self, ate: float, n_bootstrap: int = 500, alpha: float = 0.05) -> tuple[float, float]:
        """Estimate 95 % CI for ATE via residual bootstrapping (fast path).

        This is a lightweight surrogate used by the manual backend and as a
        fallback when DoWhy does not return CIs directly.
        """
        assert self._data is not None
        data = self._data
        feature_cols = self.common_causes + self.instruments
        X_feat = data[feature_cols].values if feature_cols else np.zeros((len(data), 0))
        T = data[self.treatment].values.reshape(-1, 1)
        Y = data[self.outcome].values

        if X_feat.shape[1] > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            X_design = np.hstack([X_scaled, T])
        else:
            X_design = T

        rng = np.random.default_rng(seed=42)
        boot_ates: list[float] = []
        n = len(data)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            ols_b = LinearRegression(fit_intercept=True)
            ols_b.fit(X_design[idx], Y[idx])
            boot_ates.append(float(ols_b.coef_[-1]))

        lower = float(np.percentile(boot_ates, 100 * alpha / 2))
        upper = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))
        return lower, upper

    # ── Step 4: Refute ────────────────────────────────────────────────────────

    def refute(
        self,
        methods: list[str] = _DEFAULT_REFUTATION_METHODS,
    ) -> list[dict[str, Any]]:
        """Step 4 — Falsification / robustness tests for the causal estimate.

        Runs a suite of refutation tests that challenge the validity of the
        estimated ATE under different assumption violations:

        ``random_common_cause``
            Adds a random covariate that is independent of everything.  A
            robust estimate should not change significantly.

        ``placebo_treatment``
            Replaces the real treatment with a random permutation.  The ATE
            should collapse to near-zero.

        ``data_subset``
            Re-estimates on a random 80 % subset.  A stable estimate should
            remain within sampling noise.

        Parameters
        ----------
        methods : list[str]
            Subset of ``["random_common_cause", "placebo_treatment",
            "data_subset"]`` to run.

        Returns
        -------
        list[dict[str, Any]]
            One dict per method containing:
            ``method``          — method name.
            ``new_ate``         — refuted ATE.
            ``orig_ate``        — original ATE for reference.
            ``pvalue``          — p-value (test: new_ate == orig_ate).
            ``passed``          — True if the refutation *did not* reject the
                                  original estimate (i.e. estimate is robust).

        Raises
        ------
        RuntimeError
            If ``estimate`` has not been called first.
        """
        if self._ate is None:
            raise RuntimeError("Call estimate() before refute().")

        results: list[dict[str, Any]] = []

        for method_name in methods:
            if _DOWHY_AVAILABLE and self._estimate != "_manual":
                result = self._refute_dowhy(method_name)
            else:
                result = self._refute_manual(method_name)
            results.append(result)
            logger.info(
                "Step 4/4 — refutation",
                extra={"method": method_name, "passed": result["passed"], "pvalue": result["pvalue"]},
            )

        return results

    def _refute_dowhy(self, method_name: str) -> dict[str, Any]:
        """Run a single DoWhy refutation test."""
        dowhy_name = _DOWHY_REFUTER_MAP.get(method_name, method_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = self._model.refute_estimate(
                self._estimand,
                self._estimate,
                method_name=dowhy_name,
            )

        new_ate = float(ref.new_effect)
        pvalue = float(ref.refutation_result.get("p_value", 1.0)) if hasattr(ref, "refutation_result") else 1.0
        passed = pvalue > 0.05

        return {
            "method": method_name,
            "new_ate": new_ate,
            "orig_ate": self._ate,
            "pvalue": pvalue,
            "passed": passed,
            "backend": "dowhy",
        }

    def _refute_manual(self, method_name: str) -> dict[str, Any]:
        """Manual refutation tests using sklearn and permutation logic."""
        assert self._data is not None
        data = self._data
        orig_ate = self._ate
        assert orig_ate is not None

        rng = np.random.default_rng(seed=0)

        if method_name == "random_common_cause":
            data_r = data.copy()
            data_r["_random_cause"] = rng.standard_normal(len(data_r))
            common_causes_r = self.common_causes + ["_random_cause"]
            new_ate = self._quick_ols_ate(data_r, common_causes_r)

        elif method_name == "placebo_treatment":
            data_r = data.copy()
            data_r[self.treatment] = rng.permutation(data_r[self.treatment].values)
            new_ate = self._quick_ols_ate(data_r, self.common_causes)

        elif method_name == "data_subset":
            n = len(data)
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            data_r = data.iloc[idx].reset_index(drop=True)
            new_ate = self._quick_ols_ate(data_r, self.common_causes)

        else:
            raise ValueError(f"Unknown refutation method: {method_name!r}")

        # p-value via bootstrap distribution of (new_ate - orig_ate)
        pvalue = self._refutation_pvalue(orig_ate, new_ate, method_name)
        passed = pvalue > 0.05

        return {
            "method": method_name,
            "new_ate": new_ate,
            "orig_ate": orig_ate,
            "pvalue": pvalue,
            "passed": passed,
            "backend": "manual",
        }

    def _quick_ols_ate(self, data: pd.DataFrame, common_causes: list[str]) -> float:
        """Fit OLS on a (possibly modified) dataset and return the treatment coefficient."""
        feature_cols = common_causes + self.instruments
        X_feat = data[feature_cols].values if feature_cols else np.zeros((len(data), 0))
        T = data[self.treatment].values.reshape(-1, 1)
        Y = data[self.outcome].values

        if X_feat.shape[1] > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            X_design = np.hstack([X_scaled, T])
        else:
            X_design = T

        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_design, Y)
        return float(ols.coef_[-1])

    def _refutation_pvalue(self, orig_ate: float, new_ate: float, method_name: str) -> float:
        """Approximate p-value for H0: new_ate == orig_ate.

        Uses a one-sample permutation approximation:
        - For placebo methods the null hypothesis is ATE=0 so we test whether
          ``abs(new_ate)`` is large enough to be inconsistent with zero.
        - For subset / random-cause methods we test whether the shift is within
          bootstrap variability.
        """
        if method_name == "placebo_treatment":
            # Under the null (placebo), the true ATE is 0.
            # We estimate noise from the original bootstrap CI.
            if self._ci is not None:
                half_width = (self._ci[1] - self._ci[0]) / 2
                std_approx = half_width / 1.96
            else:
                std_approx = abs(orig_ate) * 0.1 + 1e-9

            from scipy import stats  # always available via scipy dep

            t_stat = abs(new_ate) / (std_approx + 1e-12)
            pvalue = float(2 * stats.norm.sf(t_stat))
        else:
            # For other methods: test shift |new - orig| against sampling noise
            if self._ci is not None:
                half_width = (self._ci[1] - self._ci[0]) / 2
                std_approx = half_width / 1.96
            else:
                std_approx = abs(orig_ate) * 0.1 + 1e-9

            from scipy import stats

            z_stat = abs(new_ate - orig_ate) / (std_approx + 1e-12)
            pvalue = float(2 * stats.norm.sf(z_stat))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, pvalue))

    # ── Convenience: run full pipeline ────────────────────────────────────────

    def run_pipeline(
        self,
        data: pd.DataFrame,
        estimate_method: str = "backdoor.linear_regression",
        refutation_methods: list[str] = _DEFAULT_REFUTATION_METHODS,
    ) -> dict[str, Any]:
        """Run all 4 DoWhy steps in sequence and return consolidated results.

        Convenience wrapper that calls ``build_model → identify → estimate →
        refute`` and collates their outputs into a single dictionary.

        Parameters
        ----------
        data : pd.DataFrame
            Observational dataset with all required columns.
        estimate_method : str
            Passed to ``estimate(method=...)``.
        refutation_methods : list[str]
            Passed to ``refute(methods=...)``.

        Returns
        -------
        dict[str, Any]
            ``ate``          — Average Treatment Effect point estimate.
            ``ci_lower``     — 95 % CI lower bound.
            ``ci_upper``     — 95 % CI upper bound.
            ``method``       — estimation method used.
            ``backend``      — ``"dowhy"`` or ``"manual"``.
            ``refutations``  — list of refutation result dicts (see ``refute``).
            ``n_obs``        — number of observations used.
            ``treatment``    — treatment variable name.
            ``outcome``      — outcome variable name.

        Examples
        --------
        >>> pipe = DoWhyPipeline(
        ...     treatment="promotion",
        ...     outcome="units_sold",
        ...     common_causes=["price", "seasonality"],
        ... )
        >>> res = pipe.run_pipeline(df)
        >>> print(f"ATE={res['ate']:.3f}  [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
        """
        logger.info("Running DoWhy 4-step pipeline", extra={"n_obs": len(data)})

        self.build_model(data)
        self.identify()
        estimate_result = self.estimate(method=estimate_method)
        refutations = self.refute(methods=refutation_methods)

        return {
            "ate": estimate_result["ate"],
            "ci_lower": estimate_result["ci_lower"],
            "ci_upper": estimate_result["ci_upper"],
            "method": estimate_result["method"],
            "backend": estimate_result["backend"],
            "refutations": refutations,
            "n_obs": len(data),
            "treatment": self.treatment,
            "outcome": self.outcome,
        }

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ate_str = f"{self._ate:.4f}" if self._ate is not None else "not estimated"
        return (
            f"DoWhyPipeline("
            f"treatment={self.treatment!r}, "
            f"outcome={self.outcome!r}, "
            f"common_causes={self.common_causes!r}, "
            f"ate={ate_str}, "
            f"backend={'dowhy' if _DOWHY_AVAILABLE else 'manual'})"
        )
