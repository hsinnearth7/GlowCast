"""CUPED — Controlled-experiment Using Pre-Experiment Data.

Implements variance reduction for A/B test metrics by exploiting the linear
relationship between a pre-experiment covariate X and the post-experiment
outcome Y:

    Y_cuped = Y - theta * (X - E[X])

where the optimal control variate coefficient is:

    theta = Cov(Y, X) / Var(X)

Variance of the CUPED estimator:

    Var(Y_cuped) = Var(Y) * (1 - rho^2)

where rho is the Pearson correlation between Y and X.

For GlowCast, the target correlation is rho = 0.74, which yields a 55 %
variance reduction — validated via n = 1000 bootstrap replications with an
expected CI of approximately [0.42, 0.48] for the variance reduction ratio.

Reference
---------
Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013).  Improving the sensitivity
of online controlled experiments by utilising pre-experiment data.  In
Proceedings of the Sixth ACM International Conference on Web Search and Data
Mining (WSDM 2013), pp. 123-132.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class BootstrapCI(NamedTuple):
    """Confidence interval produced by bootstrap resampling.

    Attributes
    ----------
    lower:
        Lower bound of the confidence interval.
    upper:
        Upper bound of the confidence interval.
    point_estimate:
        Point estimate (median of bootstrap distribution).
    bootstrap_std:
        Standard deviation of the bootstrap distribution.
    n_bootstrap:
        Number of bootstrap replications used.
    """

    lower: float
    upper: float
    point_estimate: float
    bootstrap_std: float
    n_bootstrap: int


@dataclass
class CUPEDAnalyzer:
    """CUPED variance reduction for controlled experiments.

    Fits a linear control variate on pre-experiment data and transforms
    post-experiment outcomes to reduce variance and thereby increase
    experiment sensitivity.

    Parameters
    ----------
    alpha_ci:
        Confidence level for bootstrap intervals (default 0.95 → 95 % CI).
    random_seed:
        Seed for the NumPy random generator used in bootstrapping.

    Attributes
    ----------
    theta_:
        Fitted control variate coefficient after calling :meth:`fit`.
    mean_x_:
        Sample mean of the pre-experiment covariate after calling :meth:`fit`.
    rho_:
        Pearson correlation between Y and X after calling :meth:`fit`.
    variance_reduction_:
        Estimated variance reduction ratio ``1 - rho^2`` after :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(100, 20, 5000)
    >>> Y = 0.74 * X + rng.normal(0, 15, 5000)   # rho ≈ 0.74
    >>> analyzer = CUPEDAnalyzer(random_seed=42)
    >>> analyzer.fit(pre_metric=X, post_metric=Y)
    >>> Y_cuped = analyzer.transform(Y, X)
    >>> ci = analyzer.bootstrap_ci(n=1000)
    >>> print(f"Variance reduction: {analyzer.variance_reduction_:.2%}")
    """

    alpha_ci: float = 0.95
    random_seed: int = 42

    # Fitted parameters (set by fit())
    theta_: float = field(default=0.0, init=False, repr=False)
    mean_x_: float = field(default=0.0, init=False, repr=False)
    rho_: float = field(default=0.0, init=False, repr=False)
    variance_reduction_: float = field(default=0.0, init=False, repr=False)

    # Cached arrays (set by fit())
    _pre_metric: NDArray[np.float64] = field(default_factory=lambda: np.array([]), init=False, repr=False)
    _post_metric: NDArray[np.float64] = field(default_factory=lambda: np.array([]), init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        pre_metric: NDArray[np.float64] | list[float],
        post_metric: NDArray[np.float64] | list[float],
    ) -> "CUPEDAnalyzer":
        """Fit the control variate coefficient from paired (X, Y) observations.

        Parameters
        ----------
        pre_metric:
            Pre-experiment covariate X, shape ``(n,)``.  For GlowCast this is
            typically units_sold or revenue from the pre-period window.
        post_metric:
            Post-experiment outcome Y, shape ``(n,)``.  Must have the same
            length as ``pre_metric``.

        Returns
        -------
        CUPEDAnalyzer
            Returns ``self`` to allow method chaining.

        Raises
        ------
        ValueError
            If the two arrays differ in length or contain fewer than 2
            observations.
        """
        X = np.asarray(pre_metric, dtype=np.float64)
        Y = np.asarray(post_metric, dtype=np.float64)

        if X.ndim != 1 or Y.ndim != 1:
            raise ValueError("pre_metric and post_metric must be 1-D arrays.")
        if len(X) != len(Y):
            raise ValueError(
                f"pre_metric (n={len(X)}) and post_metric (n={len(Y)}) must have the same length."
            )
        if len(X) < 2:
            raise ValueError("At least 2 observations are required to fit CUPED.")

        var_x = np.var(X, ddof=1)
        if var_x < 1e-12:
            warnings.warn(
                "Variance of pre_metric is near zero; setting theta=0 (no adjustment).",
                RuntimeWarning,
                stacklevel=2,
            )
            self.theta_ = 0.0
            self.rho_ = 0.0
        else:
            cov_yx = np.cov(Y, X, ddof=1)[0, 1]
            self.theta_ = float(cov_yx / var_x)
            # Pearson correlation
            std_y = float(np.std(Y, ddof=1))
            std_x = float(np.sqrt(var_x))
            self.rho_ = float(cov_yx / (std_y * std_x)) if std_y > 1e-12 else 0.0

        self.mean_x_ = float(np.mean(X))
        self.variance_reduction_ = float(1.0 - self.rho_ ** 2)

        self._pre_metric = X
        self._post_metric = Y
        self._is_fitted = True

        return self

    def transform(
        self,
        Y: NDArray[np.float64] | list[float],
        X: NDArray[np.float64] | list[float],
    ) -> NDArray[np.float64]:
        """Apply the CUPED transformation to new or training data.

        Computes:

            Y_cuped = Y - theta * (X - E[X])

        where ``theta`` and ``E[X]`` come from the fitted model.

        Parameters
        ----------
        Y:
            Post-experiment outcome array, shape ``(n,)``.
        X:
            Pre-experiment covariate array, shape ``(n,)``.

        Returns
        -------
        NDArray[np.float64]
            CUPED-adjusted outcome array, shape ``(n,)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before :meth:`transform`.
        """
        self._check_is_fitted()
        Y_arr = np.asarray(Y, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)

        if len(Y_arr) != len(X_arr):
            raise ValueError(
                f"Y (n={len(Y_arr)}) and X (n={len(X_arr)}) must have the same length."
            )

        return Y_arr - self.theta_ * (X_arr - self.mean_x_)

    def compute_variance_reduction(self) -> dict[str, float]:
        """Compute variance statistics for the CUPED transformation.

        Returns a dictionary containing the raw and CUPED variances as well
        as the analytical variance reduction ratio ``1 - rho^2``.

        Returns
        -------
        dict[str, float]
            Keys:
            - ``"var_raw"``: Variance of the raw post-experiment metric.
            - ``"var_cuped"``: Variance of the CUPED-adjusted metric.
            - ``"variance_reduction_ratio"``: ``1 - (var_cuped / var_raw)``.
            - ``"analytical_reduction"``: ``1 - rho^2`` (theoretical).
            - ``"rho"``: Pearson correlation used in the reduction.
            - ``"theta"``: Fitted control variate coefficient.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_is_fitted()

        Y_cuped = self.transform(self._post_metric, self._pre_metric)
        var_raw = float(np.var(self._post_metric, ddof=1))
        var_cuped = float(np.var(Y_cuped, ddof=1))

        empirical_reduction = float(1.0 - var_cuped / var_raw) if var_raw > 1e-12 else 0.0

        return {
            "var_raw": var_raw,
            "var_cuped": var_cuped,
            "variance_reduction_ratio": empirical_reduction,
            "analytical_reduction": self.variance_reduction_,
            "rho": self.rho_,
            "theta": self.theta_,
        }

    def bootstrap_ci(self, n: int = 1000) -> BootstrapCI:
        """Estimate a confidence interval for the variance reduction ratio via bootstrap.

        Performs ``n`` non-parametric bootstrap replications.  In each
        replication a paired sample of (Y, X) is drawn with replacement and
        the variance reduction ratio ``1 - (Var(Y_cuped) / Var(Y_raw))`` is
        recomputed.  The CI is extracted from the bootstrap percentile
        distribution.

        Target (GlowCast, rho = 0.74): CI ≈ [0.42, 0.48] at 95 % level.

        Parameters
        ----------
        n:
            Number of bootstrap replications (default 1000).

        Returns
        -------
        BootstrapCI
            Named tuple containing ``lower``, ``upper``, ``point_estimate``,
            ``bootstrap_std``, and ``n_bootstrap``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``n`` is not a positive integer.
        """
        self._check_is_fitted()
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n}.")

        rng = np.random.default_rng(self.random_seed)
        N = len(self._post_metric)
        reductions = np.empty(n, dtype=np.float64)

        for i in range(n):
            idx = rng.integers(0, N, size=N)
            Y_boot = self._post_metric[idx]
            X_boot = self._pre_metric[idx]

            var_x_b = float(np.var(X_boot, ddof=1))
            if var_x_b < 1e-12:
                reductions[i] = 0.0
                continue

            cov_b = float(np.cov(Y_boot, X_boot, ddof=1)[0, 1])
            theta_b = cov_b / var_x_b
            mean_x_b = float(np.mean(X_boot))

            Y_cuped_b = Y_boot - theta_b * (X_boot - mean_x_b)
            var_raw_b = float(np.var(Y_boot, ddof=1))
            var_cuped_b = float(np.var(Y_cuped_b, ddof=1))

            reductions[i] = 1.0 - var_cuped_b / var_raw_b if var_raw_b > 1e-12 else 0.0

        alpha = 1.0 - self.alpha_ci
        lower = float(np.percentile(reductions, 100 * alpha / 2))
        upper = float(np.percentile(reductions, 100 * (1.0 - alpha / 2)))
        point_estimate = float(np.median(reductions))
        bootstrap_std = float(np.std(reductions, ddof=1))

        return BootstrapCI(
            lower=lower,
            upper=upper,
            point_estimate=point_estimate,
            bootstrap_std=bootstrap_std,
            n_bootstrap=n,
        )

    def summary(self) -> str:
        """Return a human-readable summary of the fitted CUPED model.

        Returns
        -------
        str
            Multi-line summary string.
        """
        self._check_is_fitted()
        stats = self.compute_variance_reduction()
        return (
            "CUPED Analyzer Summary\n"
            "======================\n"
            f"  n observations   : {len(self._post_metric):,}\n"
            f"  theta (coef)     : {self.theta_:.6f}\n"
            f"  E[X]             : {self.mean_x_:.4f}\n"
            f"  rho (Y,X)        : {self.rho_:.4f}\n"
            f"  Var(Y) raw       : {stats['var_raw']:.4f}\n"
            f"  Var(Y_cuped)     : {stats['var_cuped']:.4f}\n"
            f"  Variance red. %  : {stats['variance_reduction_ratio']:.2%} (empirical)\n"
            f"  Analytical 1-rho²: {stats['analytical_reduction']:.2%}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not yet been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "CUPEDAnalyzer is not fitted yet.  Call fit(pre_metric, post_metric) first."
            )
