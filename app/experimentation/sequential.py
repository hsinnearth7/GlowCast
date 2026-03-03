"""Sequential hypothesis testing with always-valid p-values.

Implements the mixture Sequential Probability Ratio Test (mSPRT) for
continuous monitoring of A/B experiments.  Unlike fixed-horizon tests,
mSPRT controls the Type-I error at level alpha *for every sample size
simultaneously*, enabling practitioners to peek at results any time
without inflating false-positive rates.

The likelihood ratio at time t under the mixture prior N(0, tau^2) over
the mean-difference parameter delta is:

    Lambda_t = integral_delta  L(delta) * phi(delta; 0, tau^2)  d_delta

For normally distributed observations this integral has a closed form:

    Lambda_t = sqrt(V / (V + t * tau^2))
               * exp( t^2 * tau^2 * delta_hat_t^2 / (2 * V * (V + t * tau^2)) )

where:
    V        = pooled variance estimate (sigma^2 per observation)
    t        = current sample size (per group)
    delta_hat_t = current observed treatment effect (mean difference)
    tau^2    = prior variance on delta (controls sensitivity)

The always-valid p-value is obtained by inverting the mixture LR:

    p_value = min(1, 1 / Lambda_t)

and an always-valid confidence sequence (CS) is derived from the same
likelihood ratio family.

Reference
---------
Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017).  Peeking at
A/B tests: Why it matters and what to do about it.  In Proceedings of
the 23rd ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining, pp. 1517-1525.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class SequentialResult(NamedTuple):
    """Snapshot result from the sequential tester.

    Attributes
    ----------
    n_control:
        Number of observations in the control group so far.
    n_treatment:
        Number of observations in the treatment group so far.
    mean_control:
        Running mean of the control group.
    mean_treatment:
        Running mean of the treatment group.
    observed_delta:
        Observed difference in means (treatment − control).
    p_value:
        Always-valid p-value (min(1, 1 / Lambda_t)).
    lambda_ratio:
        Current mixture likelihood ratio Lambda_t.
    confidence_sequence_lower:
        Lower bound of the always-valid confidence sequence for delta.
    confidence_sequence_upper:
        Upper bound of the always-valid confidence sequence for delta.
    should_stop:
        Whether the test has crossed the decision boundary at alpha=0.05.
    """

    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    observed_delta: float
    p_value: float
    lambda_ratio: float
    confidence_sequence_lower: float
    confidence_sequence_upper: float
    should_stop: bool


@dataclass
class SequentialTester:
    """Always-valid sequential hypothesis tester (mSPRT).

    Suitable for continuous monitoring of GlowCast A/B experiments
    (e.g. recommendation algorithm lift, pricing intervention effects)
    where the experiment duration is not fixed in advance.

    Parameters
    ----------
    tau_sq:
        Prior variance for the mixture prior on the mean-difference
        parameter delta.  Larger values make the test more sensitive to
        large effects at the cost of slower convergence for small effects.
        A reasonable default is the expected variance of the metric being
        tracked (default 1.0).
    pooled_variance:
        Known or estimated per-observation variance sigma^2.  If ``None``
        (default), it is estimated online from the running sample variance
        of both groups combined.

    Attributes
    ----------
    _obs_control:
        List of raw observations for the control group.
    _obs_treatment:
        List of raw observations for the treatment group.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> tester = SequentialTester(tau_sq=1.0)
    >>> for _ in range(200):
    ...     tester.update(rng.normal(10.0, 2.0), group="control")
    ...     tester.update(rng.normal(10.5, 2.0), group="treatment")
    >>> result = tester.get_result()
    >>> print(f"p-value: {result.p_value:.4f}, stop: {result.should_stop}")
    """

    tau_sq: float = 1.0
    pooled_variance: float | None = None

    _obs_control: list[float] = field(default_factory=list, init=False, repr=False)
    _obs_treatment: list[float] = field(default_factory=list, init=False, repr=False)

    # Running Welford accumulators for online variance estimation
    _n_ctrl: int = field(default=0, init=False, repr=False)
    _mean_ctrl: float = field(default=0.0, init=False, repr=False)
    _M2_ctrl: float = field(default=0.0, init=False, repr=False)

    _n_trt: int = field(default=0, init=False, repr=False)
    _mean_trt: float = field(default=0.0, init=False, repr=False)
    _M2_trt: float = field(default=0.0, init=False, repr=False)

    def update(self, observation: float, group: str) -> None:
        """Incorporate a new observation into the running statistics.

        Parameters
        ----------
        observation:
            Numeric metric value for the new observation (e.g. revenue,
            conversion indicator, units_sold).
        group:
            Either ``"control"`` or ``"treatment"``.

        Raises
        ------
        ValueError
            If ``group`` is not ``"control"`` or ``"treatment"``.
        """
        group = group.lower().strip()
        if group not in ("control", "treatment"):
            raise ValueError(f"group must be 'control' or 'treatment', got '{group}'.")

        x = float(observation)

        if group == "control":
            self._obs_control.append(x)
            self._n_ctrl, self._mean_ctrl, self._M2_ctrl = self._welford_update(
                self._n_ctrl, self._mean_ctrl, self._M2_ctrl, x
            )
        else:
            self._obs_treatment.append(x)
            self._n_trt, self._mean_trt, self._M2_trt = self._welford_update(
                self._n_trt, self._mean_trt, self._M2_trt, x
            )

    def update_batch(self, observations: NDArray[np.float64] | list[float], group: str) -> None:
        """Incorporate a batch of observations at once.

        Parameters
        ----------
        observations:
            1-D array of numeric metric values.
        group:
            Either ``"control"`` or ``"treatment"``.
        """
        for obs in observations:
            self.update(float(obs), group)

    def get_pvalue(self) -> float:
        """Compute the current always-valid p-value.

        Returns the mixture likelihood ratio p-value:

            p_value = min(1.0, 1.0 / Lambda_t)

        Returns
        -------
        float
            Always-valid p-value in ``[0, 1]``.  Returns ``1.0`` if either
            group has fewer than 2 observations.
        """
        lambda_t = self._compute_lambda()
        if lambda_t <= 0:
            return 1.0
        return float(min(1.0, 1.0 / lambda_t))

    def should_stop(self, alpha: float = 0.05) -> bool:
        """Return True if the experiment has crossed the stopping boundary.

        The always-valid stopping rule is:

            stop when  Lambda_t >= 1 / alpha  (equivalently, p_value <= alpha)

        Parameters
        ----------
        alpha:
            Significance level (default 0.05).

        Returns
        -------
        bool
            ``True`` if the current evidence is sufficient to reject H0
            while controlling the familywise Type-I error at ``alpha``.
        """
        lambda_t = self._compute_lambda()
        threshold = 1.0 / alpha
        return bool(lambda_t >= threshold)

    def get_confidence_sequence(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute the always-valid confidence sequence for the mean difference.

        Returns an interval ``(lower, upper)`` such that the probability that
        the true delta lies outside *any* element of the sequence is at most
        ``alpha``, regardless of when peeking occurs.

        The boundary is derived by inverting the mSPRT at level ``alpha``:

            half_width = sqrt( (V / t) * (V + t*tau^2) / (V * tau^2)
                               * 2 * log(1 / alpha) )

        Parameters
        ----------
        alpha:
            Familywise error rate (default 0.05).

        Returns
        -------
        tuple[float, float]
            ``(lower, upper)`` confidence sequence bounds for
            (mean_treatment - mean_control).
        """
        if self._n_ctrl < 2 or self._n_trt < 2:
            return (-np.inf, np.inf)

        t = min(self._n_ctrl, self._n_trt)
        V = self._get_pooled_variance()
        delta_hat = self._mean_trt - self._mean_ctrl

        # Invert the mSPRT likelihood ratio at level alpha
        numerator = (V / t) * (V + t * self.tau_sq)
        denominator = V * self.tau_sq

        log_threshold = np.log(1.0 / alpha)
        half_width_sq = (numerator / denominator) * 2.0 * log_threshold

        if half_width_sq < 0:
            return (-np.inf, np.inf)

        half_width = float(np.sqrt(half_width_sq))
        return (delta_hat - half_width, delta_hat + half_width)

    def get_result(self, alpha: float = 0.05) -> SequentialResult:
        """Return a full snapshot of the current test state.

        Parameters
        ----------
        alpha:
            Significance level used for the stopping rule and confidence
            sequence (default 0.05).

        Returns
        -------
        SequentialResult
            Named tuple with all current statistics.
        """
        lambda_t = self._compute_lambda()
        p_val = float(min(1.0, 1.0 / lambda_t)) if lambda_t > 0 else 1.0
        cs_lower, cs_upper = self.get_confidence_sequence(alpha=alpha)
        delta_hat = (self._mean_trt - self._mean_ctrl) if (self._n_ctrl > 0 and self._n_trt > 0) else 0.0

        return SequentialResult(
            n_control=self._n_ctrl,
            n_treatment=self._n_trt,
            mean_control=self._mean_ctrl,
            mean_treatment=self._mean_trt,
            observed_delta=float(delta_hat),
            p_value=p_val,
            lambda_ratio=float(lambda_t),
            confidence_sequence_lower=cs_lower,
            confidence_sequence_upper=cs_upper,
            should_stop=bool(lambda_t >= 1.0 / alpha),
        )

    def reset(self) -> None:
        """Reset all accumulated observations and statistics."""
        self._obs_control = []
        self._obs_treatment = []
        self._n_ctrl = 0
        self._mean_ctrl = 0.0
        self._M2_ctrl = 0.0
        self._n_trt = 0
        self._mean_trt = 0.0
        self._M2_trt = 0.0

    def summary(self) -> str:
        """Return a human-readable summary of the current test state.

        Returns
        -------
        str
            Multi-line summary string.
        """
        result = self.get_result()
        return (
            "SequentialTester (mSPRT) Summary\n"
            "=================================\n"
            f"  Control   n={result.n_control:,}  mean={result.mean_control:.4f}\n"
            f"  Treatment n={result.n_treatment:,}  mean={result.mean_treatment:.4f}\n"
            f"  Delta hat        : {result.observed_delta:+.4f}\n"
            f"  Lambda_t         : {result.lambda_ratio:.4f}\n"
            f"  Always-valid p   : {result.p_value:.4f}\n"
            f"  95% CS           : [{result.confidence_sequence_lower:.4f}, "
            f"{result.confidence_sequence_upper:.4f}]\n"
            f"  Should stop      : {result.should_stop}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _welford_update(n: int, mean: float, M2: float, x: float) -> tuple[int, float, float]:
        """One-pass Welford online algorithm for mean and variance.

        Parameters
        ----------
        n, mean, M2:
            Current accumulators.
        x:
            New observation.

        Returns
        -------
        tuple[int, float, float]
            Updated ``(n, mean, M2)``.
        """
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        return n, mean, M2

    def _get_pooled_variance(self) -> float:
        """Return the pooled per-observation variance estimate.

        If ``pooled_variance`` was supplied at construction, return it.
        Otherwise compute a pooled sample variance from both groups.

        Returns
        -------
        float
            Estimated variance (always > 0 to avoid division by zero).
        """
        if self.pooled_variance is not None:
            return max(1e-12, float(self.pooled_variance))

        n_total = self._n_ctrl + self._n_trt

        if n_total < 2:
            return 1.0  # fallback before sufficient data

        # Pooled M2 using parallel Welford combination
        var_ctrl = self._M2_ctrl / (self._n_ctrl - 1) if self._n_ctrl > 1 else 0.0
        var_trt = self._M2_trt / (self._n_trt - 1) if self._n_trt > 1 else 0.0

        # Weighted average of the two sample variances
        if self._n_ctrl > 1 and self._n_trt > 1:
            pooled = (
                (self._n_ctrl - 1) * var_ctrl + (self._n_trt - 1) * var_trt
            ) / (self._n_ctrl + self._n_trt - 2)
        elif self._n_ctrl > 1:
            pooled = var_ctrl
        else:
            pooled = var_trt

        return max(1e-12, float(pooled))

    def _compute_lambda(self) -> float:
        """Compute the current mixture likelihood ratio Lambda_t.

        For normally distributed observations, the closed-form mSPRT
        statistic is:

            Lambda_t = sqrt(V / (V + t*tau^2))
                       * exp( t^2 * tau^2 * delta_hat^2
                              / (2 * V * (V + t*tau^2)) )

        where ``t = min(n_ctrl, n_trt)`` is the smaller group size and
        ``V`` is the pooled variance per observation.

        Returns
        -------
        float
            Lambda_t (>= 0). Returns 0.0 if there is insufficient data.
        """
        if self._n_ctrl < 2 or self._n_trt < 2:
            return 0.0

        t = float(min(self._n_ctrl, self._n_trt))
        V = self._get_pooled_variance()
        delta_hat = self._mean_trt - self._mean_ctrl
        tau_sq = max(1e-12, self.tau_sq)

        denom = V + t * tau_sq

        # log-domain to avoid overflow for large t
        log_scale = 0.5 * (np.log(V) - np.log(denom))
        log_exp = (t ** 2 * tau_sq * delta_hat ** 2) / (2.0 * V * denom)

        log_lambda = log_scale + log_exp

        # Clamp to prevent overflow on exp
        log_lambda = float(np.clip(log_lambda, -500.0, 500.0))

        return float(np.exp(log_lambda))
