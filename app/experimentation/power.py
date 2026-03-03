"""Sample size and Minimum Detectable Effect (MDE) calculations.

Provides tools for experiment planning at GlowCast, including:

- **required_sample_size**: Two-sample z-test sample size for a given MDE,
  baseline mean, baseline std, alpha and power.
- **mde_table**: Tabulate required sample sizes for multiple MDE levels
  (3 %, 5 %, 10 %) with and without CUPED variance reduction.
- **cuped_adjusted_n**: Scale raw sample sizes down by the CUPED factor
  ``1 / (1 - rho^2)``.

Sample size formula (two-sided, two-sample):

    n = (z_alpha/2 + z_beta)^2 * 2 * sigma^2 / delta^2

where:
    delta   = mde * baseline_mean          (absolute effect size)
    sigma   = baseline_std                 (per-observation std)
    z_alpha/2 = quantile from N(0,1) for significance level alpha
    z_beta    = quantile from N(0,1) for power (1 - beta)

CUPED-adjusted sample size:

    n_cuped = ceil(n_raw * (1 - rho^2))

Because CUPED reduces per-observation variance by factor ``(1 - rho^2)``,
fewer observations are needed to achieve the same statistical power.

At GlowCast's target rho = 0.74 the CUPED factor is ≈ 0.452, roughly
halving the required sample size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from scipy import stats


class SampleSizeResult(NamedTuple):
    """Result of a single sample size calculation.

    Attributes
    ----------
    mde_relative:
        Minimum detectable effect as a fraction of the baseline mean.
    mde_absolute:
        Absolute effect size (mde_relative * baseline_mean).
    n_per_group_raw:
        Required observations per group (no variance reduction).
    n_total_raw:
        Total required observations across both groups.
    n_per_group_cuped:
        Required observations per group after CUPED adjustment.
    n_total_cuped:
        Total required CUPED-adjusted observations.
    cuped_reduction_pct:
        Percentage reduction in sample size due to CUPED.
    alpha:
        Significance level used.
    power:
        Statistical power used.
    rho:
        Assumed CUPED correlation coefficient.
    """

    mde_relative: float
    mde_absolute: float
    n_per_group_raw: int
    n_total_raw: int
    n_per_group_cuped: int
    n_total_cuped: int
    cuped_reduction_pct: float
    alpha: float
    power: float
    rho: float


class MDETableRow(NamedTuple):
    """One row of the MDE summary table.

    Attributes
    ----------
    mde_pct:
        MDE as a percentage of the baseline mean (e.g. 3.0, 5.0, 10.0).
    n_per_group_raw:
        Raw required sample size per group.
    n_total_raw:
        Raw total sample size (both groups).
    n_per_group_cuped:
        CUPED-adjusted sample size per group.
    n_total_cuped:
        CUPED-adjusted total sample size.
    experiment_days_raw:
        Estimated days to collect n_total_raw observations (if daily_traffic given).
    experiment_days_cuped:
        Estimated days to collect n_total_cuped observations.
    """

    mde_pct: float
    n_per_group_raw: int
    n_total_raw: int
    n_per_group_cuped: int
    n_total_cuped: int
    experiment_days_raw: float | None
    experiment_days_cuped: float | None


@dataclass
class PowerAnalyzer:
    """Sample size and MDE planning for GlowCast A/B experiments.

    Parameters
    ----------
    rho:
        Expected Pearson correlation between the pre-experiment covariate
        and the post-experiment metric (used for CUPED adjustment).
        Default is GlowCast's target value of 0.74.

    Examples
    --------
    >>> analyzer = PowerAnalyzer(rho=0.74)
    >>> result = analyzer.required_sample_size(
    ...     baseline_mean=50.0,
    ...     baseline_std=25.0,
    ...     mde=0.05,
    ... )
    >>> print(f"Raw n/group: {result.n_per_group_raw:,}")
    >>> print(f"CUPED n/group: {result.n_per_group_cuped:,}")
    >>> table = analyzer.mde_table(baseline_mean=50.0, baseline_std=25.0)
    """

    rho: float = 0.74

    def required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> SampleSizeResult:
        """Compute the required per-group sample size for detecting an MDE.

        Uses the standard two-sided, two-sample z-test formula.  The
        CUPED-adjusted size accounts for the variance reduction achieved
        when rho > 0.

        Parameters
        ----------
        baseline_mean:
            Mean of the control metric (e.g. mean units_sold per user per day).
        baseline_std:
            Standard deviation of the metric in the control group.
        mde:
            Minimum detectable effect expressed as a fraction of the baseline
            mean (e.g. 0.05 for a 5 % lift).
        alpha:
            Two-sided significance level (default 0.05).
        power:
            Desired statistical power, 1 - beta (default 0.80).

        Returns
        -------
        SampleSizeResult
            Named tuple containing raw and CUPED-adjusted sample sizes.

        Raises
        ------
        ValueError
            If ``baseline_mean <= 0``, ``baseline_std <= 0``, ``mde <= 0``,
            ``alpha`` or ``power`` are out of (0, 1), or ``rho`` is out of
            ``(-1, 1)``.
        """
        self._validate_inputs(baseline_mean, baseline_std, mde, alpha, power)

        delta_abs = float(mde * baseline_mean)
        sigma = float(baseline_std)

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))

        # Two-sample z-test: n = (z_a/2 + z_b)^2 * 2 * sigma^2 / delta^2
        n_raw_exact = (z_alpha + z_beta) ** 2 * 2.0 * sigma ** 2 / delta_abs ** 2
        n_per_group_raw = math.ceil(n_raw_exact)
        n_total_raw = 2 * n_per_group_raw

        n_per_group_cuped, n_total_cuped = self.cuped_adjusted_n(n_per_group_raw, self.rho)
        cuped_reduction_pct = float((1.0 - n_total_cuped / n_total_raw) * 100.0)

        return SampleSizeResult(
            mde_relative=mde,
            mde_absolute=delta_abs,
            n_per_group_raw=n_per_group_raw,
            n_total_raw=n_total_raw,
            n_per_group_cuped=n_per_group_cuped,
            n_total_cuped=n_total_cuped,
            cuped_reduction_pct=cuped_reduction_pct,
            alpha=alpha,
            power=power,
            rho=self.rho,
        )

    def mde_table(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde_levels: list[float] | None = None,
        alpha: float = 0.05,
        power: float = 0.80,
        daily_traffic: int | None = None,
    ) -> list[MDETableRow]:
        """Produce an MDE planning table for multiple effect sizes.

        Computes raw and CUPED-adjusted sample sizes for each MDE level.
        If ``daily_traffic`` is provided, also estimates experiment duration
        in calendar days.

        Parameters
        ----------
        baseline_mean:
            Control group metric mean.
        baseline_std:
            Control group metric standard deviation.
        mde_levels:
            List of relative MDE fractions to tabulate.
            Defaults to ``[0.03, 0.05, 0.10]`` (3 %, 5 %, 10 %).
        alpha:
            Significance level (default 0.05).
        power:
            Statistical power (default 0.80).
        daily_traffic:
            Total daily observations across both groups combined.  When
            provided, experiment duration estimates (in days) are included.

        Returns
        -------
        list[MDETableRow]
            One row per MDE level, sorted by MDE ascending.

        Raises
        ------
        ValueError
            If any MDE level is not in (0, 1).
        """
        if mde_levels is None:
            mde_levels = [0.03, 0.05, 0.10]

        for m in mde_levels:
            if not (0 < m < 1):
                raise ValueError(f"All MDE levels must be in (0, 1); got {m}.")

        rows: list[MDETableRow] = []

        for mde in sorted(mde_levels):
            result = self.required_sample_size(
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                mde=mde,
                alpha=alpha,
                power=power,
            )

            if daily_traffic is not None and daily_traffic > 0:
                days_raw = math.ceil(result.n_total_raw / daily_traffic)
                days_cuped = math.ceil(result.n_total_cuped / daily_traffic)
            else:
                days_raw = None
                days_cuped = None

            rows.append(
                MDETableRow(
                    mde_pct=round(mde * 100, 1),
                    n_per_group_raw=result.n_per_group_raw,
                    n_total_raw=result.n_total_raw,
                    n_per_group_cuped=result.n_per_group_cuped,
                    n_total_cuped=result.n_total_cuped,
                    experiment_days_raw=float(days_raw) if days_raw is not None else None,
                    experiment_days_cuped=float(days_cuped) if days_cuped is not None else None,
                )
            )

        return rows

    def cuped_adjusted_n(self, n_raw: int, rho: float) -> tuple[int, int]:
        """Compute CUPED-adjusted per-group and total sample sizes.

        CUPED reduces per-observation variance by a factor of ``(1 - rho^2)``,
        so the required sample size scales proportionally:

            n_cuped = ceil(n_raw * (1 - rho^2))

        Parameters
        ----------
        n_raw:
            Raw (unadjusted) required sample size per group.
        rho:
            Pearson correlation between the pre-experiment covariate and
            the post-experiment metric.

        Returns
        -------
        tuple[int, int]
            ``(n_per_group_cuped, n_total_cuped)``.

        Raises
        ------
        ValueError
            If ``rho`` is outside ``(-1, 1)`` or ``n_raw < 1``.
        """
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in the open interval (-1, 1), got {rho}.")
        if n_raw < 1:
            raise ValueError(f"n_raw must be >= 1, got {n_raw}.")

        cuped_factor = max(0.0, 1.0 - rho ** 2)
        n_per_group_cuped = max(1, math.ceil(n_raw * cuped_factor))
        n_total_cuped = 2 * n_per_group_cuped

        return n_per_group_cuped, n_total_cuped

    def format_mde_table(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde_levels: list[float] | None = None,
        alpha: float = 0.05,
        power: float = 0.80,
        daily_traffic: int | None = None,
    ) -> str:
        """Return the MDE table formatted as a plain-text string.

        Parameters
        ----------
        baseline_mean:
            Control group metric mean.
        baseline_std:
            Control group metric standard deviation.
        mde_levels:
            MDE fractions to tabulate (default [0.03, 0.05, 0.10]).
        alpha:
            Significance level (default 0.05).
        power:
            Statistical power (default 0.80).
        daily_traffic:
            If provided, include estimated experiment duration in days.

        Returns
        -------
        str
            ASCII table string.
        """
        rows = self.mde_table(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde_levels=mde_levels,
            alpha=alpha,
            power=power,
            daily_traffic=daily_traffic,
        )

        header_parts = [
            f"{'MDE%':>6}",
            f"{'n/grp raw':>12}",
            f"{'n total raw':>13}",
            f"{'n/grp cuped':>13}",
            f"{'n total cuped':>15}",
        ]
        if daily_traffic is not None:
            header_parts += [f"{'days raw':>10}", f"{'days cuped':>12}"]

        header = "  ".join(header_parts)
        sep = "-" * len(header)

        lines = [
            f"MDE Planning Table  (baseline_mean={baseline_mean}, "
            f"baseline_std={baseline_std}, alpha={alpha}, power={power}, rho={self.rho})",
            sep,
            header,
            sep,
        ]

        for row in rows:
            parts = [
                f"{row.mde_pct:>6.1f}",
                f"{row.n_per_group_raw:>12,}",
                f"{row.n_total_raw:>13,}",
                f"{row.n_per_group_cuped:>13,}",
                f"{row.n_total_cuped:>15,}",
            ]
            if daily_traffic is not None:
                days_r = f"{int(row.experiment_days_raw):>10,}" if row.experiment_days_raw is not None else f"{'N/A':>10}"
                days_c = f"{int(row.experiment_days_cuped):>12,}" if row.experiment_days_cuped is not None else f"{'N/A':>12}"
                parts += [days_r, days_c]
            lines.append("  ".join(parts))

        lines.append(sep)
        lines.append(
            f"  CUPED factor: 1 - rho^2 = 1 - {self.rho}^2 = {1 - self.rho**2:.4f}  "
            f"(~{(1 - (1 - self.rho**2)) * 100:.1f}% sample size reduction)"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        alpha: float,
        power: float,
    ) -> None:
        """Validate all inputs and raise ValueError if any are invalid."""
        if baseline_mean <= 0:
            raise ValueError(f"baseline_mean must be > 0, got {baseline_mean}.")
        if baseline_std <= 0:
            raise ValueError(f"baseline_std must be > 0, got {baseline_std}.")
        if not (0 < mde < 1):
            raise ValueError(f"mde must be in (0, 1), got {mde}.")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0 < power < 1):
            raise ValueError(f"power must be in (0, 1), got {power}.")
