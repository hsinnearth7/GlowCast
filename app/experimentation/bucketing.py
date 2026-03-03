"""Deterministic hash-based experiment bucketing with SRM detection.

Provides reproducible, consistent unit-of-randomization assignment for
GlowCast A/B experiments using SHA-256 hashing:

    bucket = int(SHA-256(salt + unit_id), 16) % n_buckets

Properties:
- **Deterministic**: the same (unit_id, salt, n_buckets) triple always
  maps to the same bucket.
- **Uniform**: SHA-256 output is cryptographically uniform over ``[0, 2^256)``,
  so the modulo distribution is very close to uniform for practical
  n_buckets values.
- **Isolated**: using unique salts per experiment layer prevents users
  from being correlated across simultaneous experiments.

Sample Ratio Mismatch (SRM) detection uses Pearson's chi-squared test to
check whether the observed bucket distribution matches the expected
distribution under the null hypothesis of no traffic manipulation:

    chi2 = sum((observed_i - expected_i)^2 / expected_i)

An SRM at p < 0.01 is a red flag indicating logging bugs, self-selection
bias, or implementation errors that can invalidate experiment conclusions.

Reference (SRM)
---------------
Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer, L.,
& Dmitriev, P. (2019).  Diagnosing Sample Ratio Mismatch in Online
Controlled Experiments: A Taxonomy and Rules of Thumb for Practitioners.
In Proceedings of the 25th ACM SIGKDD, pp. 2156-2164.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class BucketAssignment(NamedTuple):
    """Result of a single unit's bucket assignment.

    Attributes
    ----------
    unit_id:
        The original unit identifier string.
    bucket:
        Assigned bucket number in ``[0, n_buckets)``.
    group:
        Experiment group label (e.g. ``"control"``, ``"treatment"``, or
        ``"holdout"``).  ``None`` if the unit is not in any experiment
        (traffic allocation < 100 %).
    layer:
        Experiment layer name (for multi-layer isolation).
    salt:
        Salt string used in the hash.
    """

    unit_id: str
    bucket: int
    group: str | None
    layer: str
    salt: str


class SRMResult(NamedTuple):
    """Result of a Sample Ratio Mismatch chi-squared test.

    Attributes
    ----------
    chi2_statistic:
        Observed Pearson chi-squared statistic.
    p_value:
        Corresponding p-value.
    degrees_of_freedom:
        Degrees of freedom = len(observed_counts) - 1.
    observed_counts:
        Raw observed counts per group.
    expected_counts:
        Expected counts under the null (proportional to expected_ratio).
    is_srm:
        ``True`` if ``p_value < alpha``, indicating a likely SRM.
    alpha:
        Significance threshold used.
    """

    chi2_statistic: float
    p_value: float
    degrees_of_freedom: int
    observed_counts: list[int]
    expected_counts: list[float]
    is_srm: bool
    alpha: float


@dataclass
class BucketingAssigner:
    """Deterministic SHA-256 hash bucketing for A/B experiments.

    Supports single-unit assignment, batch assignment, traffic splitting,
    and layer isolation.  SRM detection is provided via chi-squared test.

    Parameters
    ----------
    n_buckets:
        Total number of hash buckets (default 1000).  A larger value gives
        finer-grained traffic allocation control.
    default_salt:
        Default salt string prepended to unit IDs before hashing.  Set a
        unique value per experiment to prevent cross-experiment correlation.
    layers:
        Dictionary mapping layer names to their respective salt overrides.
        Enables independent randomisation across concurrent experiment layers.
        Example: ``{"pricing": "layer_pricing_v1", "reco": "layer_reco_v2"}``

    Examples
    --------
    >>> assigner = BucketingAssigner(n_buckets=1000, default_salt="exp_001")
    >>> assignment = assigner.assign_bucket("user_12345")
    >>> print(f"bucket={assignment.bucket}")
    >>> group = assigner.assign_group(
    ...     "user_12345",
    ...     group_splits={"control": 0.50, "treatment": 0.50},
    ... )
    >>> print(f"group={group}")
    """

    n_buckets: int = 1000
    default_salt: str = ""
    layers: dict[str, str] = field(default_factory=dict)

    def assign_bucket(
        self,
        unit_id: str,
        n_buckets: int | None = None,
        salt: str | None = None,
        layer: str = "default",
    ) -> BucketAssignment:
        """Assign a single unit to a deterministic bucket.

        Computes:

            bucket = int(SHA-256(effective_salt + unit_id).hexdigest(), 16)
                     % n_buckets

        Parameters
        ----------
        unit_id:
            String identifier for the randomisation unit (e.g. user_id,
            session_id, sku_id).
        n_buckets:
            Override the instance-level n_buckets for this call.
        salt:
            Override salt for this call.  If ``None``, uses the layer-specific
            salt (if ``layer`` is in ``self.layers``) or ``self.default_salt``.
        layer:
            Experiment layer name for isolation.  If the layer is registered
            in ``self.layers``, its salt is used automatically.

        Returns
        -------
        BucketAssignment
            Named tuple with bucket number and metadata.

        Raises
        ------
        TypeError
            If ``unit_id`` is not a string.
        ValueError
            If ``n_buckets`` is not a positive integer.
        """
        if not isinstance(unit_id, str):
            raise TypeError(f"unit_id must be a string, got {type(unit_id).__name__}.")

        effective_n = n_buckets if n_buckets is not None else self.n_buckets
        if effective_n < 1:
            raise ValueError(f"n_buckets must be >= 1, got {effective_n}.")

        effective_salt = self._resolve_salt(salt, layer)
        bucket = self._hash_to_bucket(unit_id, effective_n, effective_salt)

        return BucketAssignment(
            unit_id=unit_id,
            bucket=bucket,
            group=None,
            layer=layer,
            salt=effective_salt,
        )

    def assign_group(
        self,
        unit_id: str,
        group_splits: dict[str, float],
        salt: str | None = None,
        layer: str = "default",
    ) -> str | None:
        """Assign a unit to a named experiment group based on traffic splits.

        Groups are allocated contiguous bucket ranges according to their
        specified traffic proportions.  The splits need not sum to 1.0; any
        remaining traffic is treated as holdout (returns ``None``).

        Example: ``group_splits = {"control": 0.40, "treatment": 0.40}``
        leaves 20 % as holdout.

        Parameters
        ----------
        unit_id:
            Randomisation unit identifier.
        group_splits:
            Ordered mapping of group names to traffic fractions.  The
            insertion order of the dict defines the bucket range ordering.
        salt:
            Optional salt override.
        layer:
            Experiment layer name.

        Returns
        -------
        str or None
            Group name, or ``None`` if the unit is in the holdout traffic
            (i.e. not assigned to any experiment group).

        Raises
        ------
        ValueError
            If any individual split is <= 0 or the total exceeds 1.0.
        """
        for name, fraction in group_splits.items():
            if fraction <= 0:
                raise ValueError(f"Split for group '{name}' must be > 0, got {fraction}.")

        total = sum(group_splits.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Total traffic split {total:.4f} exceeds 1.0. "
                "Reduce group proportions or leave remainder as holdout."
            )

        effective_salt = self._resolve_salt(salt, layer)
        bucket = self._hash_to_bucket(unit_id, self.n_buckets, effective_salt)

        # Map the [0, n_buckets) bucket to [0.0, 1.0) fractional position
        position = bucket / self.n_buckets
        cumulative = 0.0

        for group_name, fraction in group_splits.items():
            cumulative += fraction
            if position < cumulative:
                return group_name

        return None  # holdout

    def assign_batch(
        self,
        unit_ids: list[str],
        group_splits: dict[str, float] | None = None,
        salt: str | None = None,
        layer: str = "default",
    ) -> list[BucketAssignment]:
        """Assign a list of units to buckets (and optionally groups) in bulk.

        Parameters
        ----------
        unit_ids:
            List of unit identifier strings.
        group_splits:
            Optional group traffic splits.  If provided, each assignment's
            ``group`` field is populated.
        salt:
            Optional salt override.
        layer:
            Experiment layer name.

        Returns
        -------
        list[BucketAssignment]
            One BucketAssignment per unit, in the same order as ``unit_ids``.
        """
        effective_salt = self._resolve_salt(salt, layer)
        results: list[BucketAssignment] = []

        for uid in unit_ids:
            bucket = self._hash_to_bucket(uid, self.n_buckets, effective_salt)

            group: str | None = None
            if group_splits is not None:
                position = bucket / self.n_buckets
                cumulative = 0.0
                for group_name, fraction in group_splits.items():
                    cumulative += fraction
                    if position < cumulative:
                        group = group_name
                        break

            results.append(
                BucketAssignment(
                    unit_id=uid,
                    bucket=bucket,
                    group=group,
                    layer=layer,
                    salt=effective_salt,
                )
            )

        return results

    def check_srm(
        self,
        observed_counts: list[int] | NDArray[np.int64],
        expected_ratio: list[float] | NDArray[np.float64],
        alpha: float = 0.01,
    ) -> SRMResult:
        """Perform a chi-squared Sample Ratio Mismatch (SRM) test.

        Tests whether the observed unit counts per group match the intended
        traffic allocation under H0: no mismatch.

        Parameters
        ----------
        observed_counts:
            Observed number of units in each group.  Length must match
            ``expected_ratio``.
        expected_ratio:
            Intended traffic proportions for each group (need not sum to 1;
            they are normalised internally).
        alpha:
            Significance level for the SRM flag (default 0.01, as GlowCast
            uses a stricter threshold than typical hypothesis tests to avoid
            false SRM alerts).

        Returns
        -------
        SRMResult
            Named tuple with test statistic, p-value, expected counts, and
            SRM flag.

        Raises
        ------
        ValueError
            If ``observed_counts`` and ``expected_ratio`` have different
            lengths, any count is negative, or any ratio is non-positive.
        """
        obs = np.asarray(observed_counts, dtype=np.float64)
        exp_ratio = np.asarray(expected_ratio, dtype=np.float64)

        if obs.ndim != 1 or exp_ratio.ndim != 1:
            raise ValueError("observed_counts and expected_ratio must be 1-D arrays.")
        if len(obs) != len(exp_ratio):
            raise ValueError(
                f"observed_counts (len={len(obs)}) and expected_ratio (len={len(exp_ratio)}) "
                "must have the same length."
            )
        if np.any(obs < 0):
            raise ValueError("All observed_counts must be non-negative.")
        if np.any(exp_ratio <= 0):
            raise ValueError("All expected_ratio values must be > 0.")

        n_total = float(np.sum(obs))
        if n_total == 0:
            raise ValueError("Total observed count is 0; nothing to test.")

        # Normalise expected ratios to probabilities
        exp_prob = exp_ratio / exp_ratio.sum()
        expected_counts = exp_prob * n_total

        # Pearson chi-squared statistic
        chi2_stat = float(np.sum((obs - expected_counts) ** 2 / expected_counts))
        dof = len(obs) - 1
        p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=dof))

        return SRMResult(
            chi2_statistic=chi2_stat,
            p_value=p_value,
            degrees_of_freedom=dof,
            observed_counts=[int(x) for x in obs],
            expected_counts=[float(x) for x in expected_counts],
            is_srm=bool(p_value < alpha),
            alpha=alpha,
        )

    def get_bucket_distribution(
        self,
        unit_ids: list[str],
        salt: str | None = None,
        layer: str = "default",
    ) -> NDArray[np.int64]:
        """Return the bucket number for each unit in ``unit_ids``.

        Useful for visualising or auditing the empirical bucket distribution
        before running an experiment.

        Parameters
        ----------
        unit_ids:
            List of unit identifiers.
        salt:
            Optional salt override.
        layer:
            Experiment layer name.

        Returns
        -------
        NDArray[np.int64]
            Integer array of bucket numbers, shape ``(len(unit_ids),)``.
        """
        effective_salt = self._resolve_salt(salt, layer)
        return np.array(
            [self._hash_to_bucket(uid, self.n_buckets, effective_salt) for uid in unit_ids],
            dtype=np.int64,
        )

    def register_layer(self, layer_name: str, layer_salt: str) -> None:
        """Register an experiment layer with its dedicated salt.

        Registering a layer guarantees that all experiments within that
        layer use the same salt, providing full orthogonality against other
        layers.

        Parameters
        ----------
        layer_name:
            Human-readable layer identifier (e.g. ``"pricing"``, ``"reco"``).
        layer_salt:
            Unique salt string for the layer (e.g. ``"pricing_layer_v3"``).
        """
        self.layers[layer_name] = layer_salt

    def summary(self, unit_ids: list[str], salt: str | None = None, layer: str = "default") -> str:
        """Return a human-readable summary of the bucket distribution.

        Parameters
        ----------
        unit_ids:
            List of unit IDs to summarise.
        salt:
            Optional salt override.
        layer:
            Experiment layer name.

        Returns
        -------
        str
            Summary string including uniformity statistics.
        """
        buckets = self.get_bucket_distribution(unit_ids, salt=salt, layer=layer)
        counts = np.bincount(buckets, minlength=self.n_buckets)
        effective_salt = self._resolve_salt(salt, layer)

        expected_per_bucket = len(unit_ids) / self.n_buckets
        max_deviation_pct = float(
            np.max(np.abs(counts - expected_per_bucket)) / max(expected_per_bucket, 1e-9) * 100
        )
        chi2_stat, p_uniform = stats.chisquare(counts)

        return (
            "BucketingAssigner Distribution Summary\n"
            "=======================================\n"
            f"  Units            : {len(unit_ids):,}\n"
            f"  n_buckets        : {self.n_buckets:,}\n"
            f"  Layer            : {layer}\n"
            f"  Salt             : '{effective_salt}'\n"
            f"  Expected / bucket: {expected_per_bucket:.1f}\n"
            f"  Min bucket count : {int(counts.min()):,}\n"
            f"  Max bucket count : {int(counts.max()):,}\n"
            f"  Max deviation    : {max_deviation_pct:.2f}%\n"
            f"  Uniformity chi2  : {chi2_stat:.2f} (p={p_uniform:.4f})\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_salt(self, salt: str | None, layer: str) -> str:
        """Resolve the effective salt for a given call.

        Priority: explicit ``salt`` argument > layer-registered salt >
        ``self.default_salt``.
        """
        if salt is not None:
            return salt
        if layer in self.layers:
            return self.layers[layer]
        return self.default_salt

    @staticmethod
    def _hash_to_bucket(unit_id: str, n_buckets: int, salt: str) -> int:
        """Compute deterministic bucket via SHA-256.

        Implements:

            int(hashlib.sha256((salt + unit_id).encode()).hexdigest(), 16)
            % n_buckets

        Parameters
        ----------
        unit_id:
            Unit identifier string.
        n_buckets:
            Number of buckets.
        salt:
            Salt prepended to ``unit_id`` before hashing.

        Returns
        -------
        int
            Bucket number in ``[0, n_buckets)``.
        """
        hash_input = (salt + unit_id).encode("utf-8")
        digest = hashlib.sha256(hash_input).hexdigest()
        return int(digest, 16) % n_buckets
