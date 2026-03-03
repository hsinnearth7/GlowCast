"""Team Draft Interleaving for online ranking evaluation.

Interleaving is an efficient method for comparing two ranking systems
(e.g. recommendation algorithms, search result rankers) by combining their
output lists into a single interleaved list shown to users.  Clicks on
interleaved results are attributed to the team (A or B) that first placed
each document, and the winning ranker is determined by which team
accumulated more clicks.

Team Draft interleaving (Chapelle et al., 2012) is the de-facto standard
because it satisfies:

- **Balanced**: Each team receives approximately the same number of
  documents per interleaved list.
- **Unbiased**: Under position-dependent click models, the expected
  winning margin is 0 when both rankers are identical.
- **Sensitive**: High statistical power for detecting subtle ranking
  differences.

GlowCast application: compare the current SKU recommendation engine
(ranker A, baseline) against a new candidate ranker (ranker B) by
presenting interleaved product lists and tracking add-to-cart events.

Reference
---------
Chapelle, O., Joachims, T., Radlinski, F., & Yue, Y. (2012).  Large-scale
validation and analysis of interleaved search evaluation.  ACM Transactions
on Information Systems (TOIS), 30(1), 1-41.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from scipy import stats


class InterleavingResult(NamedTuple):
    """Outcome of a single interleaving trial or an aggregated experiment.

    Attributes
    ----------
    interleaved_list:
        The merged list of document IDs shown to the user.
    team_a:
        Set of document IDs attributed to ranker A.
    team_b:
        Set of document IDs attributed to ranker B.
    clicks_a:
        Number of clicks on documents belonging to team A.
    clicks_b:
        Number of clicks on documents belonging to team B.
    delta:
        Normalised win margin for ranker B:
        ``(clicks_b - clicks_a) / (clicks_a + clicks_b + 1e-9)``.
        Positive values favour ranker B; negative values favour ranker A.
    """

    interleaved_list: list[Any]
    team_a: set[Any]
    team_b: set[Any]
    clicks_a: int
    clicks_b: int
    delta: float


class ExperimentSummary(NamedTuple):
    """Aggregate summary across multiple interleaving trials.

    Attributes
    ----------
    n_trials:
        Number of query/session trials.
    mean_delta:
        Mean normalised win margin across trials.
    std_delta:
        Standard deviation of trial-level deltas.
    p_value:
        Two-tailed t-test p-value for H0: mean_delta = 0.
    t_statistic:
        Corresponding t-statistic.
    winner:
        ``"B"`` if mean_delta > 0 and statistically significant at alpha=0.05,
        ``"A"`` if mean_delta < 0 and significant, else ``"tie"``.
    total_clicks_a:
        Aggregate clicks attributed to ranker A.
    total_clicks_b:
        Aggregate clicks attributed to ranker B.
    """

    n_trials: int
    mean_delta: float
    std_delta: float
    p_value: float
    t_statistic: float
    winner: str
    total_clicks_a: int
    total_clicks_b: int


@dataclass
class InterleavingAnalyzer:
    """Team Draft interleaving analyzer for ranking comparison.

    Supports both single-trial team draft construction and full experiment
    orchestration over a corpus of query sessions with a simulated or
    observed click model.

    Parameters
    ----------
    random_seed:
        Seed for the NumPy RNG used to break ties during team assignment.
        Default 42.

    Examples
    --------
    >>> analyzer = InterleavingAnalyzer(random_seed=0)
    >>> list_a = [1, 2, 3, 4, 5]
    >>> list_b = [3, 1, 4, 2, 5]
    >>> interleaved, team_a, team_b = analyzer.team_draft(list_a, list_b, k=4)
    >>> clicks = {1, 3}   # user clicked items 1 and 3
    >>> result = analyzer.compute_delta(interleaved, clicks, team_a, team_b)
    >>> print(f"delta={result.delta:.3f}, winner={'B' if result.delta > 0 else 'A'}")
    """

    random_seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)

    def team_draft(
        self,
        ranking_a: list[Any],
        ranking_b: list[Any],
        k: int,
    ) -> tuple[list[Any], set[Any], set[Any]]:
        """Construct a Team Draft interleaved list from two ranked lists.

        The algorithm alternates turns between teams A and B.  On each
        turn, the team whose turn it is picks the highest-ranked document
        from its list that has not yet been added.  That document is also
        retrospectively credited to the *other* team if it appears at an
        equal or better position in the other team's ranking.

        Concretely:
        1. Flip a fair coin to decide which team goes first.
        2. The current team picks its top-remaining document ``d``.
        3. ``d`` is added to the interleaved list.
        4. ``d`` is added to the picking team's set.
        5. If ``d`` appears in the other team's list AND the other team
           has not already picked it, ``d`` is also added to both teams'
           sets (credit sharing for equally good documents).
        6. Alternate turns until ``k`` documents are collected.

        Parameters
        ----------
        ranking_a:
            Ordered list of document IDs from ranker A (best first).
        ranking_b:
            Ordered list of document IDs from ranker B (best first).
        k:
            Number of documents in the interleaved list.

        Returns
        -------
        tuple[list[Any], set[Any], set[Any]]
            - ``interleaved``: The k-document interleaved list.
            - ``team_a``: Document IDs credited to ranker A.
            - ``team_b``: Document IDs credited to ranker B.

        Raises
        ------
        ValueError
            If ``k`` exceeds the union size of both lists or is non-positive.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")

        max_possible = len(set(ranking_a) | set(ranking_b))
        if k > max_possible:
            raise ValueError(
                f"k={k} exceeds the number of unique documents in both lists "
                f"(union size={max_possible})."
            )

        # Build position lookup for fast rank retrieval
        rank_a: dict[Any, int] = {doc: pos for pos, doc in enumerate(ranking_a)}
        rank_b: dict[Any, int] = {doc: pos for pos, doc in enumerate(ranking_b)}

        interleaved: list[Any] = []
        team_a: set[Any] = set()
        team_b: set[Any] = set()
        already_placed: set[Any] = set()

        # Remaining candidates (preserve order)
        remaining_a = list(ranking_a)
        remaining_b = list(ranking_b)

        # Random coin flip to decide who goes first
        current_team = int(self._rng.integers(0, 2))  # 0 = A, 1 = B

        while len(interleaved) < k:
            if current_team == 0:
                # Team A picks
                doc = self._pick_top(remaining_a, already_placed)
                if doc is None:
                    break
                interleaved.append(doc)
                already_placed.add(doc)
                team_a.add(doc)
                # Credit B as well if B ranks doc equally or better among
                # documents not yet placed
                if doc in rank_b:
                    team_b.add(doc)
                current_team = 1
            else:
                # Team B picks
                doc = self._pick_top(remaining_b, already_placed)
                if doc is None:
                    break
                interleaved.append(doc)
                already_placed.add(doc)
                team_b.add(doc)
                if doc in rank_a:
                    team_a.add(doc)
                current_team = 0

        return interleaved, team_a, team_b

    def compute_delta(
        self,
        interleaved_list: list[Any],
        clicks: set[Any] | list[Any],
        team_a: set[Any],
        team_b: set[Any],
    ) -> InterleavingResult:
        """Compute the normalised win margin delta from a click observation.

        Clicks are attributed to the team(s) that contain the clicked
        document.  Documents credited to *both* teams (shared credit)
        contribute 0.5 to each team's click count.

        Delta is defined as:

            delta = (clicks_B - clicks_A) / (clicks_A + clicks_B + epsilon)

        A positive delta indicates that ranker B won the trial.

        Parameters
        ----------
        interleaved_list:
            The interleaved document list produced by :meth:`team_draft`.
        clicks:
            Set or list of document IDs that the user clicked.
        team_a:
            Document IDs attributed to ranker A.
        team_b:
            Document IDs attributed to ranker B.

        Returns
        -------
        InterleavingResult
            Result named tuple containing click counts and delta.
        """
        clicks_set = set(clicks)
        clicks_a_float = 0.0
        clicks_b_float = 0.0

        for doc in interleaved_list:
            if doc in clicks_set:
                in_a = doc in team_a
                in_b = doc in team_b
                if in_a and in_b:
                    clicks_a_float += 0.5
                    clicks_b_float += 0.5
                elif in_a:
                    clicks_a_float += 1.0
                elif in_b:
                    clicks_b_float += 1.0

        clicks_a = int(clicks_a_float * 2)   # stored as 2x for integer repr
        clicks_b = int(clicks_b_float * 2)

        denom = clicks_a_float + clicks_b_float + 1e-9
        delta = float((clicks_b_float - clicks_a_float) / denom)

        return InterleavingResult(
            interleaved_list=interleaved_list,
            team_a=team_a,
            team_b=team_b,
            clicks_a=clicks_a,
            clicks_b=clicks_b,
            delta=delta,
        )

    def run_experiment(
        self,
        rankings_a: list[list[Any]],
        rankings_b: list[list[Any]],
        click_model: Any,
        k: int = 10,
        alpha: float = 0.05,
    ) -> ExperimentSummary:
        """Run a full interleaving experiment over multiple query sessions.

        For each (ranking_a_i, ranking_b_i) pair:
        1. Construct the Team Draft interleaved list of length ``k``.
        2. Simulate clicks using ``click_model``.
        3. Compute and record the trial delta.

        Statistical significance is assessed via a two-tailed one-sample
        t-test of H0: mean_delta = 0.

        Parameters
        ----------
        rankings_a:
            List of per-session ranked lists from ranker A.  Length
            determines the number of trials.
        rankings_b:
            List of per-session ranked lists from ranker B.  Must have
            the same length as ``rankings_a``.
        click_model:
            Any callable with signature
            ``click_model(interleaved_list) -> set[Any]``
            returning a set of clicked document IDs.  Can be a
            position-based model, cascade model, or a replay of logged
            clicks.
        k:
            Number of documents per interleaved list (default 10).
        alpha:
            Significance level for the t-test (default 0.05).

        Returns
        -------
        ExperimentSummary
            Aggregate statistics across all trials.

        Raises
        ------
        ValueError
            If ``rankings_a`` and ``rankings_b`` have different lengths or
            either is empty.
        """
        if len(rankings_a) != len(rankings_b):
            raise ValueError(
                f"rankings_a (n={len(rankings_a)}) and rankings_b (n={len(rankings_b)}) "
                "must have the same number of sessions."
            )
        if not rankings_a:
            raise ValueError("At least one session is required to run the experiment.")

        deltas: list[float] = []
        total_clicks_a = 0
        total_clicks_b = 0

        for ra, rb in zip(rankings_a, rankings_b):
            interleaved, team_a, team_b = self.team_draft(ra, rb, k=min(k, len(set(ra) | set(rb))))
            clicks = click_model(interleaved)
            result = self.compute_delta(interleaved, clicks, team_a, team_b)
            deltas.append(result.delta)
            total_clicks_a += result.clicks_a
            total_clicks_b += result.clicks_b

        delta_arr = np.array(deltas, dtype=np.float64)
        mean_delta = float(np.mean(delta_arr))
        std_delta = float(np.std(delta_arr, ddof=1)) if len(delta_arr) > 1 else 0.0

        if len(delta_arr) > 1 and std_delta > 1e-12:
            t_stat, p_val = stats.ttest_1samp(delta_arr, popmean=0.0)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat, p_val = 0.0, 1.0

        if p_val <= alpha:
            winner = "B" if mean_delta > 0 else "A"
        else:
            winner = "tie"

        return ExperimentSummary(
            n_trials=len(deltas),
            mean_delta=mean_delta,
            std_delta=std_delta,
            p_value=p_val,
            t_statistic=t_stat,
            winner=winner,
            total_clicks_a=total_clicks_a,
            total_clicks_b=total_clicks_b,
        )

    def summary(self, experiment_result: ExperimentSummary) -> str:
        """Format an ExperimentSummary as a human-readable string.

        Parameters
        ----------
        experiment_result:
            Result from :meth:`run_experiment`.

        Returns
        -------
        str
            Multi-line summary.
        """
        res = experiment_result
        return (
            "Interleaving Experiment Summary (Team Draft)\n"
            "============================================\n"
            f"  Trials           : {res.n_trials:,}\n"
            f"  Total clicks A   : {res.total_clicks_a:,}\n"
            f"  Total clicks B   : {res.total_clicks_b:,}\n"
            f"  Mean delta       : {res.mean_delta:+.4f}\n"
            f"  Std delta        : {res.std_delta:.4f}\n"
            f"  t-statistic      : {res.t_statistic:+.4f}\n"
            f"  p-value          : {res.p_value:.4f}\n"
            f"  Winner           : {res.winner}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_top(ranked_list: list[Any], already_placed: set[Any]) -> Any | None:
        """Return the first document in ``ranked_list`` not in ``already_placed``.

        Parameters
        ----------
        ranked_list:
            Ordered list of document IDs.
        already_placed:
            Set of already-selected document IDs to skip.

        Returns
        -------
        Any or None
            The top-ranked unplaced document, or ``None`` if all are placed.
        """
        for doc in ranked_list:
            if doc not in already_placed:
                return doc
        return None
