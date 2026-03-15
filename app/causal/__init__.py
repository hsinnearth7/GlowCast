"""GlowCast causal inference module.

Provides production-quality causal reasoning tools for the GlowCast cost &
commercial analytics platform.  Two complementary paradigms are implemented:

Modules
-------
dowhy_pipeline
    DoWhyPipeline — end-to-end structural causal model workflow following the
    canonical 4-step DoWhy API: Model DAG → Identify estimand → Estimate ATE
    → Refute with falsification tests.
    Reference: Sharma & Kiciman, "DoWhy: An End-to-End Library for Causal
    Inference", arXiv 2020.

uplift
    UpliftAnalyzer — heterogeneous treatment effect estimation via four
    meta-learner architectures (S-, T-, X-, and Causal Forest).  Designed for
    the imbalanced 20 / 80 treatment / control split typical of GlowCast
    cost-reduction experiments, which is precisely the regime where the X-Learner
    outperforms simpler baselines.
    Reference: Künzel et al., "Metalearners for estimating heterogeneous
    treatment effects using machine learning", PNAS 2019.

Example
-------
>>> import pandas as pd
>>> from app.causal import DoWhyPipeline, UpliftAnalyzer

>>> # --- DoWhy workflow ---
>>> pipe = DoWhyPipeline(
...     treatment="cost_reduction_action",
...     outcome="unit_cost_change",
...     common_causes=["category", "plant_id", "commodity_index"],
... )
>>> results = pipe.run_pipeline(data=df)
>>> print(results["ate"], results["refutations"])

>>> # --- Uplift / CATE workflow ---
>>> analyzer = UpliftAnalyzer()
>>> analyzer.fit(X, treatment, outcome)
>>> cate = analyzer.predict_cate(X_new)
>>> summary = analyzer.ablation_study(X, treatment, outcome)
>>> print(summary)
"""

from __future__ import annotations

from app.causal.dowhy_pipeline import DoWhyPipeline
from app.causal.uplift import UpliftAnalyzer

__all__ = [
    "DoWhyPipeline",
    "UpliftAnalyzer",
]
