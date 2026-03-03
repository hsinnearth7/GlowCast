"""GlowCast explainability module.

Provides model-agnostic explainability and fairness tooling for the GlowCast
beauty & skincare supply chain forecasting platform.  Two complementary
explanation paradigms are implemented alongside a cross-group fairness analyzer.

Modules
-------
shap_lime
    SHAPExplainer — global feature attribution via SHAP values (Lundberg & Lee,
    NeurIPS 2017).  Uses TreeExplainer for tree-based models (LightGBM, XGBoost,
    Random Forest) and falls back to KernelExplainer for arbitrary sklearn
    estimators.  When the ``shap`` package is unavailable, permutation importance
    is substituted automatically.

    LIMEExplainer — local surrogate explanation for individual predictions
    (Ribeiro et al., KDD 2016).  When the ``lime`` package is unavailable a
    simple Gaussian-perturbation fallback is used.

    compare_explanations — side-by-side top-feature comparison across both
    paradigms, useful for sanity-checking and model-card reporting.

fairness
    FairnessAnalyzer — sliced evaluation across fulfillment-center groups and
    SKU categories.  Implements MAPE with 95 % bootstrap CI, Kruskal-Wallis
    test for cross-group differences, chi-squared test for categorical bias,
    and arbitrary segment slicing.

    Target fairness results for GlowCast v2.1.0:
    - Kruskal-Wallis: H = 2.31, p = 0.51  (no significant difference across FCs)
    - Chi-squared:    chi2 = 8.7, p = 0.12 (no significant per-category bias)

Example
-------
>>> import numpy as np
>>> from app.explain import SHAPExplainer, LIMEExplainer, compare_explanations
>>> from app.explain import FairnessAnalyzer

>>> # --- SHAP global importance ---
>>> shap_exp = SHAPExplainer(model=lgbm_model, feature_names=feature_names)
>>> shap_values = shap_exp.compute_shap_values(X_test)
>>> importance_df = shap_exp.feature_importance()

>>> # --- LIME local explanation ---
>>> lime_exp = LIMEExplainer(model=lgbm_model, feature_names=feature_names)
>>> weights = lime_exp.explain_instance(X_test[0])

>>> # --- Side-by-side comparison ---
>>> comparison_df = compare_explanations(shap_exp, lime_exp, X_test[:50])

>>> # --- Fairness evaluation ---
>>> analyzer = FairnessAnalyzer(y_true=y_true, y_pred=y_pred, group_labels=fc_labels)
>>> kw = analyzer.kruskal_wallis_test()
>>> chi = analyzer.chi_squared_test(category_labels=category_labels)

References
----------
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. NeurIPS 2017.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
  Explaining the predictions of any classifier. KDD 2016.
- Mitchell, M., Wu, S., et al. (2019). Model cards for model reporting.
  FAT* 2019.
"""

from __future__ import annotations

from app.explain.fairness import FairnessAnalyzer
from app.explain.shap_lime import LIMEExplainer, SHAPExplainer, compare_explanations

__all__ = [
    "SHAPExplainer",
    "LIMEExplainer",
    "compare_explanations",
    "FairnessAnalyzer",
]
