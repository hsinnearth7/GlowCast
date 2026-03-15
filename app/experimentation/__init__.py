"""GlowCast experimentation framework.

Provides a production-quality A/B testing and experimentation toolkit for the
GlowCast cost & commercial analytics platform.

Modules
-------
cuped
    CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
    Implements Y_cuped = Y - theta*(X - E[X]) with bootstrap CI validation.
    Reference: Deng et al., WSDM 2013.

sequential
    Always-valid inference via mixture Sequential Probability Ratio Test (mSPRT).
    Supports continuous monitoring without inflating Type-I error.
    Reference: Johari et al., 2017.

interleaving
    Team Draft interleaving for online ranking comparison.
    Efficient head-to-head evaluation of recommendation/ranking models.
    Reference: Chapelle et al., 2012.

power
    Sample size and MDE calculations with CUPED-adjusted estimates.
    Produces MDE tables at 3 %, 5 %, and 10 % effect sizes.

bucketing
    Deterministic SHA-256 hash bucketing with Sample-Ratio Mismatch (SRM)
    detection via chi-squared test and layer-isolation support.

Example
-------
>>> from app.experimentation.bucketing import BucketingAssigner
>>> from app.experimentation.power import PowerAnalyzer
>>> from app.experimentation.cuped import CUPEDAnalyzer
>>> from app.experimentation.sequential import SequentialTester
>>> from app.experimentation.interleaving import InterleavingAnalyzer
"""

from __future__ import annotations

from app.experimentation.bucketing import BucketingAssigner
from app.experimentation.cuped import CUPEDAnalyzer
from app.experimentation.interleaving import InterleavingAnalyzer
from app.experimentation.power import PowerAnalyzer
from app.experimentation.sequential import SequentialTester

__all__ = [
    "CUPEDAnalyzer",
    "SequentialTester",
    "InterleavingAnalyzer",
    "PowerAnalyzer",
    "BucketingAssigner",
]
