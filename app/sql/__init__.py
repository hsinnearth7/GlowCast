"""GlowCast SQL analysis pipeline — SQLite-backed analytical query engine.

Provides the SQLPipelineExecutor class for running star-schema analytical
queries over in-memory SQLite databases loaded from pandas DataFrames.

Pipeline modules
----------------
dos_woc            : Cost Variance Analysis (Plant × Category)
scrap_risk         : Should-Cost Gap Analysis
cross_zone_penalty : Supplier Performance Analysis
demand_anomaly     : Cost Anomaly Detection (Z-score)
"""

from app.sql.executor import SQLPipelineExecutor

__all__ = ["SQLPipelineExecutor"]
