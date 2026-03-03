"""GlowCast SQL analysis pipeline — SQLite-backed analytical query engine.

Provides the SQLPipelineExecutor class for running star-schema analytical
queries over in-memory SQLite databases loaded from pandas DataFrames.

Pipeline modules
----------------
dos_woc          : Dynamic Days of Supply / Weeks of Coverage
scrap_risk        : FIFO shelf-life scrap risk matrix
cross_zone_penalty: Cross-zone fulfillment penalty analysis
demand_anomaly    : Rolling Z-score demand anomaly detection
social_lead_lag   : Social signal cross-correlation lead/lag analysis
"""

from app.sql.executor import SQLPipelineExecutor

__all__ = ["SQLPipelineExecutor"]
