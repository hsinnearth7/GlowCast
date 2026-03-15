"""SQLPipelineExecutor — loads star-schema DataFrames into SQLite and runs analytical SQL pipelines.

Design principles
-----------------
- In-memory SQLite via sqlite3 + pandas to_sql; no external DB dependency.
- SQL files co-located in the same directory as this module (Path(__file__).parent).
- Each public method is idempotent: safe to call multiple times.
- EXPLAIN QUERY PLAN executed via explain_analyze(); results are returned as a
  human-readable string (SQLite does not support EXPLAIN ANALYZE, so we use
  EXPLAIN QUERY PLAN which describes the chosen access strategy without executing).
- Thread-safety: each call to run_pipeline / explain_analyze creates a fresh
  connection so the executor is safe for concurrent use in async contexts.

Usage
-----
    from app.sql.executor import SQLPipelineExecutor

    tables = {
        "dim_product":              dim_product_df,
        "dim_plant":                dim_plant_df,
        "dim_supplier":             dim_supplier_df,
        "fact_cost_transactions":    fact_cost_transactions_df,
        "fact_purchase_orders":      fact_purchase_orders_df,
        "fact_quality_events":       fact_quality_events_df,
    }
    executor = SQLPipelineExecutor(tables)
    executor.load_tables()

    dos_df = executor.run_pipeline("dos_woc.sql")
    all_results = executor.run_all_pipelines()
    plan_text = executor.explain_analyze("demand_anomaly.sql")
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Directory that contains this module and all co-located .sql files
_SQL_DIR: Path = Path(__file__).parent

# Ordered list of the five analytical pipeline SQL files shipped with GlowCast
PIPELINE_FILES: tuple[str, ...] = (
    "dos_woc.sql",
    "scrap_risk.sql",
    "cross_zone_penalty.sql",
    "demand_anomaly.sql",
)


class SQLPipelineExecutor:
    """Execute GlowCast analytical SQL pipelines over an in-memory SQLite database.

    Parameters
    ----------
    tables:
        Mapping of SQLite table name → pandas DataFrame.  Every DataFrame is
        loaded verbatim; column names must match the column references used in
        the .sql files.

    Attributes
    ----------
    _tables:
        The original DataFrames supplied at construction time.
    _conn:
        The persistent SQLite connection used for all load / query operations.
        Kept open for the lifetime of the executor to avoid repeated setup cost.
    _loaded:
        Whether load_tables() has been called successfully.
    """

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        if not isinstance(tables, dict):
            raise TypeError(f"tables must be a dict[str, pd.DataFrame], got {type(tables).__name__}")
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"tables['{name}'] must be a pd.DataFrame, got {type(df).__name__}")

        self._tables: dict[str, pd.DataFrame] = tables
        self._conn: sqlite3.Connection = self._make_connection()
        self._loaded: bool = False

    # ── Connection helpers ────────────────────────────────────────────────

    @staticmethod
    def _make_connection() -> sqlite3.Connection:
        """Create an in-memory SQLite connection with performance pragmas applied."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous  = OFF;")
        conn.execute("PRAGMA temp_store   = MEMORY;")
        conn.execute("PRAGMA cache_size   = -65536;")  # 64 MB page cache
        conn.row_factory = sqlite3.Row
        return conn

    def _fresh_connection(self) -> sqlite3.Connection:
        """Return the shared connection (tables must already be loaded).

        A single shared connection is used because :memory: databases are not
        shared across sqlite3 connections.
        """
        if not self._loaded:
            raise RuntimeError("Call load_tables() before running queries.")
        return self._conn

    # ── Public API ────────────────────────────────────────────────────────

    def load_tables(self) -> None:
        """Load all DataFrames into the in-memory SQLite database.

        Converts datetime columns to ISO-8601 strings (SQLite has no native
        DATETIME type) so that date arithmetic with strftime / julianday works
        correctly.  Existing tables are replaced on each call.

        Raises
        ------
        ValueError
            If any supplied table name is empty or None.
        """
        t0 = time.perf_counter()

        for table_name, df in self._tables.items():
            if not table_name:
                raise ValueError("Table name must be a non-empty string.")

            # SQLite has no native datetime type; cast to ISO strings so that
            # strftime() and julianday() work transparently in queries.
            df_to_load = df.copy()
            for col in df_to_load.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
                df_to_load[col] = df_to_load[col].dt.strftime("%Y-%m-%d")

            df_to_load.to_sql(
                name=table_name,
                con=self._conn,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=10_000,
            )
            logger.debug("Loaded table '%s' (%d rows, %d cols).", table_name, len(df), len(df.columns))

        elapsed = time.perf_counter() - t0
        self._loaded = True
        logger.info("load_tables() completed in %.3f s (%d tables).", elapsed, len(self._tables))

    def run_pipeline(self, sql_file: str) -> pd.DataFrame:
        """Execute a single SQL pipeline file and return results as a DataFrame.

        Parameters
        ----------
        sql_file:
            Filename of the SQL script (e.g. ``"dos_woc.sql"``).  The file is
            resolved relative to the same directory as this module.

        Returns
        -------
        pd.DataFrame
            Query results.  Returns an empty DataFrame (with columns) if the
            query produces no rows.

        Raises
        ------
        FileNotFoundError
            If the SQL file does not exist.
        RuntimeError
            If load_tables() has not been called.
        sqlite3.OperationalError
            If the SQL itself is malformed.
        """
        sql_path = self._resolve_sql_path(sql_file)
        sql_text = self._read_sql(sql_path)
        conn = self._fresh_connection()

        t0 = time.perf_counter()
        try:
            result = pd.read_sql_query(sql_text, conn)
        except sqlite3.OperationalError as exc:
            logger.error("SQL error in '%s': %s", sql_file, exc)
            raise
        elapsed = time.perf_counter() - t0
        logger.info("run_pipeline('%s') → %d rows in %.3f s.", sql_file, len(result), elapsed)
        return result

    def run_all_pipelines(self) -> dict[str, pd.DataFrame]:
        """Execute all bundled pipeline SQL files and return a result mapping.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are pipeline names without the ``.sql`` extension
            (e.g. ``"dos_woc"``), values are the resulting DataFrames.
            Pipelines that fail are skipped and logged at ERROR level; the
            remaining results are still returned.
        """
        results: dict[str, pd.DataFrame] = {}
        for sql_file in PIPELINE_FILES:
            pipeline_name = Path(sql_file).stem
            try:
                results[pipeline_name] = self.run_pipeline(sql_file)
            except Exception as exc:  # noqa: BLE001
                logger.error("Pipeline '%s' failed: %s", sql_file, exc)
        return results

    def explain_analyze(self, sql_file: str) -> str:
        """Return the SQLite query plan for a pipeline SQL file.

        SQLite does not implement PostgreSQL-style EXPLAIN ANALYZE (runtime
        statistics).  Instead this method uses ``EXPLAIN QUERY PLAN`` which
        describes the optimizer's chosen access strategy — index vs. full scan,
        join order, use of covering indexes, temporary B-Trees for sorting, etc.

        Parameters
        ----------
        sql_file:
            Filename of the SQL script (e.g. ``"demand_anomaly.sql"``).

        Returns
        -------
        str
            Formatted query plan text suitable for logging or display.

        Raises
        ------
        FileNotFoundError
            If the SQL file does not exist.
        RuntimeError
            If load_tables() has not been called.
        """
        sql_path = self._resolve_sql_path(sql_file)
        sql_text = self._read_sql(sql_path)
        conn = self._fresh_connection()

        plan_sql = f"EXPLAIN QUERY PLAN\n{sql_text}"
        try:
            cursor = conn.execute(plan_sql)
            rows: list[Any] = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            logger.error("EXPLAIN QUERY PLAN failed for '%s': %s", sql_file, exc)
            raise

        # Format the plan rows as a readable table
        header = f"=== EXPLAIN QUERY PLAN: {sql_file} ===\n"
        lines = [header]
        lines.append(f"{'id':>4}  {'parent':>6}  {'notused':>7}  detail")
        lines.append("-" * 80)
        for row in rows:
            row_dict = dict(row)
            lines.append(
                f"{row_dict.get('id', ''):>4}  "
                f"{row_dict.get('parent', ''):>6}  "
                f"{row_dict.get('notused', ''):>7}  "
                f"{row_dict.get('detail', '')}"
            )
        plan_text = "\n".join(lines)
        logger.debug("explain_analyze('%s'):\n%s", sql_file, plan_text)
        return plan_text

    # ── Introspection helpers ─────────────────────────────────────────────

    def table_names(self) -> list[str]:
        """Return the list of tables currently loaded in SQLite."""
        if not self._loaded:
            return []
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        return [row[0] for row in cursor.fetchall()]

    def table_info(self, table_name: str) -> pd.DataFrame:
        """Return column metadata for a loaded table (PRAGMA table_info).

        Returns
        -------
        pd.DataFrame
            Columns: cid, name, type, notnull, dflt_value, pk.
        """
        if not self._loaded:
            raise RuntimeError("Call load_tables() before inspecting tables.")
        return pd.read_sql_query(f"PRAGMA table_info('{table_name}');", self._conn)

    def row_counts(self) -> dict[str, int]:
        """Return row count for every loaded table."""
        if not self._loaded:
            return {}
        counts: dict[str, int] = {}
        for name in self.table_names():
            cursor = self._conn.execute(
                f"SELECT COUNT(*) FROM '{name}';"  # noqa: S608
            )
            counts[name] = cursor.fetchone()[0]
        return counts

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_sql_path(sql_file: str) -> Path:
        """Resolve *sql_file* relative to the sql/ package directory.

        Parameters
        ----------
        sql_file:
            Bare filename such as ``"dos_woc.sql"``.  Absolute paths are
            accepted without modification.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist on disk.
        """
        path = Path(sql_file)
        if not path.is_absolute():
            path = _SQL_DIR / sql_file
        if not path.exists():
            raise FileNotFoundError(
                f"SQL file not found: {path}\n"
                f"Expected directory: {_SQL_DIR}\n"
                f"Available .sql files: {[p.name for p in _SQL_DIR.glob('*.sql')]}"
            )
        return path

    @staticmethod
    def _read_sql(path: Path) -> str:
        """Read and return the SQL text, stripping leading/trailing whitespace."""
        sql = path.read_text(encoding="utf-8").strip()
        if not sql:
            raise ValueError(f"SQL file is empty: {path}")
        return sql

    # ── Resource management ───────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection and release memory."""
        try:
            self._conn.close()
            self._loaded = False
            logger.debug("SQLPipelineExecutor connection closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing SQLite connection: %s", exc)

    def __enter__(self) -> "SQLPipelineExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        table_count = len(self._tables)
        loaded_str = "loaded" if self._loaded else "not loaded"
        return (
            f"SQLPipelineExecutor("
            f"tables={table_count}, "
            f"status={loaded_str})"
        )
