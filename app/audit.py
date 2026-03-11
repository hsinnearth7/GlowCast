"""GlowCast Audit Logging — structured audit trail for compliance and debugging.

Records all significant API operations with:
- Who (user, role, IP)
- What (action, resource, parameters)
- When (timestamp)
- Outcome (success/failure, status code)

Usage
-----
    from app.audit import AuditLogger, AuditMiddleware

    # Middleware (auto-logs all requests)
    app.add_middleware(AuditMiddleware)

    # Manual logging
    audit = AuditLogger()
    audit.log_action(
        user="admin",
        action="pipeline.run",
        resource="glowcast_training",
        details={"n_skus": 200},
        outcome="success",
    )
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit log storage
# ---------------------------------------------------------------------------

_AUDIT_LOG_DIR = Path.home() / ".glowcast" / "audit"
_MAX_MEMORY_ENTRIES = 10000


class AuditLogger:
    """Structured audit logger with file and in-memory storage.

    Writes NDJSON audit records to ``~/.glowcast/audit/audit.jsonl``
    and maintains a bounded in-memory buffer for recent queries.

    In production, this would write to a dedicated audit database
    (e.g. PostgreSQL audit table, Elasticsearch, or CloudWatch Logs).
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or _AUDIT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "audit.jsonl"
        self._buffer: deque[dict[str, Any]] = deque(maxlen=_MAX_MEMORY_ENTRIES)

    def log_action(
        self,
        user: str,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
        outcome: str = "success",
        ip_address: str = "",
        role: str = "",
        status_code: int | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        """Record an audit event.

        Parameters
        ----------
        user : str
            Username or identifier of the actor.
        action : str
            Action performed (e.g. ``"pipeline.run"``, ``"forecast.read"``).
        resource : str
            Target resource (e.g. ``"glowcast_training"``, ``"SKU_0001"``).
        details : dict
            Additional context (parameters, payload summary, etc.).
        outcome : str
            Result: ``"success"``, ``"failure"``, ``"denied"``.
        ip_address : str
            Client IP address.
        role : str
            User's role at time of action.
        status_code : int
            HTTP response status code.
        duration_ms : float
            Request processing time in milliseconds.

        Returns
        -------
        dict
            The audit record that was logged.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "role": role,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "ip_address": ip_address,
            "details": details or {},
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        # Write to file
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)

        # Buffer in memory
        self._buffer.append(record)

        # Structured log
        logger.info(
            "AUDIT: user=%s action=%s resource=%s outcome=%s",
            user,
            action,
            resource,
            outcome,
        )

        return record

    def query_recent(
        self,
        limit: int = 100,
        user: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query recent audit entries from the in-memory buffer.

        Parameters
        ----------
        limit : int
            Maximum number of entries to return.
        user : str
            Filter by username.
        action : str
            Filter by action prefix.
        outcome : str
            Filter by outcome.

        Returns
        -------
        list[dict]
            Matching audit records, newest first.
        """
        results = []
        for record in reversed(self._buffer):
            if user and record["user"] != user:
                continue
            if action and not record["action"].startswith(action):
                continue
            if outcome and record["outcome"] != outcome:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        if not self._buffer:
            return {"total_entries": 0}

        outcomes: dict[str, int] = {}
        actions: dict[str, int] = {}
        users = set()
        for record in self._buffer:
            outcome = record.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            action = record.get("action", "unknown")
            actions[action] = actions.get(action, 0) + 1
            users.add(record.get("user", "unknown"))

        return {
            "total_entries": len(self._buffer),
            "unique_users": len(users),
            "outcomes": outcomes,
            "top_actions": dict(sorted(actions.items(), key=lambda x: -x[1])[:10]),
        }


# ---------------------------------------------------------------------------
# Global audit logger instance
# ---------------------------------------------------------------------------

_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# ---------------------------------------------------------------------------
# Audit Middleware
# ---------------------------------------------------------------------------


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that automatically logs all API requests.

    Captures:
    - Request method, path, and query parameters
    - Authenticated user (from request.state.user if set by RBACMiddleware)
    - Response status code
    - Processing duration

    Exempt paths are not logged to reduce noise.
    """

    EXEMPT_PATHS = {"/api/health", "/api/metrics", "/api/docs", "/api/openapi.json"}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path

        # Skip noisy operational endpoints
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        start = time.perf_counter()

        # Execute request
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        # Extract user info
        user_obj = getattr(request.state, "user", None)
        username = user_obj.username if user_obj else "anonymous"
        role = user_obj.role.value if user_obj else "none"

        # Determine action from method + path
        action = self._derive_action(request.method, path)

        # Determine outcome
        if response.status_code < 400:
            outcome = "success"
        elif response.status_code == 403:
            outcome = "denied"
        else:
            outcome = "failure"

        # Get client IP
        ip_address = request.client.host if request.client else "unknown"

        # Log the audit event
        audit = get_audit_logger()
        audit.log_action(
            user=username,
            action=action,
            resource=path,
            details={
                "method": request.method,
                "query_params": dict(request.query_params),
            },
            outcome=outcome,
            ip_address=ip_address,
            role=role,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response

    @staticmethod
    def _derive_action(method: str, path: str) -> str:
        """Convert HTTP method + path into an audit action string.

        Examples:
        - GET /api/forecasts/SKU_001 -> "forecast.read"
        - POST /api/pipelines/run -> "pipeline.run"
        - GET /api/drift/status -> "drift.read"
        """
        # Normalize path segments
        segments = [s for s in path.strip("/").split("/") if s and s != "api"]

        if not segments:
            return f"api.{method.lower()}"

        resource = segments[0]

        action_map = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        verb = action_map.get(method, method.lower())

        # Special cases
        if resource == "pipelines" and len(segments) > 1:
            return f"pipeline.{segments[1]}"
        if resource == "forecasts":
            return f"forecast.{verb}"
        if resource == "drift":
            return f"drift.{verb}"
        if resource == "experiments":
            return f"experiment.{verb}"

        return f"{resource}.{verb}"
