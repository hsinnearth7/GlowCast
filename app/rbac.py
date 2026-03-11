"""GlowCast RBAC — Role-Based Access Control for API endpoints.

Roles
-----
- VIEWER: Read-only access to dashboards and forecasts
- ANALYST: Can read + run experiments, view detailed metrics
- ADMIN: Full access including pipeline management and user administration

Permissions
-----------
- READ_DASHBOARD: View Streamlit dashboard
- READ_FORECASTS: Access forecast endpoints
- RUN_EXPERIMENTS: Execute A/B tests and uplift analyses
- RUN_PIPELINE: Trigger training/data pipelines
- MANAGE_USERS: Create/modify user accounts and roles

Usage
-----
    from app.rbac import RBACMiddleware, require_permission, Permission

    # In FastAPI
    app.add_middleware(RBACMiddleware)

    @app.get("/api/pipelines/run")
    async def run_pipeline(user=Depends(require_permission(Permission.RUN_PIPELINE))):
        ...
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """User roles with increasing privilege levels."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class Permission(str, Enum):
    """Granular permissions mapped to API operations."""
    READ_DASHBOARD = "read_dashboard"
    READ_FORECASTS = "read_forecasts"
    RUN_EXPERIMENTS = "run_experiments"
    RUN_PIPELINE = "run_pipeline"
    MANAGE_USERS = "manage_users"


# ---------------------------------------------------------------------------
# Role → Permission mapping
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_DASHBOARD,
        Permission.READ_FORECASTS,
    },
    Role.ANALYST: {
        Permission.READ_DASHBOARD,
        Permission.READ_FORECASTS,
        Permission.RUN_EXPERIMENTS,
    },
    Role.ADMIN: {
        Permission.READ_DASHBOARD,
        Permission.READ_FORECASTS,
        Permission.RUN_EXPERIMENTS,
        Permission.RUN_PIPELINE,
        Permission.MANAGE_USERS,
    },
}

# Endpoint → required permission mapping
ENDPOINT_PERMISSIONS: dict[str, Permission] = {
    "/api/forecasts": Permission.READ_FORECASTS,
    "/api/drift": Permission.READ_FORECASTS,
    "/api/pipelines/run": Permission.RUN_PIPELINE,
    "/api/pipelines/status": Permission.RUN_PIPELINE,
    "/api/experiments": Permission.RUN_EXPERIMENTS,
    "/api/users": Permission.MANAGE_USERS,
}


# ---------------------------------------------------------------------------
# User model (in production, this comes from a database / JWT token)
# ---------------------------------------------------------------------------


class User:
    """Represents an authenticated user with a role.

    In production, user data would be extracted from a JWT token
    or looked up from a database. This is a minimal in-memory implementation.
    """

    def __init__(self, username: str, role: Role, email: str = "") -> None:
        self.username = username
        self.role = role
        self.email = email

    @property
    def permissions(self) -> set[Permission]:
        """Get all permissions for this user's role."""
        return ROLE_PERMISSIONS.get(self.role, set())

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        """Serialize user for logging and responses."""
        return {
            "username": self.username,
            "role": self.role.value,
            "email": self.email,
            "permissions": [p.value for p in self.permissions],
        }


# ---------------------------------------------------------------------------
# In-memory user store (production: database or identity provider)
# ---------------------------------------------------------------------------

_USER_STORE: dict[str, User] = {
    "admin": User("admin", Role.ADMIN, "admin@glowcast.example.com"),
    "analyst": User("analyst", Role.ANALYST, "analyst@glowcast.example.com"),
    "viewer": User("viewer", Role.VIEWER, "viewer@glowcast.example.com"),
}


def get_user_by_api_key(api_key: str) -> User | None:
    """Resolve a user from an API key.

    In production, this would look up the API key in a database
    and return the associated user.  For now, we use a simple
    prefix-based mapping.
    """
    if not api_key:
        return None

    # Check environment variable for admin key
    admin_key = os.environ.get("API_KEY", "")
    if admin_key and api_key == admin_key:
        return _USER_STORE.get("admin")

    # Prefix-based resolution (for development)
    for role_name, user in _USER_STORE.items():
        if api_key.startswith(f"{role_name}-"):
            return user

    return None


def get_user_by_username(username: str) -> User | None:
    """Look up a user by username."""
    return _USER_STORE.get(username)


# ---------------------------------------------------------------------------
# RBAC Middleware
# ---------------------------------------------------------------------------


class RBACMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces role-based access control.

    Extracts the user from the X-API-Key header, resolves their role,
    and checks if they have permission for the requested endpoint.

    Exempt paths (no auth required):
    - /api/health
    - /api/docs
    - /api/openapi.json
    """

    EXEMPT_PATHS = {"/api/health", "/api/docs", "/api/openapi.json", "/api/metrics"}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path

        # Skip auth for exempt paths
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Extract API key
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            # No RBAC enforcement if no API key configured
            env_key = os.environ.get("API_KEY", "")
            if not env_key:
                return await call_next(request)
            raise HTTPException(status_code=401, detail="Missing X-API-Key header")

        # Resolve user
        user = get_user_by_api_key(api_key)
        if user is None:
            logger.warning("Invalid API key attempted: %s***", api_key[:4])
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Check permission for this endpoint
        required_permission = self._resolve_permission(path)
        if required_permission and not user.has_permission(required_permission):
            logger.warning(
                "Permission denied: user=%s role=%s endpoint=%s required=%s",
                user.username,
                user.role.value,
                path,
                required_permission.value,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required_permission.value}",
            )

        # Attach user to request state for downstream handlers
        request.state.user = user

        return await call_next(request)

    @staticmethod
    def _resolve_permission(path: str) -> Permission | None:
        """Find the required permission for a given path."""
        for prefix, perm in ENDPOINT_PERMISSIONS.items():
            if path.startswith(prefix):
                return perm
        return None


# ---------------------------------------------------------------------------
# Dependency for FastAPI route-level permission checks
# ---------------------------------------------------------------------------


def require_permission(permission: Permission):
    """FastAPI dependency that checks for a specific permission.

    Usage::

        @app.post("/api/pipelines/run")
        async def run_pipeline(
            user: User = Depends(require_permission(Permission.RUN_PIPELINE))
        ):
            ...
    """
    async def _check(request: Request) -> User:
        user: User | None = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value} required",
            )
        return user

    return _check
