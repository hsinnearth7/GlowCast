"""YAML-based configuration loader for GlowCast.

Loads configs/glowcast.yaml and provides structured access to all configuration values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "glowcast.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config. Defaults to configs/glowcast.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_data_config() -> dict[str, Any]:
    """Get data generation configuration."""
    return load_config().get("data", {})


def get_segment_config() -> dict[str, Any]:
    """Get segment configuration."""
    return load_config().get("segments", {})


def get_cost_config(section: str | None = None) -> dict[str, Any]:
    """Get cost analytics configuration, optionally for a specific section."""
    cost_cfg = load_config().get("cost", {})
    if section:
        return cost_cfg.get(section, {})
    return cost_cfg


def get_pricing_config() -> dict[str, Any]:
    """Get pricing / elasticity configuration."""
    return load_config().get("pricing", {})


def get_experimentation_config() -> dict[str, Any]:
    """Get experimentation / A/B testing configuration."""
    return load_config().get("experimentation", {})


def get_causal_config() -> dict[str, Any]:
    """Get causal inference configuration."""
    return load_config().get("causal", {})


def get_monitoring_config() -> dict[str, Any]:
    """Get monitoring configuration."""
    return load_config().get("monitoring", {})


def get_sql_config() -> dict[str, Any]:
    """Get SQL pipeline configuration."""
    return load_config().get("sql", {})
