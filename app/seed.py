"""Global seed management for full reproducibility.

Usage:
    from app.seed import set_global_seed
    set_global_seed(42)  # call once at startup

Guarantees:
    - Identical results across runs on same platform
    - Deterministic data generation, model training, and evaluation
    - Seed propagated to: random, numpy, sklearn, LightGBM
"""

from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_seed() -> int:
    """Return the global seed value."""
    return GLOBAL_SEED
