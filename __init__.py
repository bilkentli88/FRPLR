"""Auditable causal-factorization utilities for the TACL expansion."""

from .determinism import stable_int, stable_rng
from .geometry import (
    Condition,
    FactorialPrompt,
    UserEncoding,
    assemble_factorial,
    fractional_local_positions,
)

__all__ = [
    "Condition",
    "FactorialPrompt",
    "UserEncoding",
    "assemble_factorial",
    "fractional_local_positions",
    "stable_int",
    "stable_rng",
]

__version__ = "0.5.1"
