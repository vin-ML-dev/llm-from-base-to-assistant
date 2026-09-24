"""Tiny stats helper for Day 5 -- just the Wilson interval, nothing else."""
from __future__ import annotations


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Better than the naive +/-
    (Wald) interval for the small sample sizes an eval like this produces.
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)
