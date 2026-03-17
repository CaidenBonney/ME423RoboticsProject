from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


G = 9.81  # m/s^2


@dataclass
class InterceptResult:
    valid: bool
    t_hit_ms: Optional[float] = None
    xyz_hit: Optional[np.ndarray] = None  # shape (3,)
    xy_hit: Optional[np.ndarray] = None   # shape (2,)
    confidence: float = 0.0
    reason: str = ""


def solve_quadratic_real(a: float, b: float, c: float) -> np.ndarray:
    """Return all real roots of a*x^2 + b*x + c = 0."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return np.array([], dtype=np.float64)
        return np.array([-c / b], dtype=np.float64)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.array([], dtype=np.float64)

    s = np.sqrt(disc)
    r1 = (-b - s) / (2.0 * a)
    r2 = (-b + s) / (2.0 * a)
    return np.array([r1, r2], dtype=np.float64)


def choose_future_root(roots: np.ndarray, t_now: float, min_lead: float = 0.0) -> Optional[float]:
    """
    Choose the earliest root that lies at least min_lead into the future.
    All units must match (e.g. milliseconds).
    """
    roots = np.asarray(roots, dtype=np.float64)
    valid = roots[np.isfinite(roots) & (roots >= (t_now + min_lead))]
    if valid.size == 0:
        return None
    return float(np.min(valid))


def weighted_polyfit(t: np.ndarray, y: np.ndarray, deg: int, w: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if w is None:
        return np.polyfit(t, y, deg)
    return np.polyfit(t, y, deg, w=np.asarray(w, dtype=np.float64).reshape(-1))


def estimate_time_to_move_xy_ms(current_xy: np.ndarray, target_xy: np.ndarray, v_xy_max_mps: float = 1.0) -> float:
    """
    Crude XY travel-time estimate in milliseconds.
    """
    current_xy = np.asarray(current_xy, dtype=np.float64).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    d = np.linalg.norm(target_xy - current_xy)
    t_s = d / max(v_xy_max_mps, 1e-6)
    return 1000.0 * float(t_s)